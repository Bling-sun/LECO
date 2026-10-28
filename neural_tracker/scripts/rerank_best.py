"""用新的 checkpoint_rank 逻辑，在已存盘的 checkpoint 中重新选出 best.pth。

训练时 best 是在线判定的，改了 checkpoint_rank 不会自动重选。本脚本读取每个
存盘 checkpoint 内已记录的 behavior_metrics（验证集实测，无需重跑），按新 rank
选出最优者，覆盖 best.pth，并把结果作为新版本上传到 wandb artifact。
"""
import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from train import checkpoint_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='models/v26_newdata')
    ap.add_argument('--wandb', action='store_true', help='上传新 best 到 wandb')
    ap.add_argument('--wandb-project', default='neural_tracker')
    ap.add_argument('--run-name', default='v26_newdata')
    args = ap.parse_args()

    cfg = Config()
    # 候选：所有 epoch_*.pth（已存盘的定期检查点）以及当前 best.pth
    candidates = sorted(glob.glob(os.path.join(args.model_dir, 'epoch_*.pth')))
    cur_best = os.path.join(args.model_dir, 'best.pth')
    if os.path.exists(cur_best):
        candidates.append(cur_best)

    scored = []
    for path in candidates:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        bm = ck.get('behavior_metrics')
        if bm is None:
            print(f"skip {os.path.basename(path)}: no behavior_metrics")
            continue
        vloss = ck.get('best_val_loss')
        # best_val_loss 在每个 ckpt 里是“截至该点的历史最优”，不是本 epoch 的；
        # rank 的最后一项只在所有前序键都相等时才生效，这里影响极小，沿用即可。
        rank = checkpoint_rank(bm, vloss if vloss is not None else 0.0, cfg)
        scored.append((rank, path, bm, ck.get('epoch')))
        print(f"{os.path.basename(path):16s} epoch={ck.get('epoch')} "
              f"col={bm['collision_rate']:.4f} obs={bm['obs_err']:.4f} "
              f"hard={bm['hard_collision_rate']:.4f} rank0={rank[0]} obs_key={rank[2]:.4f}")

    if not scored:
        print("No scorable checkpoints found.")
        return

    scored.sort(key=lambda x: x[0])
    best_rank, best_path, best_bm, best_epoch = scored[0]
    print(f"\n==> 新 best: {os.path.basename(best_path)} (epoch={best_epoch}) "
          f"col={best_bm['collision_rate']:.4f} obs={best_bm['obs_err']:.4f}")

    # 更新选中 checkpoint 内的 best 元数据，再写成 best.pth
    ck = torch.load(best_path, map_location='cpu', weights_only=False)
    ck['best_behavior_metrics'] = best_bm
    ck['best_metric_rank'] = best_rank
    if ck.get('behavior_metrics'):
        # best_val_loss 对齐到该 checkpoint 自身（若有更精确来源可在此覆盖）
        pass
    out = os.path.join(args.model_dir, 'best.pth')
    # 先备份旧 best 以防误操作
    if os.path.exists(out) and os.path.abspath(best_path) != os.path.abspath(out):
        shutil.copy2(out, os.path.join(args.model_dir, 'best_prev.pth'))
    torch.save(ck, out)
    print(f"Wrote {out} (备份旧 best 为 best_prev.pth)")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"{args.run_name}_rerank",
                         job_type='rerank', reinit=True)
        run.summary['reranked_best_epoch'] = best_epoch
        for k, v in best_bm.items():
            run.summary[f'best/{k}'] = v
        art = wandb.Artifact(args.run_name, type='model',
                             description='best re-selected by v26 obs_err-priority rank')
        for f in sorted(glob.glob(os.path.join(args.model_dir, '*.pth'))):
            art.add_file(f, name=os.path.basename(f))
        run.log_artifact(art)
        run.finish()
        print(f"wandb updated: {run.url}")


if __name__ == '__main__':
    main()
