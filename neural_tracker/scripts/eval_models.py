"""
Batch evaluation across val sets and models.

Compute behavior metrics on the SAME train.py:behavior_metrics definition
(so numbers are directly comparable to TensorBoard val_eval_metric/*).

Run:
    .venv/bin/python scripts/eval_models.py \\
        --models models/v24_control_safety/best.pth models/v25_corner_recovery_controlled/best.pth \\
        --datasets data/scenarios/val.pkl data/scenarios/val_narrow.pkl
"""
import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, restore_config_from_checkpoint
from dataset import TrackingDataset
from network import TrackingPolicy
from train import behavior_metrics, load_model_state_checked


def evaluate(model, loader, cfg, max_batches=None):
    model.eval()
    totals = {
        'samples': 0,
        'collision_count': 0.0,
        'hard_collision_count': 0.0,
        'obs_err_sum': 0.0,
        'xy_rough_sum': 0.0,
        'omega_smooth_sum': 0.0,
        'omega_margin_sum': 0.0,
        'min_esdf_sum': 0.0,
        'mean_abs_v_sum': 0.0,
        'min_tracking_dist_sum': 0.0,
        'too_close_count': 0.0,
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            esdf = batch['esdf'].to(cfg.DEVICE)
            robot_state = batch['robot_state'].to(cfg.DEVICE)
            target_rel = batch['target_rel'].to(cfg.DEVICE)
            history = batch['history'].to(cfg.DEVICE)
            visibility_ray = batch['visibility_ray'].to(cfg.DEVICE)
            full_robot_state = batch['full_robot_state'].to(cfg.DEVICE)
            target_pos = batch['target_pos'].to(cfg.DEVICE)
            target_vel = batch['target_vel'].to(cfg.DEVICE)
            target_future_pos = batch['target_future_pos'].to(cfg.DEVICE)
            target_future_pos_mask = batch['target_future_pos_mask'].to(cfg.DEVICE)
            global_esdf = batch['global_esdf'].to(cfg.DEVICE)

            init_pose = full_robot_state[:, :3]
            outputs = model(esdf, robot_state, target_rel,
                            history, visibility_ray, init_pose=init_pose)
            m = behavior_metrics(
                outputs, target_pos, init_pose, global_esdf, cfg,
                target_vel=target_vel,
                target_future_pos=target_future_pos,
                target_future_pos_mask=target_future_pos_mask,
            )
            for k in totals:
                totals[k] += m[k]

    n = max(totals['samples'], 1)
    return {
        'n': totals['samples'],
        'collision_rate': totals['collision_count'] / n,
        'hard_collision_rate': totals['hard_collision_count'] / n,
        'obs_err': totals['obs_err_sum'] / n,
        'xy_roughness': totals['xy_rough_sum'] / n,
        'omega_smoothness': totals['omega_smooth_sum'] / n,
        'omega_margin': totals['omega_margin_sum'] / n,
        'min_esdf': totals['min_esdf_sum'] / n,
        'mean_abs_v': totals['mean_abs_v_sum'] / n,
        'min_tracking_dist': totals['min_tracking_dist_sum'] / n,
        'too_close_rate': totals['too_close_count'] / n,
    }


def fmt(m, key, w=8, prec=4):
    v = m[key]
    return f"{v:{w}.{prec}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+',
                        default=['data/scenarios/val.pkl', 'data/scenarios/val_narrow.pkl'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='cap per-dataset batches (smoke test)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args()

    print(f"Datasets:")
    dataset_paths = [path for path in args.datasets if os.path.exists(path)]
    for missing in set(args.datasets) - set(dataset_paths):
        print(f"  [skip] {missing} (missing)")
    if not dataset_paths:
        raise SystemExit("No datasets loaded.")

    results = {}  # results[model_path][dataset_path] = metrics
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"\n[skip] model {model_path} (missing)")
            continue
        print(f"\nLoading model {model_path}")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = restore_config_from_checkpoint(ckpt, Config(), exclude={'DEVICE'})
        if args.device is not None:
            cfg.DEVICE = args.device
        cfg.BATCH_SIZE = args.batch_size
        cfg.NUM_WORKERS = args.num_workers
        loaders = {}
        for ds_path in dataset_paths:
            dataset = TrackingDataset(ds_path, cfg, augment=False)
            loaders[ds_path] = DataLoader(
                dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=cfg.NUM_WORKERS > 0 and cfg.DEVICE == 'cuda',
            )
        model = TrackingPolicy(cfg).to(cfg.DEVICE)
        load_model_state_checked(model, ckpt['model_state_dict'])
        results[model_path] = {}
        for ds_path, loader in loaders.items():
            t0 = time.time()
            m = evaluate(model, loader, cfg, max_batches=args.max_batches)
            dt = time.time() - t0
            print(f"  {os.path.basename(ds_path):20s} n={m['n']:5d}  "
                  f"coll={m['collision_rate']:.4f}  "
                  f"hard={m['hard_collision_rate']:.4f}  "
                  f"obs={m['obs_err']:.3f}  "
                  f"minD={m['min_tracking_dist']:.3f}  "
                  f"close={m['too_close_rate']:.4f}  "
                  f"minE={m['min_esdf']:.3f}  "
                  f"rough={m['xy_roughness']:.4f}  "
                  f"om_sm={m['omega_smoothness']:.3f}  "
                  f"om_mg={m['omega_margin']:.4f}  "
                  f"|v|={m['mean_abs_v']:.3f}  "
                  f"({dt:.1f}s)")
            results[model_path][ds_path] = m

    # Pretty cross-table
    print("\n" + "=" * 78)
    print("Cross-comparison")
    print("=" * 78)
    metrics_keys = [
        ('collision_rate',     'coll',   '.4f'),
        ('hard_collision_rate','hard',   '.4f'),
        ('obs_err',            'obs',    '.3f'),
        ('min_tracking_dist',  'minD',   '.3f'),
        ('too_close_rate',     'close',  '.4f'),
        ('min_esdf',           'minE',   '.3f'),
        ('xy_roughness',       'rough',  '.4f'),
        ('omega_smoothness',   'om_sm',  '.3f'),
    ]
    for key, label, fmtstr in metrics_keys:
        print(f"\n[{label}]")
        header = f"  {'model':45s}"
        for ds in dataset_paths:
            header += f"{os.path.basename(ds):>22s}"
        print(header)
        for model_path in results:
            row = f"  {os.path.basename(os.path.dirname(model_path))+'/'+os.path.basename(model_path):45s}"
            for ds in dataset_paths:
                v = results[model_path][ds][key]
                row += f"{format(v, fmtstr):>22s}"
            print(row)
    print()


if __name__ == '__main__':
    main()
