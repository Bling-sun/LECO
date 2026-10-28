"""End-to-end direct-control tracker training.

Expert command/plan imitation is the primary supervised objective; geometric,
collision, visibility, feasibility, and smoothness losses shape safety and
closed-loop behavior around those demonstrations.
"""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    # W&B-only environments (e.g. the lightweight deployment venv) may not
    # ship TensorBoard. Keep training usable; W&B logging remains enabled.
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
from tqdm import tqdm
import numpy as np

from config import (
    Config,
    apply_g1_config,
    apply_strategy_config,
    config_to_dict,
    get_loss_weights,
    get_eval_loss_weights,
    restore_config_from_checkpoint,
    validate_config,
)
from dataset import TrackingDataset
from network import TrackingPolicy
from loss import TotalLoss

try:
    import wandb
except ImportError:
    wandb = None

#### ============================ 核心工具函数与数据适配 ============================

SUMMARY_KEYS = (
    'anchor', 'pos', 'rot', 'angle', 'dist', 'moving_dist', 'min_tracking_dist',
    'nonholo', 'lateral',
    'curvature', 'collision', 'visibility', 'feasibility', 'uniform',
    'anchor_entropy', 'smooth', 'control_smooth', 'omega_margin',
    'clearance_recovery', 'corner', 'corridor_center', 'progress_speed', 'expert_control',
    'expert_cmd_seq', 'expert_plan', 'consistency'
)
CHECKPOINT_RANK_VERSION = 2


def rescale_control_output_head(model, source_max_vel, source_max_omega,
                                target_max_vel, target_max_omega):
    """Approximately preserve physical commands when output limits change.

    The direct-control head emits ``limit * tanh(logit)``.  Legacy v43 and
    earlier checkpoints used 2.0 m/s and 1.2 rad/s; loading those logits under
    5.6/4.0 limits would otherwise multiply the initial commands by roughly
    three.  Scaling each final-layer logit row by the old/new limit ratio is
    exact in tanh's linear region and gives fine-tuning a stable starting point.
    """
    source = (float(source_max_vel), float(source_max_omega))
    target = (float(target_max_vel), float(target_max_omega))
    if min(*source, *target) <= 0.0:
        raise ValueError('control output limits must be positive')

    final_layer = model.control_head.control_head[-1]
    if not isinstance(final_layer, nn.Linear) or final_layer.out_features != 2:
        raise TypeError('expected a two-output linear control head')
    ratios = final_layer.weight.new_tensor([
        source[0] / target[0], source[1] / target[1]
    ])
    with torch.no_grad():
        final_layer.weight.mul_(ratios[:, None])
        if final_layer.bias is not None:
            final_layer.bias.mul_(ratios)
    return tuple(float(value) for value in ratios)


# 计算损失权重
def effective_loss_weights(weights):
    effective = dict(weights)
    effective['pos'] = weights.get('pos', 0.0) * weights.get('imitate', 0.0)
    effective['rot'] = weights.get('rot', 0.0) * weights.get('imitate', 0.0)
    return effective


# 格式化损失贡献
def format_loss_contributors(losses, weights, limit=8):
    """格式化输出对总 Loss 贡献最大的前几个子项（降序排列），便于 Debug 观察哪个 Loss 在起主导作用"""
    effective = effective_loss_weights(weights)
    items = []
    for key in SUMMARY_KEYS:
        if key not in losses:
            continue
        raw = float(losses[key])
        weight = float(effective.get(key, 0.0))
        weighted = raw * weight
        if weight > 0 or raw > 0:
            items.append((abs(weighted), key, raw, weight, weighted))

    items.sort(reverse=True)
    parts = [
        f"{key}={weighted:.3g} ({raw:.3g}x{weight:.3g})"
        for _, key, raw, weight, weighted in items[:limit]
    ]
    return ", ".join(parts) if parts else "none"


# 适配目标维度
def adapt_state_dict_for_target_dim(model, state_dict):
    """
    Load old checkpoints after TARGET_DIM changes.

    v26 and earlier used target_rel dim=5. New models use dim=6 with target
    speed appended. The first StateEncoder Linear therefore gains one input
    column. Copy the old columns and initialize any new columns to zero so old
    checkpoints keep the same behavior until fine-tuned.
    """
    """
    当修改了配置中的模型输入维度（如 TARGET_DIM 从 5 维扩展到 6 维增加目标速度）时，
    此函数能让模型无缝加载旧的 Checkpoint。
    它将旧权重的列复制到新权重的前几列，并将新出现的列初始化为 0，确保老模型在微调（Fine-tune）前行为一致。
    """
    model_state = model.state_dict()
    key = 'state_encoder.mlp.0.weight'
    if key not in state_dict or key not in model_state:
        return state_dict

    old_weight = state_dict[key]
    new_weight = model_state[key]
    if old_weight.shape == new_weight.shape:
        return state_dict
    if old_weight.ndim != 2 or new_weight.ndim != 2 or old_weight.shape[0] != new_weight.shape[0]:
        return state_dict

    adapted = dict(state_dict)
    merged = new_weight.clone()
    cols = min(old_weight.shape[1], new_weight.shape[1])
    merged[:, :cols] = old_weight[:, :cols]
    if new_weight.shape[1] > old_weight.shape[1]:
        merged[:, old_weight.shape[1]:] = 0.0
    adapted[key] = merged
    print(
        f"Adapted {key}: checkpoint {tuple(old_weight.shape)} -> model {tuple(new_weight.shape)}"
    )
    return adapted


def load_model_state_checked(model, state_dict):
    """Allow the documented target-dimension adaptation, but no silent gaps."""
    state_dict = adapt_state_dict_for_target_dim(model, state_dict)
    model.load_state_dict(state_dict, strict=True)

#### ============================ 行为指标评估模块（Behavior Metrics）===========================
# 评估轨迹积分
def integrate_eval_controls(init_pose, controls, cfg):
    """按部署语义积分 eval 控制量；eval forward 已经做 hard clamp。"""
    traj = [init_pose]
    for t in range(controls.shape[1]):
        prev = traj[-1]
        v = controls[:, t, 0]
        omega = controls[:, t, 1]
        yaw = prev[:, 2]
        next_x = prev[:, 0] + cfg.DT * v * torch.cos(yaw)
        next_y = prev[:, 1] + cfg.DT * v * torch.sin(yaw)
        next_yaw = torch.atan2(
            torch.sin(prev[:, 2] + cfg.DT * omega),
            torch.cos(prev[:, 2] + cfg.DT * omega)
        )
        traj.append(torch.stack([next_x, next_y, next_yaw], dim=-1))
    return torch.stack(traj, dim=1)


# 核心物理指标计算
def behavior_metrics(outputs, target_pos, init_pose, global_esdf, cfg, target_vel=None,
                     expert_cmd_seq=None, expert_cmd_seq_mask=None,
                     target_future_pos=None, target_future_pos_mask=None):
    controls = outputs['controls']
    exec_positions = integrate_eval_controls(init_pose, controls, cfg)
    B = exec_positions.shape[0]

    # Expert-command match: the primary quality signal for imitation of a
    # moving-target tracker. First step matters most for receding-horizon
    # deployment, so weight it higher (mirrors ExpertCmdSeqLoss.first_step_weight).
    # 1.专家指令匹配误差
    expert_cmd_err_sum = 0.0
    expert_cmd_first_err_sum = 0.0
    expert_cmd_count = 0.0
    expert_cmd_first_count = 0.0
    if expert_cmd_seq is not None:
        horizon = min(controls.shape[1], expert_cmd_seq.shape[1])
        pred = controls[:, :horizon]
        exp = expert_cmd_seq[:, :horizon].clone()
        exp[:, :, 0] = exp[:, :, 0].clamp(-cfg.MAX_VEL, cfg.MAX_VEL)
        exp[:, :, 1] = exp[:, :, 1].clamp(-cfg.MAX_OMEGA, cfg.MAX_OMEGA)
        err_v = torch.abs(pred[:, :, 0] - exp[:, :, 0]) / max(cfg.MAX_VEL, 1e-3)
        err_w = torch.abs(pred[:, :, 1] - exp[:, :, 1]) / max(cfg.MAX_OMEGA, 1e-3)
        step_err = 0.5 * (err_v + err_w)  # (B, horizon)
        if expert_cmd_seq_mask is not None:
            m = expert_cmd_seq_mask[:, :horizon].float()
            denom = m.sum(dim=1).clamp_min(1.0)
            per_sample = (step_err * m).sum(dim=1) / denom
            sample_valid = m.sum(dim=1) > 0
            first_valid = m[:, 0] > 0
        else:
            per_sample = step_err.mean(dim=1)
            sample_valid = torch.ones(B, dtype=torch.bool, device=controls.device)
            first_valid = sample_valid
        expert_cmd_err_sum = per_sample[sample_valid].sum().item()
        expert_cmd_first_err_sum = step_err[:, 0][first_valid].sum().item()
        expert_cmd_count = sample_valid.float().sum().item()
        expert_cmd_first_count = first_valid.float().sum().item()

    dist = torch.norm(exec_positions[:, -1, :2] - target_pos, dim=-1)
    obs_err = torch.abs(dist - cfg.OBSERVATION_DISTANCE)

    # Moving-target tracking distance: the honest metric for a moving target.
    # Averaged over the near horizon against a constant-velocity target, so it
    # reflects "how well OBSERVATION_DISTANCE is held while following", not the
    # misleading static end-point distance.
    # 2.静态与动态目标跟踪距离误差
    n_steps = exec_positions.shape[1] - 1
    horizon_m = min(10, n_steps)
    if target_vel is not None:
        t_ext = torch.arange(1, horizon_m + 1, device=exec_positions.device,
                             dtype=exec_positions.dtype) * cfg.DT
        tgt_ext = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * t_ext.view(1, horizon_m, 1)
    else:
        tgt_ext = target_pos.unsqueeze(1).expand(-1, horizon_m, -1)
    if target_future_pos is not None:
        future_horizon = min(horizon_m, target_future_pos.shape[1])
        tgt_ext = tgt_ext.clone()
        if target_future_pos_mask is None:
            tgt_ext[:, :future_horizon] = target_future_pos[:, :future_horizon]
        else:
            valid_future = target_future_pos_mask[:, :future_horizon].bool().unsqueeze(-1)
            tgt_ext[:, :future_horizon] = torch.where(
                valid_future,
                target_future_pos[:, :future_horizon],
                tgt_ext[:, :future_horizon],
            )
    moving_dist = torch.norm(exec_positions[:, 1:horizon_m + 1, :2] - tgt_ext, dim=-1)
    moving_obs_err = torch.abs(moving_dist - cfg.OBSERVATION_DISTANCE).mean(dim=1)
    min_tracking_dist = moving_dist.min(dim=1).values
    too_close = min_tracking_dist < cfg.MIN_TRACKING_DISTANCE

    H, W = global_esdf.shape[1:]

    # Occlusion: fraction of near-horizon steps where the tracker->target line of
    # sight is blocked by an obstacle (mid-ray ESDF < 0). This is the direct
    # measure of "keeps the target visible around corners" — the problem-2
    # failure of losing the target at sharp walls shows up here.
    # 3.视线遮挡率计算
    K = cfg.VISIBILITY_SAMPLES
    a_los = torch.linspace(0.0, 1.0, K + 2, device=exec_positions.device)[1:-1].view(1, 1, K, 1)
    rob_los = exec_positions[:, 1:horizon_m + 1, :2]           # (B, hm, 2)
    tgt_los = tgt_ext                                          # (B, hm, 2)
    ray_los = rob_los.unsqueeze(2) * (1 - a_los) + tgt_los.unsqueeze(2) * a_los
    los_pts = ray_los.reshape(exec_positions.shape[0], horizon_m * K, 2)
    lpx = torch.floor((los_pts[:, :, 0] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION).long()
    lpy = torch.floor((los_pts[:, :, 1] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION).long()
    los_in_bounds = (lpx >= 0) & (lpx < W) & (lpy >= 0) & (lpy < H)
    lpx = lpx.clamp(0, W - 1)
    lpy = lpy.clamp(0, H - 1)
    b_idx = torch.arange(exec_positions.shape[0], device=global_esdf.device).unsqueeze(1).expand_as(lpx)
    los_esdf = global_esdf[b_idx, lpy, lpx].masked_fill(~los_in_bounds, -1.0)
    los_esdf = los_esdf.reshape(exec_positions.shape[0], horizon_m, K)
    occluded = (los_esdf.min(dim=-1).values <= 0.0).float().mean(dim=1)  # (B,) frac of steps

    # Collision: fraction of near-horizon steps where the robot's footprint
    # 4.碰撞检测与安全边界
    # Match CollisionLoss by checking points along each segment, not only the
    # discrete trajectory nodes (thin obstacles can lie between two nodes).
    seg_alpha = torch.linspace(0.0, 1.0, K, device=exec_positions.device).view(1, 1, K, 1)
    seg_pts = (
        exec_positions[:, :-1, :2].unsqueeze(2) * (1.0 - seg_alpha)
        + exec_positions[:, 1:, :2].unsqueeze(2) * seg_alpha
    ).reshape(B, -1, 2)
    px = torch.floor((seg_pts[:, :, 0] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION).long()
    py = torch.floor((seg_pts[:, :, 1] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION).long()
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px.clamp(0, W - 1)
    py = py.clamp(0, H - 1)

    batch_idx = torch.arange(B, device=global_esdf.device).unsqueeze(1).expand_as(px)
    esdf_vals = global_esdf[batch_idx, py, px]
    collision_steps = (esdf_vals < cfg.ROBOT_RADIUS) | ~in_bounds
    collided = collision_steps.any(dim=1)
    esdf_for_min = esdf_vals.masked_fill(~in_bounds, -1.0)
    min_esdf = esdf_for_min.min(dim=1).values
    hard_collided = min_esdf < getattr(cfg, 'RECOVERY_HARD_MARGIN', 0.05)
    
    # 5.轨迹粗糙度与控制平滑度
    dxy = exec_positions[:, 1:, :2] - exec_positions[:, :-1, :2]
    if dxy.shape[1] >= 2:
        xy_rough = torch.norm(dxy[:, 1:, :] - dxy[:, :-1, :], dim=-1).mean(dim=1)
    else:
        xy_rough = torch.zeros(B, device=exec_positions.device)

    omega = controls[:, :, 1]
    v = controls[:, :, 0]
    omega_smooth = omega.std(dim=1, unbiased=False)
    omega_margin = torch.relu(torch.abs(omega) - cfg.MAX_OMEGA * cfg.OMEGA_SOFT_LIMIT_RATIO).mean(dim=1)
    mean_abs_v = torch.abs(v).mean(dim=1)

    # Node clearances remain useful for per-control corner diagnostics.
    node_px = torch.floor(
        (exec_positions[:, :, 0] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION
    ).long()
    node_py = torch.floor(
        (exec_positions[:, :, 1] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION
    ).long()
    node_in_bounds = (node_px >= 0) & (node_px < W) & (node_py >= 0) & (node_py < H)
    node_px = node_px.clamp(0, W - 1)
    node_py = node_py.clamp(0, H - 1)
    node_esdf = global_esdf[batch_idx[:, :node_px.shape[1]], node_py, node_px]
    node_esdf = node_esdf.masked_fill(~node_in_bounds, -1.0)

    # 6b. Corner-cutting: min clearance over TURNING steps (|omega| above a
    # threshold). Directly measures "trajectory painted onto the inside corner"
    # — the higher this is, the wider the robot swings around corners. Steps
    # aligned with controls (esdf_vals[:,1:] are the integrated poses 1..n).
    turn_mask = omega.abs() > (0.3 * cfg.MAX_OMEGA)             # (B, n)
    step_clear = node_esdf[:, 1:1 + omega.shape[1]]            # (B, n) clearance at poses 1..n
    big = torch.full_like(step_clear, 10.0)
    turn_clear = torch.where(turn_mask[:, :step_clear.shape[1]], step_clear, big)
    has_turn = turn_mask[:, :step_clear.shape[1]].any(dim=1)
    corner_clip = torch.where(has_turn, turn_clear.min(dim=1).values,
                              torch.full((B,), float('nan'), device=exec_positions.device))
    
    # 6.速度匹配误差
    speed_match_err = torch.zeros(B, device=exec_positions.device)
    speed_match_mask = torch.zeros(B, dtype=torch.bool, device=exec_positions.device)
    if target_vel is not None:
        target_speed = torch.norm(target_vel, dim=-1)
        target_speed = target_speed.clamp(max=getattr(cfg, 'PROGRESS_MAX_TARGET_V', cfg.MAX_VEL))
        delta0 = target_pos - init_pose[:, :2]
        bearing0 = torch.atan2(delta0[:, 1], delta0[:, 0]) - init_pose[:, 2]
        bearing0 = torch.atan2(torch.sin(bearing0), torch.cos(bearing0))
        safe_margin = getattr(cfg, 'SPEED_MATCH_SAFE_MARGIN', cfg.ROBOT_RADIUS + 0.12)
        speed_match_mask = (
            (target_speed > 0.2)
            & (torch.abs(bearing0) < getattr(cfg, 'PROGRESS_MAX_BEARING', 0.9))
            & (min_esdf > safe_margin)
        )
        horizon = min(10, v.shape[1])
        v_early = v[:, :horizon].mean(dim=1).clamp_min(0.0)
        speed_match_err = torch.abs(v_early - target_speed) / target_speed.clamp_min(0.2)

    return {
        'samples': B,
        'collision_count': collided.float().sum().item(),
        'hard_collision_count': hard_collided.float().sum().item(),
        'obs_err_sum': obs_err.sum().item(),
        'xy_rough_sum': xy_rough.sum().item(),
        'omega_smooth_sum': omega_smooth.sum().item(),
        'omega_margin_sum': omega_margin.sum().item(),
        'min_esdf_sum': min_esdf.sum().item(),
        'mean_abs_v_sum': mean_abs_v.sum().item(),
        'speed_match_err_sum': speed_match_err[speed_match_mask].sum().item(),
        'speed_match_count': speed_match_mask.float().sum().item(),
        'expert_cmd_err_sum': expert_cmd_err_sum,
        'expert_cmd_first_err_sum': expert_cmd_first_err_sum,
        'expert_cmd_count': expert_cmd_count,
        'expert_cmd_first_count': expert_cmd_first_count,
        'moving_obs_err_sum': moving_obs_err.sum().item(),
        'min_tracking_dist_sum': min_tracking_dist.sum().item(),
        'too_close_count': too_close.float().sum().item(),
        'occluded_sum': occluded.sum().item(),
        'corner_clip_sum': torch.nan_to_num(corner_clip, nan=0.0).sum().item(),
        'corner_clip_count': torch.isfinite(corner_clip).float().sum().item(),
    }


def format_behavior_metrics(metrics):
    return (
        f"collision={metrics['collision_rate']:.4f}, "
        f"obs_err={metrics['obs_err']:.4f}, "
        f"xy_rough={metrics['xy_roughness']:.4f}, "
        f"omega_smooth={metrics['omega_smoothness']:.4f}, "
        f"omega_margin={metrics['omega_margin']:.4f}, "
        f"hard_collision={metrics['hard_collision_rate']:.4f}, "
        f"min_esdf={metrics['min_esdf']:.4f}, "
        f"mean|v|={metrics['mean_abs_v']:.4f}, "
        f"moving_obs_err={metrics['moving_obs_err']:.4f}, "
        f"min_tracking_dist={metrics['min_tracking_dist']:.4f}, "
        f"too_close={metrics['too_close_rate']:.4f}, "
        f"occluded={metrics['occluded']:.4f}, "
        f"corner_clip={metrics['corner_clip']:.4f}, "
        f"expert_cmd_err={metrics['expert_cmd_err']:.4f}, "
        f"expert_first_err={metrics['expert_cmd_first_err']:.4f}, "
        f"speed_match_err={metrics['speed_match_err']:.4f}"
    )

# checkpoint 排序选择算法
def checkpoint_rank(metrics, val_losses, cfg):
    """checkpoint 排序键（越小越好）。

    v27c: safety / visibility first. Speed matching is useful only after the
    tracker is safe, visible, feasible, and close to the desired observation
    distance.

    v26: 先用碰撞率硬门槛过滤，门槛内以 obs_err（观察距离误差）为主目标排序。
    旧逻辑用 max(0, obs_err - 0.15) / max(0, 0.35 - min_esdf) 作软惩罚，但实际
    训练中 obs_err 始终 < 0.15、min_esdf 始终 ~1.28，这两项恒为 0，导致排序退化
    成几乎只看 collision_rate 的字典序，会选出训练初期“又稳又笨”（obs_err 极高、
    转向不平滑）的 checkpoint。新逻辑在保证安全（碰撞不超过阈值）的前提下优先
    跟踪精度。
    """
    collision_accept = getattr(cfg, 'BEST_COLLISION_ACCEPT', 0.0182)
    hard_accept = getattr(cfg, 'BEST_HARD_COLLISION_ACCEPT', 0.02)
    collision_gate = 1 if metrics['collision_rate'] > collision_accept else 0
    hard_over = max(0.0, metrics['hard_collision_rate'] - hard_accept)
    too_close_accept = getattr(cfg, 'BEST_TOO_CLOSE_ACCEPT', 0.0)
    too_close_rate = float(metrics.get('too_close_rate', 1.0))
    too_close_gate = 1 if too_close_rate > too_close_accept else 0
    corner_accept = getattr(cfg, 'BEST_CORNER_CLEARANCE_ACCEPT', 0.0)
    corner_value = float(metrics.get('corner_clip', float('nan')))
    corner_gate = (
        1 if np.isfinite(corner_value) and corner_value < corner_accept else 0
    )
    speed_value = float(metrics.get('speed_match_err', float('nan')))
    # Missing observations must never masquerade as a perfect zero error.
    speed_rank = speed_value if np.isfinite(speed_value) else float('inf')
    expert_first = float(metrics.get('expert_cmd_first_err', float('nan')))
    expert_all = float(metrics.get('expert_cmd_err', float('nan')))
    expert_first_rank = expert_first if np.isfinite(expert_first) else float('inf')
    expert_all_rank = expert_all if np.isfinite(expert_all) else float('inf')
    val_total = float(val_losses.get('total', 0.0))
    # Expert-imitation-primary ranking: within the collision safety gate, prefer
    # the checkpoint that best matches the expert command sequence (esp. the
    # first, deployed step) and keeps the target visible around corners.
    # Occlusion is bucketed (0.02 steps) so it dominates only on meaningful
    # differences and does not out-vote first-step command match on noise.
    # Static obs_err is deliberately NOT a primary key — it rewards sitting near
    # the initial target position for a moving target.
    occluded_bucket = round(metrics.get('occluded', 0.0) / 0.02)
    return (
        collision_gate,               # 安全硬门槛：超阈值的一律排在后面
        hard_over,                    # 硬碰撞软约束
        too_close_gate,               # 与目标距离不得低于 MIN_TRACKING_DISTANCE
        too_close_rate,
        corner_gate,                  # 转弯净空不足的一律排后
        occluded_bucket,              # 保持目标可见（过尖角不丢）优先
        speed_rank,
        expert_first_rank,
        expert_all_rank,
        metrics['collision_rate'],
        metrics['hard_collision_rate'],
        -metrics['min_esdf'],         # clearance 越大越好
        metrics['xy_roughness'],
        metrics['omega_smoothness'],
        val_total,
    )

#### ========================== 核心训练器类 ============================
# 初始化与架构重置
class Trainer:
    def __init__(self, config, use_wandb=False, run_name=None, wandb_project='neural_tracker',
                 upload_dataset=False):
        validate_config(config)
        self.cfg = config
        self.device = config.DEVICE
        self.use_wandb = use_wandb and wandb is not None
        if use_wandb and wandb is None:
            print("WARNING: wandb requested but not installed; continuing without it.")

        # 创建目录
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # 加载数据集
        print("Loading datasets...")
        train_path = os.path.join(config.DATA_DIR, 'train.pkl')
        val_path = os.path.join(config.DATA_DIR, 'val.pkl')

        train_sets = [TrackingDataset(train_path, config, augment=True)]
        val_sets = [TrackingDataset(val_path, config, augment=False)]
        narrow_train_path = os.path.join(config.DATA_DIR, 'narrow_train.pkl')
        narrow_val_path = os.path.join(config.DATA_DIR, 'narrow_val.pkl')
        if getattr(config, 'INCLUDE_NARROW_DATA', False):
            if not os.path.isfile(narrow_train_path) or not os.path.isfile(narrow_val_path):
                raise FileNotFoundError(
                    'G1 narrow data requested but narrow_train.pkl/narrow_val.pkl are missing '
                    f'in {config.DATA_DIR}'
                )
            narrow_train = TrackingDataset(narrow_train_path, config, augment=True)
            narrow_val = TrackingDataset(narrow_val_path, config, augment=False)
            unrelabeled = sum(
                sample.get('g1_expert_relabelled') is not True
                for sample in list(narrow_train.samples) + list(narrow_val.samples)
            )
            if unrelabeled:
                raise ValueError(
                    f'{unrelabeled} narrow G1 samples are not marked '
                    'g1_expert_relabelled=True; regenerate their expert_cmd_seq '
                    'and expert_plan with the corrected A*+MPC policy before '
                    'using --include-narrow'
                )
            train_sets.append(narrow_train)
            val_sets.append(narrow_val)
        self.train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        self.val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

        history_samples = sum(
            1 for dataset in train_sets for sample in dataset.samples
            if sample.get('control_history', sample.get('history')) is not None
        )
        total_train_samples = sum(len(dataset.samples) for dataset in train_sets)
        config.USE_CONTROL_HISTORY = (
            total_train_samples > 0 and history_samples == total_train_samples
        )
        if 0 < history_samples < total_train_samples:
            print(
                f"WARNING: control history is only present in {history_samples}/"
                f"{total_train_samples} training samples; deployment history remains disabled."
            )
        elif config.USE_CONTROL_HISTORY:
            print("Control history enabled: all training samples provide past commands.")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batches per epoch: {len(self.train_loader)}")

        # 创建模型
        print("\nInitializing model...")
        self.model = TrackingPolicy(config).to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

        # 损失函数
        self.criterion = TotalLoss(config).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.LR_MILESTONES,
            gamma=config.LR_GAMMA
        )

        # TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)

        # Weights & Biases (optional, real-time sync)
        self.wandb_run = None
        if self.use_wandb:
            cfg_dict = {
                k: getattr(config, k) for k in dir(config)
                if not k.startswith('__') and not callable(getattr(config, k))
            }
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=cfg_dict,
                job_type='train',
                reinit=True,
            )
            print(f"wandb run: {self.wandb_run.url}")
            if upload_dataset:
                self.upload_dataset_artifact()

        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metric_rank = None
        self.best_behavior_metrics = None
        self.global_step = 0


    # 训练单步循环
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()

        # 获取当前 epoch 的损失权重（课程学习）
        weights = get_loss_weights(epoch, self.cfg)

        epoch_losses = {key: 0.0 for key in weights.keys()}
        epoch_losses['total'] = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.NUM_EPOCHS}",
            disable=getattr(self.cfg, 'DISABLE_TQDM', False)
        )

        max_batches = getattr(self.cfg, 'MAX_TRAIN_BATCHES', None)
        processed_batches = 0
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # 数据移到 GPU
            esdf = batch['esdf'].to(self.device)
            robot_state = batch['robot_state'].to(self.device)
            target_rel = batch['target_rel'].to(self.device)
            history = batch['history'].to(self.device)
            visibility_ray = batch['visibility_ray'].to(self.device)
            full_robot_state = batch['full_robot_state'].to(self.device)
            target_pos = batch['target_pos'].to(self.device)
            target_vel = batch['target_vel'].to(self.device)
            target_future_pos = batch['target_future_pos'].to(self.device)
            target_future_pos_mask = batch['target_future_pos_mask'].to(self.device)
            global_esdf = batch['global_esdf'].to(self.device)
            expert_control = batch['expert_control'].to(self.device)
            expert_cmd_seq = batch['expert_cmd_seq'].to(self.device)
            expert_cmd_seq_mask = batch['expert_cmd_seq_mask'].to(self.device)
            expert_plan = batch['expert_plan'].to(self.device)
            expert_plan_mask = batch['expert_plan_mask'].to(self.device)
            sample_weight = batch.get('sample_weight')
            if sample_weight is not None:
                sample_weight = sample_weight.to(self.device)

            init_pose = full_robot_state[:, :3]

            # 前向传播
            outputs = self.model(esdf, robot_state, target_rel,
                                 history, visibility_ray, init_pose=init_pose)

            # 计算损失（替换 batch 里的 tensor 为 GPU 版本）
            batch_gpu = {
                'target_pos': target_pos,
                'target_vel': target_vel,
                'target_future_pos': target_future_pos,
                'target_future_pos_mask': target_future_pos_mask,
                'global_esdf': global_esdf,
                'init_pose': init_pose,
                'current_control': robot_state,
                'expert_control': expert_control,
                'expert_cmd_seq': expert_cmd_seq,
                'expert_cmd_seq_mask': expert_cmd_seq_mask,
                'expert_plan': expert_plan,
                'expert_plan_mask': expert_plan_mask,
                'sample_weight': sample_weight,
            }
            total_loss, loss_dict = self.criterion(
                outputs, batch_gpu, weights, prev_controls=None
            )

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录损失
            epoch_losses['total'] += total_loss.item()
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()

            # 更新进度条
            if batch_idx % self.cfg.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

                # TensorBoard 记录
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}_loss', value.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.wandb_run is not None:
                    log_dict = {'train/total_loss': total_loss.item(),
                                'train/lr': self.optimizer.param_groups[0]['lr']}
                    for key, value in loss_dict.items():
                        log_dict[f'train/{key}_loss'] = value.item()
                    self.wandb_run.log(log_dict, step=self.global_step)

            # 保存当前控制用于下一帧的一致性 loss
            # 注意：跨 batch 的一致性已废弃（shuffle 后 batch 间无时序关系）
            # consistency loss 现在在 loss.py 内部对 controls[:, 1:] vs [:, :-1] 做时序平滑

            self.global_step += 1
            processed_batches += 1

        # 计算平均损失
        num_batches = max(processed_batches, 1)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    # 验证与保存检查点
    @torch.no_grad()
    def validate(self, epoch, weights=None, tag='val'):
        """验证"""
        self.model.eval()

        if weights is None:
            weights = get_eval_loss_weights(self.cfg)
        val_losses = {key: 0.0 for key in weights.keys()}
        val_losses['total'] = 0.0
        metric_totals = {
            'samples': 0,
            'collision_count': 0.0,
            'hard_collision_count': 0.0,
            'obs_err_sum': 0.0,
            'xy_rough_sum': 0.0,
            'omega_smooth_sum': 0.0,
            'omega_margin_sum': 0.0,
            'min_esdf_sum': 0.0,
            'mean_abs_v_sum': 0.0,
            'speed_match_err_sum': 0.0,
            'speed_match_count': 0.0,
            'expert_cmd_err_sum': 0.0,
            'expert_cmd_first_err_sum': 0.0,
            'expert_cmd_count': 0.0,
            'expert_cmd_first_count': 0.0,
            'moving_obs_err_sum': 0.0,
            'min_tracking_dist_sum': 0.0,
            'too_close_count': 0.0,
            'occluded_sum': 0.0,
            'corner_clip_sum': 0.0,
            'corner_clip_count': 0.0,
        }

        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            disable=getattr(self.cfg, 'DISABLE_TQDM', False)
        )

        max_batches = getattr(self.cfg, 'MAX_VAL_BATCHES', None)
        processed_batches = 0
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # 数据移到 GPU
            esdf = batch['esdf'].to(self.device)
            robot_state = batch['robot_state'].to(self.device)
            target_rel = batch['target_rel'].to(self.device)
            history = batch['history'].to(self.device)
            visibility_ray = batch['visibility_ray'].to(self.device)
            full_robot_state = batch['full_robot_state'].to(self.device)
            target_pos = batch['target_pos'].to(self.device)
            target_vel = batch['target_vel'].to(self.device)
            target_future_pos = batch['target_future_pos'].to(self.device)
            target_future_pos_mask = batch['target_future_pos_mask'].to(self.device)
            global_esdf = batch['global_esdf'].to(self.device)
            expert_control = batch['expert_control'].to(self.device)
            expert_cmd_seq = batch['expert_cmd_seq'].to(self.device)
            expert_cmd_seq_mask = batch['expert_cmd_seq_mask'].to(self.device)
            expert_plan = batch['expert_plan'].to(self.device)
            expert_plan_mask = batch['expert_plan_mask'].to(self.device)
            sample_weight = batch.get('sample_weight')
            if sample_weight is not None:
                sample_weight = sample_weight.to(self.device)

            init_pose = full_robot_state[:, :3]

            # 前向传播
            outputs = self.model(esdf, robot_state, target_rel,
                                 history, visibility_ray, init_pose=init_pose)

            batch_gpu = {
                'target_pos': target_pos,
                'target_vel': target_vel,
                'target_future_pos': target_future_pos,
                'target_future_pos_mask': target_future_pos_mask,
                'global_esdf': global_esdf,
                'init_pose': init_pose,
                'current_control': robot_state,
                'expert_control': expert_control,
                'expert_cmd_seq': expert_cmd_seq,
                'expert_cmd_seq_mask': expert_cmd_seq_mask,
                'expert_plan': expert_plan,
                'expert_plan_mask': expert_plan_mask,
                'sample_weight': sample_weight,
            }
            total_loss, loss_dict = self.criterion(
                outputs, batch_gpu, weights, prev_controls=None
            )

            # 记录损失
            batch_size = int(esdf.shape[0])
            val_losses['total'] += total_loss.item() * batch_size
            for key, value in loss_dict.items():
                val_losses[key] += value.item() * batch_size

            batch_metrics = behavior_metrics(
                outputs, target_pos, init_pose, global_esdf, self.cfg, target_vel,
                expert_cmd_seq=expert_cmd_seq, expert_cmd_seq_mask=expert_cmd_seq_mask,
                target_future_pos=target_future_pos,
                target_future_pos_mask=target_future_pos_mask,
            )
            for key in metric_totals:
                metric_totals[key] += batch_metrics[key]
            processed_batches += 1

        # 计算平均损失
        evaluated_samples = max(metric_totals['samples'], 1)
        for key in val_losses:
            val_losses[key] /= evaluated_samples

        sample_count = max(metric_totals['samples'], 1)
        behavior = {
            'collision_rate': metric_totals['collision_count'] / sample_count,
            'hard_collision_rate': metric_totals['hard_collision_count'] / sample_count,
            'obs_err': metric_totals['obs_err_sum'] / sample_count,
            'xy_roughness': metric_totals['xy_rough_sum'] / sample_count,
            'omega_smoothness': metric_totals['omega_smooth_sum'] / sample_count,
            'omega_margin': metric_totals['omega_margin_sum'] / sample_count,
            'min_esdf': metric_totals['min_esdf_sum'] / sample_count,
            'mean_abs_v': metric_totals['mean_abs_v_sum'] / sample_count,
            'speed_match_err': (
                metric_totals['speed_match_err_sum'] / metric_totals['speed_match_count']
                if metric_totals['speed_match_count'] > 0 else float('nan')
            ),
            'expert_cmd_err': (
                metric_totals['expert_cmd_err_sum'] / metric_totals['expert_cmd_count']
                if metric_totals['expert_cmd_count'] > 0 else float('nan')
            ),
            'expert_cmd_first_err': (
                metric_totals['expert_cmd_first_err_sum']
                / metric_totals['expert_cmd_first_count']
                if metric_totals['expert_cmd_first_count'] > 0 else float('nan')
            ),
            'moving_obs_err': metric_totals['moving_obs_err_sum'] / sample_count,
            'min_tracking_dist': metric_totals['min_tracking_dist_sum'] / sample_count,
            'too_close_rate': metric_totals['too_close_count'] / sample_count,
            'occluded': metric_totals['occluded_sum'] / sample_count,
            'corner_clip': (
                metric_totals['corner_clip_sum'] / metric_totals['corner_clip_count']
                if metric_totals['corner_clip_count'] > 0 else float('nan')
            ),
        }

        # TensorBoard 记录
        self.writer.add_scalar(f'{tag}/total_loss', val_losses['total'], epoch)
        for key, value in val_losses.items():
            if key != 'total':
                self.writer.add_scalar(f'{tag}/{key}_loss', value, epoch)
        for key, value in behavior.items():
            self.writer.add_scalar(f'{tag}_metric/{key}', value, epoch)

        if self.wandb_run is not None:
            log_dict = {f'{tag}/total_loss': val_losses['total'], 'epoch': epoch}
            for key, value in val_losses.items():
                if key != 'total':
                    log_dict[f'{tag}/{key}_loss'] = value
            for key, value in behavior.items():
                log_dict[f'{tag}_metric/{key}'] = value
            self.wandb_run.log(log_dict, step=self.global_step)

        return val_losses, behavior

    def save_checkpoint(self, epoch, val_loss, is_best=False, behavior=None, metric_rank=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metric_rank': self.best_metric_rank,
            'best_behavior_metrics': self.best_behavior_metrics,
            'behavior_metrics': behavior,
            'metric_rank': metric_rank,
            'checkpoint_rank_version': CHECKPOINT_RANK_VERSION,
            'config_dict': config_to_dict(self.cfg),
        }

        # 保存最新的检查点
        latest_path = os.path.join(self.cfg.MODEL_DIR, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 定期保存
        if (epoch + 1) % self.cfg.SAVE_INTERVAL == 0:
            epoch_path = os.path.join(self.cfg.MODEL_DIR, f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"Saved checkpoint: {epoch_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.cfg.MODEL_DIR, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path} (rank={metric_rank}, eval_val_loss={val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        load_model_state_checked(self.model, checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        if checkpoint.get('checkpoint_rank_version') == CHECKPOINT_RANK_VERSION:
            self.best_metric_rank = checkpoint.get('best_metric_rank')
            self.best_behavior_metrics = checkpoint.get('best_behavior_metrics')
        else:
            self.best_metric_rank = None
            self.best_behavior_metrics = None
            print('Checkpoint ranking schema changed; best-rank state was reset.')

        print(f"Resumed from epoch {self.start_epoch}")

    def init_from_checkpoint(self, checkpoint_path):
        """只加载模型权重，用于新 loss/新 run 的 fine-tune。"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        print(f"Initializing model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        load_model_state_checked(self.model, checkpoint['model_state_dict'])
        source_cfg = restore_config_from_checkpoint(checkpoint, Config())
        source_limits = (source_cfg.MAX_VEL, source_cfg.MAX_OMEGA)
        target_limits = (self.cfg.MAX_VEL, self.cfg.MAX_OMEGA)
        if source_limits != target_limits:
            ratios = rescale_control_output_head(
                self.model, *source_limits, *target_limits
            )
            print(
                "Migrated control-head output scale: "
                f"|v| {source_limits[0]:.3g}->{target_limits[0]:.3g}, "
                f"|omega| {source_limits[1]:.3g}->{target_limits[1]:.3g} "
                f"(logit row ratios {ratios[0]:.4f}, {ratios[1]:.4f})"
            )
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metric_rank = None
        self.best_behavior_metrics = None
        print("Loaded model weights; optimizer/scheduler/best loss were reset.")

    def reset_prediction_head(self):
        """Reset the direct-control prediction head while keeping encoders."""
        self.model.control_head = type(self.model.control_head)(self.cfg).to(self.device)
        reset_name = 'control_head'
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            weight_decay=self.cfg.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.LR_MILESTONES,
            gamma=self.cfg.LR_GAMMA
        )
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metric_rank = None
        self.best_behavior_metrics = None
        print(f"Reset {reset_name} and optimizer/scheduler.")

    # 完整主循环控制
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        for epoch in range(self.start_epoch, self.cfg.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.cfg.NUM_EPOCHS}")
            print("-" * 60)

            # 打印当前课程学习权重
            weights = get_loss_weights(epoch, self.cfg)
            print("Current loss weights:")
            weight_line = ", ".join(f"{key}: {value:.3g}" for key, value in weights.items())
            print(f"  {weight_line}")
            print("Prediction head: direct controls")

            # 训练
            train_weights = weights
            train_losses = self.train_epoch(epoch)
            print(f"\nTrain - Total Loss: {train_losses['total']:.4f}")
            print("Train contributors:",
                  format_loss_contributors(train_losses, train_weights))

            # 验证
            if (epoch + 1) % self.cfg.VAL_INTERVAL == 0:
                eval_weights = get_eval_loss_weights(self.cfg)
                val_losses, behavior = self.validate(epoch, weights=eval_weights, tag='val_eval')
                print(f"Val(eval fixed) - Total Loss: {val_losses['total']:.4f}")
                print("Val contributors:",
                      format_loss_contributors(val_losses, eval_weights))
                print("Val behavior:", format_behavior_metrics(behavior))

                # 保存最佳模型：优先外部行为指标，而不是纯 fixed loss
                rank = checkpoint_rank(behavior, val_losses, self.cfg)
                is_best = self.best_metric_rank is None or rank < self.best_metric_rank
                if is_best:
                    self.best_metric_rank = rank
                    self.best_behavior_metrics = behavior
                    self.best_val_loss = val_losses['total']

                # Save the scheduler state for the *next* epoch so resumed
                # training follows the same LR sequence as uninterrupted runs.
                self.scheduler.step()
                self.save_checkpoint(epoch, val_losses['total'], is_best,
                                     behavior=behavior, metric_rank=rank)
            else:
                self.scheduler.step()
                self.save_checkpoint(epoch, train_losses['total'], is_best=False)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best metric rank: {self.best_metric_rank}")
        if self.best_behavior_metrics is not None:
            print(f"Best behavior: {format_behavior_metrics(self.best_behavior_metrics)}")
        print(f"Best checkpoint eval loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        self.writer.close()
        self.finalize_wandb()

    def finalize_wandb(self):
        """写入 summary、上传 checkpoint artifact 并结束 wandb run。"""
        if self.wandb_run is None:
            return
        if self.best_behavior_metrics:
            for k, v in self.best_behavior_metrics.items():
                self.wandb_run.summary[f'best/{k}'] = v
        self.wandb_run.summary['best_val_loss'] = self.best_val_loss

        import glob as _glob
        art = wandb.Artifact(
            os.path.basename(self.cfg.MODEL_DIR.rstrip('/')),
            type='model',
            description='checkpoints from this run',
        )
        for f in sorted(_glob.glob(os.path.join(self.cfg.MODEL_DIR, '*.pth'))):
            art.add_file(f, name=os.path.basename(f))
        self.wandb_run.log_artifact(art)
        self.wandb_run.finish()
        print(f"wandb run finished: {self.wandb_run.url}")

    def upload_dataset_artifact(self):
        """Upload the exact filtered files used by this run to W&B."""
        if self.wandb_run is None:
            return
        import json
        data_dir = self.cfg.DATA_DIR
        artifact_name = os.path.basename(data_dir.rstrip('/'))
        artifact = wandb.Artifact(
            artifact_name,
            type='dataset',
            description='G1 clearance-filtered neural tracker dataset used for training',
        )
        for filename in ('train.pkl', 'val.pkl', 'narrow_train.pkl', 'narrow_val.pkl',
                         'manifest.json'):
            path = os.path.join(data_dir, filename)
            if os.path.isfile(path):
                artifact.add_file(path, name=filename)
        manifest_path = os.path.join(data_dir, 'manifest.json')
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as handle:
                    manifest = json.load(handle)
                train_stats = manifest.get('splits', {}).get('train', {})
                val_stats = manifest.get('splits', {}).get('val', {})
                artifact.metadata.update({
                    'filter_version': manifest.get('filter_version'),
                    'hard_clearance_m': manifest.get('hard_clearance_m'),
                    'preferred_clearance_m': manifest.get('preferred_clearance_m'),
                    'train_normal_samples': train_stats.get('normal_samples'),
                    'train_narrow_samples': train_stats.get('narrow_samples'),
                    'train_rejected_samples': train_stats.get('rejected_samples'),
                    'val_normal_samples': val_stats.get('normal_samples'),
                    'val_narrow_samples': val_stats.get('narrow_samples'),
                    'val_rejected_samples': val_stats.get('rejected_samples'),
                })
            except (OSError, ValueError, TypeError):
                pass
        self.wandb_run.log_artifact(artifact, aliases=['latest'])
        print(f"wandb dataset artifact queued: {artifact_name}")

def main():
    # 配置
    cfg = Config()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Python/NumPy/PyTorch random seed for this invocation')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--init-from', type=str, default=None,
                        help='checkpoint path to load model weights only')
    parser.add_argument('--run-name', type=str, default=None,
                        help='write checkpoints/logs under models/<run-name> and logs/<run-name>')
    parser.add_argument('--strategy', choices=['current', 'v41', 'v43'], default=None,
                        help='training-loss recipe; v43 adds corner safety, same-speed and corridor centering')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='override batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='override learning rate')
    parser.add_argument('--num-workers', type=int, default=None, help='override dataloader workers')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='override device')
    parser.add_argument('--max-vel', type=float, default=None,
                        help='override platform |v| limit in m/s')
    parser.add_argument('--max-omega', type=float, default=None,
                        help='override platform |omega| limit in rad/s')
    parser.add_argument('--min-tracking-distance', type=float, default=None,
                        help='override one-sided minimum tracker-target distance in meters')
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='limit train batches for smoke tests')
    parser.add_argument('--max-val-batches', type=int, default=None,
                        help='limit val batches for smoke tests')
    parser.add_argument('--disable-tqdm', action='store_true', help='disable progress bars')
    parser.add_argument('--wandb', action='store_true', help='log metrics to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='neural_tracker',
                        help='wandb project name')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='dataset directory containing train.pkl and val.pkl')
    parser.add_argument('--g1', action='store_true',
                        help='use G1 clearance semantics (0.55m operational hard boundary, 0.70m preference)')
    parser.add_argument('--include-narrow', action='store_true',
                        help='mix MPC-relabelled narrow_train/val.pkl into G1 training; do not use before relabelling')
    parser.add_argument('--narrow-expert-weight', type=float, default=None,
                        help='expert imitation weight for 0.55--0.70m narrow samples')
    parser.add_argument('--wandb-upload-dataset', action='store_true',
                        help='upload the exact dataset files used by this run as a W&B artifact')
    parser.add_argument('--reset-head', action='store_true',
                        help='reset direct-control prediction head after loading checkpoint')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='override trainer start epoch after loading model weights')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.resume:
        resume_meta = torch.load(args.resume, map_location='cpu', weights_only=False)
        restore_config_from_checkpoint(resume_meta, cfg, exclude={'DEVICE'})
        print(f"Restored effective config from {args.resume}")

    cfg.SEED = args.seed
    if args.strategy is not None:
        apply_strategy_config(cfg, args.strategy)

    if args.g1:
        # 0.50m is the physical body model. Runtime planning rejects below
        # 0.55m and prefers 0.70m; these training margins match that policy.
        apply_g1_config(cfg)
        cfg.INCLUDE_NARROW_DATA = args.include_narrow
    elif args.include_narrow:
        parser.error('--include-narrow requires --g1')
    if args.narrow_expert_weight is not None:
        if not 0.0 < args.narrow_expert_weight <= 1.0:
            parser.error('--narrow-expert-weight must be in (0, 1]')
        cfg.G1_NARROW_EXPERT_WEIGHT = args.narrow_expert_weight
    if args.data_dir is not None:
        cfg.DATA_DIR = args.data_dir
    if args.wandb_upload_dataset:
        cfg.UPLOAD_DATASET_ARTIFACT = True

    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        cfg.LEARNING_RATE = args.learning_rate
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
        cfg.PIN_MEMORY = args.num_workers > 0 and cfg.DEVICE == 'cuda'
    if args.device is not None:
        cfg.DEVICE = args.device
        cfg.PIN_MEMORY = cfg.NUM_WORKERS > 0 and cfg.DEVICE == 'cuda'
    if args.max_vel is not None:
        cfg.MAX_VEL = args.max_vel
    if args.max_omega is not None:
        cfg.MAX_OMEGA = args.max_omega
    if args.min_tracking_distance is not None:
        cfg.MIN_TRACKING_DISTANCE = args.min_tracking_distance
    if args.max_train_batches is not None:
        cfg.MAX_TRAIN_BATCHES = args.max_train_batches
    if args.max_val_batches is not None:
        cfg.MAX_VAL_BATCHES = args.max_val_batches
    if args.disable_tqdm:
        cfg.DISABLE_TQDM = True
    if args.run_name:
        cfg.MODEL_DIR = os.path.join(cfg.MODEL_DIR, args.run_name)
        cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, args.run_name)

    try:
        validate_config(cfg)
    except ValueError as exc:
        parser.error(str(exc))

    print("=" * 60)
    print("Neural Tracker Training")
    print("=" * 60)
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch size: {cfg.BATCH_SIZE}")
    print(f"Learning rate: {cfg.LEARNING_RATE}")
    print(f"Epochs: {cfg.NUM_EPOCHS}")
    print(f"Strategy: {args.strategy or 'checkpoint/default'}")
    print(f"Data dir: {cfg.DATA_DIR}")
    print(f"G1 mode: {cfg.G1_MODE} (narrow data: {cfg.INCLUDE_NARROW_DATA}, "
          f"narrow expert weight: {cfg.G1_NARROW_EXPERT_WEIGHT})")
    print(f"Platform limits: |v| <= {cfg.MAX_VEL:.2f} m/s, "
          f"|omega| <= {cfg.MAX_OMEGA:.2f} rad/s")
    print(f"Tracking distance: target={cfg.OBSERVATION_DISTANCE:.2f} m, "
          f"minimum={cfg.MIN_TRACKING_DISTANCE:.2f} m")
    print(f"Model dir: {cfg.MODEL_DIR}")
    print(f"Log dir: {cfg.LOG_DIR}")

    # 检查数据是否存在
    train_path = os.path.join(cfg.DATA_DIR, 'train.pkl')
    if not os.path.exists(train_path):
        print("\n" + "=" * 60)
        print("ERROR: Training data not found!")
        print("=" * 60)
        print(f"Expected path: {train_path}")
        print("\nPlease run: python data_generator.py")
        raise SystemExit(1)

    # 创建训练器
    wandb_run_name = args.run_name if args.run_name else 'neural_tracker'
    trainer = Trainer(cfg, use_wandb=args.wandb, run_name=wandb_run_name,
                      wandb_project=args.wandb_project,
                      upload_dataset=args.wandb_upload_dataset)

    if args.resume and args.init_from:
        raise ValueError('--resume and --init-from cannot be used together')
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.init_from:
        trainer.init_from_checkpoint(args.init_from)
    if args.reset_head:
        trainer.reset_prediction_head()
    if args.start_epoch is not None:
        trainer.start_epoch = args.start_epoch
        print(f"Start epoch overridden to {trainer.start_epoch}")

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.start_epoch,
            val_loss=trainer.best_val_loss,
            is_best=False
        )
        print("Checkpoint saved. You can resume training later.")
        trainer.finalize_wandb()

if __name__ == '__main__':
    main()
