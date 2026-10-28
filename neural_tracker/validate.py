"""
模型验证与跟踪可视化

用法:
    python validate.py                    # 验证 val 集上的 5 个场景
    python validate.py --num 10           # 验证 10 个场景
    python validate.py --model best.pth   # 指定模型
    python validate.py --save             # 保存图片到 results/
"""
import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, apply_g1_config, restore_config_from_checkpoint
from network import TrackingPolicy
from data_generator import ScenarioGenerator


def adapt_state_dict_for_target_dim(model, state_dict):
    key = 'state_encoder.mlp.0.weight'
    model_state = model.state_dict()
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
    print(f"Adapted {key}: checkpoint {tuple(old_weight.shape)} -> model {tuple(new_weight.shape)}")
    return adapted


def load_model(model_path, cfg, checkpoint=None):
    model = TrackingPolicy(cfg).to(cfg.DEVICE)
    ckpt = checkpoint or torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
    state_dict = adapt_state_dict_for_target_dim(model, ckpt['model_state_dict'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded model from {model_path} (epoch {ckpt['epoch']}, val_loss={ckpt.get('best_val_loss', 'N/A')})")
    return model


def prepare_input(sample, cfg, scenario_gen):
    """从 sample 构造模型输入"""
    occupancy = sample['occupancy']
    robot_state = sample['robot_state'].copy()
    target_pos = sample['target_pos'].copy()
    target_vel = sample['target_vel']

    global_esdf = scenario_gen.compute_esdf(occupancy)

    esdf_patch = scenario_gen.extract_local_esdf(
        global_esdf, robot_state[0], robot_state[1], robot_state[2]
    )

    # Visibility ray
    K = cfg.VISIBILITY_SAMPLES
    alphas = np.linspace(0.0, 1.0, K + 2, dtype=np.float32)[1:-1]
    vis_ray = np.zeros(K, dtype=np.float32)
    for i, alpha in enumerate(alphas):
        px = robot_state[0] * (1 - alpha) + target_pos[0] * alpha
        py = robot_state[1] * (1 - alpha) + target_pos[1] * alpha
        pixel_x = int(np.floor((px + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION))
        pixel_y = int(np.floor((py + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION))
        H, W = global_esdf.shape
        if 0 <= pixel_x < W and 0 <= pixel_y < H:
            vis_ray[i] = global_esdf[pixel_y, pixel_x]
        else:
            vis_ray[i] = -1.0

    # Target relative
    rx, ry, ryaw = robot_state[:3]
    dx, dy = target_pos[0] - rx, target_pos[1] - ry
    rho = np.hypot(dx, dy)
    theta_rel = np.arctan2(dy, dx) - ryaw
    target_speed = np.hypot(target_vel[0], target_vel[1])
    theta_v = (np.arctan2(target_vel[1], target_vel[0]) - ryaw) if target_speed > 1e-3 else 0.0
    target_rel = np.array([rho, np.sin(theta_rel), np.cos(theta_rel),
                           np.sin(theta_v), np.cos(theta_v), target_speed], dtype=np.float32)

    raw_history = sample.get('control_history', sample.get('history'))
    history = np.zeros(cfg.HISTORY_DIM, dtype=np.float32)
    if raw_history is not None:
        raw_history = np.asarray(raw_history, dtype=np.float32)
        if raw_history.ndim == 2:
            raw_history = raw_history[:, :2].reshape(-1)
        else:
            raw_history = raw_history.reshape(-1)
        keep = min(raw_history.size, cfg.HISTORY_DIM)
        history[-keep:] = raw_history[-keep:]

    return {
        'esdf': torch.from_numpy(esdf_patch).unsqueeze(0).unsqueeze(0).float().to(cfg.DEVICE),
        'robot_state': torch.from_numpy(robot_state[3:]).unsqueeze(0).float().to(cfg.DEVICE),
        'target_rel': torch.from_numpy(target_rel).unsqueeze(0).float().to(cfg.DEVICE),
        'history': torch.from_numpy(history).unsqueeze(0).float().to(cfg.DEVICE),
        'visibility_ray': torch.from_numpy(vis_ray).unsqueeze(0).float().to(cfg.DEVICE),
        'init_pose': torch.from_numpy(robot_state[:3]).unsqueeze(0).float().to(cfg.DEVICE),
        'global_esdf': global_esdf,
        'occupancy': occupancy,
    }


@torch.no_grad()
def run_inference(model, inputs):
    """单帧推理"""
    outputs = model(
        inputs['esdf'],
        inputs['robot_state'],
        inputs['target_rel'],
        inputs['history'],
        inputs['visibility_ray'],
        inputs['init_pose'],
    )
    return outputs


def integrate_controls(init_pose, controls, cfg):
    """从控制序列重新积分出差速车可执行轨迹。"""
    controls_np = controls[0].detach().cpu().numpy()
    traj = np.zeros((len(controls_np) + 1, 3), dtype=np.float32)
    traj[0] = init_pose[:3]

    for t, (v, omega) in enumerate(controls_np):
        yaw = traj[t, 2]
        traj[t + 1, 0] = traj[t, 0] + cfg.DT * v * np.cos(yaw)
        traj[t + 1, 1] = traj[t, 1] + cfg.DT * v * np.sin(yaw)
        traj[t + 1, 2] = traj[t, 2] + cfg.DT * omega

    traj[:, 2] = np.arctan2(np.sin(traj[:, 2]), np.cos(traj[:, 2]))
    return traj


def trajectory_quality(pos_np, target_pos, global_esdf, cfg, target_vel=None,
                       target_future_pos=None, target_future_pos_mask=None):
    """计算一条轨迹的距离、碰撞和平滑指标。"""
    steps = len(pos_np) - 1
    times = np.arange(1, steps + 1, dtype=np.float32) * cfg.DT
    velocity = np.zeros(2, dtype=np.float32) if target_vel is None else np.asarray(target_vel)
    target_traj = np.asarray(target_pos, dtype=np.float32)[None] + times[:, None] * velocity[None]
    if target_future_pos is not None:
        future = np.asarray(target_future_pos, dtype=np.float32)
        mask = (
            np.ones(len(future), dtype=bool) if target_future_pos_mask is None
            else np.asarray(target_future_pos_mask).reshape(-1) > 0
        )
        n = min(steps, len(future), len(mask))
        valid_idx = np.flatnonzero(mask[:n])
        target_traj[valid_idx] = future[valid_idx, :2]
    distances = np.linalg.norm(pos_np[1:, :2] - target_traj, axis=1)
    dist_to_target = float(distances[-1]) if len(distances) else float(np.linalg.norm(pos_np[0, :2] - target_pos))
    signed_obs_err = dist_to_target - cfg.OBSERVATION_DISTANCE
    obs_dist_err = abs(signed_obs_err)
    moving_obs_err = float(np.mean(np.abs(distances - cfg.OBSERVATION_DISTANCE))) if len(distances) else obs_dist_err
    min_tracking_dist = float(np.min(distances)) if len(distances) else dist_to_target

    collisions = 0
    min_esdf = np.inf
    map_violations = 0
    segment_samples = max(2, cfg.VISIBILITY_SAMPLES)
    sampled_points = []
    for t in range(max(len(pos_np) - 1, 1)):
        if len(pos_np) == 1:
            sampled_points.append(pos_np[0, :2])
            continue
        alphas = np.linspace(0.0, 1.0, segment_samples, dtype=np.float32)
        sampled_points.extend(
            pos_np[t, :2][None] * (1.0 - alphas[:, None])
            + pos_np[t + 1, :2][None] * alphas[:, None]
        )
    for point in sampled_points:
        px = int(np.floor((point[0] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION))
        py = int(np.floor((point[1] + cfg.MAP_SIZE / 2) / cfg.MAP_RESOLUTION))
        H, W = global_esdf.shape
        if 0 <= px < W and 0 <= py < H:
            esdf_val = global_esdf[py, px]
            min_esdf = min(min_esdf, esdf_val)
            if esdf_val < cfg.ROBOT_RADIUS:
                collisions += 1
        else:
            collisions += 1
            map_violations += 1
            min_esdf = min(min_esdf, -1.0)

    dyaw = np.diff(pos_np[:, 2])
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    smoothness = np.std(dyaw)

    dxy = np.diff(pos_np[:, :2], axis=0)
    if len(dxy) >= 2:
        second_diff = np.diff(dxy, axis=0)
        xy_roughness = np.mean(np.linalg.norm(second_diff, axis=1))
    else:
        xy_roughness = 0.0

    return {
        'dist_to_target': dist_to_target,
        'signed_obs_err': signed_obs_err,
        'obs_dist_err': obs_dist_err,
        'moving_obs_err': moving_obs_err,
        'min_tracking_dist': min_tracking_dist,
        'below_min_tracking': min_tracking_dist < cfg.MIN_TRACKING_DISTANCE,
        'too_close_0_5m': dist_to_target < cfg.OBSERVATION_DISTANCE - 0.5,
        'too_far_0_5m': dist_to_target > cfg.OBSERVATION_DISTANCE + 0.5,
        'collisions': collisions,
        'map_violations': map_violations,
        'min_esdf': min_esdf,
        'smoothness': smoothness,
        'xy_roughness': xy_roughness,
    }


def compute_metrics(positions, controls, target_pos, robot_state, global_esdf, cfg,
                    target_vel=None, target_future_pos=None, target_future_pos_mask=None):
    """计算跟踪质量指标"""
    exec_pos_np = integrate_controls(robot_state, controls, cfg)
    controls_np = controls[0].cpu().numpy()  # (n, 2)

    exec_metrics = trajectory_quality(
        exec_pos_np, target_pos, global_esdf, cfg,
        target_vel=target_vel,
        target_future_pos=target_future_pos,
        target_future_pos_mask=target_future_pos_mask,
    )

    # 速度可行性
    dx = np.diff(exec_pos_np[:, 0])
    dy = np.diff(exec_pos_np[:, 1])
    cos_yaw = np.cos(exec_pos_np[:-1, 2])
    sin_yaw = np.sin(exec_pos_np[:-1, 2])
    v = (cos_yaw * dx + sin_yaw * dy) / cfg.DT
    max_v_violation = max(0, np.abs(v).max() - cfg.MAX_VEL)
    max_omega_violation = max(0, np.abs(controls_np[:, 1]).max() - cfg.MAX_OMEGA)
    mean_abs_v = np.mean(np.abs(controls_np[:, 0]))
    max_abs_v = np.max(np.abs(controls_np[:, 0]))
    mean_abs_omega = np.mean(np.abs(controls_np[:, 1]))

    return {
        **exec_metrics,
        'max_v_violation': max_v_violation,
        'max_omega_violation': max_omega_violation,
        'mean_abs_v': mean_abs_v,
        'max_abs_v': max_abs_v,
        'mean_abs_omega': mean_abs_omega,
    }


def visualize_scenario(sample, outputs, inputs, cfg, metrics, idx, save_path=None):
    """可视化单个跟踪场景"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    occupancy = inputs['occupancy']
    global_esdf = inputs['global_esdf']
    robot_state = sample['robot_state']
    target_pos = sample['target_pos']
    exec_positions = integrate_controls(robot_state, outputs['controls'], cfg)
    controls = outputs['controls'][0].cpu().numpy()  # (n, 2)

    # === Panel 1: 全局地图 + 轨迹 ===
    ax = axes[0]
    ax.set_title(f'Scenario #{idx} - Global View', fontsize=11)

    extent = [-cfg.MAP_SIZE/2, cfg.MAP_SIZE/2, -cfg.MAP_SIZE/2, cfg.MAP_SIZE/2]
    ax.imshow(global_esdf, origin='lower', extent=extent, cmap='RdYlGn', alpha=0.7)
    ax.contour(global_esdf, levels=[cfg.ROBOT_RADIUS], origin='lower', extent=extent,
               colors='red', linewidths=0.5)

    ax.plot(exec_positions[:, 0], exec_positions[:, 1],
            'b-', linewidth=2, label='Executed trajectory')

    # 朝向箭头 (每 5 步)
    for t in range(0, len(exec_positions), 5):
        dx = 0.15 * np.cos(exec_positions[t, 2])
        dy = 0.15 * np.sin(exec_positions[t, 2])
        ax.arrow(exec_positions[t, 0], exec_positions[t, 1], dx, dy,
                 head_width=0.05, head_length=0.03, fc='blue', ec='blue', alpha=0.6)

    # 机器人起点
    circle = plt.Circle((robot_state[0], robot_state[1]), cfg.ROBOT_RADIUS,
                        color='blue', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.plot(robot_state[0], robot_state[1], 'bo', markersize=8, label='Robot')

    # 目标
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')

    # 观察距离圈
    obs_circle = plt.Circle((target_pos[0], target_pos[1]), cfg.OBSERVATION_DISTANCE,
                            color='orange', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(obs_circle)
    min_dist_circle = plt.Circle(
        (target_pos[0], target_pos[1]), cfg.MIN_TRACKING_DISTANCE,
        color='red', fill=False, linestyle=':', linewidth=1.5,
    )
    ax.add_patch(min_dist_circle)

    # 视野范围
    view_range = 5.0
    ax.set_xlim(robot_state[0] - view_range, robot_state[0] + view_range)
    ax.set_ylim(robot_state[1] - view_range, robot_state[1] + view_range)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Panel 2: 局部 ESDF + 轨迹 ===
    ax = axes[1]
    ax.set_title('Local ESDF (Robot Frame)', fontsize=11)

    esdf_patch = inputs['esdf'][0, 0].cpu().numpy()
    ax.imshow(esdf_patch, origin='lower', cmap='RdYlGn',
              extent=[-cfg.LOCAL_MAP_SIZE/2, cfg.LOCAL_MAP_SIZE/2,
                      -cfg.LOCAL_MAP_SIZE/2, cfg.LOCAL_MAP_SIZE/2])

    # 轨迹转到局部坐标
    cos_r = np.cos(robot_state[2])
    sin_r = np.sin(robot_state[2])
    exec_dx_w = exec_positions[:, 0] - robot_state[0]
    exec_dy_w = exec_positions[:, 1] - robot_state[1]
    exec_local_x = cos_r * exec_dx_w + sin_r * exec_dy_w
    exec_local_y = -sin_r * exec_dx_w + cos_r * exec_dy_w
    ax.plot(exec_local_x, exec_local_y, 'b-', linewidth=2)
    ax.plot(0, 0, 'bo', markersize=10)

    # 目标在局部坐标
    tdx = target_pos[0] - robot_state[0]
    tdy = target_pos[1] - robot_state[1]
    t_local_x = cos_r * tdx + sin_r * tdy
    t_local_y = -sin_r * tdx + cos_r * tdy
    ax.plot(t_local_x, t_local_y, 'r*', markersize=15)

    ax.set_xlim(-cfg.LOCAL_MAP_SIZE/2, cfg.LOCAL_MAP_SIZE/2)
    ax.set_ylim(-cfg.LOCAL_MAP_SIZE/2, cfg.LOCAL_MAP_SIZE/2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # === Panel 3: 控制量 ===
    ax = axes[2]
    ax.set_title('Controls (v, ω)', fontsize=11)

    t_steps = np.arange(len(controls)) * cfg.DT
    ax.plot(t_steps, controls[:, 0], 'b-', linewidth=1.5, label=f'v (max={controls[:,0].max():.2f})')
    ax.plot(t_steps, controls[:, 1], 'r-', linewidth=1.5, label=f'ω (max={np.abs(controls[:,1]).max():.2f})')
    ax.axhline(y=cfg.MAX_VEL, color='b', linestyle='--', alpha=0.5, label=f'v_max={cfg.MAX_VEL}')
    ax.axhline(y=-cfg.MAX_VEL, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=cfg.MAX_OMEGA, color='r', linestyle='--', alpha=0.5, label=f'ω_max={cfg.MAX_OMEGA}')
    ax.axhline(y=-cfg.MAX_OMEGA, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 指标文本
    fig.suptitle(
        f"dist={metrics['dist_to_target']:.2f}m | "
        f"moving_err={metrics['moving_obs_err']:.2f}m | "
        f"min_dist={metrics['min_tracking_dist']:.2f}m | "
        f"signed={metrics['signed_obs_err']:+.2f}m | "
        f"obs_err={metrics['obs_dist_err']:.2f}m | "
        f"collisions={metrics['collisions']} oob={metrics['map_violations']} | "
        f"rough={metrics['xy_roughness']:.3f} | "
        f"mean|v|={metrics['mean_abs_v']:.2f} | "
        f"v_viol={metrics['max_v_violation']:.3f}",
        fontsize=10, y=0.02
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Neural Tracker Validation')
    parser.add_argument('--model', type=str, default='models/best.pth')
    parser.add_argument('--data', type=str, default='data/scenarios/val.pkl')
    parser.add_argument('--num', type=int, default=None,
                        help='number of samples (default: full validation set)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='explicit device override after checkpoint config restoration')
    parser.add_argument('--g1', action='store_true',
                        help='explicitly apply G1 settings for a legacy checkpoint without config_dict')
    parser.add_argument('--save', action='store_true', help='Save figures to result directory')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory for saved figures when --save is set')
    parser.add_argument('--no-plot', action='store_true', help='Only print metrics, skip figures')
    parser.add_argument('--quiet', action='store_true', help='Only print summary metrics')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    cfg = restore_config_from_checkpoint(checkpoint, Config())
    if args.g1:
        apply_g1_config(cfg)
    if args.device is not None:
        cfg.DEVICE = args.device
    model = load_model(args.model, cfg, checkpoint=checkpoint)
    scenario_gen = ScenarioGenerator(cfg)

    with open(args.data, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} validation samples")

    np.random.seed(args.seed)
    num_samples = len(samples) if args.num is None else min(args.num, len(samples))
    if num_samples <= 0:
        raise ValueError('validation requires at least one sample')
    indices = np.random.choice(len(samples), num_samples, replace=False)

    if args.save:
        os.makedirs(args.result_dir, exist_ok=True)

    all_metrics = []
    for i, idx in enumerate(indices):
        sample = samples[idx]
        inputs = prepare_input(sample, cfg, scenario_gen)

        outputs = run_inference(model, inputs)

        metrics = compute_metrics(
            outputs['positions'], outputs['controls'], sample['target_pos'],
            sample['robot_state'], inputs['global_esdf'], cfg,
            target_vel=sample.get('target_vel'),
            target_future_pos=sample.get('target_future_pos'),
            target_future_pos_mask=sample.get('target_future_pos_mask'),
        )
        all_metrics.append(metrics)

        save_path = os.path.join(args.result_dir, f'scenario_{idx:04d}.png') if args.save else None
        if not args.no_plot:
            visualize_scenario(sample, outputs, inputs, cfg, metrics, idx, save_path)

        if not args.quiet:
            print(f"[{i+1}/{len(indices)}] Scenario {idx}: "
                  f"future_dist={metrics['dist_to_target']:.2f}m, "
                  f"moving_obs_err={metrics['moving_obs_err']:.2f}m, "
                  f"min_dist={metrics['min_tracking_dist']:.2f}m, "
                  f"signed={metrics['signed_obs_err']:+.2f}m, "
                  f"obs_err={metrics['obs_dist_err']:.2f}m, "
                  f"collisions={metrics['collisions']}, "
                  f"oob={metrics['map_violations']}, "
                  f"rough={metrics['xy_roughness']:.3f}, "
                  f"mean|v|={metrics['mean_abs_v']:.3f}, "
                  f"v_viol={metrics['max_v_violation']:.3f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Validation Summary ({len(all_metrics)} scenarios)")
    print(f"{'='*60}")
    print(f"  Future-target dist:  {np.mean([m['dist_to_target'] for m in all_metrics]):.3f} ± {np.std([m['dist_to_target'] for m in all_metrics]):.3f} m")
    print(f"  Moving obs error:    {np.mean([m['moving_obs_err'] for m in all_metrics]):.3f} ± {np.std([m['moving_obs_err'] for m in all_metrics]):.3f} m")
    print(f"  Min tracking dist:   {np.mean([m['min_tracking_dist'] for m in all_metrics]):.3f} m mean-min")
    print(f"  Below {cfg.MIN_TRACKING_DISTANCE:.2f}m:       {sum(1 for m in all_metrics if m['below_min_tracking'])}/{len(all_metrics)}")
    print(f"  Signed obs error:    {np.mean([m['signed_obs_err'] for m in all_metrics]):+.3f} ± {np.std([m['signed_obs_err'] for m in all_metrics]):.3f} m")
    print(f"  Final obs error:     {np.mean([m['obs_dist_err'] for m in all_metrics]):.3f} ± {np.std([m['obs_dist_err'] for m in all_metrics]):.3f} m")
    print(f"  Too close >0.5m:     {sum(1 for m in all_metrics if m['too_close_0_5m'])}/{len(all_metrics)}")
    print(f"  Too far >0.5m:       {sum(1 for m in all_metrics if m['too_far_0_5m'])}/{len(all_metrics)}")
    print(f"  Collision rate:      {sum(1 for m in all_metrics if m['collisions']>0)}/{len(all_metrics)}")
    print(f"  Map violation rate:  {sum(1 for m in all_metrics if m['map_violations']>0)}/{len(all_metrics)}")
    print(f"  Smoothness:          {np.mean([m['smoothness'] for m in all_metrics]):.4f}")
    print(f"  XY roughness:        {np.mean([m['xy_roughness'] for m in all_metrics]):.4f}")
    print(f"  Max vel violation:   {np.mean([m['max_v_violation'] for m in all_metrics]):.4f} m/s")
    print(f"  Max omega violation: {np.mean([m['max_omega_violation'] for m in all_metrics]):.4f} rad/s")
    print(f"  Mean |v|:            {np.mean([m['mean_abs_v'] for m in all_metrics]):.4f} m/s")
    print(f"  Max |v|:             {np.mean([m['max_abs_v'] for m in all_metrics]):.4f} m/s")
    print(f"  Mean |ω|:            {np.mean([m['mean_abs_omega'] for m in all_metrics]):.4f} rad/s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
