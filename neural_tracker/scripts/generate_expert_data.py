"""
Expert trajectory dataset generator for DiffTrack neural tracker.

Generates:
1. Random 50x50m map with obstacles
2. Target trajectory via A* + B-spline smoothing (green)
3. Expert tracker trajectory via MPC (blue)
4. Visualization to verify quality

Usage:
    python scripts/generate_expert_data.py --visualize
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Map Generation
# ============================================================

class MapGenerator:
    def __init__(self, size=20.0, resolution=0.05):
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)  # 400 pixels for 20m @ 0.05
        self.offset = self.grid_size // 2

    def generate(self, num_obstacles=30, seed=None):
        if seed is not None:
            np.random.seed(seed)

        occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Add border walls (1m thick for 20m map)
        border_px = int(1.0 / self.resolution)
        occupancy[:border_px, :] = 1
        occupancy[-border_px:, :] = 1
        occupancy[:, :border_px] = 1
        occupancy[:, -border_px:] = 1

        # Add random obstacles (各种形状、大小随机，适配 20m 地图)
        for _ in range(num_obstacles):
            shape = np.random.choice(['rect', 'circle', 'L', 'elongated'])
            cx = np.random.uniform(-self.size/2 + 2, self.size/2 - 2)
            cy = np.random.uniform(-self.size/2 + 2, self.size/2 - 2)

            if shape == 'rect':
                w = np.random.uniform(0.4, 2.0)
                h = np.random.uniform(0.4, 2.0)
                angle = np.random.uniform(0, np.pi)
                self._draw_rotated_rect(occupancy, cx, cy, w, h, angle)
            elif shape == 'circle':
                r = np.random.uniform(0.3, 1.2)
                self._draw_circle(occupancy, cx, cy, r)
            elif shape == 'L':
                w = np.random.uniform(1.0, 2.5)
                h = np.random.uniform(1.0, 2.5)
                self._draw_L(occupancy, cx, cy, w, h)
            else:  # elongated (走廊型障碍)
                length = np.random.uniform(2.0, 4.0)
                width = np.random.uniform(0.4, 0.8)
                angle = np.random.uniform(0, np.pi)
                self._draw_rotated_rect(occupancy, cx, cy, length, width, angle)

        return occupancy

    def compute_esdf(self, occupancy):
        free = (occupancy == 0).astype(np.float32)
        dist_free = distance_transform_edt(free) * self.resolution
        dist_occ = distance_transform_edt(1 - free) * self.resolution
        esdf = dist_free - dist_occ
        return esdf.astype(np.float32)

    def world_to_grid(self, x, y):
        gx = int(x / self.resolution + self.offset)
        gy = int(y / self.resolution + self.offset)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = (gx - self.offset) * self.resolution
        y = (gy - self.offset) * self.resolution
        return x, y

    def _draw_rotated_rect(self, occ, cx, cy, w, h, angle):
        pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        pts = (R @ pts.T).T + np.array([cx, cy])
        pts_px = np.array([[int(p[0]/self.resolution + self.offset),
                            int(p[1]/self.resolution + self.offset)] for p in pts])
        cv2.fillPoly(occ, [pts_px], 1)

    def _draw_circle(self, occ, cx, cy, r):
        gx, gy = self.world_to_grid(cx, cy)
        r_px = int(r / self.resolution)
        cv2.circle(occ, (gx, gy), r_px, 1, -1)

    def _draw_L(self, occ, cx, cy, w, h):
        t = min(w, h) * 0.4
        self._draw_rotated_rect(occ, cx, cy, w, t, 0)
        self._draw_rotated_rect(occ, cx - w/2 + t/2, cy + h/2 - t/2, t, h, 0)


# ============================================================
# A* Path Search
# ============================================================

def astar(occupancy, start_grid, goal_grid, safe_dist_px):
    """8-connected A* on inflated occupancy grid"""
    H, W = occupancy.shape
    # Inflate obstacles by safe_dist
    from scipy.ndimage import binary_dilation
    inflated = binary_dilation(occupancy.astype(bool),
                               iterations=int(safe_dist_px))

    sx, sy = start_grid
    gx, gy = goal_grid
    if not (0 <= sx < W and 0 <= sy < H) or inflated[sy, sx]:
        return None
    if not (0 <= gx < W and 0 <= gy < H) or inflated[gy, gx]:
        return None

    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def h(p):
        return np.hypot(p[0]-gx, p[1]-gy)

    open_heap = [(h((sx,sy)), 0.0, (sx,sy))]
    came_from = {}
    g_score = {(sx,sy): 0.0}
    closed = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cx, cy = current
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if inflated[ny, nx]:
                continue
            step = np.hypot(dx, dy)
            ng = g + step
            if ng < g_score.get((nx, ny), np.inf):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = current
                heapq.heappush(open_heap, (ng + h((nx, ny)), ng, (nx, ny)))
    return None


def simplify_path(path, occupancy, safe_dist_px):
    """按弧长均匀降采样 A* 路径，保留曲率信息，不做 line-of-sight 切角"""
    if path is None or len(path) <= 2:
        return path
    pts = np.array(path, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 2:
        return path
    # 每 ~5 px (0.25m @ 0.05 res) 取一个控制点
    step_px = 5.0
    n_keep = max(int(total / step_px), 4)
    s_sample = np.linspace(0, total, n_keep)
    kept = []
    for s in s_sample:
        idx = np.searchsorted(cum, s)
        idx = min(max(idx, 1), len(pts) - 1)
        t = (s - cum[idx-1]) / (cum[idx] - cum[idx-1] + 1e-9)
        p = pts[idx-1] * (1-t) + pts[idx] * t
        kept.append((int(p[0]), int(p[1])))
    return kept


# ============================================================
# Target Trajectory: A* + B-spline smoothing
# ============================================================

def trajectory_collides(traj_world, esdf, map_gen, safe_dist):
    """检查轨迹是否撞障碍物。返回 (is_collision, worst_idx)"""
    min_val = np.inf
    worst_idx = -1
    for i, (x, y) in enumerate(traj_world):
        val, _, _ = esdf_value_and_grad(esdf, x, y,
                                         map_gen.resolution, map_gen.offset)
        if val < min_val:
            min_val = val
            worst_idx = i
    return min_val < safe_dist, worst_idx, min_val


def plan_target_trajectory(map_gen, occupancy, waypoints_world, esdf,
                           dt=0.1, max_vel=1.5, safe_dist=0.5):
    """
    Plan target trajectory through multiple waypoints.

    策略：A* 找路 → 密集采样 → 移动平均平滑（保证不偏离太远）→ 碰撞验证。
    不使用三次样条（样条振荡会切角穿障碍物）。

    Returns:
        traj_xy: (N, 2) trajectory points sampled at dt, or None if failed
    """
    safe_dist_px = int(safe_dist / map_gen.resolution)
    all_pts = []

    for i in range(len(waypoints_world) - 1):
        start = map_gen.world_to_grid(*waypoints_world[i])
        goal = map_gen.world_to_grid(*waypoints_world[i+1])
        path_grid = astar(occupancy, start, goal, safe_dist_px)
        if path_grid is None:
            return None
        path_world = np.array([map_gen.grid_to_world(p[0], p[1])
                                for p in path_grid])
        if i > 0:
            path_world = path_world[1:]
        all_pts.append(path_world)

    full_path = np.concatenate(all_pts, axis=0)  # (M, 2)

    # 按弧长均匀重采样（间距 = max_vel * dt）
    diffs = np.diff(full_path, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    if total_len < 1.0:
        return None

    step_size = max_vel * dt  # 每步行走距离
    n_steps = int(total_len / step_size)
    if n_steps < 10:
        return None
    s_samples = np.linspace(0, total_len, n_steps)

    # 线性插值重采样（不做样条，避免振荡）
    traj_x = np.interp(s_samples, cum_len, full_path[:, 0])
    traj_y = np.interp(s_samples, cum_len, full_path[:, 1])
    traj = np.stack([traj_x, traj_y], axis=-1)

    # 温和的移动平均平滑（窗口小，偏移有限）
    kernel_size = 7
    if len(traj) > kernel_size * 2:
        kernel = np.ones(kernel_size) / kernel_size
        traj_smooth_x = np.convolve(traj[:, 0], kernel, mode='same')
        traj_smooth_y = np.convolve(traj[:, 1], kernel, mode='same')
        # 两端不平滑（保持起止点精确）
        margin = kernel_size
        traj_smooth_x[:margin] = traj[:margin, 0]
        traj_smooth_x[-margin:] = traj[-margin:, 0]
        traj_smooth_y[:margin] = traj[:margin, 1]
        traj_smooth_y[-margin:] = traj[-margin:, 1]
        traj = np.stack([traj_smooth_x, traj_smooth_y], axis=-1)

    # 碰撞验证：确保每个点的 ESDF > robot_radius
    min_esdf = np.inf
    res = map_gen.resolution
    offset = map_gen.offset
    for x, y in traj:
        val, _, _ = esdf_value_and_grad(esdf, x, y, res, offset)
        min_esdf = min(min_esdf, val)
    if min_esdf < 0.2:
        # 平滑后仍有碰撞，用更大 inflate 重试
        return None

    return traj



# ============================================================
# ESDF query with bilinear interpolation
# ============================================================

def esdf_value_and_grad(esdf, x, y, resolution, offset):
    """Bilinear interpolation of ESDF value and its spatial gradient"""
    H, W = esdf.shape
    px = x / resolution + offset
    py = y / resolution + offset
    px = np.clip(px, 0, W - 2)
    py = np.clip(py, 0, H - 2)
    x1 = int(px); y1 = int(py)
    x2 = x1 + 1; y2 = y1 + 1
    u = px - x1; v = py - y1
    v00 = esdf[y1, x1]; v10 = esdf[y1, x2]
    v01 = esdf[y2, x1]; v11 = esdf[y2, x2]
    val = (1-u)*(1-v)*v00 + u*(1-v)*v10 + (1-u)*v*v01 + u*v*v11
    grad_x = ((1-v)*(v10-v00) + v*(v11-v01)) / resolution
    grad_y = ((1-u)*(v01-v00) + u*(v11-v10)) / resolution
    return val, grad_x, grad_y


# ============================================================
# MPC Tracker (simplified, scipy-based)
# ============================================================

class MPCTracker:
    """
    完整的 6 项 MPC，复现 mpc.cpp 的 visibility mode。
    Cost = λ_a·J_angle + λ_d·J_distance + λ_o·J_occlusion +
           λ_c·J_collision + λ_f·J_feasibility + λ_s·J_smoothness
    """
    def __init__(self, map_gen, esdf, horizon=30, dt=0.1,
                 max_vel=5.6, max_omega=4.0, robot_radius=0.15,
                 observation_distance=2.5, min_tracking_distance=0.7,
                 lambda_a=1.0, lambda_d=1.0, lambda_o=1.0,
                 lambda_c=1e9, lambda_f=1e5, lambda_s=1e3,
                 lambda_min_distance=1e4,
                 kappa=8, visibility_samples=3):
        self.map_gen = map_gen
        self.esdf = esdf
        self.n = horizon
        self.dt = dt
        self.max_vel = max_vel
        self.max_omega = max_omega
        self.radius = robot_radius
        self.ob = observation_distance
        self.min_tracking_distance = min_tracking_distance
        self.lambda_a = lambda_a
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f
        self.lambda_s = lambda_s
        self.lambda_min_distance = lambda_min_distance
        self.kappa = 1.0 / kappa  # collision 采样步长
        self.vis_samples = visibility_samples

    def propagate(self, x0, controls):
        n = controls.shape[0]
        traj = np.zeros((n + 1, 3))
        traj[0] = x0
        for t in range(n):
            v, w = controls[t]
            cur_yaw = traj[t, 2]
            traj[t+1, 2] = cur_yaw + self.dt * w
            traj[t+1, 0] = traj[t, 0] + self.dt * v * np.cos(cur_yaw)
            traj[t+1, 1] = traj[t, 1] + self.dt * v * np.sin(cur_yaw)
        return traj

    def cost(self, control_flat, x0, target_ref):
        """
        target_ref: (horizon, 2) 目标的未来位置（lookahead）
        """
        controls = control_flat.reshape(self.n, 2)
        traj = self.propagate(x0, controls)
        res = self.map_gen.resolution
        offset = self.map_gen.offset

        cost = 0.0

        # Visibility mode costs (per timestep)
        for t in range(1, self.n + 1):
            robot_xy = traj[t, :2]
            robot_yaw = traj[t, 2]
            target_xy = target_ref[t-1]

            # J_occlusion: 采样 robot → target 连线上的 ESDF
            occlusion = 0.0
            for k_frac in np.linspace(self.kappa, 1.0, self.vis_samples):
                inter = k_frac * target_xy + (1 - k_frac) * robot_xy
                val, _, _ = esdf_value_and_grad(self.esdf, inter[0], inter[1],
                                                 res, offset)
                viol = self.radius - val
                if viol > 0:
                    occlusion += viol ** 2
            cost += self.lambda_o * occlusion

            # J_distance: (||d|| - ob)² / ||d||
            dx = target_xy[0] - robot_xy[0]
            dy = target_xy[1] - robot_xy[1]
            dist = np.sqrt(dx**2 + dy**2 + 1e-8)
            cost += self.lambda_d * ((dist - self.ob) ** 2 / dist)
            min_dist_violation = max(self.min_tracking_distance - dist, 0.0)
            cost += self.lambda_min_distance * min_dist_violation ** 2

            # J_angle: 1 - cos(yaw_err)
            target_angle = np.arctan2(dy, dx)
            ang_err = robot_yaw - target_angle
            cost += self.lambda_a * (1 - np.cos(ang_err))

        # J_collision: 沿轨迹段 t→t+1 采样
        for t in range(self.n):
            for k_frac in np.arange(0, 1, self.kappa):
                inter = k_frac * traj[t+1, :2] + (1-k_frac) * traj[t, :2]
                val, _, _ = esdf_value_and_grad(self.esdf, inter[0], inter[1],
                                                 res, offset)
                viol = self.radius - val
                if viol > 0:
                    weight = (self.n - t) / self.n  # 后期权重递减
                    cost += self.lambda_c * weight * (viol ** 2)

        # J_smoothness: 3 阶位置差分（类似原实现的 dp3）
        if self.n >= 4:
            dp3 = (traj[3:, :2] - 3*traj[2:-1, :2] + 3*traj[1:-2, :2] - traj[:-3, :2])
            cost += self.lambda_s * np.sum(dp3 ** 2)

        # J_feasibility: 速度/加速度违反
        v_viol = np.maximum(np.abs(controls[:, 0]) - self.max_vel, 0)
        w_viol = np.maximum(np.abs(controls[:, 1]) - self.max_omega, 0)
        cost += self.lambda_f * (np.sum(v_viol**2) + np.sum(w_viol**2))

        dv = np.diff(controls[:, 0]) / self.dt
        dw = np.diff(controls[:, 1]) / self.dt
        max_acc = 3.0; max_alpha = 2.0
        acc_viol = np.maximum(np.abs(dv) - max_acc, 0)
        ang_viol = np.maximum(np.abs(dw) - max_alpha, 0)
        cost += self.lambda_f * (np.sum(acc_viol**2) + np.sum(ang_viol**2))

        return cost

    def solve(self, x0, target_ref, init_controls=None):
        if init_controls is None:
            init_controls = np.zeros((self.n, 2))
            init_controls[:, 0] = 0.5
        x_init = init_controls.flatten()

        res = minimize(
            self.cost, x_init, args=(x0, target_ref),
            method='L-BFGS-B', jac=None,
            options={'maxiter': 15, 'gtol': 1e-4}
        )
        controls = res.x.reshape(self.n, 2)
        controls[:, 0] = np.clip(controls[:, 0], -self.max_vel, self.max_vel)
        controls[:, 1] = np.clip(controls[:, 1], -self.max_omega, self.max_omega)
        return controls


def simulate_tracker(map_gen, esdf, target_traj, init_pose,
                     dt=0.1, horizon=30, observation_distance=2.5,
                     min_tracking_distance=0.7,
                     max_vel=5.6, max_omega=4.0, robot_radius=0.15,
                     sample_every=1):
    """
    真实 MPC visibility mode 跟踪。每步 lookahead horizon 步的目标未来位置。
    MPC 内部用 6 项 cost（angle, distance, occlusion, collision, feasibility, smoothness）
    自然保持 observation_distance 并避障。

    sample_every: MPC 求解间隔。每 N 步才重新调用一次 MPC，
                  中间步直接使用前一次解出的控制序列，显著加速。
                  sample_every=1 表示每步都求解（标准 MPC）。
    """
    mpc = MPCTracker(map_gen, esdf, horizon=horizon, dt=dt,
                     max_vel=max_vel, max_omega=max_omega,
                     robot_radius=robot_radius,
                     observation_distance=observation_distance,
                     min_tracking_distance=min_tracking_distance,
                     lambda_a=1.0, lambda_d=1.0, lambda_o=1.0,
                     lambda_c=1e9, lambda_f=1e5, lambda_s=1e3,
                     kappa=8, visibility_samples=3)

    T = len(target_traj)
    states = [np.array(init_pose, dtype=np.float64)]
    vels = []
    prev_controls = None

    for t in range(T):
        x0 = states[-1]

        # 是否在本步重新求解 MPC
        resolve = (t % sample_every == 0) or (prev_controls is None)

        if resolve:
            # Lookahead: 目标的未来 horizon 步位置
            end_idx = min(t + horizon, T - 1)
            if end_idx - t < horizon:
                ref = np.concatenate([
                    target_traj[t+1:end_idx+1],
                    np.tile(target_traj[end_idx:end_idx+1], (horizon - (end_idx - t), 1))
                ], axis=0)
            else:
                ref = target_traj[t+1:t+horizon+1]
            if len(ref) < horizon:
                ref = np.concatenate([ref,
                    np.tile(ref[-1:], (horizon - len(ref), 1))], axis=0)

            if prev_controls is not None:
                init_c = np.roll(prev_controls, -1, axis=0)
                init_c[-1] = prev_controls[-1]
            else:
                init_c = None

            controls = mpc.solve(x0, ref, init_controls=init_c)
            prev_controls = controls.copy()
            ctrl_idx = 0
        else:
            # 复用上一次求解的控制序列
            ctrl_idx = t % sample_every
            ctrl_idx = min(ctrl_idx, len(prev_controls) - 1)

        v, w = prev_controls[ctrl_idx]
        cur_yaw = x0[2]
        new_state = np.array([
            x0[0] + dt * v * np.cos(cur_yaw),
            x0[1] + dt * v * np.sin(cur_yaw),
            cur_yaw + dt * w
        ])
        states.append(new_state)
        vels.append([v, w])

    return np.array(states), np.array(vels)


# ============================================================
# Visualization
# ============================================================

def visualize(occupancy, esdf, target_traj, tracker_traj, waypoints,
              map_gen, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: occupancy + trajectories
    ax = axes[0]
    extent = [-map_gen.size/2, map_gen.size/2, -map_gen.size/2, map_gen.size/2]
    ax.imshow(occupancy, extent=extent, origin='lower',
              cmap='gray_r', alpha=0.8)

    ax.plot(target_traj[:, 0], target_traj[:, 1], '-', color='#00cc00',
            lw=3.0, label='Target trajectory (A* + smooth)', zorder=3)
    ax.plot(tracker_traj[:, 0], tracker_traj[:, 1], '-', color='#0066ff',
            lw=2.0, alpha=0.7,
            label='Expert tracker trajectory (MPC)', zorder=4)

    # Waypoints
    wp = np.array(waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], c='red', s=80, marker='*',
               edgecolors='black', linewidths=1.0, zorder=5,
               label='Target waypoints')
    ax.scatter([target_traj[0, 0]], [target_traj[0, 1]], c='lime', s=120,
               marker='o', edgecolors='black', linewidths=1.0, zorder=5,
               label='Target start')
    ax.scatter([tracker_traj[0, 0]], [tracker_traj[0, 1]], c='cyan', s=120,
               marker='s', edgecolors='black', linewidths=1.0, zorder=5,
               label='Tracker start')

    ax.set_xlim(-map_gen.size/2, map_gen.size/2)
    ax.set_ylim(-map_gen.size/2, map_gen.size/2)
    ax.set_aspect('equal')
    ax.set_title(f'Map {map_gen.size}m × {map_gen.size}m: target (green) vs expert tracker (blue)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Right: ESDF
    ax = axes[1]
    im = ax.imshow(esdf, extent=extent, origin='lower',
                   cmap='RdYlGn', vmin=-2, vmax=5)
    ax.plot(target_traj[:, 0], target_traj[:, 1], 'g-', lw=2)
    ax.plot(tracker_traj[:, 0], tracker_traj[:, 1], 'b-', lw=1.5)
    ax.set_aspect('equal')
    ax.set_title('ESDF (red=obstacle, green=free)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


# ============================================================
# Main
# ============================================================

def generate_one_sample(seed=42, num_waypoints=5, save_dir='./data/preview'):
    os.makedirs(save_dir, exist_ok=True)

    map_gen = MapGenerator(size=20.0, resolution=0.05)
    print(f"[seed={seed}] Generating map {map_gen.grid_size}x{map_gen.grid_size} (20m)...")
    occupancy = map_gen.generate(num_obstacles=12, seed=seed)
    esdf = map_gen.compute_esdf(occupancy)
    print(f"  ESDF range: [{esdf.min():.2f}, {esdf.max():.2f}]")

    # Sample waypoints in free space (>= 1m from obstacles)
    rng = np.random.RandomState(seed)
    free_threshold = 1.0
    waypoints = []
    attempts = 0
    while len(waypoints) < num_waypoints and attempts < 500:
        x = rng.uniform(-map_gen.size/2 + 3, map_gen.size/2 - 3)
        y = rng.uniform(-map_gen.size/2 + 3, map_gen.size/2 - 3)
        val, _, _ = esdf_value_and_grad(esdf, x, y, map_gen.resolution, map_gen.offset)
        if val > free_threshold:
            # Also keep waypoints reasonably spaced
            if not waypoints or all(np.hypot(x-w[0], y-w[1]) > 4.0 for w in waypoints):
                waypoints.append((x, y))
        attempts += 1
    if len(waypoints) < num_waypoints:
        print(f"  Only got {len(waypoints)} waypoints — retry with another seed")
        return False
    print(f"  Sampled {len(waypoints)} waypoints")

    # Plan target trajectory (try increasing safe_dist until collision-free)
    print("  Planning target trajectory (A* + moving-avg smooth)...")
    target_traj = None
    for sd in [0.5, 0.7, 0.9, 1.2]:
        target_traj = plan_target_trajectory(map_gen, occupancy, waypoints, esdf,
                                              dt=0.1, max_vel=1.2, safe_dist=sd)
        if target_traj is not None:
            break
    if target_traj is None:
        print("  Target trajectory planning failed (all safe_dist tried)")
        return False
    print(f"  Target trajectory: {len(target_traj)} steps, "
          f"length={np.sum(np.linalg.norm(np.diff(target_traj, axis=0), axis=1)):.1f}m")

    # 诊断：目标轨迹的最小 ESDF
    target_min_esdf = np.inf
    for x, y in target_traj:
        v, _, _ = esdf_value_and_grad(esdf, x, y, map_gen.resolution, map_gen.offset)
        target_min_esdf = min(target_min_esdf, v)
    print(f"  Target min ESDF: {target_min_esdf:.3f}m (>0 = no collision)")

    # Tracker 起始位置：沿目标轨迹反方向 obs_dist 处，面向目标
    obs_dist = 2.5
    init_dir_t = target_traj[1] - target_traj[0]
    init_dir_t /= np.linalg.norm(init_dir_t) + 1e-6

    # 尝试沿反向放置
    cand_x = target_traj[0, 0] - obs_dist * init_dir_t[0]
    cand_y = target_traj[0, 1] - obs_dist * init_dir_t[1]
    val, _, _ = esdf_value_and_grad(esdf, cand_x, cand_y,
                                     map_gen.resolution, map_gen.offset)
    if val > 0.5:
        cand_yaw = np.arctan2(init_dir_t[1], init_dir_t[0])
        init_pose = (cand_x, cand_y, cand_yaw)
    else:
        # 退化：环绕目标找 free 点
        init_pose = None
        for theta in np.linspace(0, 2*np.pi, 16, endpoint=False):
            cx = target_traj[0, 0] - obs_dist * np.cos(theta)
            cy = target_traj[0, 1] - obs_dist * np.sin(theta)
            v, _, _ = esdf_value_and_grad(esdf, cx, cy,
                                           map_gen.resolution, map_gen.offset)
            if v > 0.5:
                cyaw = np.arctan2(target_traj[0, 1] - cy,
                                   target_traj[0, 0] - cx)
                init_pose = (cx, cy, cyaw)
                break
    if init_pose is None:
        print("  No collision-free tracker start found")
        return False
    print(f"  Tracker init: ({init_pose[0]:.2f}, {init_pose[1]:.2f}, {init_pose[2]:.2f})")

    # Run MPC tracker (快速版：减小 horizon 和迭代次数)
    print("  Running MPC tracker (fast mode: horizon=15, maxiter=15)...")
    tracker_traj, tracker_vel = simulate_tracker(
        map_gen, esdf, target_traj, init_pose,
        dt=0.1, horizon=15, observation_distance=2.5,
        min_tracking_distance=0.7,
        max_vel=5.6, max_omega=4.0, robot_radius=0.15
    )
    print(f"  Tracker trajectory: {len(tracker_traj)} steps")

    # Visualize
    save_path = os.path.join(save_dir, f'sample_seed{seed:04d}.png')
    visualize(occupancy, esdf, target_traj, tracker_traj, waypoints,
              map_gen, save_path)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./data/preview')
    args = parser.parse_args()

    success = generate_one_sample(seed=args.seed, save_dir=args.save_dir)
    if not success:
        # Try a couple more seeds
        for s in [args.seed + 1, args.seed + 2, args.seed + 3]:
            if generate_one_sample(seed=s, save_dir=args.save_dir):
                break
