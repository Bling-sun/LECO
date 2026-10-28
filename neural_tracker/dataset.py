"""
PyTorch Dataset - 加载训练数据
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import cv2
from collections import OrderedDict
from config import Config
from data_generator import ScenarioGenerator

class TrackingDataset(Dataset):
    def __init__(self, data_path, config, augment=False):
        """
        参数：
            data_path: .pkl 文件路径
            config: Config 对象
            augment: 是否使用数据增强
        """
        self.cfg = config
        self.augment = augment
        self.scenario_gen = ScenarioGenerator(config)
        self._esdf_cache = OrderedDict()

        # 加载数据
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def compute_visibility_ray(self, robot_pos, target_pos, global_esdf, map_size, resolution):
        """
        计算自身→目标连线上的 ESDF 采样

        返回：visibility_ray (8,)
        """
        # Only sample the interior of the line of sight.  The robot and target
        # endpoints describe footprint/goal clearance, not geometric occlusion.
        K = self.cfg.VISIBILITY_SAMPLES
        alphas = np.linspace(0.0, 1.0, K + 2, dtype=np.float32)[1:-1]

        ray_samples = np.zeros(K, dtype=np.float32)

        for i, alpha in enumerate(alphas):
            # 插值点
            px = robot_pos[0] * (1 - alpha) + target_pos[0] * alpha
            py = robot_pos[1] * (1 - alpha) + target_pos[1] * alpha

            # 转换到像素坐标
            pixel_x = int(np.floor((px + map_size / 2) / resolution))
            pixel_y = int(np.floor((py + map_size / 2) / resolution))

            # 边界检查
            H, W = global_esdf.shape
            if 0 <= pixel_x < W and 0 <= pixel_y < H:
                ray_samples[i] = global_esdf[pixel_y, pixel_x]
            else:
                ray_samples[i] = -1.0  # 超出地图

        return ray_samples

    def compute_target_relative(self, robot_state, target_pos, target_vel):
        """
        计算目标相对信息（极坐标）

        返回：target_rel (6,) [ρ, sin(θ), cos(θ), sin(θ_v), cos(θ_v), |v_target|]
        """
        rx, ry, ryaw = robot_state[:3]
        tx, ty = target_pos
        tvx, tvy = target_vel

        # 相对位置
        dx = tx - rx
        dy = ty - ry

        # 极坐标
        rho = np.hypot(dx, dy)
        theta_world = np.arctan2(dy, dx)
        theta_rel = theta_world - ryaw  # 相对机器人朝向

        # 目标速度方向
        target_speed = np.hypot(tvx, tvy)
        if target_speed > 1e-3:
            theta_v = np.arctan2(tvy, tvx) - ryaw
        else:
            theta_v = 0.0

        target_rel = np.array([
            rho,
            np.sin(theta_rel),
            np.cos(theta_rel),
            np.sin(theta_v),
            np.cos(theta_v),
            target_speed
        ], dtype=np.float32)

        return target_rel

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 提取数据
        occupancy = sample['occupancy']  # (400, 400) uint8
        robot_state = sample['robot_state'].copy()  # [x, y, yaw, v, ω]
        target_pos = sample['target_pos'].copy()    # [x, y]
        target_vel = sample['target_vel']           # [vx, vy]
        has_expert_supervision = any(
            key in sample for key in ('expert_cmd', 'expert_cmd_seq', 'expert_plan')
        )

        # Prefer a precomputed ESDF when data generation saved one.  Legacy
        # datasets remain supported, but do not force every dataset access to
        # recompute an already available distance field.
        if 'global_esdf' in sample:
            global_esdf = np.asarray(sample['global_esdf'], dtype=np.float32)
        else:
            cache_key = id(occupancy)
            global_esdf = self._esdf_cache.get(cache_key)
            if global_esdf is None:
                global_esdf = self.scenario_gen.compute_esdf(occupancy)  # (400, 400) float32
                cache_size = max(0, int(getattr(self.cfg, 'ESDF_CACHE_SIZE', 32)))
                if cache_size > 0:
                    self._esdf_cache[cache_key] = global_esdf
                    while len(self._esdf_cache) > cache_size:
                        self._esdf_cache.popitem(last=False)
            else:
                self._esdf_cache.move_to_end(cache_key)

        if self.augment and not has_expert_supervision and getattr(self.cfg, 'USE_HARD_CASE_AUG', False):
            if np.random.rand() < getattr(self.cfg, 'HARD_CASE_PROB', 0.0):
                robot_state = self._sample_hard_start(robot_state, target_pos, global_esdf)

        # 数据增强：里程计扰动放在提取局部 patch 之前
        # Pose/target perturbations change the supervised problem.  Applying
        # them while retaining commands or a world-frame expert plan creates
        # contradictory labels, so geometry augmentation is restricted to
        # genuinely unsupervised samples.  Sensor-only ESDF noise below is safe.
        if self.augment and self.cfg.USE_DATA_AUG and not has_expert_supervision:
            if np.random.rand() < 0.5:
                robot_state[0] += np.random.randn() * self.cfg.ODOM_NOISE_POS
                robot_state[1] += np.random.randn() * self.cfg.ODOM_NOISE_POS
                robot_state[2] += np.random.randn() * self.cfg.ODOM_NOISE_YAW

            target_noise_prob = getattr(
                self.cfg, 'TARGET_POS_NOISE_PROB', getattr(self.cfg, 'TARGET_DROPOUT', 0.0)
            )
            if np.random.rand() < target_noise_prob:
                target_pos = target_pos + np.random.randn(2).astype(np.float32) * 0.2

        # 提取以机器人为中心、按朝向对齐的局部 ESDF
        esdf = self.scenario_gen.extract_local_esdf(
            global_esdf, robot_state[0], robot_state[1], robot_state[2]
        )  # (80, 80)

        # ESDF 噪声放在 patch 上即可
        if self.augment and self.cfg.USE_DATA_AUG and np.random.rand() < 0.5:
            esdf = esdf + np.random.randn(*esdf.shape).astype(np.float32) * self.cfg.ESDF_NOISE_STD

        # 计算 visibility ray
        visibility_ray = self.compute_visibility_ray(
            robot_state[:2], target_pos, global_esdf,
            self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
        )

        # 计算目标相对信息
        target_rel = self.compute_target_relative(robot_state, target_pos, target_vel)

        history = self._extract_control_history(sample)

        # 转换为 tensor
        esdf_tensor = torch.from_numpy(esdf).unsqueeze(0).float()  # (1, 80, 80)
        robot_state_tensor = torch.from_numpy(robot_state[3:]).float()  # [v, ω]
        target_rel_tensor = torch.from_numpy(target_rel).float()
        history_tensor = torch.from_numpy(history).float()
        visibility_ray_tensor = torch.from_numpy(visibility_ray).float()

        # 保留完整信息用于 loss 计算
        full_robot_state = torch.from_numpy(robot_state).float()
        target_pos_tensor = torch.from_numpy(target_pos).float()
        target_vel_tensor = torch.from_numpy(target_vel).float()
        global_esdf_tensor = torch.from_numpy(global_esdf).float()
        raw_expert_cmd = sample.get('expert_cmd')
        expert_cmd = (
            np.asarray(raw_expert_cmd, dtype=np.float32).reshape(-1)[:2]
            if raw_expert_cmd is not None else np.zeros(2, dtype=np.float32)
        )
        expert_cmd_seq, expert_cmd_seq_mask = self._extract_cmd_seq(sample, expert_cmd)
        expert_plan, expert_plan_mask = self._extract_expert_plan(sample, robot_state)
        target_future_pos, target_future_pos_mask = self._extract_target_future(sample)

        expert_control_tensor = torch.from_numpy(expert_cmd.copy()).float()
        expert_cmd_seq_tensor = torch.from_numpy(expert_cmd_seq).float()
        expert_cmd_seq_mask_tensor = torch.from_numpy(expert_cmd_seq_mask).float()
        expert_plan_tensor = torch.from_numpy(expert_plan).float()
        expert_plan_mask_tensor = torch.from_numpy(expert_plan_mask).float()
        target_future_pos_tensor = torch.from_numpy(target_future_pos).float()
        target_future_pos_mask_tensor = torch.from_numpy(target_future_pos_mask).float()

        # Samples produced by filter_g1_clearance.py carry a clearance class.
        # Narrow (0.55--0.70 m) labels are useful for corridor behavior only
        # after the corrected A*+MPC expert has replaced their trajectories;
        # they are intentionally down-weighted for direct imitation.
        clearance_class = str(sample.get('g1_clearance_class', 'normal'))
        sample_weight = (
            float(getattr(self.cfg, 'G1_NARROW_EXPERT_WEIGHT', 0.25))
            if clearance_class == 'narrow' else 1.0
        )

        return {
            # 网络输入
            'esdf': esdf_tensor,
            'robot_state': robot_state_tensor,
            'target_rel': target_rel_tensor,
            'history': history_tensor,
            'visibility_ray': visibility_ray_tensor,

            # 用于 loss 计算
            'full_robot_state': full_robot_state,
            'target_pos': target_pos_tensor,
            'target_vel': target_vel_tensor,
            'target_future_pos': target_future_pos_tensor,
            'target_future_pos_mask': target_future_pos_mask_tensor,
            'global_esdf': global_esdf_tensor,
            'expert_control': expert_control_tensor,
            'expert_cmd_seq': expert_cmd_seq_tensor,
            'expert_cmd_seq_mask': expert_cmd_seq_mask_tensor,
            'expert_plan': expert_plan_tensor,
            'expert_plan_mask': expert_plan_mask_tensor,
            'sample_weight': torch.tensor(sample_weight, dtype=torch.float32),
        }

    def _extract_cmd_seq(self, sample, expert_cmd):
        seq = sample.get('expert_cmd_seq')
        if seq is None:
            out = np.zeros((self.cfg.HORIZON, 2), dtype=np.float32)
            out_mask = np.zeros(self.cfg.HORIZON, dtype=np.float32)
            # A single expert command only supervises the deployed first step;
            # it is not evidence that the command stays constant for 3 seconds.
            if sample.get('expert_cmd') is not None and expert_cmd.size >= 2:
                out[0] = expert_cmd[:2]
                out_mask[0] = 1.0
            return out, out_mask
        else:
            seq = np.asarray(seq, dtype=np.float32)
            if seq.ndim != 2 or seq.shape[1] < 2:
                raise ValueError(
                    f"expert_cmd_seq must have shape (N, >=2), got {seq.shape}"
                )
            mask = np.asarray(
                sample.get('expert_cmd_seq_mask', np.ones(seq.shape[0], dtype=np.float32)),
                dtype=np.float32
            ).reshape(-1)

        out = np.zeros((self.cfg.HORIZON, 2), dtype=np.float32)
        out_mask = np.zeros(self.cfg.HORIZON, dtype=np.float32)
        n = min(self.cfg.HORIZON, seq.shape[0], mask.shape[0])
        out[:n] = seq[:n, :2]
        out_mask[:n] = mask[:n]
        return out, out_mask

    def _extract_expert_plan(self, sample, robot_state):
        plan = sample.get('expert_plan')
        if plan is None:
            plan = np.repeat(robot_state[:3].reshape(1, 3), self.cfg.HORIZON + 1, axis=0)
            mask = np.zeros(self.cfg.HORIZON + 1, dtype=np.float32)
        else:
            plan = np.asarray(plan, dtype=np.float32)
            if plan.ndim != 2 or plan.shape[1] < 3:
                raise ValueError(f"expert_plan must have shape (N, >=3), got {plan.shape}")
            mask = np.asarray(
                sample.get('expert_plan_mask', np.ones(plan.shape[0], dtype=np.float32)),
                dtype=np.float32
            ).reshape(-1)

        out = np.zeros((self.cfg.HORIZON + 1, 3), dtype=np.float32)
        out[:, :] = robot_state[:3]
        out_mask = np.zeros(self.cfg.HORIZON + 1, dtype=np.float32)
        n = min(self.cfg.HORIZON + 1, plan.shape[0], mask.shape[0])
        out[:n] = plan[:n, :3]
        out_mask[:n] = mask[:n]
        return out, out_mask

    def _extract_control_history(self, sample):
        """Return oldest-to-newest [v, omega] history with left zero padding."""
        raw = sample.get('control_history', sample.get('history'))
        out = np.zeros(self.cfg.HISTORY_DIM, dtype=np.float32)
        if raw is None:
            return out
        history = np.asarray(raw, dtype=np.float32)
        if history.ndim == 2:
            if history.shape[1] < 2:
                raise ValueError(f"control_history must have shape (N, >=2), got {history.shape}")
            history = history[:, :2].reshape(-1)
        else:
            history = history.reshape(-1)
        if history.size % 2 != 0:
            raise ValueError(f"control_history must contain [v, omega] pairs, got {history.size} values")
        keep = min(history.size, self.cfg.HISTORY_DIM)
        out[-keep:] = history[-keep:]
        return out

    def _extract_target_future(self, sample):
        """Load future target positions for t=1..H, with an explicit mask."""
        out = np.zeros((self.cfg.HORIZON, 2), dtype=np.float32)
        out_mask = np.zeros(self.cfg.HORIZON, dtype=np.float32)
        raw = sample.get('target_future_pos')
        if raw is None:
            return out, out_mask
        future = np.asarray(raw, dtype=np.float32)
        if future.ndim != 2 or future.shape[1] < 2:
            raise ValueError(f"target_future_pos must have shape (N, >=2), got {future.shape}")
        mask = np.asarray(
            sample.get('target_future_pos_mask', np.ones(future.shape[0], dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        n = min(self.cfg.HORIZON, future.shape[0], mask.shape[0])
        out[:n] = future[:n, :2]
        out_mask[:n] = mask[:n]
        return out, out_mask

    def _sample_hard_start(self, robot_state, target_pos, global_esdf):
        """Place robot near low-clearance free space to create corner/narrow training cases."""
        H, W = global_esdf.shape
        min_c = getattr(self.cfg, 'HARD_CASE_MIN_CLEARANCE', 0.05)
        max_c = getattr(self.cfg, 'HARD_CASE_MAX_CLEARANCE', 0.45)
        attempts = getattr(self.cfg, 'HARD_CASE_MAX_ATTEMPTS', 80)

        for _ in range(attempts):
            px = np.random.randint(0, W)
            py = np.random.randint(0, H)
            clearance = global_esdf[py, px]
            if clearance < min_c or clearance > max_c:
                continue
            if not self._is_hard_context(px, py, global_esdf):
                continue

            x = px * self.cfg.MAP_RESOLUTION - self.cfg.MAP_SIZE / 2
            y = py * self.cfg.MAP_RESOLUTION - self.cfg.MAP_SIZE / 2
            dist = np.linalg.norm(target_pos - np.array([x, y], dtype=np.float32))
            if dist < self.cfg.MIN_TARGET_DIST or dist > self.cfg.MAX_TARGET_DIST:
                continue

            hard_state = robot_state.copy()
            hard_state[0] = x
            hard_state[1] = y
            yaw_to_target = np.arctan2(target_pos[1] - y, target_pos[0] - x)
            hard_state[2] = yaw_to_target + np.random.uniform(-0.8, 0.8)
            hard_state[2] = np.arctan2(np.sin(hard_state[2]), np.cos(hard_state[2]))
            hard_state[3:] = 0.0
            return hard_state.astype(np.float32)

        return robot_state

    def _is_hard_context(self, px, py, global_esdf):
        """
        Prefer near-obstacle poses that look like corners or narrow passages.

        A single wall nearby is easy and tends to teach wall-following/crawling.
        Multiple occupied/low-clearance sectors around the robot better match
        the deployment failures at obstacle corners and corridor mouths.
        """
        H, W = global_esdf.shape
        safe_margin = self.cfg.ROBOT_RADIUS + 0.12
        sector_hits = 0
        sectors = 8
        radius_px = max(3, int(0.7 / self.cfg.MAP_RESOLUTION))

        for i in range(sectors):
            theta = 2.0 * np.pi * i / sectors
            sx = int(round(px + np.cos(theta) * radius_px))
            sy = int(round(py + np.sin(theta) * radius_px))
            if sx < 0 or sx >= W or sy < 0 or sy >= H:
                sector_hits += 1
                continue
            if global_esdf[sy, sx] < safe_margin:
                sector_hits += 1

        if sector_hits >= 2:
            return True

        # Also keep very tight but valid free-space samples.
        return global_esdf[py, px] < self.cfg.ROBOT_RADIUS + 0.08

def test_dataset():
    """测试 dataset"""
    from torch.utils.data import DataLoader

    cfg = Config()

    # 需要先生成数据
    import os
    train_path = os.path.join(cfg.DATA_DIR, 'train.pkl')
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}")
        print("Please run: python data_generator.py")
        return

    dataset = TrackingDataset(train_path, cfg, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("=" * 60)
    print("Dataset test")
    print("=" * 60)

    batch = next(iter(loader))

    print("Batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)

if __name__ == '__main__':
    test_dataset()
