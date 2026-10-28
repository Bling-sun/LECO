"""
损失函数 - 将 MPC cost 翻译为可微的 PyTorch loss

包含：
1. MPC 原始 6 项：角度、距离、遮挡、碰撞、可行性、平滑
2. DPtraj 几何约束：非完整、曲率、弧长均匀
3. 新增项：滚动一致性、模仿学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

def normalize_angle(angle):
    """将角度归一化到 [-π, π]"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def positiveSmoothedL1(x, epsilon=1e-4):
    """
    平滑的 ReLU（参考 DPtraj）
    在 x=0 处用三次多项式平滑，避免零梯度
    """
    half = 0.5 * epsilon
    f3c = 1.0 / (epsilon * epsilon)
    f4c = -0.5 * f3c / epsilon

    b1 = x <= 0.0
    b2 = (x > 0.0) & (x < epsilon)
    b3 = x >= epsilon

    a1 = torch.zeros_like(x)
    a2 = (f4c * x + f3c) * x * x * x
    a3 = x - half

    loss = a1 * b1 + a2 * b2 + a3 * b3
    return loss

def positiveSmoothedL2(x):
    """平滑的平方惩罚"""
    x_pos = F.relu(x)
    return x_pos * x_pos

def grid_sample_esdf(positions, esdf_map, map_size, resolution, return_in_bounds=False):
    """
    在 ESDF 地图上批量采样（双线性插值）

    参数：
        positions: (B, N, 2) 世界坐标 [x, y]
        esdf_map: (B, H, W) ESDF 地图
        map_size: 地图尺寸（米）
        resolution: 分辨率（米/像素）

    返回：
        values: (B, N) 采样的 ESDF 值（保留梯度，用于 loss 反传）
        gradients: (B, N, 2) ESDF 空间梯度 [∂/∂x, ∂/∂y]（纯物理量，已 detach）
        in_bounds: optional (B, N) mask when return_in_bounds=True
    """
    B, N, _ = positions.shape
    H, W = esdf_map.shape[1:]

    esdf_expanded = esdf_map.unsqueeze(1)  # (B, 1, H, W)
    half = map_size / 2

    def sample_at(query):
        pixel_x = (query[:, :, 0] + half) / resolution
        pixel_y = (query[:, :, 1] + half) / resolution
        in_map = (
            (pixel_x >= 0.0) & (pixel_x < W)
            & (pixel_y >= 0.0) & (pixel_y < H)
        )
        norm_x = pixel_x / max(W - 1, 1) * 2.0 - 1.0
        norm_y = pixel_y / max(H - 1, 1) * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(
            esdf_expanded, grid, mode='bilinear',
            padding_mode='border', align_corners=True,
        ).squeeze(1).squeeze(1)

        # Out-of-map space is occupied.  The signed penetration term also
        # provides a gradient that points predictions back into the map.
        outside_x = torch.maximum(-pixel_x, pixel_x - (W - 1)).clamp_min(0.0) * resolution
        outside_y = torch.maximum(-pixel_y, pixel_y - (H - 1)).clamp_min(0.0) * resolution
        oob_value = -torch.maximum(outside_x, outside_y) - resolution
        return torch.where(in_map, sampled, oob_value), in_map

    values, in_bounds = sample_at(positions)

    # 有限差分求 ESDF 空间梯度（纯物理量，不参与反向传播）
    with torch.no_grad():
        delta = resolution * 0.5

        pos_dx = positions.detach().clone()
        pos_dx[:, :, 0] += delta
        values_dx, _ = sample_at(pos_dx)

        pos_dy = positions.detach().clone()
        pos_dy[:, :, 1] += delta
        values_dy, _ = sample_at(pos_dy)

        grad_x = (values_dx - values.detach()) / delta
        grad_y = (values_dy - values.detach()) / delta
        gradients = torch.stack([grad_x, grad_y], dim=-1)  # (B, N, 2)

    if return_in_bounds:
        return values, gradients, in_bounds
    return values, gradients


def target_visibility_gate(positions, target_pos, esdf_map, cfg, target_vel=None,
                           target_future_pos=None, target_future_mask=None):
    """
    Soft gate for task losses that directly point at the target.

    If the line segment from tracker to target is occluded by obstacles, angle /
    distance / speed objectives should not pull the robot straight through the
    wall. The returned gate has shape (B, n) for positions[:, 1:].
    """
    B, n_plus_1, _ = positions.shape
    n = n_plus_1 - 1
    K = cfg.VISIBILITY_SAMPLES
    device = positions.device

    robot_pos = positions[:, 1:, :2]
    t = torch.arange(1, n + 1, device=device, dtype=positions.dtype) * cfg.DT
    if target_vel is not None:
        target_exp = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * t.view(1, n, 1)
    else:
        target_exp = target_pos.unsqueeze(1).expand(-1, n, -1)
    if target_future_pos is not None:
        m = min(n, target_future_pos.shape[1])
        future = target_future_pos[:, :m]
        target_exp = target_exp.clone()
        if target_future_mask is None:
            target_exp[:, :m] = future
        else:
            valid = target_future_mask[:, :m].to(dtype=torch.bool).unsqueeze(-1)
            target_exp[:, :m] = torch.where(valid, future, target_exp[:, :m])
    alphas = torch.linspace(0.0, 1.0, K + 2, device=device)[1:-1].view(1, 1, K, 1)
    ray = robot_pos.unsqueeze(2) * (1 - alphas) + target_exp.unsqueeze(2) * alphas
    sample_pts = ray.reshape(B, n * K, 2)
    esdf_val, _ = grid_sample_esdf(sample_pts, esdf_map, cfg.MAP_SIZE, cfg.MAP_RESOLUTION)
    min_ray_clear = esdf_val.reshape(B, n, K).min(dim=-1).values.detach()

    margin = getattr(cfg, 'VISIBILITY_CLEARANCE_MARGIN', cfg.ROBOT_RADIUS + 0.05)
    temp = max(getattr(cfg, 'VISIBILITY_GATE_TEMP', 0.06), 1e-3)
    return torch.sigmoid((min_ray_clear - margin) / temp)

class AngleLoss(nn.Module):
    """角度对齐损失（MPC 的 λ_a）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, positions, target_pos, step_weight=None):
        """
        参数：
            positions: (B, n+1, 3) [x, y, yaw]
            target_pos: (B, 2) [x, y]

        返回：
            loss: scalar
        """
        B, n_plus_1, _ = positions.shape
        n = n_plus_1 - 1

        # 计算每个时间步的角度误差
        robot_pos = positions[:, 1:, :2]  # (B, n, 2)，跳过初始点
        robot_yaw = positions[:, 1:, 2]   # (B, n)

        # 目标方向
        target_expanded = target_pos.unsqueeze(1).expand(-1, n, -1)  # (B, n, 2)
        delta = target_expanded - robot_pos  # (B, n, 2)
        target_angle = torch.atan2(delta[:, :, 1], delta[:, :, 0])  # (B, n)

        # 角度差
        angle_error = normalize_angle(robot_yaw - target_angle)  # (B, n)

        # 损失：1 - cos(θ)
        loss = 1 - torch.cos(angle_error)  # (B, n)
        if step_weight is not None:
            loss = loss * step_weight

        return loss.mean()

class DistanceLoss(nn.Module):
    """跟踪距离损失（MPC 的 λ_d）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ob = cfg.OBSERVATION_DISTANCE

    def forward(self, positions, target_pos, step_weight=None):
        """
        Match mpc.cpp td(): (distance - ob)^2 / distance.
        """
        B, n_plus_1, _ = positions.shape
        n = n_plus_1 - 1

        robot_pos = positions[:, 1:, :2]  # (B, n, 2)
        target_expanded = target_pos.unsqueeze(1).expand(-1, n, -1)

        delta = target_expanded - robot_pos  # (B, n, 2)
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-8)  # (B, n)
        step_loss = (dist - self.ob).square() / dist.clamp_min(1e-3)
        if step_weight is not None:
            step_loss = step_loss * step_weight

        return step_loss.mean()

class MovingTargetDistanceLoss(nn.Module):
    """Keep OBSERVATION_DISTANCE from the MOVING target.

    The static DistanceLoss uses target_pos at t=0, which contradicts tracking a
    target that moves ~1 m/s (it rewards sitting near the initial target
    position). This version extrapolates the target with constant velocity, so
    the objective is "stay OBSERVATION_DISTANCE behind the target as it moves",
    compatible with the expert demonstrations. Early steps are weighted higher
    because only the near horizon is executed under receding-horizon control.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ob = cfg.OBSERVATION_DISTANCE

    def forward(self, positions, target_pos, target_vel=None, step_weight=None,
                target_future_pos=None, target_future_mask=None):
        B, n_plus_1, _ = positions.shape
        n = n_plus_1 - 1
        device = positions.device

        robot_pos = positions[:, 1:, :2]  # (B, n, 2)
        # Constant-velocity target trajectory for t=1..n.
        t = torch.arange(1, n + 1, device=device, dtype=positions.dtype) * self.cfg.DT  # (n,)
        if target_vel is not None:
            tgt = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * t.view(1, n, 1)
        else:
            tgt = target_pos.unsqueeze(1).expand(-1, n, -1)
        if target_future_pos is not None:
            m = min(n, target_future_pos.shape[1])
            if target_future_mask is None:
                tgt = tgt.clone()
                tgt[:, :m] = target_future_pos[:, :m]
            else:
                valid = target_future_mask[:, :m].to(dtype=torch.bool).unsqueeze(-1)
                tgt = tgt.clone()
                tgt[:, :m] = torch.where(valid, target_future_pos[:, :m], tgt[:, :m])

        delta = tgt - robot_pos
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-8)
        step_loss = (dist - self.ob).square() / dist.clamp_min(1e-3)  # (B, n)

        # Emphasize the near horizon (linearly decaying weight, like mpc.cpp).
        horizon_w = (n - torch.arange(n, device=device, dtype=positions.dtype)) / n
        step_loss = step_loss * horizon_w.unsqueeze(0)
        if step_weight is not None:
            step_loss = step_loss * step_weight

        return step_loss.mean()


class MinTrackingDistanceLoss(nn.Module):
    """One-sided barrier that keeps tracker-target distance at or above a minimum."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.min_distance = cfg.MIN_TRACKING_DISTANCE

    def forward(self, positions, target_pos, target_vel=None,
                target_future_pos=None, target_future_mask=None):
        n = positions.shape[1] - 1
        device = positions.device
        dtype = positions.dtype
        t = torch.arange(1, n + 1, device=device, dtype=dtype) * self.cfg.DT
        if target_vel is not None:
            target = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * t.view(1, n, 1)
        else:
            target = target_pos.unsqueeze(1).expand(-1, n, -1)
        if target_future_pos is not None:
            horizon = min(n, target_future_pos.shape[1])
            target = target.clone()
            if target_future_mask is None:
                target[:, :horizon] = target_future_pos[:, :horizon]
            else:
                valid = target_future_mask[:, :horizon].bool().unsqueeze(-1)
                target[:, :horizon] = torch.where(
                    valid, target_future_pos[:, :horizon], target[:, :horizon]
                )

        distance = torch.norm(positions[:, 1:, :2] - target, dim=-1)
        violation = F.relu(self.min_distance - distance) / max(self.min_distance, 1e-3)
        penalty = violation.square()
        # The worst predicted instant must not disappear in a horizon average.
        return penalty.mean() + penalty.max(dim=1).values.mean()


class VisibilityLoss(nn.Module):
    """遮挡损失（MPC 的 λ_o），向量化"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.radius = cfg.ROBOT_RADIUS

    def forward(self, positions, target_pos, esdf_map, target_vel=None,
                target_future_pos=None, target_future_mask=None):
        """
        Match mpc.cpp occlusion term: accumulate positive ray violations per
        timestep, then square that occlusion amount.

        For a moving target the line of sight must stay clear as the target
        moves, so the target is extrapolated at constant velocity per step
        (target_vel given). This is the key signal for keeping the target
        visible around corners; without it the policy drives straight at the
        best observation pose and loses the target when it rounds a wall.

        参数：
            positions: (B, n+1, 3)
            target_pos: (B, 2)
            esdf_map: (B, H, W)
        """
        B, n_plus_1, _ = positions.shape
        n = n_plus_1 - 1
        K = self.cfg.VISIBILITY_SAMPLES
        device = positions.device

        robot_pos = positions[:, 1:, :2]                   # (B, n, 2)
        if target_vel is not None:
            t = torch.arange(1, n + 1, device=device, dtype=positions.dtype) * self.cfg.DT
            target_exp = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * t.view(1, n, 1)
        else:
            target_exp = target_pos.unsqueeze(1).expand(-1, n, -1)  # (B, n, 2)
        if target_future_pos is not None:
            m = min(n, target_future_pos.shape[1])
            if target_future_mask is None:
                target_exp = target_exp.clone()
                target_exp[:, :m] = target_future_pos[:, :m]
            else:
                valid = target_future_mask[:, :m].to(dtype=torch.bool).unsqueeze(-1)
                target_exp = target_exp.clone()
                target_exp[:, :m] = torch.where(
                    valid, target_future_pos[:, :m], target_exp[:, :m]
                )

        alphas = torch.linspace(0.0, 1.0, K + 2, device=device)[1:-1].view(1, 1, K, 1)
        ray = robot_pos.unsqueeze(2) * (1 - alphas) + target_exp.unsqueeze(2) * alphas
        # ray: (B, n, K, 2)

        sample_pts = ray.reshape(B, n * K, 2)  # (B, n*K, 2)
        esdf_val, _ = grid_sample_esdf(sample_pts, esdf_map,
                                       self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION)
        # esdf_val: (B, n*K)

        los_margin = getattr(self.cfg, 'LOS_CLEARANCE_MARGIN', 0.0)
        violation = F.relu(los_margin - esdf_val).reshape(B, n, K)
        ray_violation = violation.sum(dim=-1)
        return ray_violation.square().mean()

class CollisionLoss(nn.Module):
    """Collision loss matching mpc.cpp segment ESDF cost."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.radius = cfg.ROBOT_RADIUS
        self.margin = getattr(cfg, 'COLLISION_SAFETY_MARGIN', self.radius)
        self.topk_ratio = getattr(cfg, 'COLLISION_TOPK_RATIO', 0.0)
        self.hard_weight = getattr(cfg, 'COLLISION_HARD_WEIGHT', 1.0)
        self.clearance_weight = getattr(cfg, 'COLLISION_CLEARANCE_WEIGHT', 1.0)

    def forward(self, positions, esdf_map):
        B, n_plus_1, _ = positions.shape
        n = n_plus_1 - 1
        K = self.cfg.VISIBILITY_SAMPLES
        device = positions.device

        start = positions[:, :-1, :2]
        end = positions[:, 1:, :2]
        alphas = torch.linspace(0.0, 1.0, K, device=device).view(1, 1, K, 1)
        segment_pts = start.unsqueeze(2) * (1.0 - alphas) + end.unsqueeze(2) * alphas

        sample_pts = segment_pts.reshape(B, n * K, 2)
        esdf_val, _ = grid_sample_esdf(sample_pts, esdf_map,
                                       self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION)
        clearance_violation = F.relu(self.margin - esdf_val).reshape(B, n, K)
        hard_violation = F.relu(self.radius - esdf_val).reshape(B, n, K)

        clearance_penalty = clearance_violation.square()
        hard_penalty = hard_violation.square().mean(dim=-1)
        mean_penalty = clearance_penalty.mean(dim=-1)

        if self.topk_ratio > 0:
            topk = max(1, int(round(K * self.topk_ratio)))
            worst_penalty = clearance_penalty.topk(topk, dim=-1).values.mean(dim=-1)
        else:
            worst_penalty = torch.zeros_like(mean_penalty)

        penalty = (
            self.clearance_weight * mean_penalty
            + worst_penalty
            + self.hard_weight * hard_penalty
        )

        # mpc.cpp weights earlier segments more strongly: (n - t) / n.
        t_idx = torch.arange(1, n + 1, device=positions.device, dtype=positions.dtype)
        weight = (n - t_idx + 1) / n                          # (n,)
        return (penalty * weight.unsqueeze(0)).mean()

class FeasibilityLoss(nn.Module):
    """可行性损失（MPC 的 λ_f）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_vel = cfg.MAX_VEL
        self.max_omega = cfg.MAX_OMEGA
        self.max_acc = cfg.MAX_ACC
        self.max_alpha = cfg.MAX_ALPHA
        self.dt = cfg.DT

    def forward(self, controls, current_control=None):
        """
        惩罚超出运动学约束的控制量

        参数：
            controls: (B, n, 2) [v, ω]
        """
        B, n, _ = controls.shape

        v = controls[:, :, 0]      # (B, n)
        omega = controls[:, :, 1]  # (B, n)

        # ControlPolicyHead already applies MAX_VEL*tanh / MAX_OMEGA*tanh, so the
        # velocity-limit violations are identically zero here. Keeping them only
        # added dead terms, so v/omega limits are dropped; acc/alpha still matter
        # because integration of adjacent bounded controls can exceed them.
        #
        # Normalize violations by the squared limits so each term is O(1) and the
        # feasibility weight lives on the same scale as the other task losses,
        # instead of being amplified by 1/dt**2 (=100) and dominating the total.
        if current_control is not None:
            current_control = current_control[:, :2].to(device=controls.device, dtype=controls.dtype)
            v = torch.cat([current_control[:, 0:1], v], dim=1)
            omega = torch.cat([current_control[:, 1:2], omega], dim=1)

        if v.shape[1] > 1:
            acc = (v[:, 1:] - v[:, :-1]) / self.dt
            alpha = (omega[:, 1:] - omega[:, :-1]) / self.dt

            acc_loss = F.relu(acc.square() / (self.max_acc ** 2) - 1.0).mean()
            alpha_loss = F.relu(alpha.square() / (self.max_alpha ** 2) - 1.0).mean()
        else:
            acc_loss = torch.tensor(0.0, device=controls.device)
            alpha_loss = torch.tensor(0.0, device=controls.device)

        return acc_loss + alpha_loss

class SmoothLoss(nn.Module):
    """MPC smoothness loss: third finite difference of pose."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, positions):
        B, n_plus_1, _ = positions.shape

        if n_plus_1 < 4:
            return torch.tensor(0.0, device=positions.device)

        xy = positions[:, :, :2]
        xy_dp3 = 3.0 * xy[:, 1:-2] - 3.0 * xy[:, 2:-1] + xy[:, 3:] - xy[:, :-3]

        # Difference wrapped increments instead of wrapped absolute yaw.  A
        # smooth crossing from +pi to -pi must not look like a 2*pi jerk.
        yaw_step = normalize_angle(positions[:, 1:, 2] - positions[:, :-1, 2])
        yaw_dp3 = yaw_step[:, 2:] - 2.0 * yaw_step[:, 1:-1] + yaw_step[:, :-2]
        return (xy_dp3.square().sum(dim=-1) + yaw_dp3.square()).mean()


class ControlSmoothLoss(nn.Module):
    """控制序列平滑损失：v 轻约束，omega 强约束。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, controls):
        if controls.shape[1] < 2:
            return torch.tensor(0.0, device=controls.device)

        v = controls[:, :, 0] / max(self.cfg.MAX_VEL, 1e-3)
        omega = controls[:, :, 1] / max(self.cfg.MAX_OMEGA, 1e-3)

        dv = v[:, 1:] - v[:, :-1]
        domega = omega[:, 1:] - omega[:, :-1]
        first_loss = 0.25 * dv.square().mean() + 4.0 * domega.square().mean()

        if domega.shape[1] >= 2:
            ddv = dv[:, 1:] - dv[:, :-1]
            ddomega = domega[:, 1:] - domega[:, :-1]
            second_loss = 0.10 * ddv.square().mean() + 2.5 * ddomega.square().mean()
        else:
            second_loss = torch.tensor(0.0, device=controls.device)

        return first_loss + second_loss


class OmegaMarginLoss(nn.Module):
    """惩罚 omega 贴近上限，避免长期贴边转向。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, controls):
        omega = controls[:, :, 1]
        soft_limit = self.cfg.MAX_OMEGA * getattr(self.cfg, 'OMEGA_SOFT_LIMIT_RATIO', 0.75)
        return positiveSmoothedL2(torch.abs(omega) - soft_limit).mean()


class ExpertControlLoss(nn.Module):
    """Imitate the expert/current controller command stored in robot_state[v, omega]."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, controls, expert_control):
        pred = controls[:, 0, :]
        pred_norm = torch.stack([
            pred[:, 0] / max(self.cfg.MAX_VEL, 1e-3),
            pred[:, 1] / max(self.cfg.MAX_OMEGA, 1e-3),
        ], dim=-1)
        expert_norm = torch.stack([
            expert_control[:, 0] / max(self.cfg.MAX_VEL, 1e-3),
            expert_control[:, 1] / max(self.cfg.MAX_OMEGA, 1e-3),
        ], dim=-1)
        return F.smooth_l1_loss(pred_norm, expert_norm)


class ExpertCmdSeqLoss(nn.Module):
    """Imitate the expert future command sequence [v, omega]."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.first_step_weight = 2.0

    def forward(self, controls, expert_cmd_seq, mask=None, sample_weight=None):
        horizon = min(controls.shape[1], expert_cmd_seq.shape[1])
        pred = controls[:, :horizon, :]
        expert = expert_cmd_seq[:, :horizon, :].clone()
        expert[:, :, 0] = expert[:, :, 0].clamp(-self.cfg.MAX_VEL, self.cfg.MAX_VEL)
        expert[:, :, 1] = expert[:, :, 1].clamp(-self.cfg.MAX_OMEGA, self.cfg.MAX_OMEGA)

        pred_norm = torch.stack([
            pred[:, :, 0] / max(self.cfg.MAX_VEL, 1e-3),
            pred[:, :, 1] / max(self.cfg.MAX_OMEGA, 1e-3),
        ], dim=-1)
        expert_norm = torch.stack([
            expert[:, :, 0] / max(self.cfg.MAX_VEL, 1e-3),
            expert[:, :, 1] / max(self.cfg.MAX_OMEGA, 1e-3),
        ], dim=-1)

        step_loss = F.smooth_l1_loss(pred_norm, expert_norm, reduction='none').mean(dim=-1)
        if mask is None:
            valid = torch.ones_like(step_loss)
        else:
            valid = mask[:, :horizon].float()

        weights = torch.ones(horizon, device=controls.device, dtype=controls.dtype)
        weights[0] = self.first_step_weight
        valid = valid * weights.unsqueeze(0)
        valid_count = valid.sum(dim=1)
        sample_valid = valid_count > 0
        per_sample = (step_loss * valid).sum(dim=1) / valid_count.clamp_min(1.0)
        if sample_weight is not None:
            sample_weights = sample_weight.to(
                device=per_sample.device, dtype=per_sample.dtype
            ).reshape(-1) * sample_valid.to(per_sample.dtype)
            return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        return per_sample[sample_valid].mean() if sample_valid.any() else step_loss.sum() * 0.0


class ExpertPlanLoss(nn.Module):
    """Auxiliary imitation of the expert integrated path."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, positions, expert_plan, mask=None, sample_weight=None):
        horizon = min(positions.shape[1], expert_plan.shape[1])
        pred = positions[:, :horizon, :]
        expert = expert_plan[:, :horizon, :]

        xy_loss = F.smooth_l1_loss(pred[:, :, :2], expert[:, :, :2], reduction='none').mean(dim=-1)
        yaw_err = normalize_angle(pred[:, :, 2] - expert[:, :, 2])
        yaw_loss = 0.25 * (1.0 - torch.cos(yaw_err))
        step_loss = xy_loss + yaw_loss

        if mask is None:
            valid = torch.ones_like(step_loss)
        else:
            valid = mask[:, :horizon].float()
        valid_count = valid.sum(dim=1)
        sample_valid = valid_count > 0
        per_sample = (step_loss * valid).sum(dim=1) / valid_count.clamp_min(1.0)
        if sample_weight is not None:
            sample_weights = sample_weight.to(
                device=per_sample.device, dtype=per_sample.dtype
            ).reshape(-1) * sample_valid.to(per_sample.dtype)
            return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        return per_sample[sample_valid].mean() if sample_valid.any() else step_loss.sum() * 0.0


class ClearanceRecoveryLoss(nn.Module):
    """v25: low-clearance corner/narrow-passage recovery objective."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.margin = getattr(cfg, 'RECOVERY_CLEARANCE_MARGIN', 0.35)
        self.hard_margin = getattr(cfg, 'RECOVERY_HARD_MARGIN', 0.05)
        self.danger_margin = getattr(cfg, 'RECOVERY_DANGER_MARGIN', cfg.ROBOT_RADIUS + 0.06)
        self.min_turn = getattr(cfg, 'RECOVERY_MIN_OMEGA', 0.25)

    def forward(self, positions, controls, esdf_map):
        pts = positions[:, :, :2]
        values, grads = grid_sample_esdf(
            pts, esdf_map, self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
        )

        # Basic soft clearance margin. This shapes the trajectory without forcing
        # emergency recovery behavior in every narrow passage.
        clearance_all = positiveSmoothedL2(self.margin - values)
        clearance_loss = clearance_all.mean()
        topk = max(1, int(round(values.shape[1] * 0.25)))
        worst_clearance_loss = clearance_all.topk(topk, dim=1).values.mean()

        low = F.relu(self.margin - values[:, :-1]).detach() / max(self.margin, 1e-3)
        low = low.clamp(max=1.0)
        danger = F.relu(self.danger_margin - values[:, :-1]).detach() / max(self.danger_margin, 1e-3)
        danger = danger.clamp(max=1.5)

        v = controls[:, :, 0]
        omega = controls[:, :, 1]

        # Only suppress forward motion in true near-collision states. Penalizing
        # all low-clearance motion makes the robot crawl in narrow corridors.
        forward_loss = (danger * F.relu(v)).mean()

        # A stuck robot near a corner may need rotation, but making every
        # low-clearance sample rotate creates the omega spikes seen in v25.
        forward_intent = (F.relu(v).detach() / max(self.cfg.MAX_VEL, 1e-3)).clamp(max=1.0)
        turn_loss = (danger * forward_intent * F.relu(self.min_turn - torch.abs(omega))).mean()

        # Encourage movement toward increasing ESDF when clearance is low.
        dxy = pts[:, 1:, :] - pts[:, :-1, :]
        move_dir = F.normalize(dxy, dim=-1)
        grad_dir = F.normalize(grads[:, :-1, :].detach(), dim=-1)
        outward = (move_dir * grad_dir).sum(dim=-1)
        outward_loss = (danger * F.relu(0.10 - outward)).mean()

        # If the first pose is already near an obstacle, final clearance should improve.
        init_clear = values[:, 0].detach()
        final_clear = values[:, -1]
        near_init = (init_clear < self.margin).float()
        improve_loss = (
            near_init * F.relu(init_clear + self.hard_margin - final_clear)
        ).mean()

        # v41: re-balanced sub-terms to reduce wall-climbing WITHOUT the
        # "far & slow" degradation v40 showed. Penalizing all low clearance
        # (clearance/forward) pushed the policy a meter off walls and slowed it,
        # fighting expert wall-hugging. Instead we down-weight "don't be near a
        # wall" and up-weight the DIRECTIONAL escape terms (outward=move toward
        # increasing ESDF, improve=end farther from obstacle than start,
        # worst=fix the single worst trajectory point). These cut wall-climbing
        # by shaping HOW to leave, not by forbidding proximity. Coefficients are
        # config-overridable for tuning.
        c = self.cfg
        return (
            getattr(c, 'RECOVERY_W_CLEARANCE', 0.30) * clearance_loss
            + getattr(c, 'RECOVERY_W_WORST', 1.20) * worst_clearance_loss
            + getattr(c, 'RECOVERY_W_FORWARD', 0.15) * forward_loss
            + getattr(c, 'RECOVERY_W_TURN', 0.20) * turn_loss
            + getattr(c, 'RECOVERY_W_OUTWARD', 0.60) * outward_loss
            + getattr(c, 'RECOVERY_W_IMPROVE', 2.00) * improve_loss
        )


class CornerClearanceLoss(nn.Module):
    """v42: curvature-gated clearance — kills "trajectory painted onto the corner".

    The corner failure is a GEOMETRY failure: while turning hard (high |omega|),
    the path clips the inside apex of a convex obstacle. Existing terms penalize
    *proximity* (average / worst clearance) and are cancelled at corners by the
    wall-hugging expert imitation. This term penalizes low clearance ONLY where
    the robot is turning, so:
      - straight-line wall-following (omega ~= 0) is barely penalized -> no
        "far & slow" degradation like v40;
      - turning-into-a-wall (the corner-cut) gets a strong, discriminative
        gradient that pushes the turn to swing wider.

    Normal motion is gated by stabilized curvature |omega|/max(|v|, v_low).
    Near zero speed, a separate normalized rotation gate avoids a singular
    curvature while still requesting clearance for in-place turns.  The result
    is reduced with an optional top-k on the worst turning samples.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.margin = getattr(cfg, 'CORNER_MARGIN', cfg.ROBOT_RADIUS + 0.10)
        self.topk_ratio = getattr(cfg, 'CORNER_TOPK_RATIO', 0.25)
        self.max_omega = max(cfg.MAX_OMEGA, 1e-3)
        self.curvature_scale = max(getattr(cfg, 'CORNER_CURVATURE_SCALE', 1.0), 1e-3)
        self.low_speed = max(getattr(cfg, 'CORNER_LOW_SPEED', 0.15), 1e-3)

    def forward(self, positions, controls, esdf_map):
        # Clearance sampled along the executed trajectory (skip the fixed init
        # pose; align with the per-step controls that produce steps 1..n).
        pts = positions[:, 1:, :2]                              # (B, n, 2)
        values, _ = grid_sample_esdf(
            pts, esdf_map, self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
        )                                                       # (B, n)

        n = min(values.shape[1], controls.shape[1])
        values = values[:, :n]
        v = controls[:, :n, 0].abs()
        omega = controls[:, :n, 1].abs()

        clearance_violation = F.relu(self.margin - values)     # (B, n)
        curvature = omega / v.clamp_min(self.low_speed)
        moving_gate = curvature / (curvature + self.curvature_scale)
        rotation_gate = (
            (omega / self.max_omega).clamp(max=1.0)
            * torch.sigmoid((self.low_speed - v) / (0.25 * self.low_speed))
        )
        turn_gate = torch.maximum(moving_gate, rotation_gate).clamp(max=1.0)
        penalty = clearance_violation.square() * turn_gate     # (B, n)

        mean_penalty = penalty.mean()
        if self.topk_ratio > 0:
            k = max(1, int(round(n * self.topk_ratio)))
            worst = penalty.topk(k, dim=1).values.mean()
        else:
            worst = torch.zeros_like(mean_penalty)
        return mean_penalty + worst


class CorridorCenteringLoss(nn.Module):
    """Center the executed path only when obstacles are close on both sides.

    ESDF is sampled at symmetric lateral probes in the robot frame. A soft
    bilateral gate suppresses the objective in open space and during ordinary
    single-wall following; in a narrow corridor it minimizes the left/right
    clearance imbalance without directly penalizing forward velocity.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.offset = getattr(cfg, 'CORRIDOR_PROBE_OFFSET', cfg.ROBOT_RADIUS + 0.10)
        self.side_max = getattr(cfg, 'CORRIDOR_SIDE_MAX', 0.45)
        self.gate_temp = getattr(cfg, 'CORRIDOR_GATE_TEMP', 0.06)

    def forward(self, positions, esdf_map):
        poses = positions[:, 1:, :]
        yaw = poses[..., 2]
        lateral = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        left_pts = poses[..., :2] + self.offset * lateral
        right_pts = poses[..., :2] - self.offset * lateral
        left, _ = grid_sample_esdf(
            left_pts, esdf_map, self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
        )
        right, _ = grid_sample_esdf(
            right_pts, esdf_map, self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
        )

        # Both probes must be near obstacles. max(left, right) prevents a lone
        # wall from activating the centering objective.
        bilateral = torch.sigmoid(
            (self.side_max - torch.maximum(left, right)) / max(self.gate_temp, 1e-3)
        )
        imbalance = F.smooth_l1_loss(left, right, reduction='none')
        denom = bilateral.sum().clamp_min(1.0)
        return (imbalance * bilateral).sum() / denom


class ProgressSpeedLoss(nn.Module):
    """v25/v27: match target forward speed when it is safe to do so."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.deadband = getattr(cfg, 'PROGRESS_DEADBAND', 0.25)
        self.max_target_v = getattr(cfg, 'PROGRESS_MAX_TARGET_V', 0.8)
        self.kp = getattr(cfg, 'PROGRESS_SPEED_KP', 0.6)
        self.max_bearing = getattr(cfg, 'PROGRESS_MAX_BEARING', 0.9)
        self.safe_margin = getattr(cfg, 'SPEED_MATCH_SAFE_MARGIN', cfg.ROBOT_RADIUS + 0.12)
        self.safe_temp = getattr(cfg, 'SPEED_MATCH_SAFE_TEMP', 0.05)

    def forward(self, controls, init_pose, target_pos, target_vel=None,
                positions=None, esdf_map=None, visibility_gate=None):
        delta = target_pos - init_pose[:, :2]
        dist = torch.norm(delta, dim=-1)
        bearing = normalize_angle(torch.atan2(delta[:, 1], delta[:, 0]) - init_pose[:, 2])

        signed_err = dist - self.cfg.OBSERVATION_DISTANCE
        far_err = F.relu(signed_err - self.deadband)
        close_err = F.relu(-signed_err - self.deadband)
        dist_correction = self.kp * (far_err - close_err)

        if target_vel is not None:
            target_speed = torch.norm(target_vel, dim=-1)
        else:
            target_speed = torch.zeros_like(dist)

        desired_v = (target_speed + dist_correction).clamp(min=0.0, max=self.max_target_v)
        visible_forward = (torch.abs(bearing) < self.max_bearing).float()

        safe_gate = torch.ones_like(desired_v)
        if positions is not None and esdf_map is not None:
            horizon = min(self.cfg.HORIZON + 1, positions.shape[1])
            esdf_val, _ = grid_sample_esdf(
                positions[:, :horizon, :2], esdf_map,
                self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION
            )
            min_clear = esdf_val.min(dim=1).values.detach()
            safe_gate = torch.sigmoid((min_clear - self.safe_margin) / max(self.safe_temp, 1e-3))

        # First second matters most for receding-horizon deployment.
        horizon = min(10, controls.shape[1])
        v_early = controls[:, :horizon, 0]
        loss = F.smooth_l1_loss(
            v_early,
            desired_v.unsqueeze(1).expand_as(v_early),
            reduction='none',
        )
        gate = visible_forward.unsqueeze(1) * safe_gate.unsqueeze(1)
        if visibility_gate is not None:
            gate = gate * visibility_gate[:, :horizon]
        return (loss * gate).mean()


def integrate_controls(controls, init_pose, cfg):
    """
    用控制量积分出可执行轨迹。

    ControlPolicyHead already returns physically bounded controls in m/s and
    rad/s. Do not apply another tanh here, or the loss trajectory will see a
    compressed control range and distorted gradients.
    """
    v = controls[:, :, 0]
    omega = controls[:, :, 1]

    poses = [init_pose]
    for t in range(controls.shape[1]):
        prev = poses[-1]
        yaw = prev[:, 2]
        next_x = prev[:, 0] + cfg.DT * v[:, t] * torch.cos(yaw)
        next_y = prev[:, 1] + cfg.DT * v[:, t] * torch.sin(yaw)
        next_yaw = normalize_angle(prev[:, 2] + cfg.DT * omega[:, t])
        poses.append(torch.stack([next_x, next_y, next_yaw], dim=-1))

    return torch.stack(poses, dim=1)

class NonholoLoss(nn.Module):
    """非完整约束损失（DPtraj）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.threshold = 0.067  # 差速车不能横移

    def forward(self, positions):
        """
        惩罚横向运动（arc_dir × yaw_dir）
        """
        B, n_plus_1, _ = positions.shape

        if n_plus_1 < 2:
            return torch.tensor(0.0, device=positions.device)

        pts = positions[:, :, :2]  # (B, n+1, 2)
        yaw = positions[:, :, 2]   # (B, n+1)

        # 运动方向
        arcs = pts[:, 1:, :] - pts[:, :-1, :]  # (B, n, 2)
        arcs_norm = F.normalize(arcs, dim=-1)  # (B, n, 2)

        # 朝向方向（跳过最后一个）
        yaw_dir = torch.stack([torch.cos(yaw[:, :-1]), torch.sin(yaw[:, :-1])], dim=-1)  # (B, n, 2)

        # 叉积（横向分量）
        cross = arcs_norm[:, :, 0] * yaw_dir[:, :, 1] - arcs_norm[:, :, 1] * yaw_dir[:, :, 0]

        # 惩罚
        violation = cross ** 2 - self.threshold
        loss = positiveSmoothedL1(violation).mean()

        return loss

class CurvatureLoss(nn.Module):
    """曲率约束损失（DPtraj）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.kmax = 1.67  # 最大曲率

    def forward(self, positions):
        """
        约束：Δθ² ≤ k_max² · arc²
        """
        B, n_plus_1, _ = positions.shape

        if n_plus_1 < 2:
            return torch.tensor(0.0, device=positions.device)

        pts = positions[:, :, :2]  # (B, n+1, 2)
        yaw = positions[:, :, 2]   # (B, n+1)

        # 弧长
        arcs = torch.norm(pts[:, 1:, :] - pts[:, :-1, :], dim=-1) + 1e-6  # (B, n)

        # 角度变化
        delta_yaw = normalize_angle(yaw[:, 1:] - yaw[:, :-1])  # (B, n)

        # 约束：Δθ² ≤ k_max² · arc²
        violation = delta_yaw ** 2 - self.kmax ** 2 * arcs ** 2
        loss = positiveSmoothedL1(violation).mean()

        return loss

class UniformArcLoss(nn.Module):
    """弧长均匀性损失（DPtraj）"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, positions):
        """
        惩罚弧长方差
        """
        B, n_plus_1, _ = positions.shape

        if n_plus_1 < 2:
            return torch.tensor(0.0, device=positions.device)

        pts = positions[:, :, :2]  # (B, n+1, 2)
        arcs = torch.norm(pts[:, 1:, :] - pts[:, :-1, :], dim=-1)  # (B, n)

        # 方差 / 总弧长
        arc_std = torch.std(arcs, dim=1, unbiased=False)  # (B,)
        arc_sum = torch.sum(arcs, dim=1) + 1e-6  # (B,)

        loss = (arc_std / arc_sum).mean()

        return loss

class ConsistencyLoss(nn.Module):
    """时序平滑损失：惩罚相邻时间步控制量的突变"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, controls_current, controls_prev=None):
        """
        在 batch 内对控制序列做一阶差分平滑。
        不再依赖跨 batch 的 prev_controls（shuffle 后无时序关系）。

        参数：
            controls_current: (B, n, 2) [v, ω]
            controls_prev: 已废弃，保留参数兼容性
        """
        # 相邻时间步的控制量差
        diff = controls_current[:, 1:, :] - controls_current[:, :-1, :]
        loss = (diff ** 2).mean()
        return loss

def build_reference_trajectory(init_pose, target_pos, cfg, target_vel=None):
    """
    构造模仿学习用的参考轨迹：
        - 沿"机器人 ↔ 目标"连线移动；
        - 参考速度由 target 标量速度 + 观察距离误差修正组成；
        - 太远时比 target 略快，太近时比 target 略慢/后退；
        - yaw 始终面向目标。

    输入：
        init_pose: (B, 3) [x, y, yaw]，世界坐标
        target_pos: (B, 2) 世界坐标
        cfg: Config
        target_vel: (B, 2) 世界坐标目标速度，可选

    返回：
        ref_positions: (B, n+1, 3)
    """
    B = init_pose.shape[0]
    n = cfg.HORIZON
    device = init_pose.device

    # 朝向目标
    delta = target_pos - init_pose[:, :2]                    # (B, 2)
    dist = torch.norm(delta, dim=-1, keepdim=True) + 1e-6    # (B, 1)
    direction = delta / dist                                  # (B, 2)
    target_yaw = torch.atan2(delta[:, 1], delta[:, 0])       # (B,)

    signed_err = dist.squeeze(-1) - cfg.OBSERVATION_DISTANCE  # (B,)
    deadband = getattr(cfg, 'PROGRESS_DEADBAND', 0.25)
    far_err = F.relu(signed_err - deadband)
    close_err = F.relu(-signed_err - deadband)
    dist_correction = getattr(cfg, 'PROGRESS_SPEED_KP', 0.6) * (far_err - close_err)

    if target_vel is not None:
        target_speed = torch.norm(target_vel, dim=-1)
    else:
        target_speed = torch.zeros(B, device=device, dtype=init_pose.dtype)

    max_ref_v = getattr(cfg, 'PROGRESS_MAX_TARGET_V', cfg.MAX_VEL * 0.6)
    min_ref_v = -0.4 * cfg.MAX_VEL
    ref_speed = (target_speed + dist_correction).clamp(min=min_ref_v, max=max_ref_v)
    travel = (ref_speed * cfg.DT * n).unsqueeze(-1)  # (B, 1)
    max_local_travel = cfg.LOCAL_MAP_SIZE * 0.5 - cfg.ANCHOR_CELL_SIZE
    travel = travel.clamp(min=-max_local_travel, max=max_local_travel)

    # t = 0..n 的归一化弧长
    s = torch.linspace(0.0, 1.0, n + 1, device=device).unsqueeze(0)  # (1, n+1)
    arc = travel * s                                                  # (B, n+1)

    ref_xy = init_pose[:, :2].unsqueeze(1) + direction.unsqueeze(1) * arc.unsqueeze(-1)
    ref_yaw = target_yaw.unsqueeze(1).expand(-1, n + 1)               # (B, n+1)

    return torch.cat([ref_xy, ref_yaw.unsqueeze(-1)], dim=-1)         # (B, n+1, 3)


class ImitationLoss(nn.Module):
    """模仿参考轨迹（warm-up 用）：位置 + 朝向"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, positions, init_pose, target_pos, target_vel=None, step_weight=None):
        ref = build_reference_trajectory(init_pose, target_pos, self.cfg, target_vel)  # (B, n+1, 3)

        pos_err = (positions[:, 1:, :2] - ref[:, 1:, :2]).square().mean(dim=-1)
        rot_err = normalize_angle(positions[:, 1:, 2] - ref[:, 1:, 2])
        rot_step = 1 - torch.cos(rot_err)
        if step_weight is not None:
            pos_err = pos_err * step_weight
            rot_step = rot_step * step_weight
        pos_loss = pos_err.mean()
        rot_loss = rot_step.mean()

        return pos_loss, rot_loss


class AnchorSupervisionLoss(nn.Module):
    """
    Anchor 显式监督：用 imitation reference 轨迹算出每步 ground-truth
    anchor 格子，对 anchor_logits 做交叉熵。

    避免 anchor head 在只有位置 loss 时退化为均匀分布或塌缩到一格。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, anchor_logits, init_pose, target_pos, target_vel=None, visibility_gate=None):
        """
        anchor_logits: (B, n, G, G)  原始 logits（未 softmax）
        init_pose:     (B, 3)  世界坐标 [x, y, yaw]
        target_pos:    (B, 2)  世界坐标

        注意：decode_positions 定义网络输出为"相对于 t=0 局部系的绝对位置"，
        所以监督标签也必须用 init_yaw 投影到 t=0 局部系。
        """
        cfg = self.cfg
        B, n, G, _ = anchor_logits.shape
        device = anchor_logits.device

        # 1) reference 轨迹（世界系）
        ref = build_reference_trajectory(init_pose, target_pos, cfg, target_vel)  # (B, n+1, 3)

        # 2) 将 ref 的每个时间步转到 t=0 局部系（与 decode_positions 语义一致）
        init_xy = init_pose[:, :2].unsqueeze(1)  # (B, 1, 2)
        init_yaw = init_pose[:, 2]               # (B,)
        cos_init = torch.cos(init_yaw).unsqueeze(1)  # (B, 1)
        sin_init = torch.sin(init_yaw).unsqueeze(1)  # (B, 1)

        # 世界坐标 → t=0 局部坐标
        d = ref[:, 1:, :2] - init_xy  # (B, n, 2) 相对于起点的世界位移
        local_x =  cos_init * d[..., 0] + sin_init * d[..., 1]  # (B, n)
        local_y = -sin_init * d[..., 0] + cos_init * d[..., 1]  # (B, n)

        # 3) 局部坐标 -> grid 索引
        gx_f = local_x / cfg.ANCHOR_CELL_SIZE + G / 2
        gy_f = local_y / cfg.ANCHOR_CELL_SIZE + G / 2

        gx = gx_f.round().long()
        gy = gy_f.round().long()

        valid = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)  # (B, n)
        if visibility_gate is not None:
            valid = valid & (visibility_gate > 0.5)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 4) flat index for cross_entropy
        gx_c = gx.clamp(0, G - 1)
        gy_c = gy.clamp(0, G - 1)
        target_idx = gy_c * G + gx_c                          # (B, n)

        logits_flat = anchor_logits.reshape(B * n, G * G)
        target_flat = target_idx.reshape(B * n)
        valid_flat = valid.reshape(B * n)

        loss = F.cross_entropy(logits_flat[valid_flat],
                               target_flat[valid_flat],
                               reduction='mean')
        return loss

class LateralLoss(nn.Module):
    """侧移惩罚：差速机器人不能横移，显式约束侧向速度为零"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg.DT

    def forward(self, positions):
        """
        参数：
            positions: (B, n+1, 3) [x, y, yaw]
        """
        dx = positions[:, 1:, 0] - positions[:, :-1, 0]  # (B, n)
        dy = positions[:, 1:, 1] - positions[:, :-1, 1]
        cos_yaw = torch.cos(positions[:, :-1, 2])
        sin_yaw = torch.sin(positions[:, :-1, 2])

        # 侧向速度 = 垂直于车头方向的位移分量 / dt
        v_lateral = (-sin_yaw * dx + cos_yaw * dy) / self.dt
        return (v_lateral ** 2).mean()


class AnchorEntropyLoss(nn.Module):
    """
    Anchor 熵正则化：避免分布塌缩到单一格子。

    当平均熵低于 target_entropy 时施加惩罚（hinge loss）；
    高于阈值时 loss=0，不强行让分布变得无意义地平。

    target_entropy 默认 2.5（20×20 均匀分布的熵约 5.99，1 个格子 = 0）
    """
    def __init__(self, cfg, target_entropy=2.5):
        super().__init__()
        self.cfg = cfg
        self.target_entropy = target_entropy

    def forward(self, anchor_probs):
        """
        anchor_probs: (B, n, G, G) 概率分布
        """
        B, n = anchor_probs.shape[:2]
        probs_flat = anchor_probs.reshape(B, n, -1)  # (B, n, G*G)
        # 防止 log(0) 数值不稳定
        entropy = -(probs_flat * torch.log(probs_flat + 1e-9)).sum(dim=-1)  # (B, n)
        mean_entropy = entropy.mean()
        # hinge：低于目标值才惩罚
        loss = F.relu(self.target_entropy - mean_entropy)
        return loss

class TotalLoss(nn.Module):
    """总损失"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 实例化所有损失
        self.angle_loss = AngleLoss(cfg)
        self.dist_loss = DistanceLoss(cfg)
        self.moving_dist_loss = MovingTargetDistanceLoss(cfg)
        self.min_tracking_dist_loss = MinTrackingDistanceLoss(cfg)
        self.visibility_loss = VisibilityLoss(cfg)
        self.collision_loss = CollisionLoss(cfg)
        self.feasibility_loss = FeasibilityLoss(cfg)
        self.smooth_loss = SmoothLoss(cfg)
        self.nonholo_loss = NonholoLoss(cfg)
        self.curvature_loss = CurvatureLoss(cfg)
        self.uniform_arc_loss = UniformArcLoss(cfg)
        self.consistency_loss = ConsistencyLoss(cfg)
        self.control_smooth_loss = ControlSmoothLoss(cfg)
        self.omega_margin_loss = OmegaMarginLoss(cfg)
        self.clearance_recovery_loss = ClearanceRecoveryLoss(cfg)
        self.corner_loss = CornerClearanceLoss(cfg)
        self.corridor_center_loss = CorridorCenteringLoss(cfg)
        self.progress_speed_loss = ProgressSpeedLoss(cfg)
        self.expert_control_loss = ExpertControlLoss(cfg)
        self.expert_cmd_seq_loss = ExpertCmdSeqLoss(cfg)
        self.expert_plan_loss = ExpertPlanLoss(cfg)
        self.imitation_loss = ImitationLoss(cfg)
        self.anchor_sup_loss = AnchorSupervisionLoss(cfg)
        self.anchor_entropy_loss = AnchorEntropyLoss(cfg)
        self.lateral_loss = LateralLoss(cfg)

    def forward(self, outputs, batch, weights, prev_controls=None):
        """
        参数：
            outputs: 网络输出字典
            batch: 数据 batch
            weights: 损失权重字典（来自课程学习）
            prev_controls: 上一帧的控制（用于一致性）

        返回：
            total_loss: scalar
            loss_dict: 各项损失的字典（用于日志）
        """
        positions = outputs['positions']  # (B, n+1, 3)
        controls = outputs['controls']    # (B, n, 2)
        anchor_probs = outputs.get('anchor_probs')

        target_pos = batch['target_pos']  # (B, 2)
        global_esdf = batch['global_esdf']  # (B, H, W)
        exec_positions = positions
        if 'init_pose' in batch:
            exec_positions = integrate_controls(controls, batch['init_pose'], self.cfg)

        if getattr(self.cfg, 'MPC_COST_ONLY', False):
            visibility_gate = target_visibility_gate(
                exec_positions, target_pos, global_esdf, self.cfg,
                target_vel=batch.get('target_vel'),
                target_future_pos=batch.get('target_future_pos'),
                target_future_mask=batch.get('target_future_pos_mask'),
            )
            loss_dict = {}
            if weights.get('angle', 0.0) > 0:
                loss_dict['angle'] = self.angle_loss(
                    exec_positions, target_pos, step_weight=visibility_gate
                )
            if weights.get('dist', 0.0) > 0:
                loss_dict['dist'] = self.dist_loss(
                    exec_positions, target_pos, step_weight=visibility_gate
                )
            if weights.get('visibility', 0.0) > 0:
                loss_dict['visibility'] = self.visibility_loss(
                    exec_positions, target_pos, global_esdf,
                    target_vel=batch.get('target_vel'),
                    target_future_pos=batch.get('target_future_pos'),
                    target_future_mask=batch.get('target_future_pos_mask'),
                )
            if weights.get('collision', 0.0) > 0:
                loss_dict['collision'] = self.collision_loss(exec_positions, global_esdf)
            # Self-supervised recovery: penalizes hugging walls without leaving
            # (outward/improve terms) and pushes trajectories off obstacle
            # corners (worst-clearance). Does not need expert demonstrations, so
            # it teaches wall-escape behavior the imitation data never shows.
            if weights.get('clearance_recovery', 0.0) > 0:
                loss_dict['clearance_recovery'] = self.clearance_recovery_loss(
                    exec_positions, controls, global_esdf
                )
            # Curvature-gated clearance: penalizes low clearance only while
            # turning, targeting the "trajectory cuts the inside corner" failure
            # without slowing straight-line wall-following.
            if weights.get('corner', 0.0) > 0:
                loss_dict['corner'] = self.corner_loss(
                    exec_positions, controls, global_esdf
                )
            if weights.get('corridor_center', 0.0) > 0:
                loss_dict['corridor_center'] = self.corridor_center_loss(
                    exec_positions, global_esdf
                )
            if weights.get('feasibility', 0.0) > 0:
                loss_dict['feasibility'] = self.feasibility_loss(
                    controls, current_control=batch.get('current_control')
                )
            if weights.get('smooth', 0.0) > 0:
                loss_dict['smooth'] = self.smooth_loss(exec_positions)
            if weights.get('progress_speed', 0.0) > 0 and 'init_pose' in batch:
                loss_dict['progress_speed'] = self.progress_speed_loss(
                    controls,
                    batch['init_pose'],
                    target_pos,
                    target_vel=batch.get('target_vel'),
                    positions=exec_positions,
                    esdf_map=global_esdf,
                    visibility_gate=visibility_gate,
                )

            # Expert supervision. Without it a direct-control policy collapses to
            # the v≈0, omega≈0 trivial optimum of the pure cost objective (all
            # collision/feasibility/smooth terms vanish when the robot never
            # moves). These anchor the policy to expert behavior early and are
            # decayed by the curriculum so cost terms take over shaping.
            if weights.get('expert_cmd_seq', 0.0) > 0 and 'expert_cmd_seq' in batch:
                loss_dict['expert_cmd_seq'] = self.expert_cmd_seq_loss(
                    controls,
                    batch['expert_cmd_seq'],
                    batch.get('expert_cmd_seq_mask'),
                    batch.get('sample_weight'),
                )
            if weights.get('expert_plan', 0.0) > 0 and 'expert_plan' in batch:
                loss_dict['expert_plan'] = self.expert_plan_loss(
                    exec_positions,
                    batch['expert_plan'],
                    batch.get('expert_plan_mask'),
                    batch.get('sample_weight'),
                )

            # Moving-target distance keeping (problem: tracks from too far away).
            # Gated by visibility so it does not pull the robot through walls.
            if weights.get('moving_dist', 0.0) > 0:
                loss_dict['moving_dist'] = self.moving_dist_loss(
                    exec_positions, target_pos,
                    target_vel=batch.get('target_vel'),
                    step_weight=visibility_gate,
                    target_future_pos=batch.get('target_future_pos'),
                    target_future_mask=batch.get('target_future_pos_mask'),
                )
            if weights.get('min_tracking_dist', 0.0) > 0:
                loss_dict['min_tracking_dist'] = self.min_tracking_dist_loss(
                    exec_positions, target_pos,
                    target_vel=batch.get('target_vel'),
                    target_future_pos=batch.get('target_future_pos'),
                    target_future_mask=batch.get('target_future_pos_mask'),
                )

            total_loss = sum(
                weights.get(key, 0.0) * torch.clamp(value, max=1e6)
                for key, value in loss_dict.items()
            )
            return total_loss, loss_dict

        visibility_gate = target_visibility_gate(
            exec_positions, target_pos, global_esdf, self.cfg,
            target_vel=batch.get('target_vel'),
            target_future_pos=batch.get('target_future_pos'),
            target_future_mask=batch.get('target_future_pos_mask'),
        )

        loss_dict = {}

        # MPC 原始 6 项：任务/安全项看控制积分轨迹，更贴近部署时实际执行结果。
        if weights['angle'] > 0:
            loss_dict['angle'] = self.angle_loss(exec_positions, target_pos, step_weight=visibility_gate)
        else:
            loss_dict['angle'] = torch.tensor(0.0, device=positions.device)

        if weights['dist'] > 0:
            loss_dict['dist'] = self.dist_loss(exec_positions, target_pos, step_weight=visibility_gate)
        else:
            loss_dict['dist'] = torch.tensor(0.0, device=positions.device)

        if weights.get('min_tracking_dist', 0.0) > 0:
            loss_dict['min_tracking_dist'] = self.min_tracking_dist_loss(
                exec_positions, target_pos,
                target_vel=batch.get('target_vel'),
                target_future_pos=batch.get('target_future_pos'),
                target_future_mask=batch.get('target_future_pos_mask'),
            )
        else:
            loss_dict['min_tracking_dist'] = torch.tensor(0.0, device=positions.device)

        if weights['visibility'] > 0:
            loss_dict['visibility'] = self.visibility_loss(
                exec_positions, target_pos, global_esdf,
                target_vel=batch.get('target_vel'),
                target_future_pos=batch.get('target_future_pos'),
                target_future_mask=batch.get('target_future_pos_mask'),
            )
        else:
            loss_dict['visibility'] = torch.tensor(0.0, device=positions.device)

        if weights['collision'] > 0:
            loss_dict['collision'] = self.collision_loss(exec_positions, global_esdf)
        else:
            loss_dict['collision'] = torch.tensor(0.0, device=positions.device)

        if weights['feasibility'] > 0:
            loss_dict['feasibility'] = self.feasibility_loss(
                controls, current_control=batch.get('current_control')
            )
        else:
            loss_dict['feasibility'] = torch.tensor(0.0, device=positions.device)

        if weights['smooth'] > 0:
            loss_dict['smooth'] = 0.5 * self.smooth_loss(positions) + self.smooth_loss(exec_positions)
        else:
            loss_dict['smooth'] = torch.tensor(0.0, device=positions.device)

        # DPtraj 几何项
        if weights['nonholo'] > 0:
            loss_dict['nonholo'] = self.nonholo_loss(positions)
        else:
            loss_dict['nonholo'] = torch.tensor(0.0, device=positions.device)

        if weights['curvature'] > 0:
            loss_dict['curvature'] = self.curvature_loss(positions)
        else:
            loss_dict['curvature'] = torch.tensor(0.0, device=positions.device)

        if weights['uniform'] > 0:
            loss_dict['uniform'] = self.uniform_arc_loss(positions)
        else:
            loss_dict['uniform'] = torch.tensor(0.0, device=positions.device)

        # 新增项
        if weights['consistency'] > 0:
            loss_dict['consistency'] = self.consistency_loss(controls, prev_controls)
        else:
            loss_dict['consistency'] = torch.tensor(0.0, device=positions.device)

        if weights.get('control_smooth', 0.0) > 0:
            loss_dict['control_smooth'] = self.control_smooth_loss(controls)
        else:
            loss_dict['control_smooth'] = torch.tensor(0.0, device=positions.device)

        if weights.get('omega_margin', 0.0) > 0:
            loss_dict['omega_margin'] = self.omega_margin_loss(controls)
        else:
            loss_dict['omega_margin'] = torch.tensor(0.0, device=positions.device)

        if weights.get('clearance_recovery', 0.0) > 0:
            loss_dict['clearance_recovery'] = self.clearance_recovery_loss(
                exec_positions, controls, global_esdf
            )
        else:
            loss_dict['clearance_recovery'] = torch.tensor(0.0, device=positions.device)

        if weights.get('progress_speed', 0.0) > 0 and 'init_pose' in batch:
            loss_dict['progress_speed'] = self.progress_speed_loss(
                controls, batch['init_pose'], target_pos,
                target_vel=batch.get('target_vel'),
                positions=exec_positions,
                esdf_map=global_esdf,
                visibility_gate=visibility_gate
            )
        else:
            loss_dict['progress_speed'] = torch.tensor(0.0, device=positions.device)

        if weights.get('expert_control', 0.0) > 0 and 'expert_control' in batch:
            loss_dict['expert_control'] = self.expert_control_loss(
                controls, batch['expert_control']
            )
        else:
            loss_dict['expert_control'] = torch.tensor(0.0, device=positions.device)

        if weights.get('expert_cmd_seq', 0.0) > 0 and 'expert_cmd_seq' in batch:
            loss_dict['expert_cmd_seq'] = self.expert_cmd_seq_loss(
                controls,
                batch['expert_cmd_seq'],
                batch.get('expert_cmd_seq_mask'),
                batch.get('sample_weight'),
            )
        else:
            loss_dict['expert_cmd_seq'] = torch.tensor(0.0, device=positions.device)

        if weights.get('expert_plan', 0.0) > 0 and 'expert_plan' in batch:
            loss_dict['expert_plan'] = self.expert_plan_loss(
                exec_positions,
                batch['expert_plan'],
                batch.get('expert_plan_mask'),
                batch.get('sample_weight'),
            )
        else:
            loss_dict['expert_plan'] = torch.tensor(0.0, device=positions.device)

        # 模仿学习（warm-up）：imitate = 总开关；pos/rot = block 内的相对权重
        if weights['imitate'] > 0 and 'init_pose' in batch:
            pos_loss, rot_loss = self.imitation_loss(
                positions, batch['init_pose'], target_pos,
                target_vel=batch.get('target_vel'),
                step_weight=visibility_gate
            )
            loss_dict['pos'] = pos_loss
            loss_dict['rot'] = rot_loss
        else:
            loss_dict['pos'] = torch.tensor(0.0, device=positions.device)
            loss_dict['rot'] = torch.tensor(0.0, device=positions.device)

        # imitate 这一项本身无独立 loss，置 0；通过 pos/rot 反传
        loss_dict['imitate'] = torch.tensor(0.0, device=positions.device)

        # Anchor 监督（与 imitation 共享 reference，用 anchor 权重单独控制）
        if weights.get('anchor', 0.0) > 0 and 'init_pose' in batch and 'anchor_logits' in outputs:
            loss_dict['anchor'] = self.anchor_sup_loss(
                outputs['anchor_logits'], batch['init_pose'], target_pos,
                target_vel=batch.get('target_vel'),
                visibility_gate=visibility_gate
            )
        else:
            loss_dict['anchor'] = torch.tensor(0.0, device=positions.device)

        # Anchor 熵正则化：防止分布塌缩
        if weights.get('anchor_entropy', 0.0) > 0 and anchor_probs is not None:
            loss_dict['anchor_entropy'] = self.anchor_entropy_loss(anchor_probs)
        else:
            loss_dict['anchor_entropy'] = torch.tensor(0.0, device=positions.device)

        # 侧移惩罚：差速机器人不能横移
        if weights.get('lateral', 0.0) > 0:
            loss_dict['lateral'] = self.lateral_loss(positions)
        else:
            loss_dict['lateral'] = torch.tensor(0.0, device=positions.device)

        # 加权求和：pos/rot 在 weights 字典里被 imitate 这个总开关 gate
        gated_weights = dict(weights)
        gated_weights['pos'] = weights.get('pos', 0.0) * weights.get('imitate', 0.0)
        gated_weights['rot'] = weights.get('rot', 0.0) * weights.get('imitate', 0.0)

        total_loss = sum(gated_weights.get(key, 0.0) * torch.clamp(loss_dict[key], max=1e4)
                         for key in loss_dict.keys())

        return total_loss, loss_dict

if __name__ == '__main__':
    # 测试损失函数
    from config import Config, get_loss_weights

    cfg = Config()
    device = cfg.DEVICE

    # 模拟数据
    B, n = 4, 30
    positions = torch.randn(B, n + 1, 3).to(device)
    controls = torch.randn(B, n, 2).to(device)
    anchor_probs = torch.randn(B, n, 20, 20).softmax(dim=-1).to(device)

    target_pos = torch.randn(B, 2).to(device)
    global_esdf = torch.randn(B, 400, 400).to(device)

    outputs = {
        'positions': positions,
        'controls': controls,
        'anchor_probs': anchor_probs
    }

    batch = {
        'target_pos': target_pos,
        'global_esdf': global_esdf
    }

    # 测试
    criterion = TotalLoss(cfg).to(device)
    weights = get_loss_weights(epoch=5)

    total_loss, loss_dict = criterion(outputs, batch, weights)

    print("=" * 60)
    print("Loss function test")
    print("=" * 60)
    print(f"Total loss: {total_loss.item():.4f}")
    print("\nIndividual losses:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
