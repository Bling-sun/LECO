"""Direct-control neural tracking policy.

The policy encodes the local ESDF, target visibility, and robot/target state,
then predicts the exact [v, omega] sequence executed by deployment.  Positions
are produced only by integrating those controls with the shared Euler model.
"""
import torch
import torch.nn as nn
import numpy as np
from config import Config

#### ============================== 基础特征提取模块 =================================================
# 双层卷积模块
## 常用于 CNN 特征提取的基础模块，类似于 UNet 的基本单元
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一层：3x3 卷积，不使用 bias（因为后面接了 BatchNorm）
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层：相同的 3x3 卷积，进一步提取特征
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 空间注意力机制
## 让模型聚焦于图像（或 ESDF 障碍物图）中更重要的空间区域
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # 经过 sigmoid 将权重限制在 0~1 之间
        attn = torch.sigmoid(self.conv(x))
        # 将权重乘回原特征图，实现“重要区域放大，次要区域缩小
        return x * attn

# 位置编码
# 由于后面使用了 Transformer 结构，Transformer 本身无法感知序列的位置顺序，需要人为加上一维或二维的位置编码（这里使用的是经典的正余弦位置编码）。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


#### ============================== 编码器模块 (感知与状态提取) =================================================
# ESDF 编码器
## 将环境的 2D ESDF 地图转化为高维特征向量
class ESDFEncoder(nn.Module):
    """v1 ESDF 编码器：CNN + Spatial Attention + Self-Attention"""
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = DoubleConv(1, cfg.CNN_CHANNELS[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(cfg.CNN_CHANNELS[0], cfg.CNN_CHANNELS[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(cfg.CNN_CHANNELS[1], cfg.CNN_CHANNELS[2])
        self.spatial_attn = SpatialAttention(cfg.CNN_CHANNELS[2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.pos_enc = PositionalEncoding(cfg.CNN_CHANNELS[2])
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.CNN_CHANNELS[2],
                nhead=cfg.ATTN_HEADS,
                dim_feedforward=cfg.ATTN_DIM,
                dropout=0.1,
                batch_first=True
            ) for _ in range(cfg.ATTN_LAYERS)
        ])
        self.layer_norm = nn.LayerNorm(cfg.CNN_CHANNELS[2])

    def forward(self, esdf):
        x = self.pool1(self.conv1(esdf))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.spatial_attn(x)
        feature_map = x
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_flat = self.pos_enc(x_flat)
        for attn_layer in self.attn_layers:
            x_flat = attn_layer(x_flat)
        x_flat = self.layer_norm(x_flat)
        x_attn = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        global_feat = self.gap(x_attn).squeeze(-1).squeeze(-1)
        return feature_map, global_feat

# VisibilityEncoder 与 StateEncoder（能见度与状态编码器）
## 将低维的一维传感器或状态数据映射到统一的隐空间。
class VisibilityEncoder(nn.Module):
    """v1 原始版本: 2层 MLP"""
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.VISIBILITY_SAMPLES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, visibility_ray):
        return self.mlp(visibility_ray)


class StateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.STATE_DIM + cfg.TARGET_DIM + cfg.HISTORY_DIM
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

    def forward(self, robot_state, target_rel, history):
        x = torch.cat([robot_state, target_rel, history], dim=1)
        return self.mlp(x)

# 直接控制策略输出头
class ControlPolicyHead(nn.Module):
    """Direct control-sequence policy head.

    v30 removes the anchor/residual intermediate representation. The network
    predicts the exact controls that deployment will execute, and losses are
    applied to the trajectory obtained by integrating these controls.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.HORIZON
        self.time_embed_dim = cfg.TIME_EMBED_DIM
        self.time_embed = nn.Embedding(cfg.HORIZON, self.time_embed_dim)

        fusion_dim = cfg.CNN_CHANNELS[2] + 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(128 + self.time_embed_dim, cfg.GRU_HIDDEN, batch_first=True)
        self.control_head = nn.Sequential(
            nn.Linear(cfg.GRU_HIDDEN, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, esdf_feat, vis_feat, state_feat):
        B = esdf_feat.size(0)
        device = esdf_feat.device

        fused = torch.cat([esdf_feat, vis_feat, state_feat], dim=1)
        fused = self.fusion(fused)

        time_ids = torch.arange(self.horizon, device=device)
        time_emb = self.time_embed(time_ids).unsqueeze(0).expand(B, -1, -1)
        fused_expand = fused.unsqueeze(1).expand(-1, self.horizon, -1)
        gru_input = torch.cat([fused_expand, time_emb], dim=-1)

        gru_out, _ = self.gru(gru_input)
        raw = self.control_head(gru_out)

        v = self.cfg.MAX_VEL * torch.tanh(raw[:, :, 0])
        omega = self.cfg.MAX_OMEGA * torch.tanh(raw[:, :, 1])
        return torch.stack([v, omega], dim=-1)

#### ============================= 核心策略组装模块  =================================================
class TrackingPolicy(nn.Module):
    """v30 direct-control tracking policy."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.esdf_encoder = ESDFEncoder(cfg)
        self.vis_encoder = VisibilityEncoder(cfg)
        self.state_encoder = StateEncoder(cfg)
        self.control_head = ControlPolicyHead(cfg)

    # 前向运动学，将控制量积分为坐标轨迹
    # 核心物理仿真方法：integrate_controls (控制量积分)根据非圆轮单驱（差速）机器人运动学模型（Kinematic Bicycle/Unicycle Model），将预测的 $v$ 和 $\omega$ 积分转换为未来的 $x, y, \theta$ 轨迹。
    def integrate_controls(self, init_pose, controls):
        poses = [init_pose]
        for t in range(controls.shape[1]):
            prev = poses[-1]
            yaw = prev[:, 2]
            v = controls[:, t, 0]
            omega = controls[:, t, 1]
            next_x = prev[:, 0] + self.cfg.DT * v * torch.cos(yaw)
            next_y = prev[:, 1] + self.cfg.DT * v * torch.sin(yaw)
            next_yaw = torch.atan2(
                torch.sin(prev[:, 2] + self.cfg.DT * omega),
                torch.cos(prev[:, 2] + self.cfg.DT * omega)
            )
            poses.append(torch.stack([next_x, next_y, next_yaw], dim=-1))
        return torch.stack(poses, dim=1)

    # 主前向函数，整合所有模块，输出最终的控制量和轨迹
    def forward(self, esdf, robot_state, target_rel, history, visibility_ray, init_pose=None):
        esdf_map, esdf_feat = self.esdf_encoder(esdf)
        vis_feat = self.vis_encoder(visibility_ray)
        state_feat = self.state_encoder(robot_state, target_rel, history)

        controls = self.control_head(
            esdf_feat, vis_feat, state_feat
        )

        if init_pose is not None:
            positions = self.integrate_controls(init_pose, controls)
        else:
            zero_pose = torch.zeros(esdf.size(0), 3, device=esdf.device, dtype=controls.dtype)
            positions = self.integrate_controls(zero_pose, controls)

        return {
            'positions': positions,
            'controls': controls,
            'esdf_map': esdf_map,
        }


# 测试入口 (test_network)
def test_network():
    cfg = Config()
    model = TrackingPolicy(cfg).to(cfg.DEVICE)
    B = 4
    esdf = torch.randn(B, 1, 80, 80).to(cfg.DEVICE)
    robot_state = torch.randn(B, 2).to(cfg.DEVICE)
    target_rel = torch.randn(B, cfg.TARGET_DIM).to(cfg.DEVICE)
    history = torch.randn(B, 8).to(cfg.DEVICE)
    visibility_ray = torch.randn(B, 8).to(cfg.DEVICE)
    outputs = model(esdf, robot_state, target_rel, history, visibility_ray)
    print("=" * 60)
    print("Direct-control network forward pass test")
    print("=" * 60)
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
    print("Test passed!")


if __name__ == '__main__':
    test_network()
