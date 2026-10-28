"""
配置文件 - 所有超参数集中管理
"""
import copy
import torch


# Checkpoints written before config_dict only pickled instance overrides; class
# defaults such as velocity limits were absent.  Preserve their original output
# scaling instead of silently applying today's faster platform limits.
LEGACY_CHECKPOINT_DEFAULTS = {
    'MAX_VEL': 2.0,
    'MAX_OMEGA': 1.2,
    'PROGRESS_MAX_TARGET_V': 1.2,
}

class Config:
    # ==================== 机器人参数（从 planner.yaml 复制） ====================
    ROBOT_RADIUS = 0.25          # 机器人半径 (m)
    MAX_VEL = 5.6                # 平台线速度硬上限 |v| (m/s)
    MAX_OMEGA = 4.0              # 平台角速度硬上限 |ω| (rad/s)
    MAX_ACC = 3.0                # 最大线加速度 (m/s²)
    MAX_ALPHA = 2.0              # 最大角加速度 (rad/s²)

    # ==================== 地图参数 ====================
    MAP_SIZE = 20.0              # 全局地图大小 (m)
    MAP_RESOLUTION = 0.05        # 地图分辨率 (m/pixel)

    # 局部 ESDF patch（机器人坐标系）
    LOCAL_MAP_SIZE = 8.0         # v27b: 局部地图大小 (m)，覆盖 8m×8m，支持同速预测
    LOCAL_MAP_PIXELS = 80        # 80×80 像素
    LOCAL_MAP_RES = LOCAL_MAP_SIZE / LOCAL_MAP_PIXELS  # 0.1 m/pixel

    # Inactive legacy-reference parameters. The deployed v30+ policy predicts
    # controls directly and strict loading intentionally rejects anchor models.
    ANCHOR_GRID_SIZE = 20        # 20×20 anchor grid
    ANCHOR_CELL_SIZE = LOCAL_MAP_SIZE / ANCHOR_GRID_SIZE  # 0.4 m/cell

    # ==================== MPC 参数 ====================
    DT = 0.1                     # 控制周期 (s)
    HORIZON = 30                 # 预测步数（3 秒）

    # ==================== 网络架构参数 ====================
    # 输入通道
    ESDF_CHANNELS = 1            # ESDF 距离场

    # Visibility ray 采样
    VISIBILITY_SAMPLES = 8       # 自身→目标连线上采样点数

    # 状态维度
    STATE_DIM = 2                # [v, ω]
    TARGET_DIM = 6               # [ρ, sin(θ), cos(θ), sin(θ_v), cos(θ_v), |v_target|]
    HISTORY_DIM = 8              # 历史 4 步控制 [v, ω] × 4

    # 网络隐藏层
    CNN_CHANNELS = [16, 32, 128]  # CNN 通道数（最后一层 128 保证 ESDF 特征在融合时不被稀释）
    ATTN_DIM = 256               # Self-attention FFN 维度
    ATTN_HEADS = 4               # Attention 头数
    ATTN_LAYERS = 2              # Attention 层数
    GRU_HIDDEN = 128             # GRU 隐藏层

    # Anchor-residual 头
    # 残差只负责半个格子内的精细微调，格间跳跃由分类头（Gumbel-Softmax）负责
    RESIDUAL_RANGE = 0.2         # ±0.2m = ANCHOR_CELL_SIZE / 2 for 8m/20 anchor grid

    # Gumbel-Softmax 温度（余弦退火）
    GUMBEL_TAU_START = 1.0
    GUMBEL_TAU_END = 0.35

    # 时间步编码维度
    TIME_EMBED_DIM = 16

    # ==================== 损失函数权重（初始值，会被课程学习覆盖） ====================
    # Keep the core objective aligned with src/tracker/src/planner/mpc.cpp:
    # angle, tracking distance, occlusion, collision, feasibility, smoothness.
    # The neural trainer also uses visibility-gated speed matching so the model
    # learns to move with the target when that command is safe.
    MPC_COST_ONLY = True

    # MPC 原始 6 项
    LAMBDA_A = 1.0               # 角度对齐
    LAMBDA_D = 1.0               # 跟踪距离
    LAMBDA_O = 1.0               # 遮挡
    LAMBDA_C = 1e3               # 碰撞（会 ramp-up 到 1e9）
    LAMBDA_F = 1e5               # 可行性
    LAMBDA_S = 1e3               # 平滑

    # DPtraj 几何项
    LAMBDA_NONHOLO = 500.0       # 非完整约束
    LAMBDA_CURVATURE = 500.0     # 曲率约束
    LAMBDA_UNIFORM_ARC = 100.0   # 弧长均匀性

    # 新增项
    LAMBDA_CONSISTENCY = 0.01    # 滚动一致性
    LAMBDA_IMITATE = 1.0         # 模仿学习（warm-up 用，会退火到 0）

    # 观察距离（跟踪目标的期望距离）
    OBSERVATION_DISTANCE = 2.5   # m
    MIN_TRACKING_DISTANCE = 0.70 # 与移动目标之间不可低于的安全距离 (m)
    BEST_TOO_CLOSE_ACCEPT = 0.0  # checkpoint 不接受任何低于最小距离的轨迹
    OMEGA_SOFT_LIMIT_RATIO = 0.75  # v24: 超过 75% 角速度上限就开始软惩罚
    BEST_OBS_ERR_ACCEPT = 0.15     # checkpoint 选择: 观察距离误差小于此值视为可接受
    # v26: checkpoint 选择改为"安全门槛 + obs_err 主排序"。
    # collision_rate 必须不超过 BEST_COLLISION_ACCEPT，门槛内再按 obs_err 取最优，
    # 避免旧逻辑（obs_err/clearance 项因阈值过宽恒为 0）退化成纯 collision 字典序。
    BEST_COLLISION_ACCEPT = 0.0100   # v31b: 安全门槛内优先选更近的跟踪距离
    BEST_HARD_COLLISION_ACCEPT = 0.0020  # 硬碰撞软约束阈值

    # v25: narrow passage / corner / near-collision recovery
    RECOVERY_CLEARANCE_MARGIN = 0.35
    RECOVERY_HARD_MARGIN = 0.05
    RECOVERY_DANGER_MARGIN = ROBOT_RADIUS + 0.06
    RECOVERY_MIN_OMEGA = 0.25
    # Lowered from +0.12 to +0.04: the expert grazes obstacles (min clearance
    # p10 ~= -0.05) to follow wall-hugging targets. A 0.37m clearance margin
    # penalized behavior the expert routinely does, causing wall hesitation.
    # Kept just above ROBOT_RADIUS so the soft clearance term still nudges away
    # from walls without forbidding wall-following. Hard collision (< radius) is
    # handled separately with COLLISION_HARD_WEIGHT.
    # v40b: +0.05 (between v39's 0.29 which climbed walls, and v40's 0.33 which
    # was over-conservative: min_esdf ~0.95m, mean|v| stuck ~0.5). +0.05
    # discourages climbing without pushing the tracker a full meter off walls,
    # so it can still follow wall-hugging targets. clearance_recovery handles
    # escape; hard collision (< radius) still guards true contact.
    COLLISION_SAFETY_MARGIN = ROBOT_RADIUS + 0.05
    COLLISION_TOPK_RATIO = 0.25
    COLLISION_HARD_WEIGHT = 4.0
    COLLISION_CLEARANCE_WEIGHT = 0.6
    # v42: curvature-gated corner clearance. margin slightly larger than the
    # collision margin so hard turns leave extra room off convex apexes.
    CORNER_MARGIN = ROBOT_RADIUS + 0.10
    CORNER_TOPK_RATIO = 0.25
    # v43: lateral ESDF probes identify true narrow corridors (obstacles on
    # both sides) and penalize left/right clearance imbalance. Unlike a generic
    # clearance maximum this does not push wall-following trajectories away
    # from a single wall or suppress forward speed.
    CORRIDOR_PROBE_OFFSET = ROBOT_RADIUS + 0.10
    CORRIDOR_SIDE_MAX = 0.45
    CORRIDOR_GATE_TEMP = 0.06
    VISIBILITY_CLEARANCE_MARGIN = ROBOT_RADIUS + 0.06
    LOS_CLEARANCE_MARGIN = 0.0   # geometric LoS obstruction; safety gate is separate
    VISIBILITY_GATE_TEMP = 0.06
    PROGRESS_DEADBAND = 0.25
    PROGRESS_SPEED_KP = 0.6
    PROGRESS_MAX_TARGET_V = MAX_VEL
    PROGRESS_MAX_BEARING = 0.9
    SPEED_MATCH_SAFE_MARGIN = ROBOT_RADIUS + 0.12
    SPEED_MATCH_SAFE_TEMP = 0.05

    # ==================== 训练参数 ====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    # Select the pre-v42 v41 training recipe when requested by train.py.
    USE_CORNER_LOSS = True
    USE_V43_OBJECTIVES = False
    CORNER_CURVATURE_SCALE = 1.0
    CORNER_LOW_SPEED = 0.15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # 学习率调度（滞后于后期约束注入，避免刚切目标就降学习率）
    LR_MILESTONES = [28, 33]     # epoch 28 和 33 时降低学习率
    LR_GAMMA = 0.1               # 学习率衰减因子

    # 数据增强
    USE_DATA_AUG = True
    ESDF_NOISE_STD = 0.02        # ESDF 噪声标准差 (m)
    ESDF_CACHE_SIZE = 32         # per-worker cache for episodes sharing one occupancy map
    ODOM_NOISE_POS = 0.05        # 里程计位置噪声 (m)
    ODOM_NOISE_YAW = 0.05        # 里程计朝向噪声 (rad)
    TARGET_POS_NOISE_PROB = 0.1  # 无监督样本的目标位置噪声概率（不是 dropout）
    TARGET_DROPOUT = TARGET_POS_NOISE_PROB  # deprecated compatibility alias
    USE_HARD_CASE_AUG = True     # v25: sample near-obstacle starts for corner/narrow recovery
    HARD_CASE_PROB = 0.35
    HARD_CASE_MIN_CLEARANCE = ROBOT_RADIUS + 0.02
    HARD_CASE_MAX_CLEARANCE = 0.65
    HARD_CASE_MAX_ATTEMPTS = 160

    # G1 clearance-filtered training. Narrow samples are retained for
    # corridor/low-speed behavior, but their wall-hugging expert labels are
    # down-weighted for direct imitation.
    G1_MODE = False
    USE_CONTROL_HISTORY = False  # set from dataset coverage and saved in checkpoints
    G1_NARROW_EXPERT_WEIGHT = 0.25
    INCLUDE_NARROW_DATA = False
    UPLOAD_DATASET_ARTIFACT = False

    # ==================== 数据生成参数 ====================
    NUM_TRAIN_SAMPLES = 100000   # 训练样本数
    NUM_VAL_SAMPLES = 5000       # 验证样本数

    # 障碍物参数
    MIN_OBSTACLES = 5
    MAX_OBSTACLES = 30
    OBSTACLE_MIN_SIZE = 0.3      # m
    OBSTACLE_MAX_SIZE = 2.0      # m

    # 目标轨迹参数（Bezier）
    BEZIER_CONTROL_POINTS = [3, 4, 5]  # 随机选择控制点数
    BEZIER_MIN_DURATION = 3.0    # 最短轨迹时长 (s)
    BEZIER_MAX_DURATION = 10.0   # 最长轨迹时长 (s)

    # 目标距离范围
    MIN_TARGET_DIST = 1.5        # m
    MAX_TARGET_DIST = 6.0        # m

    # 课程学习难度分级
    CURRICULUM_STAGES = {
        'easy': {
            'epochs': [0, 5],
            'obstacle_density': 0.3,     # 30% 最大障碍数
            'max_curvature': 0.3,        # 大半径曲线
            'target_speed_range': [0.5, 1.0],
        },
        'medium': {
            'epochs': [5, 12],
            'obstacle_density': 0.6,
            'max_curvature': 0.8,
            'target_speed_range': [0.3, 1.5],
        },
        'hard': {
            'epochs': [12, 20],
            'obstacle_density': 1.0,
            'max_curvature': 1.5,        # 急转
            'target_speed_range': [0.0, 2.0],  # 包括静止和快速
        }
    }

    # ==================== 设备配置 ====================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 8              # DataLoader 工作线程数
    PIN_MEMORY = True

    # ==================== 日志和保存 ====================
    LOG_DIR = './logs'
    MODEL_DIR = './models'
    DATA_DIR = './data/scenarios'

    SAVE_INTERVAL = 5            # 每 5 个 epoch 保存一次
    LOG_INTERVAL = 100           # 每 100 个 batch 打印一次
    VAL_INTERVAL = 1             # 每 1 个 epoch 验证一次

    # ==================== 导出配置 ====================
    ONNX_OPSET = 17              # ≥16 避免 Transformer MHA 在 TensorRT FP16 下的 shape/overflow 问题
    TENSORRT_FP16 = True
    TENSORRT_WORKSPACE = 1 << 30  # 1GB

# 课程学习权重调度（线性插值，避免阶跃跳升导致梯度爆炸）
def _lerp_schedule(epoch, keyframes):
    """在 keyframes [(epoch, value), ...] 之间做线性插值"""
    if epoch <= keyframes[0][0]:
        return keyframes[0][1]
    if epoch >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        e0, v0 = keyframes[i]
        e1, v1 = keyframes[i + 1]
        if e0 <= epoch < e1:
            t = (epoch - e0) / (e1 - e0)
            return v0 + t * (v1 - v0)
    return keyframes[-1][1]


def config_to_dict(config):
    """Serialize the effective uppercase configuration without pickling a class instance."""
    return {
        key: copy.deepcopy(getattr(config, key))
        for key in dir(config)
        if key.isupper() and not callable(getattr(config, key))
    }


def validate_config(config):
    """Fail early on physically inconsistent training limits."""
    if config.MAX_VEL <= 0 or config.MAX_OMEGA <= 0:
        raise ValueError('MAX_VEL and MAX_OMEGA must be positive magnitudes')
    if config.MIN_TRACKING_DISTANCE <= 0:
        raise ValueError('MIN_TRACKING_DISTANCE must be positive')
    if config.OBSERVATION_DISTANCE < config.MIN_TRACKING_DISTANCE:
        raise ValueError(
            'OBSERVATION_DISTANCE must be >= MIN_TRACKING_DISTANCE '
            f'({config.OBSERVATION_DISTANCE} < {config.MIN_TRACKING_DISTANCE})'
        )
    return config


def restore_config_from_checkpoint(checkpoint, config=None, exclude=()):
    """Restore effective training settings from new or legacy checkpoints."""
    config = Config() if config is None else config
    values = checkpoint.get('config_dict')
    if values is None:
        legacy = checkpoint.get('config')
        if isinstance(legacy, dict):
            values = dict(LEGACY_CHECKPOINT_DEFAULTS)
            values.update(legacy)
        elif legacy is not None:
            values = dict(LEGACY_CHECKPOINT_DEFAULTS)
            values.update(vars(legacy))
    if not isinstance(values, dict):
        return config
    excluded = set(exclude)
    for key, value in values.items():
        if key.isupper() and key not in excluded:
            setattr(config, key, copy.deepcopy(value))
    return config


def apply_strategy_config(config, strategy):
    """Apply a named loss/safety recipe to one config instance."""
    if strategy == 'current':
        config.USE_CORNER_LOSS = True
        config.USE_V43_OBJECTIVES = False
    elif strategy == 'v41':
        config.USE_CORNER_LOSS = False
        config.USE_V43_OBJECTIVES = False
    elif strategy == 'v43':
        config.USE_CORNER_LOSS = True
        config.USE_V43_OBJECTIVES = True
        config.COLLISION_HARD_WEIGHT = 8.0
        config.COLLISION_CLEARANCE_WEIGHT = 0.8
        config.CORNER_MARGIN = config.ROBOT_RADIUS + 0.15
        config.BEST_CORNER_CLEARANCE_ACCEPT = config.ROBOT_RADIUS + 0.25
    else:
        raise ValueError(f'unknown strategy: {strategy}')
    return config


def apply_g1_config(config):
    """Apply the effective G1 footprint and clearance policy."""
    config.G1_MODE = True
    config.ROBOT_RADIUS = 0.55
    config.COLLISION_SAFETY_MARGIN = 0.70
    config.CORNER_MARGIN = 0.80
    config.CORRIDOR_PROBE_OFFSET = 0.65
    config.VISIBILITY_CLEARANCE_MARGIN = 0.61
    config.SPEED_MATCH_SAFE_MARGIN = 0.67
    config.RECOVERY_DANGER_MARGIN = 0.61
    config.HARD_CASE_MIN_CLEARANCE = 0.57
    config.HARD_CASE_MAX_CLEARANCE = 0.80
    config.BEST_CORNER_CLEARANCE_ACCEPT = 0.80
    config.OBSERVATION_DISTANCE = 2.50
    return config


def get_loss_weights(epoch, cfg=None):
    """
    MPC-cost-only schedule.

    Active terms intentionally match ModelPredictiveControl::cost() in
    src/tracker/src/planner/mpc.cpp: A/D/O/C/F/S. Extra terms from previous
    experiments are disabled so gradients do not fight each other.

    The raw neural losses are on very different scales from the C++ costs.
    Collision/feasibility therefore stay numerically dominant, matching the
    traditional MPC priority instead of copying its literal lambda values.
    """
    cfg = Config() if cfg is None else cfg
    return {
        # Expert-imitation-primary schedule.
        #
        # The expert demonstrations track a MOVING target (target |v|~1 m/s), so
        # the expert trajectory legitimately ends far from the target's t=0
        # position. The static-target terms (dist/angle/visibility/progress_speed)
        # instead reward sitting near the initial target position and directly
        # fight the expert, producing the passive "stand still" policy that had
        # to be patched with deploy-time speed assist. They are therefore
        # disabled; the policy distills the expert MPC, with collision kept as a
        # safety backstop (hard-case augmentation still creates near-obstacle
        # starts the expert data alone may not cover).
        'expert_cmd_seq': 20.0,   # primary objective (first step extra-weighted in the loss)
        'expert_plan': 8.0,       # trajectory-shape anchor

        # Moving-target distance keeping: fixes "tracks from too far away".
        # Extrapolates the target at constant velocity and holds
        # OBSERVATION_DISTANCE behind it.
        #
        # v37 showed moving_obs_err completely unresponsive at peak 3.0 (vs
        # expert_cmd_seq 20): the distance gradient was drowned by imitation.
        # Raised to the expert's order of magnitude and ramped earlier so it can
        # actually pull the tracker closer, while still staying below expert so
        # imitation remains the shape anchor.
        'moving_dist': _lerp_schedule(epoch, [(0, 2.0), (4, 5.0), (12, 8.0)]),
        # Independent one-sided safety barrier.  OBSERVATION_DISTANCE remains
        # the 2.5m optimum; this term forbids imitation/task trade-offs from
        # accepting any predicted tracker-target distance below 0.7m.
        'min_tracking_dist': 100.0,

        # From-scratch cold start: keep collision gentle for the first few epochs
        # so the random-init policy can first learn to move like the expert
        # without being crushed back into "afraid to move" by huge early
        # collision gradients, then ramp safety up.
        #
        # Reduced from 4000: the expert legitimately grazes obstacles (22% of
        # expert points sit inside the old 0.37m margin, 13% inside ROBOT_RADIUS)
        # to follow wall-hugging targets. The old high weight + wide margin
        # fought the imitation signal near walls and produced the "hesitation /
        # can't get through" behavior. Hard-collision (esdf < ROBOT_RADIUS,
        # weight COLLISION_HARD_WEIGHT inside CollisionLoss) still guards true
        # collisions; COLLISION_SAFETY_MARGIN is lowered toward ROBOT_RADIUS.
        'collision': (_lerp_schedule(epoch, [(0, 300.0), (4, 1000.0), (10, 2200.0), (18, 3000.0)])
                      if cfg.USE_V43_OBJECTIVES else
                      _lerp_schedule(epoch, [(0, 100.0), (4, 450.0), (8, 1200.0), (20, 1800.0)])),

        # Self-supervised wall-escape / corner recovery (does NOT use expert).
        # Addresses v39 deploy failures: climbing onto walls and getting stuck,
        # and planning trajectories onto obstacle corners. The outward/improve
        # terms push the robot off walls when clearance is low; worst-clearance
        # pulls trajectory samples off corners.
        # v40b: peak 24 -> 12. At 24 (with the +0.08 collision margin) the policy
        # went over-conservative (min_esdf ~0.95m, mean|v| ~0.5) — good at NOT
        # climbing walls but too far/slow to track wall-hugging targets. 12 keeps
        # the escape behavior without dominating imitation/distance keeping.
        # v41: overall weight raised (12 -> 20) because the sub-terms were
        # re-balanced toward directional escape (outward/improve/worst) and away
        # from "stay far / slow" (clearance/forward). A higher overall weight now
        # makes the escape terms bite on wall-climbing without the far&slow
        # degradation, since proximity itself is barely penalized anymore.
        'clearance_recovery': _lerp_schedule(epoch, [(0, 3.0), (6, 12.0), (15, 20.0)]),

        # v42: curvature-gated clearance ("don't cut the inside corner"). Only
        # penalizes low clearance while turning, so it targets corner-cutting
        # without slowing straight wall-following. Fine-tune term ramped from a
        # moderate base (this is a warm-start refinement, not from-scratch).
        'corner': ((_lerp_schedule(epoch, [(0, 8.0), (5, 20.0), (12, 32.0)])
                    if cfg.USE_V43_OBJECTIVES else
                    _lerp_schedule(epoch, [(0, 4.0), (6, 10.0), (12, 14.0)]))
                   if cfg.USE_CORNER_LOSS else 0.0),

        # v43 deployment fixes: explicit same-speed tracking and narrow-passage
        # centering. ProgressSpeedLoss is visibility/clearance gated, so it does
        # not pull through corners; corridor_center only activates with nearby
        # obstacles on both sides.
        'progress_speed': (_lerp_schedule(epoch, [(0, 3.0), (4, 8.0), (10, 14.0)])
                           if cfg.USE_V43_OBJECTIVES else 0.0),
        'corridor_center': (_lerp_schedule(epoch, [(0, 2.0), (5, 8.0), (12, 14.0)])
                            if cfg.USE_V43_OBJECTIVES else 0.0),

        # FeasibilityLoss / SmoothLoss are normalized to O(1); small quality terms.
        'feasibility': _lerp_schedule(epoch, [(0, 1.0), (8, 2.0), (20, 4.0)]),
        'smooth': _lerp_schedule(epoch, [(0, 0.15), (8, 0.25), (20, 0.4)]),

        # Visibility (occlusion) is NOT a static-target term: it penalizes the
        # tracker->target line of sight being blocked by obstacles, evaluated
        # against the moving (extrapolated) target. 85% of expert trajectories
        # experience mid-line occlusion (l_shapes/corridors dominate the data),
        # and the expert detours to keep sight around corners. With this at 0 the
        # policy drove straight at the best observation pose and lost the target
        # whenever it rounded a sharp wall. Re-enabled and ramped so it shapes
        # corner behavior without overriding imitation early.
        # v40: strengthened (6 -> 10) since v39 only partially fixed the corner
        # behavior (trajectory still planned onto the corner ~sometimes).
        'visibility': _lerp_schedule(epoch, [(0, 1.5), (6, 5.0), (15, 10.0)]),

        # Static-target terms disabled: they contradict moving-target tracking.
        # (moving_dist above replaces the static dist objective.)
        'angle': 0.0,
        'dist': 0.0,
    }


def get_eval_loss_weights(cfg=None):
    """
    固定验证权重。

    训练权重会随 epoch 变化，不能直接用它的 total loss 选 best checkpoint。
    这里用稳定的一套目标做横向比较，避免早期模型因为约束尚未启用而被误选。
    """
    cfg = Config() if cfg is None else cfg
    return {
        # Fixed eval objective mirrors the expert-imitation-primary training
        # objective so the eval total loss is comparable across checkpoints.
        # Checkpoint ranking (checkpoint_rank) is driven by expert-command match
        # + collision, NOT by the static-target obs_err, which is misleading for
        # a moving-target receding-horizon tracker.
        'expert_cmd_seq': 20.0,
        'expert_plan': 8.0,
        'moving_dist': 8.0,
        'min_tracking_dist': 100.0,
        'visibility': 10.0,
        'collision': 3000.0 if cfg.USE_V43_OBJECTIVES else 1800.0,
        'clearance_recovery': 20.0,
        'corner': ((32.0 if cfg.USE_V43_OBJECTIVES else 14.0)
                   if cfg.USE_CORNER_LOSS else 0.0),
        'corridor_center': 14.0 if cfg.USE_V43_OBJECTIVES else 0.0,
        'progress_speed': 14.0 if cfg.USE_V43_OBJECTIVES else 0.0,
        'feasibility': 4.0,
        'smooth': 0.4,
    }

if __name__ == '__main__':
    # 测试配置
    cfg = Config()
    print(f"Device: {cfg.DEVICE}")
    print(f"Local map: {cfg.LOCAL_MAP_PIXELS}×{cfg.LOCAL_MAP_PIXELS} @ {cfg.LOCAL_MAP_RES}m/pixel")
    print(f"Anchor grid: {cfg.ANCHOR_GRID_SIZE}×{cfg.ANCHOR_GRID_SIZE}")
    print(f"Horizon: {cfg.HORIZON} steps × {cfg.DT}s = {cfg.HORIZON * cfg.DT}s")

    print("\n课程学习权重调度:")
    for ep in [0, 2, 5, 8, 12, 15, 18]:
        w = get_loss_weights(ep)
        print(f"Epoch {ep}: " + ", ".join(f"{key}={value:.3g}" for key, value in w.items()))
