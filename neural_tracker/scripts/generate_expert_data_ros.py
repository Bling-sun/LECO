"""
Expert trajectory dataset generator via ROS2 — connects to the real MPC tracker.

Architecture:
    simulator (real env)  ──┐
                            ├─→ /tracker/odom, /map, /target/detection
                            ↓
    prediction node ──→ /target/bezier
                            ↓
    tracker node (REAL MPC) ──→ /tracker/plan, /tracker/cmd_vel
                            ↓
    [this recorder]  subscribes everything, dumps pkl

Usage:
    # Terminal 1 — run the full DiffTrack stack
    cd <DiffTrack root> && source install/setup.bash
    ros2 launch simulator simulation.launch.py &
    ros2 launch tracker tracking.launch.py &

    # Terminal 2 — record a few samples
    python scripts/generate_expert_data_ros.py --num_samples 5 \
                                                --episode_seconds 20 \
                                                --save_dir ./data/expert_ros

The recorder waits until odom + map + detection are all live, then captures
a fixed-length episode. Each sample is saved as one .pkl per episode and a
.png preview.
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist


def quat_to_yaw(q):
    sinp = 2 * (q.w * q.y - q.x * q.z)
    if abs(sinp) >= 1 - 1e-6:
        return float(np.copysign(2 * np.arctan2(q.y, q.x), sinp))
    return float(np.arctan2(2 * (q.x * q.y + q.z * q.w),
                             q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z))


class ExpertRecorder(Node):
    """Subscribes to tracker/simulator topics and stores aligned samples."""

    def __init__(self, episode_seconds=20.0, sample_hz=10.0,
                 expert_horizon=30, expert_plan_horizon=None,
                 scenario_type="unknown"):
        super().__init__('expert_recorder')

        self.episode_seconds = episode_seconds
        self.sample_dt = 1.0 / sample_hz
        self.expert_horizon = expert_horizon
        self.expert_plan_horizon = expert_plan_horizon or expert_horizon + 1
        self.scenario_type = scenario_type

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.last_odom = None      # [x, y, yaw, v, w]
        self.last_target = None    # [tx, ty, valid]
        self.last_plan = None      # (N, 3) [x, y, yaw]  predicted trajectory
        self.last_cmd = None       # [v, w]
        self.map_grid = None       # OccupancyGrid (kept once)
        self.map_meta = None       # (width, height, resolution, origin_x, origin_y)

        self.create_subscription(Odometry, '/tracker/odom', self._odom_cb, qos)
        self.create_subscription(PoseStamped, '/target/detection',
                                 self._target_cb, qos)
        self.create_subscription(Path, '/tracker/plan', self._plan_cb, qos)
        self.create_subscription(Twist, '/tracker/cmd_vel', self._cmd_cb, qos)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)

        self.frames = []
        self._recording = False
        self._t_start = None

    def _odom_cb(self, msg: Odometry):
        self.last_odom = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quat_to_yaw(msg.pose.pose.orientation),
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        ], dtype=np.float64)

    def _target_cb(self, msg: PoseStamped):
        valid = msg.pose.orientation.w > 0
        self.last_target = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            1.0 if valid else 0.0,
        ], dtype=np.float64)

    def _plan_cb(self, msg: Path):
        if not msg.poses:
            return
        n = len(msg.poses)
        plan = np.zeros((n, 3))
        for i, p in enumerate(msg.poses):
            plan[i, 0] = p.pose.position.x
            plan[i, 1] = p.pose.position.y
            plan[i, 2] = 2 * np.arctan2(p.pose.orientation.z,
                                         p.pose.orientation.w)
        self.last_plan = plan

    def _cmd_cb(self, msg: Twist):
        self.last_cmd = np.array([msg.linear.x, msg.angular.z],
                                  dtype=np.float64)

    def _map_cb(self, msg: OccupancyGrid):
        if self.map_grid is not None:
            return
        w, h = msg.info.width, msg.info.height
        self.map_grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.map_meta = (
            w, h, msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y,
        )
        self.get_logger().info(
            f'Map received: {w}x{h} @ {msg.info.resolution:.3f}m'
        )

    def ready(self):
        return all(x is not None for x in (
            self.last_odom, self.last_target, self.map_grid
        ))

    def start_episode(self):
        self.frames = []
        self._t_start = time.monotonic()
        self._recording = True

    def step_record(self):
        if not self._recording or not self.ready():
            return False
        self.frames.append({
            't': time.monotonic() - self._t_start,
            'odom': self.last_odom.copy(),
            'target': self.last_target.copy(),
            'plan': self.last_plan.copy() if self.last_plan is not None
                                          else np.zeros((0, 3)),
            'cmd': self.last_cmd.copy() if self.last_cmd is not None
                                         else np.zeros(2),
        })
        if (time.monotonic() - self._t_start) >= self.episode_seconds:
            self._recording = False
            return True
        return False

    def dump(self, path):
        import cv2
        # Must match simulator.yaml (width=20, height=20, resolution=0.05) → 400×400
        MAP_SIZE, MAP_RES = 20.0, 0.05
        TARGET = int(MAP_SIZE / MAP_RES)  # 400

        # --- 1. Normalize map: int8 (-1/0/100) → uint8 (0/0/255), pad to square, resize ---
        h, w = self.map_grid.shape
        res_orig = self.map_meta[2]
        ox, oy = self.map_meta[3], self.map_meta[4]

        if (not np.isclose(w * res_orig, MAP_SIZE)
                or not np.isclose(h * res_orig, MAP_SIZE)):
            raise ValueError(
                f"Recorder requires a {MAP_SIZE}x{MAP_SIZE} m map; got "
                f"{w * res_orig}x{h * res_orig}. Use the expert_collection.yaml config."
            )

        occ_bin = np.where(self.map_grid == 100, np.uint8(255), np.uint8(0))
        side = max(h, w)
        padded = np.zeros((side, side), dtype=np.uint8)
        r0, c0 = (side - h) // 2, (side - w) // 2
        padded[r0:r0 + h, c0:c0 + w] = occ_bin
        occ = cv2.resize(padded, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)

        # New origin and resolution after pad + resize
        new_res = (side * res_orig) / TARGET
        new_ox = ox - c0 * res_orig
        new_oy = oy - r0 * res_orig

        # Map center in world coords → becomes (0, 0) after normalization
        cx = new_ox + TARGET * new_res / 2
        cy = new_oy + TARGET * new_res / 2

        # --- 2. target_vel via finite diff ---
        frames = self.frames
        vels = [np.zeros(2, dtype=np.float32)]
        for i in range(1, len(frames)):
            dt = frames[i]['t'] - frames[i - 1]['t']
            if (dt > 1e-6
                    and frames[i]['target'][2] > 0.5
                    and frames[i - 1]['target'][2] > 0.5):
                v = (frames[i]['target'][:2] - frames[i - 1]['target'][:2]) / dt
            else:
                v = np.zeros(2)
            vels.append(v.astype(np.float32))

        # --- 3. Future expert control sequence from MPC cmd_vel ---
        cmd_seq = []
        control_histories = []
        for i in range(len(frames)):
            controls = np.zeros((self.expert_horizon, 2), dtype=np.float32)
            mask = np.zeros(self.expert_horizon, dtype=np.float32)
            for j in range(self.expert_horizon):
                k = i + j
                if k >= len(frames):
                    break
                controls[j] = frames[k]['cmd'].astype(np.float32)
                mask[j] = 1.0
            cmd_seq.append((controls, mask))
            history = np.zeros((4, 2), dtype=np.float32)
            start = max(0, i - len(history))
            previous = [frames[k]['cmd'].astype(np.float32) for k in range(start, i)]
            if previous:
                history[-len(previous):] = np.asarray(previous, dtype=np.float32)
            control_histories.append(history)

        # --- 4. Build flat list: one dict per frame, keys matching dataset.py ---
        samples = []
        for i, f in enumerate(frames):
            robot = f['odom'].astype(np.float32).copy()
            robot[0] -= cx
            robot[1] -= cy
            tgt = f['target'][:2].astype(np.float32).copy()
            tgt[0] -= cx
            tgt[1] -= cy
            expert_cmd = f['cmd'].astype(np.float32).copy()
            expert_cmd_seq, expert_cmd_seq_mask = cmd_seq[i]
            target_future_pos = np.zeros((self.expert_horizon, 2), dtype=np.float32)
            target_future_pos_mask = np.zeros(self.expert_horizon, dtype=np.float32)
            for j in range(self.expert_horizon):
                k = i + j + 1
                if k >= len(frames) or frames[k]['target'][2] <= 0.5:
                    break
                future_target = frames[k]['target'][:2].astype(np.float32).copy()
                future_target[0] -= cx
                future_target[1] -= cy
                target_future_pos[j] = future_target
                target_future_pos_mask[j] = 1.0
            raw_plan = f['plan'].astype(np.float32).copy()
            expert_plan = np.zeros((self.expert_plan_horizon, 3), dtype=np.float32)
            expert_plan_mask = np.zeros(self.expert_plan_horizon, dtype=np.float32)
            if raw_plan.size:
                raw_plan[:, 0] -= cx
                raw_plan[:, 1] -= cy
                n = min(len(raw_plan), self.expert_plan_horizon)
                expert_plan[:n] = raw_plan[:n]
                expert_plan_mask[:n] = 1.0
            samples.append({
                'occupancy': occ,                          # (1000,1000) uint8
                'robot_state': robot,                      # [x,y,yaw,v,ω] centered
                'target_pos': tgt,                         # [x,y] centered
                'target_vel': vels[i],                     # [vx,vy]
                'target_future_pos': target_future_pos,    # t=1..H measured targets
                'target_future_pos_mask': target_future_pos_mask,
                'target_valid': float(f['target'][2] > 0.5),
                'expert_cmd': expert_cmd,                  # [v, w] from MPC tracker
                'control_history': control_histories[i],   # (4, 2), oldest -> newest
                'expert_cmd_seq': expert_cmd_seq,          # (H, 2) future [v, w]
                'expert_cmd_seq_mask': expert_cmd_seq_mask,
                'expert_plan': expert_plan,                # (H, 3) [x,y,yaw], centered
                'expert_plan_mask': expert_plan_mask,      # (H,) valid plan points
                'scenario_type': self.scenario_type,
            })

        with open(path, 'wb') as f:
            pickle.dump(samples, f)


def visualize(samples, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MAP_SIZE = 20.0
    half = MAP_SIZE / 2
    extent = [-half, half, -half, half]

    grid = samples[0]['occupancy']
    odom = np.array([s['robot_state'] for s in samples])
    tgt = np.array([s['target_pos'] for s in samples])
    valid = np.array([s['target_valid'] for s in samples]) > 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow((grid > 0).astype(np.uint8), extent=extent, origin='lower',
              cmap='gray_r', alpha=0.7)

    ax.plot(tgt[valid, 0], tgt[valid, 1], 'g-', lw=2.5,
            label='target trajectory')
    ax.plot(odom[:, 0], odom[:, 1], 'b-', lw=1.8,
            label='expert tracker (real MPC)')
    ax.scatter(odom[0, 0], odom[0, 1], c='cyan', s=120,
               marker='s', edgecolors='black', zorder=5)
    if valid.any():
        i0 = int(np.argmax(valid))
        ax.scatter(tgt[i0, 0], tgt[i0, 1], c='lime', s=120,
                   marker='o', edgecolors='black', zorder=5)

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right')
    ax.set_title(f'episode: {len(samples)} frames, '
                 f'{samples[-1]["robot_state"][0]:.1f}s')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--episode_seconds', type=float, default=20.0)
    parser.add_argument('--sample_hz', type=float, default=10.0)
    parser.add_argument('--save_dir', type=str, default='./data/expert_ros')
    parser.add_argument('--ready_timeout', type=float, default=30.0,
                        help='How long to wait for /odom + /map + /detection.')
    parser.add_argument('--expert_horizon', type=int, default=30,
                        help='Number of future cmd_vel steps saved per frame.')
    parser.add_argument('--expert_plan_horizon', type=int, default=None,
                        help='Number of /tracker/plan points saved per frame.')
    parser.add_argument('--scenario_type', type=str, default='unknown',
                        help='Scenario label saved in every frame.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Resume: continue from the next index after existing samples
    import glob, re
    existing = glob.glob(os.path.join(args.save_dir, 'sample_*.pkl'))
    indices = [int(m.group(1)) for p in existing
               if (m := re.search(r'sample_(\d+)\.pkl$', p))]
    start_idx = max(indices) + 1 if indices else 0

    rclpy.init()
    node = ExpertRecorder(
        episode_seconds=args.episode_seconds,
        sample_hz=args.sample_hz,
        expert_horizon=args.expert_horizon,
        expert_plan_horizon=args.expert_plan_horizon,
        scenario_type=args.scenario_type,
    )

    try:
        for k in range(args.num_samples):
            i = start_idx + k
            node.get_logger().info(
                f'[sample {k+1}/{args.num_samples} → idx {i:04d}] waiting for topics...'
            )
            t0 = time.monotonic()
            while rclpy.ok() and not node.ready():
                rclpy.spin_once(node, timeout_sec=0.1)
                if time.monotonic() - t0 > args.ready_timeout:
                    node.get_logger().error(
                        'Timed out waiting for topics. Is simulator + tracker '
                        'launched?\n'
                        '  ros2 launch simulator simulation.launch.py\n'
                        '  ros2 launch tracker tracking.launch.py\n'
                        'Required: /tracker/odom, /map, /target/detection'
                    )
                    return 1

            node.get_logger().info(
                f'[sample {k+1}/{args.num_samples}] recording '
                f'{args.episode_seconds:.1f}s...'
            )
            node.start_episode()
            done = False
            next_t = time.monotonic()
            while rclpy.ok() and not done:
                rclpy.spin_once(node, timeout_sec=0.01)
                if time.monotonic() >= next_t:
                    done = node.step_record()
                    next_t += node.sample_dt

            pkl_path = os.path.join(args.save_dir, f'sample_{i:04d}.pkl')
            png_path = os.path.join(args.save_dir, f'sample_{i:04d}.png')
            with open(pkl_path, 'rb' if False else 'wb') as _:
                pass
            node.dump(pkl_path)
            with open(pkl_path, 'rb') as f:
                visualize(pickle.load(f), png_path)
            node.get_logger().info(
                f'[sample {k+1} → idx {i:04d}] saved {len(node.frames)} frames → '
                f'{pkl_path}'
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
