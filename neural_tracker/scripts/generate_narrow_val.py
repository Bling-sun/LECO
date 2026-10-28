"""
Generate a narrow-passage / corner / near-obstacle validation set.

Reuses occupancies from data/scenarios/val.pkl, but resamples each sample's
robot pose into a hard-context location (low clearance with multiple
near-obstacle sectors). Target pose stays close to the original where
possible; otherwise we re-pick a target inside a valid distance band.

Output: data/scenarios/val_narrow.pkl

Run:
    .venv/bin/python scripts/generate_narrow_val.py --num 5000
"""
import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_generator import ScenarioGenerator


def is_hard_context(px, py, esdf, cfg):
    H, W = esdf.shape
    safe_margin = cfg.ROBOT_RADIUS + 0.12
    sectors = 8
    radius_px = max(3, int(0.7 / cfg.MAP_RESOLUTION))
    sector_hits = 0
    for i in range(sectors):
        theta = 2.0 * np.pi * i / sectors
        sx = int(round(px + np.cos(theta) * radius_px))
        sy = int(round(py + np.sin(theta) * radius_px))
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            sector_hits += 1
            continue
        if esdf[sy, sx] < safe_margin:
            sector_hits += 1
    if sector_hits >= 2:
        return True
    return esdf[py, px] < cfg.ROBOT_RADIUS + 0.08


def sample_target_in_band(rx, ry, esdf, cfg, rng,
                          min_dist=None, max_dist=None,
                          min_target_clearance=0.4, attempts=120):
    """Pick a target pose with clearance > threshold and in range [min_dist, max_dist]."""
    H, W = esdf.shape
    if min_dist is None:
        min_dist = cfg.MIN_TARGET_DIST
    if max_dist is None:
        max_dist = cfg.MAX_TARGET_DIST
    for _ in range(attempts):
        tpx = rng.integers(0, W)
        tpy = rng.integers(0, H)
        if esdf[tpy, tpx] < min_target_clearance:
            continue
        tx = tpx * cfg.MAP_RESOLUTION - cfg.MAP_SIZE / 2
        ty = tpy * cfg.MAP_RESOLUTION - cfg.MAP_SIZE / 2
        d = np.hypot(tx - rx, ty - ry)
        if d < min_dist or d > max_dist:
            continue
        return np.array([tx, ty], dtype=np.float32)
    return None


def find_hard_pose(occupancy, esdf, cfg, rng,
                   min_clearance, max_clearance, max_attempts=400):
    H, W = esdf.shape
    for _ in range(max_attempts):
        px = rng.integers(0, W)
        py = rng.integers(0, H)
        clearance = esdf[py, px]
        if clearance < min_clearance or clearance > max_clearance:
            continue
        if not is_hard_context(px, py, esdf, cfg):
            continue
        x = px * cfg.MAP_RESOLUTION - cfg.MAP_SIZE / 2
        y = py * cfg.MAP_RESOLUTION - cfg.MAP_SIZE / 2
        return np.array([x, y], dtype=np.float32), float(clearance)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/scenarios/val.pkl')
    parser.add_argument('--dst', default='data/scenarios/val_narrow.pkl')
    parser.add_argument('--num', type=int, default=5000)
    parser.add_argument('--min-clearance', type=float, default=0.27,
                        help='lower bound on robot init clearance (m). Default = ROBOT_RADIUS+0.02')
    parser.add_argument('--max-clearance', type=float, default=0.65,
                        help='upper bound on robot init clearance (m)')
    parser.add_argument('--seed', type=int, default=20260625)
    args = parser.parse_args()

    cfg = Config()
    gen = ScenarioGenerator(cfg)
    rng = np.random.default_rng(args.seed)

    print(f"Loading source samples from {args.src} ...")
    with open(args.src, 'rb') as f:
        src_samples = pickle.load(f)
    print(f"  loaded {len(src_samples)}")

    out = []
    n_attempts = 0
    n_hard = 0
    n_no_target = 0
    clearances = []
    src_indices = rng.permutation(len(src_samples))

    print(f"Generating up to {args.num} narrow-passage samples ...")
    for sidx in src_indices:
        if len(out) >= args.num:
            break
        n_attempts += 1
        src = src_samples[sidx]
        occupancy = src['occupancy']
        esdf = gen.compute_esdf(occupancy)

        pose_xy, clearance = find_hard_pose(
            occupancy, esdf, cfg, rng,
            min_clearance=args.min_clearance,
            max_clearance=args.max_clearance,
        )
        if pose_xy is None:
            continue
        n_hard += 1

        rx, ry = pose_xy
        target_pos = sample_target_in_band(rx, ry, esdf, cfg, rng)
        if target_pos is None:
            n_no_target += 1
            continue

        yaw_to_target = np.arctan2(target_pos[1] - ry, target_pos[0] - rx)
        yaw = yaw_to_target + float(rng.uniform(-0.8, 0.8))
        yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

        robot_state = np.array([rx, ry, yaw, 0.0, 0.0], dtype=np.float32)

        target_speed = float(rng.uniform(0.0, 1.0))
        target_heading = float(rng.uniform(-np.pi, np.pi))
        target_vel = np.array([
            target_speed * np.cos(target_heading),
            target_speed * np.sin(target_heading),
        ], dtype=np.float32)

        out.append({
            'occupancy': occupancy,
            'robot_state': robot_state,
            'target_pos': target_pos.astype(np.float32),
            'target_vel': target_vel,
            'target_valid': True,
        })
        clearances.append(clearance)

        if len(out) % 500 == 0:
            print(f"  collected {len(out)}/{args.num}  "
                  f"(scanned {n_attempts} src, hard_pose_hits={n_hard}, "
                  f"target_fail={n_no_target})")

    if not out:
        raise RuntimeError("No narrow-passage samples generated. Loosen clearance band?")

    clearances = np.array(clearances)
    print("\n=== generation summary ===")
    print(f"  output samples: {len(out)}")
    print(f"  source scanned: {n_attempts}, hard_pose_hits: {n_hard}, "
          f"target_fail: {n_no_target}")
    print(f"  init clearance:  min={clearances.min():.3f} "
          f"median={np.median(clearances):.3f} "
          f"mean={clearances.mean():.3f} "
          f"max={clearances.max():.3f}")
    print(f"  <0.35: {(clearances<0.35).mean()*100:.1f}%   "
          f"<0.45: {(clearances<0.45).mean()*100:.1f}%   "
          f"<0.55: {(clearances<0.55).mean()*100:.1f}%")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = os.path.getsize(args.dst) / 1024 / 1024
    print(f"\nWrote {args.dst} ({sz:.1f} MB)")


if __name__ == '__main__':
    main()
