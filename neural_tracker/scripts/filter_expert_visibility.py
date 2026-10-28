#!/usr/bin/env python3
"""Find and optionally remove ROS expert samples with poor target visibility or motion."""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


SAMPLE_RE = re.compile(r"^sample_(\d{4})\.pkl$")


def load_frames(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("frames"), list):
        return obj["frames"]
    raise ValueError(f"Unsupported sample format: {path}")


def frame_visible(frame):
    if isinstance(frame, dict):
        if "target_valid" in frame:
            return float(frame["target_valid"]) > 0.5
        if "target" in frame and len(frame["target"]) >= 3:
            return float(frame["target"][2]) > 0.5
    return False


def longest_false_run(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def target_position(frame):
    if isinstance(frame, dict):
        if "target_pos" in frame:
            pos = np.asarray(frame["target_pos"], dtype=float).reshape(-1)
            if pos.size >= 2:
                return pos[:2]
        if "target" in frame:
            target = np.asarray(frame["target"], dtype=float).reshape(-1)
            if target.size >= 2:
                return target[:2]
    return None


def tracker_position(frame):
    if isinstance(frame, dict):
        if "tracker_pos" in frame:
            pos = np.asarray(frame["tracker_pos"], dtype=float).reshape(-1)
            if pos.size >= 2:
                return pos[:2]
        if "robot_state" in frame:
            state = np.asarray(frame["robot_state"], dtype=float).reshape(-1)
            if state.size >= 2:
                return state[:2]
        if "tracker" in frame:
            tracker = np.asarray(frame["tracker"], dtype=float).reshape(-1)
            if tracker.size >= 2:
                return tracker[:2]
    return None


def path_length(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(np.stack(points), axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def net_displacement(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def longest_stuck_run(points, radius):
    if not points:
        return 0

    start = 0
    longest = 1
    for end in range(1, len(points)):
        while start < end and np.linalg.norm(points[end] - points[start]) > radius:
            start += 1
        longest = max(longest, end - start + 1)
    return longest


def collision_stats(frames, collision_distance):
    min_distance = float("inf")
    collision_count = 0
    for frame in frames:
        target = target_position(frame)
        tracker = tracker_position(frame)
        if target is None or tracker is None:
            continue
        if not np.isfinite(target).all() or not np.isfinite(tracker).all():
            continue

        distance = float(np.linalg.norm(target - tracker))
        min_distance = min(min_distance, distance)
        if distance < collision_distance:
            collision_count += 1

    if min_distance == float("inf"):
        min_distance = float("nan")
    return min_distance, collision_count


def sample_index(path):
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def scan_samples(
    data_dir,
    max_lost_ratio,
    max_consecutive_lost,
    min_frames,
    min_path_length,
    min_net_displacement,
    max_stuck_frames,
    stuck_radius,
    collision_distance,
):
    rows = []
    for path in sorted(data_dir.glob("sample_*.pkl")):
        idx = sample_index(path)
        if idx is None:
            continue

        try:
            frames = load_frames(path)
            visible = [frame_visible(frame) for frame in frames]
            total = len(visible)
            visible_count = sum(visible)
            lost_count = total - visible_count
            lost_ratio = lost_count / total if total else 1.0
            max_lost_run = longest_false_run(visible)
            positions = [
                pos for pos in (target_position(frame) for frame in frames)
                if pos is not None and np.isfinite(pos).all()
            ]
            target_path_length = path_length(positions)
            target_net_displacement = net_displacement(positions)
            target_stuck_run = longest_stuck_run(positions, stuck_radius)
            min_target_tracker_distance, collision_count = collision_stats(
                frames, collision_distance
            )
            error = ""
        except Exception as exc:
            total = 0
            visible_count = 0
            lost_count = 0
            lost_ratio = 1.0
            max_lost_run = 0
            target_path_length = 0.0
            target_net_displacement = 0.0
            target_stuck_run = 0
            min_target_tracker_distance = float("nan")
            collision_count = 0
            error = str(exc)

        bad = (
            bool(error)
            or lost_ratio > max_lost_ratio
            or max_lost_run > max_consecutive_lost
            or total < min_frames
            or target_path_length < min_path_length
            or target_net_displacement < min_net_displacement
            or target_stuck_run > max_stuck_frames
            or collision_count > 0
        )
        rows.append({
            "idx": idx,
            "path": path,
            "frames": total,
            "visible": visible_count,
            "lost": lost_count,
            "lost_ratio": lost_ratio,
            "max_lost_run": max_lost_run,
            "path_length": target_path_length,
            "net_displacement": target_net_displacement,
            "stuck_run": target_stuck_run,
            "min_distance": min_target_tracker_distance,
            "collision_count": collision_count,
            "bad": bad,
            "error": error,
        })
    return rows


def parse_ids(text):
    ids = set()
    for chunk in re.split(r"[\s,]+", text.strip()):
        if not chunk:
            continue
        ids.add(int(chunk))
    return ids


def sample_files_for_index(data_dir, idx):
    return sorted(data_dir.glob(f"sample_{idx:04d}.*"))


def delete_ids(data_dir, ids):
    deleted = []
    for idx in sorted(ids):
        files = sample_files_for_index(data_dir, idx)
        if not files:
            print(f"[warn] sample_{idx:04d} not found, skipping")
            continue
        for path in files:
            path.unlink()
            deleted.append(path.name)
    return deleted


def renumber_contiguous(data_dir):
    pkl_files = sorted(
        path for path in data_dir.glob("sample_*.pkl")
        if sample_index(path) is not None
    )
    old_indices = [sample_index(path) for path in pkl_files]
    mapping = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(old_indices)
        if old_idx != new_idx
    }
    if not mapping:
        return mapping

    temp_moves = []
    final_moves = []
    for old_idx, new_idx in mapping.items():
        old_prefix = f"sample_{old_idx:04d}"
        new_prefix = f"sample_{new_idx:04d}"
        for path in sorted(data_dir.glob(f"{old_prefix}.*")):
            suffix = path.name[len(old_prefix):]
            tmp = data_dir / f".renumber_tmp_{old_idx:04d}_{path.name}"
            final = data_dir / f"{new_prefix}{suffix}"
            temp_moves.append((path, tmp))
            final_moves.append((tmp, final))

    for src, tmp in temp_moves:
        os.replace(src, tmp)
    for tmp, final in final_moves:
        os.replace(tmp, final)
    return mapping


def contiguous_report(rows):
    indices = [row["idx"] for row in rows]
    if not indices:
        return False, []
    expected = set(range(min(indices), max(indices) + 1))
    missing = sorted(expected.difference(indices))
    return min(indices) == 0 and not missing, missing


def print_report(
    rows,
    max_lost_ratio,
    max_consecutive_lost,
    min_frames,
    min_path_length,
    min_net_displacement,
    max_stuck_frames,
    stuck_radius,
    collision_distance,
    limit,
):
    bad_rows = [row for row in rows if row["bad"]]
    contiguous, missing = contiguous_report(rows)

    print("Visibility filter")
    if rows:
        print(f"  dir: {rows[0]['path'].parent}")
    print(f"  samples: {len(rows)}")
    print(
        "  threshold: "
        f"lost_ratio > {max_lost_ratio:.2f} OR "
        f"max_consecutive_lost > {max_consecutive_lost} OR "
        f"frames < {min_frames} OR "
        f"path_length < {min_path_length:.2f}m OR "
        f"net_displacement < {min_net_displacement:.2f}m OR "
        f"stuck_run > {max_stuck_frames} frames within {stuck_radius:.2f}m OR "
        f"target_tracker_distance < {collision_distance:.2f}m"
    )
    print(f"  bad candidates: {len(bad_rows)}")
    print(f"  numbering continuous from 0000: {'yes' if contiguous else 'no'}")
    if missing:
        print("  missing ids:", ", ".join(f"{idx:04d}" for idx in missing[:50]))
        if len(missing) > 50:
            print(f"  ... {len(missing) - 50} more missing ids")
    print()

    if not bad_rows:
        print("No bad candidates found.")
        return

    print("Bad candidates:")
    print("  id     lost%   max_lost  frames  path_m  net_m  stuck  min_d  hit  reason")
    shown = bad_rows if limit <= 0 else bad_rows[:limit]
    for row in shown:
        reasons = []
        if row["error"]:
            reasons.append("load_error")
        if row["lost_ratio"] > max_lost_ratio:
            reasons.append("lost_ratio")
        if row["max_lost_run"] > max_consecutive_lost:
            reasons.append("max_consecutive")
        if row["frames"] < min_frames:
            reasons.append("short")
        if row["path_length"] < min_path_length:
            reasons.append("low_motion")
        if row["net_displacement"] < min_net_displacement:
            reasons.append("stuck")
        if row["stuck_run"] > max_stuck_frames:
            reasons.append("long_stay")
        if row["collision_count"] > 0:
            reasons.append("collision")
        print(
            f"  {row['idx']:04d}  "
            f"{row['lost_ratio'] * 100:6.1f}  "
            f"{row['max_lost_run']:8d}  "
            f"{row['frames']:6d}  "
            f"{row['path_length']:6.2f}  "
            f"{row['net_displacement']:5.2f}  "
            f"{row['stuck_run']:5d}  "
            f"{row['min_distance']:5.2f}  "
            f"{row['collision_count']:3d}  "
            f"{'+'.join(reasons)}"
        )
    if limit > 0 and len(bad_rows) > limit:
        print(f"  ... {len(bad_rows) - limit} more")
    print()
    print("IDs:")
    print(",".join(f"{row['idx']:04d}" for row in bad_rows))
    print()
    print("After you confirm, delete with:")
    script = Path(__file__).as_posix()
    print(
        f"  python3 {script} "
        f"--data_dir {str(rows[0]['path'].parent)} "
        "--delete-ids <comma_ids> --yes"
    )


def sample_dirs(root_dir):
    dirs = {
        path.parent
        for path in root_dir.rglob("sample_*.pkl")
        if path.is_file()
    }
    return sorted(dirs)


def bad_ids(rows):
    return {row["idx"] for row in rows if row["bad"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./data/expert_ros"))
    parser.add_argument("--max-lost-ratio", type=float, default=0.20)
    parser.add_argument("--max-consecutive-lost", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--min-path-length", type=float, default=1.0)
    parser.add_argument("--min-net-displacement", type=float, default=0.5)
    parser.add_argument("--max-stuck-frames", type=int, default=20)
    parser.add_argument("--stuck-radius", type=float, default=0.2)
    parser.add_argument("--collision-distance", type=float, default=0.35)
    parser.add_argument("--recursive", action="store_true",
                        help="Scan each sample-containing child directory separately.")
    parser.add_argument("--delete-bad", action="store_true",
                        help="Delete all samples that fail the configured thresholds.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit displayed bad rows. 0 means show all.")
    parser.add_argument("--delete-ids", type=str, default="",
                        help="Comma/space separated sample ids to delete.")
    parser.add_argument("--yes", action="store_true",
                        help="Required with --delete-ids.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    dirs = sample_dirs(data_dir) if args.recursive else [data_dir]

    if args.delete_bad:
        if not args.yes:
            raise SystemExit("--delete-bad requires --yes")
        total_ids = 0
        total_files = 0
        total_renumbered = 0
        for path in dirs:
            rows = scan_samples(
                path,
                max_lost_ratio=args.max_lost_ratio,
                max_consecutive_lost=args.max_consecutive_lost,
                min_frames=args.min_frames,
                min_path_length=args.min_path_length,
                min_net_displacement=args.min_net_displacement,
                max_stuck_frames=args.max_stuck_frames,
                stuck_radius=args.stuck_radius,
                collision_distance=args.collision_distance,
            )
            ids = bad_ids(rows)
            if not ids:
                print(f"{path}: no bad samples")
                continue
            deleted = delete_ids(path, ids)
            mapping = renumber_contiguous(path)
            total_ids += len(ids)
            total_files += len(deleted)
            total_renumbered += len(mapping)
            print(
                f"{path}: deleted {len(ids)} sample id(s), "
                f"{len(deleted)} file(s), renumbered {len(mapping)} id(s)"
            )
        print(
            f"Done. Deleted {total_ids} sample id(s), "
            f"{total_files} file(s), renumbered {total_renumbered} id(s)."
        )
        return 0

    if args.delete_ids:
        if args.recursive:
            raise SystemExit("--delete-ids is for one directory at a time; omit --recursive")
        if not args.yes:
            raise SystemExit("--delete-ids requires --yes")
        ids = parse_ids(args.delete_ids)
        deleted = delete_ids(data_dir, ids)
        mapping = renumber_contiguous(data_dir)
        print(f"Deleted {len(deleted)} file(s) for {len(ids)} sample id(s).")
        print(f"Renumbered {len(mapping)} sample id(s).")
        if mapping:
            first = min(mapping)
            last = max(mapping)
            print(f"Renumber range touched old ids {first:04d}..{last:04d}.")
        return 0

    for i, path in enumerate(dirs):
        if i:
            print()
        rows = scan_samples(
            path,
            max_lost_ratio=args.max_lost_ratio,
            max_consecutive_lost=args.max_consecutive_lost,
            min_frames=args.min_frames,
            min_path_length=args.min_path_length,
            min_net_displacement=args.min_net_displacement,
            max_stuck_frames=args.max_stuck_frames,
            stuck_radius=args.stuck_radius,
            collision_distance=args.collision_distance,
        )
        print_report(
            rows,
            max_lost_ratio=args.max_lost_ratio,
            max_consecutive_lost=args.max_consecutive_lost,
            min_frames=args.min_frames,
            min_path_length=args.min_path_length,
            min_net_displacement=args.min_net_displacement,
            max_stuck_frames=args.max_stuck_frames,
            stuck_radius=args.stuck_radius,
            collision_distance=args.collision_distance,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
