#!/usr/bin/env python3
"""Filter tracker data by current distance and optional target-speed validity.

The source ``train.pkl`` and ``val.pkl`` are never modified.  A sample is
rejected when its current ``robot_state[:2]`` and ``target_pos[:2]`` are
strictly closer than ``--min-distance``.  A small numerical tolerance prevents
an exact boundary value (for example 0.6 m stored as float32) from being
rejected by rounding noise.  ``--max-target-speed`` can additionally reject
finite-difference velocity spikes before they destabilize moving-target losses.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np


FILTER_VERSION = "tracker_target_distance_and_speed_v2"
COMPARISON_ATOL_M = 1e-6


def current_tracking_distance(sample: dict[str, Any]) -> float:
    """Return the current tracker-target distance, or NaN if unavailable."""
    robot = np.asarray(sample.get("robot_state", []), dtype=np.float64).reshape(-1)
    target = np.asarray(sample.get("target_pos", []), dtype=np.float64).reshape(-1)
    if len(robot) < 2 or len(target) < 2:
        return float("nan")
    if not np.isfinite(robot[:2]).all() or not np.isfinite(target[:2]).all():
        return float("nan")
    return float(np.linalg.norm(robot[:2] - target[:2]))


def current_target_speed(sample: dict[str, Any]) -> float:
    """Return target speed, or NaN if the two velocity components are invalid."""
    velocity = np.asarray(sample.get("target_vel", []), dtype=np.float64).reshape(-1)
    if len(velocity) < 2 or not np.isfinite(velocity[:2]).all():
        return float("nan")
    return float(np.linalg.norm(velocity[:2]))


def _atomic_pickle(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_split(
    split: str, source: Path, output_dir: Path, threshold: float,
    max_target_speed: float | None,
) -> dict[str, Any]:
    print(f"Loading {source} ...", flush=True)
    with source.open("rb") as handle:
        samples: list[dict[str, Any]] = pickle.load(handle)
    print(f"  loaded {len(samples):,} samples", flush=True)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unauditable = 0
    finite_distances: list[float] = []
    finite_speeds: list[float] = []
    distance_rejections = 0
    target_speed_rejections = 0
    overlapping_rejections = 0

    for index, sample in enumerate(samples):
        distance = current_tracking_distance(sample)
        target_speed = current_target_speed(sample)
        distance_bad = (
            np.isfinite(distance)
            and distance < threshold - COMPARISON_ATOL_M
        )
        speed_bad = (
            max_target_speed is not None
            and (
                not np.isfinite(target_speed)
                or target_speed > max_target_speed + COMPARISON_ATOL_M
            )
        )
        if not np.isfinite(distance):
            unauditable += 1
            kept.append(sample)
        elif distance_bad or speed_bad:
            distance_rejections += int(distance_bad)
            target_speed_rejections += int(speed_bad)
            overlapping_rejections += int(distance_bad and speed_bad)
            reasons = []
            if distance_bad:
                reasons.append("tracker_target_distance_below_minimum")
            if speed_bad:
                reasons.append("target_speed_above_maximum_or_invalid")
            rejected.append({
                "source_index": index,
                "tracker_target_distance_m": distance,
                "target_speed_mps": target_speed,
                "reasons": reasons,
                "scenario_type": str(sample.get("scenario_type", "unknown")),
            })
        else:
            kept.append(sample)
        if np.isfinite(distance):
            finite_distances.append(distance)
        if np.isfinite(target_speed):
            finite_speeds.append(target_speed)

        if (index + 1) % 100000 == 0 or index + 1 == len(samples):
            print(
                f"  {index + 1:,}/{len(samples):,}: "
                f"kept={len(kept):,}, rejected={len(rejected):,}",
                flush=True,
            )

    _atomic_pickle(kept, output_dir / f"{split}.pkl")
    _atomic_jsonl(rejected, output_dir / f"rejected_{split}.jsonl")
    distances = np.asarray(finite_distances, dtype=np.float32)
    speeds = np.asarray(finite_speeds, dtype=np.float32)
    return {
        "source": str(source),
        "source_samples": len(samples),
        "kept_samples": len(kept),
        "rejected_samples": len(rejected),
        "distance_rejections": distance_rejections,
        "target_speed_rejections": target_speed_rejections,
        "overlapping_rejections": overlapping_rejections,
        "unauditable_samples_kept": unauditable,
        "current_distance_quantiles_m": (
            {str(q): float(np.quantile(distances, q)) for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)}
            if len(distances) else {}
        ),
        "target_speed_quantiles_mps": (
            {str(q): float(np.quantile(speeds, q)) for q in (0.0, 0.5, 0.95, 0.99, 0.999, 1.0)}
            if len(speeds) else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=Path("data/scenarios"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/scenarios_min_distance_0p6"))
    parser.add_argument("--min-distance", type=float, default=0.6)
    parser.add_argument(
        "--max-target-speed", type=float, default=None,
        help="also reject invalid target velocity or speed above this value",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.min_distance <= 0.0:
        parser.error("--min-distance must be positive")
    if args.max_target_speed is not None and args.max_target_speed <= 0.0:
        parser.error("--max-target-speed must be positive")
    sources = {split: args.src_dir / f"{split}.pkl" for split in ("train", "val")}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        parser.error("missing source pickle(s): " + ", ".join(missing))
    outputs = [args.out_dir / f"{split}.pkl" for split in sources]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error(f"output already exists in {args.out_dir}; use --overwrite explicitly")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "filter_version": FILTER_VERSION,
        "source_dir": str(args.src_dir),
        "minimum_tracking_distance_m": args.min_distance,
        "comparison": "strictly_less_than",
        "comparison_absolute_tolerance_m": COMPARISON_ATOL_M,
        "maximum_target_speed_mps": args.max_target_speed,
        "splits": {},
    }
    description = f"reject current distance < {args.min_distance:.3f}m"
    if args.max_target_speed is not None:
        description += f" or target speed > {args.max_target_speed:.3f}m/s"
    print(f"Tracker data filter: {description}", flush=True)
    for split, source in sources.items():
        manifest["splits"][split] = process_split(
            split, source, args.out_dir, args.min_distance, args.max_target_speed
        )
    _atomic_json(manifest, args.out_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
