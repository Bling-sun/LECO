#!/usr/bin/env python3
"""Generate reproducible, obstacle-safe Goal sequences from a custom map JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


GOAL_FORMAT = "neural-tracker-goal-sequence"
GOAL_VERSION = 1
SUPPORTED_MAP_FORMATS = {"neural-tracker-map", "neural-tracker-forest-map"}
COMPOSITE_SHAPES = {
    "l_shape", "e_shape", "t_shape", "h_shape", "s_shape", "f_shape",
}
SEGMENTED_SHAPES = COMPOSITE_SHAPES | {"arch"}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_map(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"map JSON does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid map JSON {path}: {error}") from error
    if payload.get("format") not in SUPPORTED_MAP_FORMATS:
        raise ValueError(
            "unsupported map format; expected neural-tracker-map or "
            "neural-tracker-forest-map"
        )
    config = payload.get("map")
    if not isinstance(config, dict):
        raise ValueError("map JSON must contain a map object")
    for name in ("width", "height", "resolution"):
        if finite_number(config.get(name), f"map.{name}") <= 0.0:
            raise ValueError(f"map.{name} must be positive")
    obstacles = payload.get("obstacles")
    if not isinstance(obstacles, list):
        raise ValueError("map obstacles must be an array")
    return payload


def local_rectangles(
    shape: str, width: float, depth: float, thickness: float
) -> list[tuple[float, float, float, float]]:
    left = -(width - thickness) * 0.5
    right = (width - thickness) * 0.5
    top = (depth - thickness) * 0.5
    bottom = -(depth - thickness) * 0.5
    if shape == "l_shape":
        return [
            (0.0, bottom, width, thickness),
            (left, 0.0, thickness, depth),
        ]
    if shape == "e_shape":
        return [
            (left, 0.0, thickness, depth),
            (0.0, top, width, thickness),
            (0.0, 0.0, width, thickness),
            (0.0, bottom, width, thickness),
        ]
    if shape == "t_shape":
        return [
            (0.0, top, width, thickness),
            (
                0.0,
                -thickness * 0.5,
                thickness,
                max(thickness, depth - thickness),
            ),
        ]
    if shape == "h_shape":
        return [
            (left, 0.0, thickness, depth),
            (right, 0.0, thickness, depth),
            (0.0, 0.0, width, thickness),
        ]
    if shape == "s_shape":
        connector_depth = max(
            thickness, (depth - 3.0 * thickness) * 0.5 + thickness
        )
        return [
            (0.0, top, width, thickness),
            (0.0, 0.0, width, thickness),
            (0.0, bottom, width, thickness),
            (left, depth * 0.25, thickness, connector_depth),
            (right, -depth * 0.25, thickness, connector_depth),
        ]
    if shape == "f_shape":
        return [
            (left, 0.0, thickness, depth),
            (0.0, top, width, thickness),
            (0.0, 0.0, width, thickness),
        ]
    if shape == "arch":
        return [
            (left, 0.0, thickness, depth),
            (right, 0.0, thickness, depth),
        ]
    raise ValueError(f"unsupported composite shape: {shape}")


def obstacle_geometry(
    obstacle: dict[str, Any], resolution: float, index: int
) -> tuple[str, float, float, float, float, float, list[tuple[float, float, float, float]]]:
    shape = str(obstacle.get("shape", "wall"))
    center_x = finite_number(obstacle.get("x", 0.0), f"obstacle {index}.x")
    center_y = finite_number(obstacle.get("y", 0.0), f"obstacle {index}.y")
    angle = math.radians(
        finite_number(obstacle.get("angle", 0.0), f"obstacle {index}.angle")
    )
    width = finite_number(obstacle.get("width", 0.0), f"obstacle {index}.width")
    depth = finite_number(obstacle.get("depth", 0.0), f"obstacle {index}.depth")
    radius = finite_number(obstacle.get("radius", 0.0), f"obstacle {index}.radius")

    if shape == "tree":
        if not bool(obstacle.get("collision", True)):
            return ("skip", center_x, center_y, angle, 0.0, 0.0, [])
        trunk = obstacle.get("trunk")
        if trunk is not None:
            if not isinstance(trunk, dict):
                raise ValueError(f"obstacle {index}.trunk must be an object")
            if not bool(trunk.get("collision", True)):
                return ("skip", center_x, center_y, angle, 0.0, 0.0, [])
            if trunk.get("primitive", "cylinder") != "cylinder":
                raise ValueError(f"obstacle {index} tree trunk must be a cylinder")
            radius = finite_number(
                trunk.get("radius", 0.0), f"obstacle {index}.trunk.radius"
            )
        shape = "cylinder"

    if shape == "cylinder":
        if radius <= 0.0:
            raise ValueError(f"obstacle {index} cylinder radius must be positive")
        return (shape, center_x, center_y, angle, radius, radius, [])

    if width <= 0.0 or depth <= 0.0:
        raise ValueError(f"obstacle {index} width and depth must be positive")
    if shape in SEGMENTED_SHAPES:
        thickness = finite_number(
            obstacle.get("thickness", 0.0), f"obstacle {index}.thickness"
        )
        thickness = max(resolution, min(thickness, width, depth))
        if shape == "arch":
            thickness = min(thickness, width / 3.0)
            height = finite_number(
                obstacle.get("height", 0.0), f"obstacle {index}.height"
            )
            clearance = finite_number(
                obstacle.get("clearance", 0.0),
                f"obstacle {index}.clearance",
            )
            if height <= 0.0:
                raise ValueError(f"obstacle {index} arch height must be positive")
            if clearance <= 0.0 or clearance >= height:
                raise ValueError(
                    f"obstacle {index} arch clearance must be positive "
                    "and below its height"
                )
        rectangles = local_rectangles(shape, width, depth, thickness)
        return (shape, center_x, center_y, angle, width, depth, rectangles)
    if shape == "frustum":
        top_scale = finite_number(
            obstacle.get("topScale", 0.0), f"obstacle {index}.topScale"
        )
        if top_scale <= 0.0 or top_scale > 1.0:
            raise ValueError(
                f"obstacle {index} frustum topScale must be in (0, 1]"
            )
    elif shape not in {"wall", "box"}:
        raise ValueError(f"unsupported obstacle shape: {shape}")
    return (shape, center_x, center_y, angle, width, depth, [])


def rasterize_map(
    payload: dict[str, Any]
) -> tuple[np.ndarray, float, float, float]:
    config = payload["map"]
    width = float(config["width"])
    height = float(config["height"])
    resolution = float(config["resolution"])
    columns = int(round(width / resolution))
    rows = int(round(height / resolution))
    if not math.isclose(columns * resolution, width, abs_tol=1e-9):
        raise ValueError("map width must be an integer multiple of resolution")
    if not math.isclose(rows * resolution, height, abs_tol=1e-9):
        raise ValueError("map height must be an integer multiple of resolution")
    origin_x = -width * 0.5
    origin_y = -height * 0.5
    occupied = np.zeros((rows, columns), dtype=bool)

    for index, raw_obstacle in enumerate(payload["obstacles"], 1):
        if not isinstance(raw_obstacle, dict):
            raise ValueError(f"obstacle {index} must be an object")
        shape, center_x, center_y, angle, extent_x, extent_y, rectangles = (
            obstacle_geometry(raw_obstacle, resolution, index)
        )
        if shape == "skip":
            continue
        extent = (
            extent_x
            if shape == "cylinder"
            else math.hypot(extent_x * 0.5, extent_y * 0.5)
        )
        x_min = max(0, int(math.floor((center_x - extent - origin_x) / resolution)))
        x_max = min(
            columns - 1,
            int(math.ceil((center_x + extent - origin_x) / resolution)),
        )
        y_min = max(0, int(math.floor((center_y - extent - origin_y) / resolution)))
        y_max = min(
            rows - 1,
            int(math.ceil((center_y + extent - origin_y) / resolution)),
        )
        if x_min > x_max or y_min > y_max:
            continue
        world_x = origin_x + (np.arange(x_min, x_max + 1) + 0.5) * resolution
        world_y = origin_y + (np.arange(y_min, y_max + 1) + 0.5) * resolution
        dx = world_x[np.newaxis, :] - center_x
        dy = world_y[:, np.newaxis] - center_y
        cosine = math.cos(angle)
        sine = math.sin(angle)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        if shape == "cylinder":
            mask = local_x * local_x + local_y * local_y <= extent_x * extent_x
        elif shape in SEGMENTED_SHAPES:
            mask = np.zeros(local_x.shape, dtype=bool)
            for rect_x, rect_y, rect_width, rect_depth in rectangles:
                mask |= (
                    (np.abs(local_x - rect_x) <= rect_width * 0.5)
                    & (np.abs(local_y - rect_y) <= rect_depth * 0.5)
                )
        else:
            mask = (
                (np.abs(local_x) <= extent_x * 0.5)
                & (np.abs(local_y) <= extent_y * 0.5)
            )
        occupied[y_min:y_max + 1, x_min:x_max + 1] |= mask

    boundary = payload.get("boundary") or {}
    if not isinstance(boundary, dict):
        raise ValueError("map boundary must be an object")
    if bool(boundary.get("enabled", False)):
        thickness = finite_number(
            boundary.get("thickness", resolution), "boundary.thickness"
        )
        if thickness <= 0.0:
            raise ValueError("boundary.thickness must be positive")
        cells = max(1, int(math.ceil(thickness / resolution)))
        occupied[:cells, :] = True
        occupied[-cells:, :] = True
        occupied[:, :cells] = True
        occupied[:, -cells:] = True
    if not occupied.any():
        raise ValueError("map contains no collision cells")
    return occupied, origin_x, origin_y, resolution


def world_to_cell(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    shape: tuple[int, int],
) -> tuple[int, int]:
    column = int(math.floor((x - origin_x) / resolution))
    row = int(math.floor((y - origin_y) / resolution))
    if row < 0 or column < 0 or row >= shape[0] or column >= shape[1]:
        raise ValueError(f"start point ({x:.3f}, {y:.3f}) is outside the map")
    return row, column


def generate_points(
    candidates: np.ndarray,
    start: tuple[float, float],
    count: int,
    minimum_step: float,
    maximum_step: float,
    rng: random.Random,
    max_restarts: int,
    coverage_grid: int,
    candidate_pool: int,
) -> np.ndarray:
    coordinate_minimum = candidates.min(axis=0)
    coordinate_span = np.maximum(candidates.max(axis=0) - coordinate_minimum, 1e-9)
    candidate_bins_xy = np.floor(
        (candidates - coordinate_minimum) / coordinate_span * coverage_grid
    ).astype(int)
    candidate_bins_xy = np.clip(candidate_bins_xy, 0, coverage_grid - 1)
    candidate_bins = (
        candidate_bins_xy[:, 1] * coverage_grid + candidate_bins_xy[:, 0]
    )
    minimum_squared = minimum_step * minimum_step
    maximum_squared = maximum_step * maximum_step
    best_length = 0
    for _ in range(max_restarts):
        route_indices: list[int] = []
        used = np.zeros(len(candidates), dtype=bool)
        bin_visits = np.zeros(coverage_grid * coverage_grid, dtype=int)
        previous = np.asarray(start, dtype=float)
        for _goal in range(count):
            delta = candidates - previous
            squared = np.einsum("ij,ij->i", delta, delta)
            choices = np.flatnonzero(
                (~used)
                & (squared >= minimum_squared - 1e-9)
                & (squared <= maximum_squared + 1e-9)
            )
            if choices.size == 0:
                break
            if choices.size > candidate_pool:
                pool = np.asarray(
                    rng.sample(choices.tolist(), candidate_pool), dtype=int
                )
            else:
                pool = choices

            history = np.vstack((
                np.asarray(start, dtype=float),
                candidates[np.asarray(route_indices, dtype=int)]
                if route_indices else np.empty((0, 2), dtype=float),
            ))
            history_delta = candidates[pool, np.newaxis, :] - history[np.newaxis, :, :]
            nearest_history = np.sqrt(
                np.min(np.einsum("ijk,ijk->ij", history_delta, history_delta), axis=1)
            )

            # Penalize a candidate segment that lies close and parallel to a
            # previously travelled segment. A crossing is not considered a
            # repeat: the parallel term keeps its penalty small.
            overlap_penalty = np.zeros(len(pool), dtype=float)
            if route_indices:
                route_points = candidates[np.asarray(route_indices, dtype=int)]
                segment_starts = np.vstack((np.asarray(start, dtype=float), route_points[:-1]))
                segment_ends = route_points
                old_vectors = segment_ends - segment_starts
                old_lengths = np.linalg.norm(old_vectors, axis=1)
                old_units = old_vectors / np.maximum(old_lengths[:, np.newaxis], 1e-9)
                old_midpoints = 0.5 * (segment_starts + segment_ends)
                new_vectors = candidates[pool] - previous
                new_lengths = np.linalg.norm(new_vectors, axis=1)
                new_units = new_vectors / np.maximum(new_lengths[:, np.newaxis], 1e-9)
                new_midpoints = 0.5 * (previous + candidates[pool])
                midpoint_delta = (
                    new_midpoints[:, np.newaxis, :] - old_midpoints[np.newaxis, :, :]
                )
                midpoint_distance = np.sqrt(
                    np.einsum("ijk,ijk->ij", midpoint_delta, midpoint_delta)
                )
                parallel = np.abs(new_units @ old_units.T)
                closeness = np.exp(-np.square(midpoint_distance / 5.0))
                overlap_penalty = np.max(closeness * parallel, axis=1)

            coverage_penalty = bin_visits[candidate_bins[pool]].astype(float)
            jitter = np.asarray([rng.random() for _ in range(len(pool))])
            scores = (
                -4.0 * coverage_penalty
                + 2.0 * nearest_history / maximum_step
                - 3.0 * overlap_penalty
                + 0.05 * jitter
            )
            selected = int(pool[int(np.argmax(scores))])
            route_indices.append(selected)
            used[selected] = True
            bin_visits[candidate_bins[selected]] += 1
            previous = candidates[selected]
        best_length = max(best_length, len(route_indices))
        if len(route_indices) == count:
            return candidates[np.asarray(route_indices, dtype=int)]
    raise ValueError(
        f"could not generate {count} goals after {max_restarts} restarts "
        f"(best route contained {best_length}); reduce --count/--min-step/"
        "--min-clearance or choose a larger connected map"
    )


def build_sequence(
    map_path: Path,
    map_payload: dict[str, Any],
    output: Path,
    count: int,
    seed: int,
    minimum_clearance: float,
    path_clearance: float,
    minimum_step: float,
    maximum_step: float,
    start: tuple[float, float],
    nominal_speed: float,
    topic: str,
    frame_id: str,
    run_label: str,
    max_restarts: int,
    coverage_grid: int,
    candidate_pool: int,
) -> dict[str, Any]:
    occupied, origin_x, origin_y, resolution = rasterize_map(map_payload)
    # The EDT measures cell-centre distance. Subtract half a cell diagonal so
    # the reported clearance is a conservative lower bound to obstacle surface.
    clearance = ndimage.distance_transform_edt(~occupied) * resolution
    clearance = np.maximum(0.0, clearance - resolution / math.sqrt(2.0))
    path_free = clearance >= path_clearance
    labels, _ = ndimage.label(path_free, structure=np.ones((3, 3), dtype=np.uint8))
    start_row, start_column = world_to_cell(
        start[0], start[1], origin_x, origin_y, resolution, occupied.shape
    )
    start_component = int(labels[start_row, start_column])
    if start_component == 0:
        raise ValueError(
            f"start point ({start[0]:.3f}, {start[1]:.3f}) does not have "
            f"the requested {path_clearance:.3f} m path clearance"
        )
    candidate_rows, candidate_columns = np.nonzero(
        (labels == start_component) & (clearance >= minimum_clearance)
    )
    if candidate_rows.size == 0:
        raise ValueError("no candidate cells satisfy the clearance constraints")
    candidate_x = origin_x + (candidate_columns.astype(float) + 0.5) * resolution
    candidate_y = origin_y + (candidate_rows.astype(float) + 0.5) * resolution
    candidates = np.column_stack((candidate_x, candidate_y))
    points = generate_points(
        candidates,
        start,
        count,
        minimum_step,
        maximum_step,
        random.Random(seed),
        max_restarts,
        coverage_grid,
        candidate_pool,
    )

    point_rows = np.floor((points[:, 1] - origin_y) / resolution).astype(int)
    point_columns = np.floor((points[:, 0] - origin_x) / resolution).astype(int)
    point_clearances = clearance[point_rows, point_columns]
    previous_points = np.vstack((np.asarray(start, dtype=float), points[:-1]))
    step_distances = np.linalg.norm(points - previous_points, axis=1)
    segment_vectors = points - previous_points
    segment_units = segment_vectors / np.maximum(
        step_distances[:, np.newaxis], 1e-9
    )
    segment_midpoints = 0.5 * (points + previous_points)
    midpoint_delta = (
        segment_midpoints[:, np.newaxis, :] - segment_midpoints[np.newaxis, :, :]
    )
    midpoint_distance = np.sqrt(
        np.einsum("ijk,ijk->ij", midpoint_delta, midpoint_delta)
    )
    segment_parallel = np.abs(segment_units @ segment_units.T)
    non_adjacent = np.triu(np.ones(midpoint_distance.shape, dtype=bool), k=2)
    near_parallel_repeats = int(np.count_nonzero(
        non_adjacent & (midpoint_distance < 5.0) & (segment_parallel > 0.9)
    ))
    coordinate_minimum = candidates.min(axis=0)
    coordinate_span = np.maximum(candidates.max(axis=0) - coordinate_minimum, 1e-9)
    all_bins_xy = np.floor(
        (candidates - coordinate_minimum) / coordinate_span * coverage_grid
    ).astype(int)
    route_bins_xy = np.floor(
        (points - coordinate_minimum) / coordinate_span * coverage_grid
    ).astype(int)
    all_bins_xy = np.clip(all_bins_xy, 0, coverage_grid - 1)
    route_bins_xy = np.clip(route_bins_xy, 0, coverage_grid - 1)
    available_bins = set(map(tuple, all_bins_xy.tolist()))
    visited_bins = set(map(tuple, route_bins_xy.tolist()))
    goals: list[dict[str, Any]] = []
    elapsed = 0.0
    for index, point in enumerate(points):
        if index + 1 < len(points):
            direction = points[index + 1] - point
        else:
            direction = point - previous_points[index]
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        if index > 0:
            elapsed += float(step_distances[index]) / nominal_speed
        goals.append({
            "index": index + 1,
            "time": round(elapsed, 9),
            "frame_id": frame_id,
            "position": {
                "x": round(float(point[0]), 9),
                "y": round(float(point[1]), 9),
                "z": 0.0,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(yaw * 0.5),
                "w": math.cos(yaw * 0.5),
            },
            "yaw": yaw,
        })
    return {
        "format": GOAL_FORMAT,
        "version": GOAL_VERSION,
        "topic": topic,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "run_directory": str(output.parent),
        "time_basis": "seconds since first received goal",
        "generator": {
            "name": "random_goal_sequence",
            "map_file": str(map_path),
            "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            "seed": seed,
            "count": count,
            "start": {"x": start[0], "y": start[1]},
            "minimum_obstacle_clearance_m": minimum_clearance,
            "path_connectivity_clearance_m": path_clearance,
            "minimum_step_distance_m": minimum_step,
            "maximum_step_distance_m": maximum_step,
            "nominal_speed_mps_for_timestamps": nominal_speed,
            "map_resolution_m": resolution,
            "candidate_cells": int(candidate_rows.size),
            "coverage_grid": [coverage_grid, coverage_grid],
            "available_coverage_cells": len(available_bins),
            "visited_coverage_cells": len(visited_bins),
            "coverage_ratio_percent": (
                100.0 * len(visited_bins) / len(available_bins)
                if available_bins else 0.0
            ),
            "candidate_pool_per_step": candidate_pool,
            "unique_goal_points": int(len(np.unique(points, axis=0))),
            "near_parallel_repeated_segments": near_parallel_repeats,
            "repeated_segment_definition": (
                "non-adjacent segment midpoints <5m and |direction dot|>0.9"
            ),
            "achieved_minimum_clearance_m": float(point_clearances.min()),
            "achieved_minimum_step_distance_m": float(step_distances.min()),
            "achieved_maximum_step_distance_m": float(step_distances.max()),
        },
        "goals": goals,
    }


def safe_run_label(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("--run-label must not be empty")
    return cleaned


def resolve_map_path(requested: Path, workspace: Path) -> Path:
    names = [requested]
    if requested.suffix.lower() != ".json":
        names.append(requested.with_suffix(".json"))
    candidates: list[Path] = []
    for name in names:
        if name.is_absolute():
            candidates.append(name)
        else:
            candidates.extend((workspace / name, workspace / "src/simulator/map" / name))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(path.expanduser().resolve()) for path in candidates)
    raise ValueError(f"map JSON not found; checked: {checked}")


def next_automatic_output(experiment_root: Path, map_stem: str) -> Path:
    experiment_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(map_stem)}_goal(\d+)\.json$")
    used = {
        int(match.group(1))
        for path in experiment_root.glob(f"{map_stem}_goal*.json")
        if (match := pattern.match(path.name)) is not None
    }
    index = 0
    while index in used:
        index += 1
    return experiment_root / f"{map_stem}_goal{index}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random replay Goals with obstacle clearance and bounded "
            "Euclidean distance between consecutive points"
        )
    )
    parser.add_argument(
        "map_json",
        type=Path,
        help=(
            "map path or selector such as maze2, maze2.json, or "
            "src/simulator/map/maze2.json"
        ),
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        type=Path,
        help=(
            "output Goal JSON; default: experiments/<map>_goal<N>.json using "
            "the first unused index"
        ),
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, help="fixed seed; default: random seed")
    parser.add_argument("--min-clearance", type=float, default=1.0)
    parser.add_argument("--path-clearance", type=float, default=0.3)
    parser.add_argument("--min-step", type=float, default=25.0)
    parser.add_argument("--max-step", type=float, default=30.0)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--nominal-speed", type=float, default=1.7)
    parser.add_argument("--topic", default="/goal_pose")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--run-label")
    parser.add_argument("--max-restarts", type=int, default=200)
    parser.add_argument(
        "--coverage-grid", type=int, default=6,
        help="map partition per axis used to spread Goals (default: 6)",
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=4096,
        help="maximum random feasible candidates scored per Goal (default: 4096)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    try:
        map_path = resolve_map_path(args.map_json, workspace)
    except ValueError as error:
        parser.error(str(error))
    experiment_root = args.experiment_root.expanduser()
    if not experiment_root.is_absolute():
        experiment_root = workspace / experiment_root
    experiment_root = experiment_root.resolve()
    automatic_output = args.output_json is None
    if automatic_output:
        output = next_automatic_output(experiment_root, map_path.stem)
    else:
        output = args.output_json.expanduser()
        if not output.is_absolute():
            output = workspace / output
        output = output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"output exists: {output}; choose another file or use --overwrite")
    if automatic_output and args.overwrite:
        parser.error("--overwrite is only meaningful with an explicit output_json")
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.min_clearance <= 0.0 or args.path_clearance <= 0.0:
        parser.error("--min-clearance and --path-clearance must be positive")
    if args.min_step <= 0.0 or args.max_step < args.min_step:
        parser.error("require 0 < --min-step <= --max-step")
    if args.nominal_speed <= 0.0:
        parser.error("--nominal-speed must be positive")
    if args.max_restarts <= 0:
        parser.error("--max-restarts must be positive")
    if args.coverage_grid < 2:
        parser.error("--coverage-grid must be at least 2")
    if args.candidate_pool < 64:
        parser.error("--candidate-pool must be at least 64")
    if (args.start_x is None) != (args.start_y is None):
        parser.error("--start-x and --start-y must be provided together")
    try:
        payload = load_map(map_path)
        if args.start_x is None:
            target = payload.get("target")
            if not isinstance(target, dict):
                raise ValueError(
                    "map has no target start; pass --start-x and --start-y"
                )
            start = (
                finite_number(target.get("x"), "target.x"),
                finite_number(target.get("y"), "target.y"),
            )
        else:
            start = (float(args.start_x), float(args.start_y))
        seed = args.seed if args.seed is not None else secrets.randbits(63)
        label = safe_run_label(args.run_label or output.stem)
        sequence = build_sequence(
            map_path,
            payload,
            output,
            args.count,
            seed,
            args.min_clearance,
            args.path_clearance,
            args.min_step,
            args.max_step,
            start,
            args.nominal_speed,
            args.topic,
            args.frame_id,
            label,
            args.max_restarts,
            args.coverage_grid,
            args.candidate_pool,
        )
    except ValueError as error:
        parser.error(str(error))
    atomic_write(output, sequence)
    metadata = sequence["generator"]
    print(f"Generated {len(sequence['goals'])} goals: {output}")
    print(f"  map: {map_path}")
    print(f"  seed: {seed}")
    print(
        "  clearance: min %.3f m (required >= %.3f m)"
        % (
            metadata["achieved_minimum_clearance_m"],
            metadata["minimum_obstacle_clearance_m"],
        )
    )
    print(
        "  step distance: %.3f .. %.3f m (required %.3f .. %.3f m)"
        % (
            metadata["achieved_minimum_step_distance_m"],
            metadata["achieved_maximum_step_distance_m"],
            metadata["minimum_step_distance_m"],
            metadata["maximum_step_distance_m"],
        )
    )
    print(
        "  map coverage: %d/%d cells (%.1f%% on %dx%d grid)"
        % (
            metadata["visited_coverage_cells"],
            metadata["available_coverage_cells"],
            metadata["coverage_ratio_percent"],
            metadata["coverage_grid"][0],
            metadata["coverage_grid"][1],
        )
    )
    print(
        "  repeated near-parallel segments: %d"
        % metadata["near_parallel_repeated_segments"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
