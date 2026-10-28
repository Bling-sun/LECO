#!/usr/bin/env python3
"""Generate a difficult, strictly non-overlapping random comparison map."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SEED = 20261003
WIDTH = 42
HEIGHT = 24
MIN_BOUNDING_GAP = 0.12
MAX_PLACEMENT_ATTEMPTS = 200_000
SHAPE_COUNTS = {
    "box": 18,
    "cylinder": 12,
    "wall": 8,
    "l_shape": 6,
    "t_shape": 6,
}


def make_specification(shape: str, number: int, rng: random.Random) -> dict:
    if shape == "box":
        width = rng.uniform(1.20, 3.00)
        depth = rng.uniform(1.00, 2.40)
        thickness = min(width, depth)
        bounding_radius = 0.5 * math.hypot(width, depth)
    elif shape == "cylinder":
        bounding_radius = rng.uniform(0.50, 1.10)
        width = depth = 2.0 * bounding_radius
        thickness = bounding_radius
    elif shape == "wall":
        width = rng.uniform(1.80, 4.00)
        depth = rng.uniform(0.25, 0.65)
        thickness = depth
        bounding_radius = 0.5 * math.hypot(width, depth)
    else:
        width = rng.uniform(1.50, 3.00)
        depth = rng.uniform(1.50, 3.00)
        thickness = rng.uniform(0.25, 0.55)
        bounding_radius = 0.5 * math.hypot(width, depth)
    return {
        "shape": shape,
        "number": number,
        "width": width,
        "depth": depth,
        "thickness": thickness,
        "bounding_radius": bounding_radius,
    }


def generate() -> dict:
    rng = random.Random(SEED)
    shapes = [shape for shape, count in SHAPE_COUNTS.items() for _ in range(count)]
    rng.shuffle(shapes)
    counters = {shape: 0 for shape in SHAPE_COUNTS}
    specifications = []
    for shape in shapes:
        counters[shape] += 1
        specifications.append(make_specification(shape, counters[shape], rng))

    # Place large footprints first. Every obstacle is contained by its bounding
    # circle, so disjoint circles prove the rendered geometries cannot overlap.
    specifications.sort(key=lambda item: item["bounding_radius"], reverse=True)
    placed: list[tuple[float, float, float, dict]] = []
    actors = ((-18.5, 0.0), (-16.0, 0.0))
    for specification in specifications:
        radius = specification["bounding_radius"]
        position = None
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = (
                rng.uniform(-20.00 + radius, 20.00 - radius),
                rng.uniform(-11.00 + radius, 11.00 - radius),
            )
            if any(math.dist(candidate, actor) < radius + 1.70 for actor in actors):
                continue
            if all(
                math.dist(candidate, (x, y))
                >= radius + other_radius + MIN_BOUNDING_GAP
                for x, y, other_radius, _ in placed
            ):
                position = candidate
                break
        if position is None:
            raise RuntimeError(f"could not place {specification['shape']} obstacle")
        placed.append((position[0], position[1], radius, specification))

    rng.shuffle(placed)
    obstacles = []
    for x, y, bounding_radius, specification in placed:
        shape = specification["shape"]
        number = specification["number"]
        width = specification["width"]
        depth = specification["depth"]
        thickness = specification["thickness"]
        display_radius = bounding_radius if shape == "cylinder" else width / 2.0
        obstacles.append(
            {
                "id": f"hard-{shape}-{number:03d}",
                "name": f"{shape}_{number:03d}",
                "shape": shape,
                "x": round(x, 3),
                "y": round(y, 3),
                "angle": rng.randrange(0, 180, 5),
                "width": round(width, 3),
                "depth": round(depth, 3),
                "height": round(rng.uniform(4.0, 6.0), 3),
                "thickness": round(thickness, 3),
                "radius": round(display_radius, 3),
                "topScale": round(rng.uniform(0.35, 0.85), 3),
                "clearance": round(rng.uniform(0.40, 0.90), 3),
                "crownRadius": round(display_radius, 3),
                "bounding_radius": round(bounding_radius, 6),
            }
        )

    return {
        "format": "neural-tracker-map",
        "version": 1,
        "name": "dense_hard_50_nonoverlap_20261003",
        "map": {"width": WIDTH, "height": HEIGHT, "resolution": 0.05},
        "boundary": {"enabled": True, "thickness": 0.15, "height": 5.0},
        "tracker": {"x": -18.5, "y": 0, "yaw": 0},
        "target": {"x": -16, "y": 0, "yaw": 0},
        "generation": {
            "seed": SEED,
            "method": "strict non-overlapping random circumscribed-footprint rejection",
            "minimum_bounding_gap_m": MIN_BOUNDING_GAP,
            "height_range_m": [4.0, 6.0],
            "description": "50 large varied obstacles in a difficult irregular non-overlapping field",
            "counts": SHAPE_COUNTS,
        },
        "obstacles": obstacles,
    }


def main() -> None:
    default_output = Path(__file__).with_name("dense_small_random.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(generate(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({sum(SHAPE_COUNTS.values())} obstacles, seed={SEED})")


if __name__ == "__main__":
    main()
