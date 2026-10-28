#!/usr/bin/env python3
"""Plot target-position heatmaps in the tracker's horizontal camera FOV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
try:
    import cmocean
except ImportError:  # optional dependency; matplotlib fallback below
    cmocean = None


REQUIRED_COLUMNS = {
    "relative_forward_m",
    "relative_left_m",
    "detector_visible",
    "inside_geometric_fov",
    "sensor_fov_deg",
    "sensor_range_m",
}


def discover_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".csv":
            return [input_path]
        if input_path.suffix.lower() == ".json":
            candidate = input_path.with_name(input_path.stem + "_target_fov.csv")
            return [candidate] if candidate.is_file() else []
        return []
    return sorted(input_path.glob("**/*_target_fov.csv"))


def report_identity(csv_path: Path) -> Optional[tuple[Any, ...]]:
    suffix = "_target_fov"
    if not csv_path.stem.endswith(suffix):
        return None
    report_path = csv_path.with_name(csv_path.stem[: -len(suffix)] + ".json")
    if not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = data.get("configuration", {})
    provenance = data.get("provenance", {})
    map_info = provenance.get("map") or {}
    policy_info = provenance.get("policy") or {}
    return (
        provenance.get("algorithm_label"),
        map_info.get("sha256"),
        policy_info.get("sha256"),
        config.get("sensor_fov_deg"),
        config.get("sensor_range_m"),
    )


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_samples(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    rejected = 0
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - fields
                if missing:
                    raise ValueError("missing columns: %s" % ", ".join(sorted(missing)))
                for row in reader:
                    try:
                        sample = {
                            "forward": float(row["relative_forward_m"]),
                            "left": float(row["relative_left_m"]),
                            "visible": parse_bool(row["detector_visible"]),
                            "inside": parse_bool(row["inside_geometric_fov"]),
                            "fov_deg": float(row["sensor_fov_deg"]),
                            "range_m": float(row["sensor_range_m"]),
                            "source": str(path),
                        }
                        numeric = (
                            sample["forward"], sample["left"],
                            sample["fov_deg"], sample["range_m"],
                        )
                        if not all(math.isfinite(value) for value in numeric):
                            raise ValueError("non-finite value")
                        samples.append(sample)
                    except (KeyError, TypeError, ValueError):
                        rejected += 1
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot parse {path}: {error}") from error
    return samples, rejected


def consistent_sensor_value(
    samples: list[dict[str, Any]], key: str, override: Optional[float]
) -> float:
    if override is not None:
        return override
    values = sorted({round(float(sample[key]), 9) for sample in samples})
    if len(values) != 1:
        raise ValueError(
            f"input mixes {key} values {values}; split the configurations or "
            "pass an explicit override"
        )
    return values[0]


def default_output_prefix(input_path: Path, visibility: str) -> Path:
    if input_path.is_dir():
        return input_path / "summary" / f"target_fov_heatmap_{visibility}"
    stem = input_path.stem
    if stem.endswith("_target_fov"):
        stem = stem[: -len("_target_fov")]
    return input_path.with_name(f"{stem}_target_fov_heatmap_{visibility}")


def plot_heatmap(
    samples: list[dict[str, Any]],
    output_prefix: Path,
    visibility: str,
    fov_deg: float,
    sensor_range: float,
    bins: int,
    smooth_sigma: float,
    vmax_percent: Optional[float],
    title: Optional[str],
) -> dict[str, Any]:
    in_fov = [sample for sample in samples if sample["inside"]]
    if visibility == "visible":
        selected = [sample for sample in in_fov if sample["visible"]]
    elif visibility == "occluded":
        selected = [sample for sample in in_fov if not sample["visible"]]
    else:
        selected = in_fov
    if not selected:
        raise ValueError(
            f"no {visibility} target samples remain inside the geometric FOV"
        )

    half_angle = math.radians(fov_deg) * 0.5
    lateral_limit = sensor_range * math.sin(half_angle)
    x_edges = np.linspace(0.0, sensor_range, bins + 1)
    y_edges = np.linspace(-lateral_limit, lateral_limit, bins + 1)
    forward = np.asarray([sample["forward"] for sample in selected])
    left = np.asarray([sample["left"] for sample in selected])
    histogram, _, _ = np.histogram2d(forward, left, bins=(x_edges, y_edges))
    if smooth_sigma > 0.0:
        histogram = gaussian_filter(histogram, sigma=smooth_sigma, mode="constant")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    center_x, center_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    geometric_mask = (
        (np.hypot(center_x, center_y) <= sensor_range)
        & (np.abs(np.arctan2(center_y, center_x)) <= half_angle)
    )
    histogram[~geometric_mask] = 0.0
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("histogram is empty after applying the FOV mask")
    heat = 100.0 * histogram / total
    # A square-root display scale keeps sparse but important regions visible
    # without allowing the single densest bin to dominate the paper figure.
    # Keep zero-density bins inside the sensing sector: the pale end of the
    # colormap provides a continuous background instead of a hard white cut.
    # Render the complete rectangular canvas.  Samples outside the geometric
    # sector remain at the pale low-density end of the scale; the FOV rays and
    # arc communicate the valid sensing region without a hard polygon cut.
    masked_heat = np.ma.asarray(heat)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    # Paper-friendly sequential heatmap: low density fades to warm white/yellow,
    # while the highest-density region is red.
    # Force the zero-density end to pure white, then transition gently through
    # very pale yellow before reaching orange/red at the hotspot.
    color_map = LinearSegmentedColormap.from_list(
        "white_yellow_orange_red",
        ["#ffffff", "#fffde6", "#fff7bc", "#fed976", "#f46d43", "#b30000"],
        N=256,
    )
    color_map.set_bad("white")
    color_map.set_under("white")
    positive = heat[geometric_mask & (heat > 0.0)]
    vmax = float(vmax_percent if vmax_percent is not None else np.percentile(positive, 99.5))
    vmin = 0.0
    vmax = max(vmax, 1e-6)
    mesh = axis.imshow(
        masked_heat.T,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=color_map,
        norm=PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax),
        interpolation="bicubic",
        aspect="auto",
        rasterized=True,
    )

    ray_x = sensor_range * math.cos(half_angle)
    ray_y = sensor_range * math.sin(half_angle)
    axis.plot([0.0, ray_x], [0.0, ray_y], color="#202020", linewidth=1.1)
    axis.plot([0.0, ray_x], [0.0, -ray_y], color="#202020", linewidth=1.1)
    arc_angle = np.linspace(-half_angle, half_angle, 240)
    axis.plot(
        sensor_range * np.cos(arc_angle),
        sensor_range * np.sin(arc_angle),
        color="#202020",
        linewidth=1.0,
    )
    axis.scatter([0.0], [0.0], marker="^", s=34, color="#087e8b", zorder=4)
    axis.axhline(0.0, color="gray", alpha=0.5, linewidth=0.7, linestyle="--")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.0, sensor_range)
    axis.set_ylim(-lateral_limit, lateral_limit)
    axis.margins(0)
    axis.set_facecolor("#eaf6fb")
    axis.set_xlabel("Relative forward position (m)")
    axis.set_ylabel("Relative left position (m)")
    axis.grid(color="#777777", alpha=0.18, linewidth=0.5)
    axis.tick_params(direction="out", length=3)
    axis.set_title(
        title
        or "Target position in tracker FOV — %s (N=%d)"
        % (visibility, len(selected))
    )
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.025)
    colorbar.set_label("Relative sample density (%)")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return {
        "input_samples": len(samples),
        "inside_fov_samples": len(in_fov),
        "selected_samples": len(selected),
        "visibility_filter": visibility,
        "sensor_fov_deg": fov_deg,
        "sensor_range_m": sensor_range,
        "bins": bins,
        "smooth_sigma_bins": smooth_sigma,
        "maximum_bin_percent": float(heat.max()),
        "color_scale_vmax_percent": vmax,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a paper-ready target-position heatmap in tracker FOV"
    )
    parser.add_argument("input", type=Path, help="run JSON/CSV or experiment directory")
    parser.add_argument(
        "--visibility", choices=("visible", "occluded", "all"), default="visible",
        help="detector visibility subset; every mode is clipped to geometric FOV",
    )
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--smooth-sigma", type=float, default=1.0,
        help="Gaussian smoothing sigma in histogram bins; 0 disables smoothing",
    )
    parser.add_argument("--fov-deg", type=float, help="override recorded FOV")
    parser.add_argument("--sensor-range", type=float, help="override recorded range")
    parser.add_argument(
        "--vmax-percent", type=float,
        help="fixed colorbar maximum for comparable multi-algorithm figures",
    )
    parser.add_argument("--title", help="custom figure title")
    parser.add_argument(
        "--output-prefix", type=Path,
        help="output path without suffix; both PNG and PDF are written",
    )
    parser.add_argument(
        "--allow-mixed-configuration", action="store_true",
        help="allow combining different algorithm/map/policy identities",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        parser.error(f"input does not exist: {input_path}")
    if args.bins < 8:
        parser.error("--bins must be at least 8")
    if args.smooth_sigma < 0.0:
        parser.error("--smooth-sigma must be non-negative")
    if args.fov_deg is not None and not 0.0 < args.fov_deg < 180.0:
        parser.error("--fov-deg must be in (0, 180)")
    if args.sensor_range is not None and args.sensor_range <= 0.0:
        parser.error("--sensor-range must be positive")
    if args.vmax_percent is not None and args.vmax_percent <= 0.0:
        parser.error("--vmax-percent must be positive")

    paths = discover_csv_files(input_path)
    if not paths:
        print(
            "错误：没有找到 *_target_fov.csv。旧版数据只记录 visible=0/1，"
            "无法恢复 target 在 FOV 内的位置；请用新版 neural_tracker_metrics 重新采集。",
            file=sys.stderr,
        )
        return 1
    identities = {identity for path in paths if (identity := report_identity(path))}
    if len(identities) > 1 and not args.allow_mixed_configuration:
        parser.error(
            "input contains different algorithms, maps or policies; plot each "
            "configuration separately, or explicitly pass --allow-mixed-configuration"
        )
    try:
        samples, rejected = load_samples(paths)
        if not samples:
            raise ValueError("all FOV rows are invalid or empty")
        fov_deg = consistent_sensor_value(samples, "fov_deg", args.fov_deg)
        sensor_range = consistent_sensor_value(
            samples, "range_m", args.sensor_range
        )
        prefix = (
            args.output_prefix.expanduser().resolve()
            if args.output_prefix is not None
            else default_output_prefix(input_path, args.visibility)
        )
        result = plot_heatmap(
            samples,
            prefix,
            args.visibility,
            fov_deg,
            sensor_range,
            args.bins,
            args.smooth_sigma,
            args.vmax_percent,
            args.title,
        )
    except ValueError as error:
        parser.error(str(error))
    result["source_csv_files"] = [str(path) for path in paths]
    result["rejected_rows"] = rejected
    metadata_path = prefix.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FOV heatmap: %d selected / %d total samples" % (
        result["selected_samples"], result["input_samples"]
    ))
    print("  %s" % result["png"])
    print("  %s" % result["pdf"])
    print("  %s" % metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
