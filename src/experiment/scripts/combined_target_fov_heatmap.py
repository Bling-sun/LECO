#!/usr/bin/env python3
"""Replot three recorded FOV distributions, ordered on one shared plotting axis.

Run from any directory with Python providing numpy, scipy and matplotlib.
Each segment retains its own local forward coordinates; density is normalized
within each experiment. Original heatmap metadata supplies all histogram settings
and each experiment's original color scale. Dividing by those original scale
limits preserves the colors while allowing one shared normalized density bar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from neural_tracker_fov_heatmap import (
    LinearSegmentedColormap,
    PowerNorm,
    consistent_sensor_value,
    gaussian_filter,
    load_samples,
    np,
    plt,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = [
    ("2026-08-22/mess2_neural_run01", "(a) mess2"),
    ("2026-08-22/maze4_neural_run01", "(b) maze4"),
    ("2026-08-23/maze1_random_obstacles_neural_run03", "(c) maze1 - random obstacles"),
]
OUTPUT = ROOT / "output/pdf/target_fov_heatmap_visible_combined"


def load_distribution(run: str) -> dict:
    directory = ROOT / "experiments" / run
    metadata = json.loads(
        (directory / "summary/target_fov_heatmap_visible.json").read_text()
    )
    paths = [Path(path) for path in metadata["source_csv_files"]]
    samples, rejected = load_samples(paths)
    fov = consistent_sensor_value(samples, "fov_deg", None)
    radius = consistent_sensor_value(samples, "range_m", None)
    selected = [sample for sample in samples if sample["inside"] and sample["visible"]]
    if not selected:
        raise ValueError(f"No visible samples in {run}")
    half_angle = math.radians(fov) / 2
    lateral = radius * math.sin(half_angle)
    bins = metadata["bins"]
    x_edges = np.linspace(0, radius, bins + 1)
    y_edges = np.linspace(-lateral, lateral, bins + 1)
    histogram, _, _ = np.histogram2d(
        [sample["forward"] for sample in selected],
        [sample["left"] for sample in selected],
        bins=(x_edges, y_edges),
    )
    sigma = metadata["smooth_sigma_bins"]
    if sigma > 0:
        histogram = gaussian_filter(histogram, sigma=sigma, mode="constant")
    x, y = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
        indexing="ij",
    )
    mask = (np.hypot(x, y) <= radius) & (np.abs(np.arctan2(y, x)) <= half_angle)
    histogram[~mask] = 0
    heat = 100 * histogram / histogram.sum()
    # Confirm the raw-data reconstruction reproduces the original statistics.
    assert len(samples) == metadata["input_samples"], run
    assert len(selected) == metadata["selected_samples"], run
    assert rejected == metadata["rejected_rows"], run
    assert np.isclose(heat.max(), metadata["maximum_bin_percent"]), run
    assert np.isclose(heat.sum(), 100), run
    return dict(
        heat=heat, half_angle=half_angle, radius=radius, lateral=lateral,
        metadata=metadata, run=run,
    )


def main() -> None:
    distributions = [load_distribution(run) for run, _ in RUNS]
    first = distributions[0]
    for distribution in distributions[1:]:
        for key in ("half_angle", "radius", "lateral"):
            assert np.isclose(distribution[key], first[key]), key
        assert distribution["metadata"]["bins"] == first["metadata"]["bins"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "font.weight": "bold", "axes.labelsize": 15,
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cmap = LinearSegmentedColormap.from_list(
        "white_yellow_orange_red",
        ["#ffffff", "#fffde6", "#fff7bc", "#fed976", "#f46d43", "#b30000"],
        N=256,
    )
    cmap.set_bad("white")
    cmap.set_under("white")
    norm = PowerNorm(gamma=0.65, vmin=0, vmax=1)
    fig, axis = plt.subplots(figsize=(13.8, 6.0), constrained_layout=True)
    radius, lateral = first["radius"], first["lateral"]
    gap = 0.6
    total_width = len(RUNS) * radius + (len(RUNS) - 1) * gap
    ticks, labels = [], []
    for index, (distribution, (_, label)) in enumerate(zip(distributions, RUNS)):
        offset = index * (radius + gap)
        # Preserve the original mapping, including saturation above its vmax.
        vmax = distribution["metadata"]["color_scale_vmax_percent"]
        mesh = axis.imshow(
            (distribution["heat"] / vmax).T, origin="lower",
            extent=(offset, offset + radius, -lateral, lateral),
            cmap=cmap, norm=norm, interpolation="bicubic", rasterized=True,
            zorder=1,
        )
        angle = distribution["half_angle"]
        ray_x, ray_y = radius * math.cos(angle), radius * math.sin(angle)
        for sign in (-1, 1):
            axis.plot([offset, offset + ray_x], [0, sign * ray_y],
                      color="#202020", linewidth=1.1)
        theta = np.linspace(-angle, angle, 240)
        axis.plot(offset + radius * np.cos(theta), radius * np.sin(theta),
                  color="#202020", linewidth=1.0)
        axis.scatter([offset], [0], marker="^", s=34, color="#087e8b",
                     zorder=5, clip_on=False)
        axis.plot([offset, offset + radius], [0, 0], color="gray",
                  alpha=0.5, linewidth=0.7, linestyle="--")
        for x in np.arange(0, radius + 0.01, 1):
            ticks.append(offset + x)
            labels.append(f"{x:g}")
            axis.plot([offset + x, offset + x], [-lateral, lateral],
                      color="#777777", alpha=0.18, linewidth=0.5)
        for y in range(math.ceil(-lateral), math.floor(lateral) + 1):
            axis.plot([offset, offset + radius], [y, y], color="#777777",
                      alpha=0.18, linewidth=0.5)
        if index:
            axis.axvline(offset - gap / 2, color="#bbbbbb", linewidth=0.7)
        count = distribution["metadata"]["selected_samples"]
        axis.text(offset + radius / 2, 1.025, f"{label}\nN = {count:,}",
                  transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=15, fontweight="bold", linespacing=1.5)
    axis.set(xlim=(0, total_width), ylim=(-lateral, lateral),
             xlabel="Relative forward position within each experiment (m)",
             ylabel="Relative left position (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(ticks, labels)
    axis.tick_params(direction="out", length=3)
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.02, fraction=0.025, aspect=27)
    colorbar.set_label("Normalized density (per experiment)")
    colorbar.set_ticks(np.linspace(0, 1, 6))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "layout": "One plotting axis; experiments translated left to right",
        "x_coordinates": "Repeated local forward distance in each segment",
        "normalization": "Each experiment sums to 100 percent independently",
        "color_scale": {
            "mode": "Shared normalized density scale preserving original colors",
            "vmin": 0,
            "vmax": 1,
            "normalization": "Density divided by each original color scale maximum; values above 1 saturate",
            "reference_vmax_percent_by_experiment": [
                d["metadata"]["color_scale_vmax_percent"] for d in distributions
            ],
            "gamma": 0.65,
        },
        "segment_gap_m": gap,
        "experiments": [dict(run=d["run"], **d["metadata"]) for d in distributions],
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Verified original counts and density peaks for all {len(distributions)} runs.")
    print(f"PNG: {OUTPUT.with_suffix('.png')}")
    print(f"PDF: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
