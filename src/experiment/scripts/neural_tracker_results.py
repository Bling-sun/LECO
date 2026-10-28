#!/usr/bin/env python3
"""Aggregate final neural_tracker metric reports from an experiment folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


BRANCH_NAMES = {0: "neural", 1: "traditional", 2: "stationary_hold"}


def percentile(values: list[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


def safe_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def positive_number(value: Any, name: str) -> float:
    number = safe_number(value)
    if number is None or number <= 0.0:
        raise ValueError("%s must be a positive finite number" % name)
    return number


def read_control_samples(report_path: Path) -> Optional[list[tuple[float, float]]]:
    """Read one run's final linear/angular velocity commands."""
    path = report_path.with_name(report_path.stem + "_commands.csv")
    if not path.is_file():
        print("警告：缺少控制指令文件：%s" % path, file=sys.stderr)
        return None
    samples: list[tuple[float, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            expected_sample = 1
            for row in csv.DictReader(stream):
                sample = int(row["sample"])
                if sample != expected_sample:
                    raise ValueError(
                        "non-contiguous sample index %d (expected %d)"
                        % (sample, expected_sample)
                    )
                v = float(row["v_mps"])
                omega = float(row["omega_radps"])
                if not math.isfinite(v) or not math.isfinite(omega):
                    raise ValueError("non-finite control command")
                samples.append((v, omega))
                expected_sample += 1
    except (KeyError, OSError, ValueError) as error:
        print("警告：无法解析 %s：%s" % (path, error), file=sys.stderr)
        return None
    return samples


def control_quality_for_report(report: dict[str, Any]) -> dict[str, Any]:
    """Compute the linear-velocity RMS jerk energy for one run."""
    data = report["data"]
    config = data.get("configuration", {})
    dt = positive_number(config.get("control_dt_s"), "control_dt_s")
    commands = read_control_samples(report["path"])
    if commands is None:
        raise ValueError("raw command CSV is required for corrected control metrics")
    expected = int(data.get("control", {}).get("samples_K", 0))
    if len(commands) != expected:
        raise ValueError(
            "command sample count %d differs from report K=%d"
            % (len(commands), expected)
        )

    trials = data.get("trials", [])
    trial_sample_counts = [int(trial.get("command_samples", 0)) for trial in trials]
    if trial_sample_counts and sum(trial_sample_counts) != len(commands):
        raise ValueError(
            "sum of trial command samples %d differs from report K=%d"
            % (sum(trial_sample_counts), len(commands))
        )
    if not trial_sample_counts:
        trial_sample_counts = [len(commands)]

    jerk_linear_squared_sum = 0.0
    energy_samples = 0
    offset = 0
    for count in trial_sample_counts:
        segment = commands[offset:offset + count]
        offset += count
        energy_samples += max(0, count - 2)
        for older, previous, current in zip(segment, segment[1:], segment[2:]):
            linear_jerk = (
                current[0] - 2.0 * previous[0] + older[0]
            ) / (dt * dt)
            jerk_linear_squared_sum += linear_jerk * linear_jerk
    return {
        "samples_K": len(commands),
        "energy_samples": energy_samples,
        "energy_linear_squared_sum": jerk_linear_squared_sum,
        "rms_jerk_ev_mps3": (
            math.sqrt(jerk_linear_squared_sum / energy_samples)
            if energy_samples else None
        ),
    }


def corrected_control_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool sufficient statistics without differencing across run boundaries."""
    per_run = []
    for report in reports:
        metrics = control_quality_for_report(report)
        report["corrected_control"] = metrics
        per_run.append(metrics)

    energy_samples = sum(item["energy_samples"] for item in per_run)
    linear_squared_sum = sum(
        item["energy_linear_squared_sum"] for item in per_run
    )
    return {
        "samples_K_total": sum(item["samples_K"] for item in per_run),
        "energy_transition_samples": energy_samples,
        "rms_jerk_ev_mps3": (
            math.sqrt(linear_squared_sum / energy_samples)
            if energy_samples else None
        ),
        "definitions": {
            "rms_jerk_ev": (
                "sqrt(sum(((v_(k+1)-2*v_k+v_(k-1))/dt^2)^2)/(K-2))"
            ),
            "unit": "m/s^3",
        },
    }


def weighted_metric(
    reports: list[dict[str, Any]], key: str, denominator_offset: int,
    count_key: Optional[str] = None,
) -> tuple[Optional[float], int]:
    numerator = 0.0
    denominator = 0
    for report in reports:
        control = report["data"].get("control", {})
        samples = int(control.get("samples_K", 0))
        weight = (
            int(control[count_key])
            if count_key and count_key in control
            else max(0, samples - denominator_offset)
        )
        value = safe_number(control.get(key))
        if value is not None and weight > 0:
            numerator += value * weight
            denominator += weight
    return (numerator / denominator if denominator else None, denominator)


def weighted_component(
    reports: list[dict[str, Any]], group: str, key: str, denominator_offset: int,
    count_key: Optional[str] = None,
) -> Optional[float]:
    numerator = 0.0
    denominator = 0
    for report in reports:
        control = report["data"].get("control", {})
        samples = int(control.get("samples_K", 0))
        weight = (
            int(control[count_key])
            if count_key and count_key in control
            else max(0, samples - denominator_offset)
        )
        value = safe_number(control.get(group, {}).get(key))
        if value is not None and weight > 0:
            numerator += value * weight
            denominator += weight
    return numerator / denominator if denominator else None


def read_planning_samples(report_path: Path) -> Optional[list[float]]:
    path = report_path.with_name(report_path.stem + "_planning_time.csv")
    if not path.is_file():
        return None
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                value = float(row["planning_time_ms"])
                if math.isfinite(value) and value >= 0.0:
                    values.append(value)
    except (KeyError, OSError, ValueError) as error:
        print("警告：无法解析 %s：%s" % (path, error), file=sys.stderr)
        return None
    return values


def pooled_planning_stats(reports: list[dict[str, Any]]) -> dict[str, Any]:
    raw_by_run = [read_planning_samples(report["path"]) for report in reports]
    if all(values is not None for values in raw_by_run):
        combined = [value for values in raw_by_run for value in (values or [])]
        result = distribution(combined)
        result["source"] = "raw_planning_time_csv"
        return result

    count = 0
    total = 0.0
    second_moment_total = 0.0
    minima: list[float] = []
    maxima: list[float] = []
    for report in reports:
        timing = report["data"].get("planning_time_ms", {})
        samples = int(timing.get("samples", 0))
        mean = safe_number(timing.get("mean"))
        std = safe_number(timing.get("std"))
        if samples <= 0 or mean is None:
            continue
        count += samples
        total += samples * mean
        second_moment_total += samples * (mean * mean + (std or 0.0) ** 2)
        minimum = safe_number(timing.get("min"))
        maximum = safe_number(timing.get("max"))
        if minimum is not None:
            minima.append(minimum)
        if maximum is not None:
            maxima.append(maximum)

    mean = total / count if count else None
    variance = (
        max(0.0, second_moment_total / count - mean * mean)
        if count and mean is not None
        else None
    )
    single_timing = reports[0]["data"].get("planning_time_ms", {}) if len(reports) == 1 else {}
    return {
        "samples": count,
        "mean": mean,
        "std": math.sqrt(variance) if variance is not None else None,
        "min": min(minima) if minima else None,
        "p50": safe_number(single_timing.get("p50")),
        "p95": safe_number(single_timing.get("p95")),
        "max": max(maxima) if maxima else None,
        "source": "pooled_json_summary",
        "note": (
            None
            if len(reports) == 1
            else "Aggregate P50/P95 require every run's *_planning_time.csv"
        ),
    }


def read_branch_planning_samples(
    report_path: Path,
) -> Optional[dict[str, list[float]]]:
    planning_path = report_path.with_name(report_path.stem + "_planning_time.csv")
    branch_path = report_path.with_name(report_path.stem + "_branches.csv")
    if not planning_path.is_file() or not branch_path.is_file():
        return None
    try:
        with planning_path.open("r", encoding="utf-8", newline="") as stream:
            planning = list(csv.DictReader(stream))
        with branch_path.open("r", encoding="utf-8", newline="") as stream:
            branches = list(csv.DictReader(stream))
        if len(planning) != len(branches):
            raise ValueError("planning/branch sample counts differ")
        grouped: dict[str, list[float]] = {}
        for timing_row, branch_row in zip(planning, branches):
            if timing_row.get("sample") != branch_row.get("sample"):
                raise ValueError("planning/branch sample indices differ")
            value = float(timing_row["planning_time_ms"])
            if math.isfinite(value) and value >= 0.0:
                grouped.setdefault(branch_row["branch"], []).append(value)
        return grouped
    except (KeyError, OSError, ValueError) as error:
        print(
            "警告：无法按分支解析 %s：%s" % (report_path, error),
            file=sys.stderr,
        )
        return None


def pooled_branch_planning_stats(
    reports: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = [read_branch_planning_samples(report["path"]) for report in reports]
    if all(item is not None for item in raw):
        grouped: dict[str, list[float]] = {}
        for item in raw:
            for name, values in (item or {}).items():
                grouped.setdefault(name, []).extend(values)
        by_branch = {
            name: distribution(grouped.get(name, []))
            for name in sorted(set(BRANCH_NAMES.values()) | set(grouped))
        }
        algorithm_values = grouped.get("neural", []) + grouped.get("traditional", [])
        algorithm = distribution(algorithm_values)
        algorithm["source"] = "aligned_raw_planning_and_branch_csv"
        return by_branch, algorithm

    def pool_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
        count = 0
        total = 0.0
        second_moment_total = 0.0
        minima: list[float] = []
        maxima: list[float] = []
        for summary in summaries:
            samples = int(summary.get("samples", 0))
            mean = safe_number(summary.get("mean"))
            std = safe_number(summary.get("std"))
            if samples <= 0 or mean is None:
                continue
            count += samples
            total += samples * mean
            second_moment_total += samples * (mean * mean + (std or 0.0) ** 2)
            minimum = safe_number(summary.get("min"))
            maximum = safe_number(summary.get("max"))
            if minimum is not None:
                minima.append(minimum)
            if maximum is not None:
                maxima.append(maximum)
        mean = total / count if count else None
        variance = (
            max(0.0, second_moment_total / count - mean * mean)
            if count and mean is not None
            else None
        )
        single = summaries[0] if len(summaries) == 1 else {}
        return {
            "samples": count,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "min": min(minima) if minima else None,
            "p50": safe_number(single.get("p50")),
            "p95": safe_number(single.get("p95")),
            "max": max(maxima) if maxima else None,
            "source": "pooled_json_summary",
            "note": (
                None if len(summaries) == 1 else
                "Aggregate P50/P95 require aligned raw timing/branch CSV files"
            ),
        }

    by_branch: dict[str, dict[str, Any]] = {}
    for name in BRANCH_NAMES.values():
        summaries = [
            report["data"].get("planning_time_by_branch_ms", {}).get(name, {})
            for report in reports
        ]
        by_branch[name] = pool_summaries(summaries)
    algorithm = pool_summaries(
        [
            report["data"].get(
                "planning_time_algorithm_excluding_hold_ms", {}
            )
            for report in reports
        ]
    )
    return by_branch, algorithm


def across_run_distribution(
    reports: list[dict[str, Any]], *keys: str
) -> dict[str, Any]:
    values: list[float] = []
    for report in reports:
        value: Any = report["data"]
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else None
        number = safe_number(value)
        if number is not None:
            values.append(number)
    return distribution(values)


def wilson_interval(successful: int, completed: int) -> dict[str, Optional[float]]:
    if completed <= 0:
        return {"low_percent": None, "high_percent": None}
    z = 1.959963984540054
    p = successful / completed
    denominator = 1.0 + z * z / completed
    center = (p + z * z / (2.0 * completed)) / denominator
    margin = z * math.sqrt(
        p * (1.0 - p) / completed + z * z / (4.0 * completed * completed)
    ) / denominator
    return {
        "low_percent": 100.0 * (center - margin),
        "high_percent": 100.0 * (center + margin),
    }


def branch_summary(counts: Counter[str]) -> dict[str, Any]:
    neural = counts.get("neural", 0)
    traditional = counts.get("traditional", 0)
    hold = counts.get("stationary_hold", 0)
    total = sum(counts.values())
    algorithm_total = neural + traditional
    return {
        "total_control_cycles": total,
        "counts": dict(sorted(counts.items())),
        "algorithm_cycles_excluding_hold": algorithm_total,
        "neural_share_percent_excluding_hold": (
            100.0 * neural / algorithm_total if algorithm_total else None
        ),
        "traditional_share_percent_excluding_hold": (
            100.0 * traditional / algorithm_total if algorithm_total else None
        ),
        "neural_share_percent_of_all_cycles": (
            100.0 * neural / total if total else None
        ),
        "traditional_share_percent_of_all_cycles": (
            100.0 * traditional / total if total else None
        ),
        "stationary_hold_share_percent_of_all_cycles": (
            100.0 * hold / total if total else None
        ),
    }


def report_validation_errors(data: dict[str, Any]) -> list[str]:
    errors = []
    control = data.get("control", {})
    visibility = data.get("visibility", {})
    planning = data.get("planning_time_ms", {})
    branch = data.get("branch_usage", {})
    sampling = data.get("sampling", {})
    target_fov = data.get("target_fov", {})
    validation = data.get("validation", {})
    k = int(control.get("samples_K", 0))
    counts = (
        k,
        int(visibility.get("N_all", 0)),
        int(planning.get("samples", 0)),
        int(branch.get("total_control_cycles", 0)),
    )
    if k <= 0:
        errors.append("K=0")
    if len(set(counts)) != 1:
        errors.append("K/visibility/planning/branch sample counts differ")
    for field in (
        "active_missed_sequence_samples",
        "active_duplicate_sequence_samples",
        "active_out_of_order_sequence_samples",
    ):
        if int(sampling.get(field, 0)) != 0:
            errors.append("%s != 0" % field)
    fov_valid = int(target_fov.get("samples", 0))
    fov_missing = int(target_fov.get("missing_or_stale_pose_samples", 0))
    if fov_valid + fov_missing != k:
        errors.append("FOV valid+missing samples != K")
    max_missing = float(
        data.get("configuration", {}).get("max_fov_missing_rate", 0.01)
    )
    if (fov_missing / k if k > 0 else 1.0) > max_missing:
        errors.append("FOV missing rate exceeds threshold")
    required = validation.get("required_inputs_received", {})
    if not validation:
        errors.append("metrics validation block is missing")
    elif not validation.get("startup_ready", False):
        errors.append("metrics startup readiness was not established")
    if validation.get("runtime_input_failure"):
        errors.append("required input publisher was lost during the active route")
    if set(required) != {"metrics", "odom", "target_truth", "detection", "map"}:
        errors.append("required input readiness fields are incomplete")
    elif not all(required.values()):
        errors.append("required input was not received")
    if validation and not validation.get("valid", False):
        errors.append("metrics validation.valid is false")
    trials = data.get("trials", [])
    trial_mode = data.get("configuration", {}).get("trial_mode", "route")
    if not trials:
        errors.append("no trial")
    expected_goal_trials = 0
    for trial in trials:
        if trial.get("status") in {"incomplete", "invalid"}:
            errors.append("trial status is %s" % trial.get("status"))
        metadata = trial.get("route_metadata", {})
        expected = int(metadata.get("goal_count", 0))
        expected_goal_trials = max(expected_goal_trials, expected)
        if expected <= 0:
            errors.append("route goal_count metadata is missing")
        elif trial_mode == "route" and (
            trial.get("status") == "success"
            and int(trial.get("goal_count", 0)) != expected
        ):
            errors.append("trial goal_count differs from route goal_count")
        elif trial_mode == "per-goal" and int(trial.get("goal_count", 0)) != 1:
            errors.append("per-goal trial goal_count must be 1")
    if trial_mode == "per-goal" and (
        expected_goal_trials <= 0 or len(trials) != expected_goal_trials
    ):
        errors.append(
            "per-goal trial count %d differs from expected %d"
            % (len(trials), expected_goal_trials)
        )
    return errors


def discover_reports(
    path: Path,
    include_partial: bool,
    include_legacy: bool = False,
    include_invalid: bool = False,
) -> list[dict[str, Any]]:
    if path.is_file():
        candidates = [path]
    elif (path / "metrics.json").is_file():
        # A run directory is an explicit single-run input.  Do not recursively
        # pick up summaries or reports from any nested diagnostic directory.
        candidates = [path / "metrics.json"]
    else:
        candidates = sorted(path.glob("**/*.json"))
    reports: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            print("警告：跳过无法读取的 JSON %s：%s" % (candidate, error), file=sys.stderr)
            continue
        if not isinstance(data, dict) or not all(
            key in data
            for key in ("control", "visibility", "planning_time_ms", "success_rate")
        ):
            continue
        # Do not recursively treat this script's previous aggregate as a raw
        # run when the same directory is summarized again.
        if "source_reports" in data and "run_count" in data:
            continue
        schema = int(data.get("schema_version", 1))
        if schema < 4 and not include_legacy:
            print("跳过旧版 schema %d 评测：%s" % (schema, candidate), file=sys.stderr)
            continue
        if not include_partial and not data.get("final", False):
            print("跳过未结束的评测：%s" % candidate, file=sys.stderr)
            continue
        validation_errors = report_validation_errors(data) if schema >= 4 else []
        if validation_errors and not include_invalid:
            print(
                "跳过无效评测 %s：%s"
                % (candidate, "; ".join(validation_errors)),
                file=sys.stderr,
            )
            continue
        reports.append({"path": candidate.resolve(), "data": data})
    return reports


def methodology_signature(report: dict[str, Any]) -> tuple[Any, ...]:
    data = report["data"]
    config = data.get("configuration", {})
    provenance = data.get("provenance", {})
    map_info = provenance.get("map") or {}
    policy_info = provenance.get("policy") or {}
    replay_methods = tuple(sorted({
        (
            trial.get("route_metadata", {}).get("timing"),
            trial.get("route_metadata", {}).get("goal_tolerance_m"),
            trial.get("route_metadata", {}).get("waypoint_timeout_s"),
            trial.get("route_metadata", {}).get("continuous_waypoint_advance"),
        )
        for trial in data.get("trials", [])
    }, key=repr))
    return (
        config.get("trial_mode"),
        config.get("control_dt_s"),
        config.get("goal_tolerance_m"),
        config.get("route_timeout_s"),
        config.get("lost_timeout_s"),
        config.get("max_tracking_distance_m"),
        config.get("distance_timeout_s"),
        config.get("linear_control_scale_mps"),
        config.get("angular_control_scale_radps"),
        config.get("sensor_fov_deg"),
        config.get("sensor_range_m"),
        config.get("pose_max_age_s"),
        config.get("max_fov_missing_rate"),
        provenance.get("algorithm_label"),
        map_info.get("sha256"),
        policy_info.get("sha256"),
        replay_methods,
    )


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    control_quality = corrected_control_metrics(reports)

    n_all = 0
    n_occ = 0
    loss_count = 0
    recovery_count = 0
    branch_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    trials: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for report in reports:
        data = report["data"]
        visibility = data.get("visibility", {})
        n_all += int(visibility.get("N_all", 0))
        n_occ += int(visibility.get("N_occ", 0))
        loss_count += int(visibility.get("target_loss_count", 0))
        recovery_count += int(visibility.get("target_recovery_count", 0))

        branch_counts.update(data.get("branch_usage", {}).get("counts", {}))
        success = data.get("success_rate", {})
        for key in ("total_started", "completed", "successful", "failed", "incomplete"):
            success_counts[key] += int(success.get(key, 0))
        failure_reasons.update(success.get("failure_reasons", {}))

        run_trials = data.get("trials", [])
        for trial in run_trials:
            copied = dict(trial)
            copied["source_run"] = (
                data.get("run_label") or report["path"].parent.name
            )
            trials.append(copied)

        per_run.append(run_row(report))

    completed = success_counts["completed"]
    planning = pooled_planning_stats(reports)
    planning_by_branch, planning_algorithm = pooled_branch_planning_stats(reports)
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_reports": [str(report["path"]) for report in reports],
        "run_count": len(reports),
        "configuration": reports[0]["data"].get("configuration", {}),
        "provenance": reports[0]["data"].get("provenance", {}),
        "aggregation": {
            "rms_jerk_ev": "pooled RMS linear jerk from raw command CSV files",
            "occlusion_rate": "100*sum(N_occ)/sum(N_all)",
            "branch_share": "recomputed from summed branch cycle counts",
            "measurement_window": "active route only for current schema reports",
        },
        "control": control_quality,
        "visibility": {
            "N_all": n_all,
            "N_occ": n_occ,
            "occlusion_rate_percent": 100.0 * n_occ / n_all if n_all else None,
            "target_loss_count": loss_count,
            "target_recovery_count": recovery_count,
            "unrecovered_loss_count": max(0, loss_count - recovery_count),
        },
        "planning_time_ms": planning,
        "planning_time_by_branch_ms": planning_by_branch,
        "planning_time_algorithm_excluding_hold_ms": planning_algorithm,
        "branch_usage": branch_summary(branch_counts),
        "success_rate": {
            "total_started": success_counts["total_started"],
            "completed": completed,
            "successful": success_counts["successful"],
            "failed": success_counts["failed"],
            "incomplete": success_counts["incomplete"],
            "success_rate_percent": (
                100.0 * success_counts["successful"] / completed
                if completed
                else None
            ),
            "wilson_95_percent": wilson_interval(
                success_counts["successful"], completed
            ),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "across_run": {
            "rms_jerk_ev_mps3": distribution([
                report["corrected_control"]["rms_jerk_ev_mps3"]
                for report in reports
                if report["corrected_control"]["rms_jerk_ev_mps3"] is not None
            ]),
            "occlusion_rate_percent": across_run_distribution(
                reports, "visibility", "occlusion_rate_percent"
            ),
            "algorithm_planning_mean_ms": across_run_distribution(
                reports, "planning_time_algorithm_excluding_hold_ms", "mean"
            ),
            "success_rate_percent": across_run_distribution(
                reports, "success_rate", "success_rate_percent"
            ),
        },
        "per_run": per_run,
        "trials": trials,
    }


def run_row(report: dict[str, Any]) -> dict[str, Any]:
    data = report["data"]
    control = report["corrected_control"]
    visibility = data.get("visibility", {})
    planning = data.get("planning_time_ms", {})
    algorithm_planning = data.get(
        "planning_time_algorithm_excluding_hold_ms", {}
    )
    planning_by_branch = data.get("planning_time_by_branch_ms", {})
    branch = data.get("branch_usage", {})
    success = data.get("success_rate", {})
    counts = branch.get("counts", {})
    provenance = data.get("provenance", {})
    map_info = provenance.get("map") or {}
    policy_info = provenance.get("policy") or {}
    return {
        "run": data.get("run_label") or report["path"].parent.name,
        "source_file": report["path"].name,
        "schema_version": data.get("schema_version", 1),
        "trial_mode": data.get("configuration", {}).get("trial_mode", "unknown"),
        "algorithm_label": provenance.get("algorithm_label"),
        "map_sha256": map_info.get("sha256"),
        "policy_sha256": policy_info.get("sha256"),
        "rms_jerk_ev_mps3": control.get("rms_jerk_ev_mps3"),
        "N_all": visibility.get("N_all", 0),
        "N_occ": visibility.get("N_occ", 0),
        "occlusion_rate_percent": visibility.get("occlusion_rate_percent"),
        "target_loss_count": visibility.get("target_loss_count", 0),
        "target_recovery_count": visibility.get("target_recovery_count", 0),
        "planning_time_mean_ms": planning.get("mean"),
        "planning_time_p95_ms": planning.get("p95"),
        "algorithm_planning_time_mean_ms": algorithm_planning.get("mean"),
        "algorithm_planning_time_p95_ms": algorithm_planning.get("p95"),
        "neural_planning_time_mean_ms": planning_by_branch.get(
            "neural", {}
        ).get("mean"),
        "traditional_planning_time_mean_ms": planning_by_branch.get(
            "traditional", {}
        ).get("mean"),
        "stationary_hold_time_mean_ms": planning_by_branch.get(
            "stationary_hold", {}
        ).get("mean"),
        "neural_cycles": counts.get("neural", 0),
        "traditional_cycles": counts.get("traditional", 0),
        "stationary_hold_cycles": counts.get("stationary_hold", 0),
        "neural_share_percent": branch.get("neural_share_percent_excluding_hold"),
        "traditional_share_percent": branch.get(
            "traditional_share_percent_excluding_hold"
        ),
        "total_started": success.get("total_started", 0),
        "successful": success.get("successful", 0),
        "failed": success.get("failed", 0),
        "incomplete": success.get("incomplete", 0),
        "truth_fallback_seen": data.get("observation", {}).get(
            "truth_fallback_seen", False
        ),
        "success_rate_percent": success.get("success_rate_percent"),
    }


def fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    number = safe_number(value)
    return "N/A" if number is None else (f"{number:.{digits}f}{suffix}")


def markdown_report(result: dict[str, Any]) -> str:
    control = result["control"]
    visibility = result["visibility"]
    timing = result["planning_time_ms"]
    algorithm_timing = result["planning_time_algorithm_excluding_hold_ms"]
    timing_by_branch = result["planning_time_by_branch_ms"]
    branch = result["branch_usage"]
    success = result["success_rate"]
    single_run = result.get("calculation_scope") == "single_run"
    lines = [
        (
            "# neural_tracker 单轮次评测结果"
            if single_run else "# neural_tracker 汇总评测结果"
        ),
        "",
        "- 计算范围：%s"
        % ("单轮次（仅当前运行目录）" if single_run else "多轮次汇总"),
        "- 有效评测文件：%d" % result["run_count"],
        "- RMS jerk E_v：%s m/s^3"
        % fmt(control["rms_jerk_ev_mps3"]),
        "- 遮挡率：%s（%d/%d 周期）"
        % (
            fmt(visibility["occlusion_rate_percent"], 2, "%"),
            visibility["N_occ"],
            visibility["N_all"],
        ),
        "- 目标丢失/找回：%d/%d 次"
        % (visibility["target_loss_count"], visibility["target_recovery_count"]),
        "- 全部控制周期平均/P95耗时（含静止保持）：%s / %s ms"
        % (fmt(timing["mean"], 3), fmt(timing["p95"], 3)),
        "- 实际规划平均/P95耗时（神经+传统，不含静止保持）：%s / %s ms"
        % (
            fmt(algorithm_timing["mean"], 3),
            fmt(algorithm_timing["p95"], 3),
        ),
        "- 神经/传统/静止保持周期平均耗时：%s / %s / %s ms"
        % (
            fmt(timing_by_branch.get("neural", {}).get("mean"), 3),
            fmt(timing_by_branch.get("traditional", {}).get("mean"), 3),
            fmt(timing_by_branch.get("stationary_hold", {}).get("mean"), 3),
        ),
        "- 神经/传统分支占比（不含静止保持）：%s / %s"
        % (
            fmt(branch["neural_share_percent_excluding_hold"], 2, "%"),
            fmt(branch["traditional_share_percent_excluding_hold"], 2, "%"),
        ),
        "- 成功率：%s（成功 %d，失败 %d，未完成 %d，总启动 %d）"
        % (
            fmt(success["success_rate_percent"], 2, "%"),
            success["successful"],
            success["failed"],
            success["incomplete"],
            success["total_started"],
        ),
        "- 成功率 Wilson 95%% 置信区间：[%s, %s]"
        % (
            fmt(success["wilson_95_percent"]["low_percent"], 2, "%"),
            fmt(success["wilson_95_percent"]["high_percent"], 2, "%"),
        ),
        "",
        "## 失败原因",
        "",
    ]
    reasons = success["failure_reasons"]
    if reasons:
        lines.extend("- `%s`：%d" % item for item in sorted(reasons.items()))
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 各次运行",
            "",
            "| 运行 | 模式 | RMS jerk E_v(m/s^3) | 遮挡率 | 丢失/找回 | 实际规划均值(ms) | 神经占比 | 成功率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in result["per_run"]:
        lines.append(
            "| %s | %s | %s | %s | %d/%d | %s | %s | %s |"
            % (
                run["run"],
                run["trial_mode"],
                fmt(run["rms_jerk_ev_mps3"]),
                fmt(run["occlusion_rate_percent"], 2, "%"),
                run["target_loss_count"],
                run["target_recovery_count"],
                fmt(run["algorithm_planning_time_mean_ms"], 3),
                fmt(run["neural_share_percent"], 2, "%"),
                fmt(run["success_rate_percent"], 2, "%"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    aggregate_row = {
        "run": "ALL",
        "source_file": "",
        "schema_version": result["schema_version"],
        "trial_mode": result.get("configuration", {}).get("trial_mode", "unknown"),
        "algorithm_label": result.get("provenance", {}).get("algorithm_label"),
        "map_sha256": (result.get("provenance", {}).get("map") or {}).get(
            "sha256"
        ),
        "policy_sha256": (result.get("provenance", {}).get("policy") or {}).get(
            "sha256"
        ),
        "rms_jerk_ev_mps3": result["control"]["rms_jerk_ev_mps3"],
        "N_all": result["visibility"]["N_all"],
        "N_occ": result["visibility"]["N_occ"],
        "occlusion_rate_percent": result["visibility"]["occlusion_rate_percent"],
        "target_loss_count": result["visibility"]["target_loss_count"],
        "target_recovery_count": result["visibility"]["target_recovery_count"],
        "planning_time_mean_ms": result["planning_time_ms"]["mean"],
        "planning_time_p95_ms": result["planning_time_ms"]["p95"],
        "algorithm_planning_time_mean_ms": result[
            "planning_time_algorithm_excluding_hold_ms"
        ]["mean"],
        "algorithm_planning_time_p95_ms": result[
            "planning_time_algorithm_excluding_hold_ms"
        ]["p95"],
        "neural_planning_time_mean_ms": result[
            "planning_time_by_branch_ms"
        ].get("neural", {}).get("mean"),
        "traditional_planning_time_mean_ms": result[
            "planning_time_by_branch_ms"
        ].get("traditional", {}).get("mean"),
        "stationary_hold_time_mean_ms": result[
            "planning_time_by_branch_ms"
        ].get("stationary_hold", {}).get("mean"),
        "neural_cycles": result["branch_usage"]["counts"].get("neural", 0),
        "traditional_cycles": result["branch_usage"]["counts"].get(
            "traditional", 0
        ),
        "stationary_hold_cycles": result["branch_usage"]["counts"].get(
            "stationary_hold", 0
        ),
        "neural_share_percent": result["branch_usage"][
            "neural_share_percent_excluding_hold"
        ],
        "traditional_share_percent": result["branch_usage"][
            "traditional_share_percent_excluding_hold"
        ],
        "total_started": result["success_rate"]["total_started"],
        "successful": result["success_rate"]["successful"],
        "failed": result["success_rate"]["failed"],
        "incomplete": result["success_rate"]["incomplete"],
        "truth_fallback_seen": any(
            bool(row.get("truth_fallback_seen", False)) for row in result["per_run"]
        ),
        "success_rate_percent": result["success_rate"]["success_rate_percent"],
    }
    if result.get("calculation_scope") == "single_run":
        # One command for one run should produce exactly one CSV data row.
        aggregate_row["run"] = result["per_run"][0]["run"]
        aggregate_row["source_file"] = result["per_run"][0]["source_file"]
        rows = [aggregate_row]
    else:
        rows = [aggregate_row, *result["per_run"]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_row))
        writer.writeheader()
        writer.writerows(rows)


def output_prefix(input_path: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if input_path.is_dir():
        try:
            datetime.strptime(input_path.name, "%Y-%m-%d")
            return input_path / "summary" / "final_evaluation"
        except ValueError:
            try:
                datetime.strptime(input_path.parent.name, "%Y-%m-%d")
                if input_path.name != "summary":
                    return input_path / "final_evaluation"
            except ValueError:
                pass
            dated_directory = input_path / datetime.now().astimezone().strftime(
                "%Y-%m-%d"
            )
            return dated_directory / "summary" / "final_evaluation"
    return input_path.with_name(input_path.stem + "_summary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and aggregate final neural_tracker evaluation results"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("experiments"),
        help="metrics JSON file or experiment directory (default: ./experiments)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="output prefix; writes .json, .csv and .md",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="include reports whose final field is false",
    )
    parser.add_argument(
        "--include-legacy", action="store_true",
        help="include pre-schema-4 reports without current FOV sampling/methodology",
    )
    parser.add_argument(
        "--include-invalid", action="store_true",
        help="include schema-4 runs that fail formal data validation (diagnostics only)",
    )
    parser.add_argument(
        "--allow-mixed-schema", action="store_true",
        help="allow pooling reports produced by different metric schemas",
    )
    parser.add_argument(
        "--allow-mixed-configuration", action="store_true",
        help="allow pooling different algorithms/maps/policies/metric thresholds",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print("错误：输入不存在：%s" % input_path, file=sys.stderr)
        return 2
    reports = discover_reports(
        input_path, args.include_partial, args.include_legacy, args.include_invalid
    )
    if not reports:
        print("错误：没有找到有效的最终评测 JSON：%s" % input_path, file=sys.stderr)
        return 1
    schemas = sorted({int(report["data"].get("schema_version", 1)) for report in reports})
    if len(schemas) > 1 and not args.allow_mixed_schema:
        print(
            "错误：发现混合评测 schema %s；请按同一版本分目录汇总，或显式传入 "
            "--allow-mixed-schema" % schemas,
            file=sys.stderr,
        )
        return 2
    methodologies = {methodology_signature(report) for report in reports}
    if len(methodologies) > 1 and not args.allow_mixed_configuration:
        print(
            "错误：发现不同算法、地图、策略模型或评测阈值；请分别汇总，或仅在明确"
            "需要合并时传入 --allow-mixed-configuration",
            file=sys.stderr,
        )
        return 2

    try:
        result = aggregate(reports)
    except ValueError as error:
        print("错误：无法计算控制指标：%s" % error, file=sys.stderr)
        return 2
    single_run = input_path.is_file() or (
        input_path.is_dir() and (input_path / "metrics.json").is_file()
    )
    result["calculation_scope"] = "single_run" if single_run else "aggregate"
    prefix = output_prefix(input_path, args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_csv(csv_path, result)
    markdown = markdown_report(result)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("结果文件：")
    print("  %s" % json_path)
    print("  %s" % csv_path)
    print("  %s" % markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
