#!/usr/bin/env python3
"""Aggregate traditional-tracker reports without neural branch statistics."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Optional


_base = Path(__file__).resolve().parent
_results_path = _base / "neural_tracker_results.py"
if not _results_path.exists():
    _results_path = _base / "neural_tracker_results"
_loader = SourceFileLoader("experiment_results_core", str(_results_path))
_spec = importlib.util.spec_from_loader("experiment_results_core", _loader)
_core = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_core)


def traditionalize(result: dict[str, Any]) -> None:
    result["tracker_family"] = "traditional"
    result.get("aggregation", {}).pop("branch_share", None)
    result.pop("planning_time_by_branch_ms", None)
    result.pop("planning_time_algorithm_excluding_hold_ms", None)
    result.pop("branch_usage", None)
    result.get("across_run", {}).pop("algorithm_planning_mean_ms", None)
    branch_fields = {
        "algorithm_planning_time_mean_ms",
        "algorithm_planning_time_p95_ms",
        "neural_planning_time_mean_ms",
        "traditional_planning_time_mean_ms",
        "stationary_hold_time_mean_ms",
        "neural_cycles",
        "traditional_cycles",
        "stationary_hold_cycles",
        "neural_share_percent",
        "traditional_share_percent",
    }
    for run in result.get("per_run", []):
        for name in branch_fields:
            run.pop(name, None)


def markdown_report(result: dict[str, Any]) -> str:
    control = result["control"]
    visibility = result["visibility"]
    timing = result["planning_time_ms"]
    success = result["success_rate"]
    single_run = result.get("calculation_scope") == "single_run"
    lines = [
        "# traditional_tracker %s评测结果"
        % ("单轮次" if single_run else "汇总"),
        "",
        "- 计算范围：%s"
        % ("单轮次（仅当前运行目录）" if single_run else "多轮次汇总"),
        "- 有效评测文件：%d" % result["run_count"],
        "- RMS jerk E_v：%s m/s^3"
        % _core.fmt(control["rms_jerk_ev_mps3"]),
        "- 遮挡率：%s（%d/%d 周期）"
        % (
            _core.fmt(visibility["occlusion_rate_percent"], 2, "%"),
            visibility["N_occ"],
            visibility["N_all"],
        ),
        "- 目标丢失/找回：%d/%d 次"
        % (visibility["target_loss_count"], visibility["target_recovery_count"]),
        "- 控制周期平均/P95耗时：%s / %s ms"
        % (_core.fmt(timing["mean"], 3), _core.fmt(timing["p95"], 3)),
        "- 成功率：%s（成功 %d，失败 %d，未完成 %d，总启动 %d）"
        % (
            _core.fmt(success["success_rate_percent"], 2, "%"),
            success["successful"],
            success["failed"],
            success["incomplete"],
            success["total_started"],
        ),
        "- 成功率 Wilson 95%% 置信区间：[%s, %s]"
        % (
            _core.fmt(success["wilson_95_percent"]["low_percent"], 2, "%"),
            _core.fmt(success["wilson_95_percent"]["high_percent"], 2, "%"),
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
    lines.extend([
        "",
        "## 各次运行",
        "",
        "| 运行 | 模式 | RMS jerk E_v(m/s^3) | 遮挡率 | 丢失/找回 | 控制周期均值(ms) | 成功率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for run in result["per_run"]:
        lines.append(
            "| %s | %s | %s | %s | %d/%d | %s | %s |"
            % (
                run["run"],
                run["trial_mode"],
                _core.fmt(run["rms_jerk_ev_mps3"]),
                _core.fmt(run["occlusion_rate_percent"], 2, "%"),
                run["target_loss_count"],
                run["target_recovery_count"],
                _core.fmt(run["planning_time_mean_ms"], 3),
                _core.fmt(run["success_rate_percent"], 2, "%"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def csv_row(result: dict[str, Any], run: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    provenance = result.get("provenance", {})
    map_info = provenance.get("map") or {}
    row = {
        "run": "ALL",
        "source_file": "",
        "schema_version": result["schema_version"],
        "trial_mode": result.get("configuration", {}).get("trial_mode", "unknown"),
        "algorithm_label": provenance.get("algorithm_label"),
        "map_sha256": map_info.get("sha256"),
        "rms_jerk_ev_mps3": result["control"]["rms_jerk_ev_mps3"],
        "N_all": result["visibility"]["N_all"],
        "N_occ": result["visibility"]["N_occ"],
        "occlusion_rate_percent": result["visibility"]["occlusion_rate_percent"],
        "target_loss_count": result["visibility"]["target_loss_count"],
        "target_recovery_count": result["visibility"]["target_recovery_count"],
        "planning_time_mean_ms": result["planning_time_ms"]["mean"],
        "planning_time_p95_ms": result["planning_time_ms"]["p95"],
        "total_started": result["success_rate"]["total_started"],
        "successful": result["success_rate"]["successful"],
        "failed": result["success_rate"]["failed"],
        "incomplete": result["success_rate"]["incomplete"],
        "truth_fallback_seen": any(
            bool(item.get("truth_fallback_seen", False))
            for item in result["per_run"]
        ),
        "success_rate_percent": result["success_rate"]["success_rate_percent"],
    }
    if run is not None:
        for name in tuple(row):
            if name in run:
                row[name] = run[name]
    return row


def write_csv(path: Path, result: dict[str, Any]) -> None:
    aggregate_row = csv_row(result)
    if result.get("calculation_scope") == "single_run":
        rows = [csv_row(result, result["per_run"][0])]
    else:
        rows = [aggregate_row] + [csv_row(result, run) for run in result["per_run"]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_row))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = _core.build_parser()
    parser.description = "Parse and aggregate traditional_tracker evaluation results"
    for action in parser._actions:
        if action.dest == "input":
            action.default = Path("experiments_tradition")
            action.help = (
                "metrics JSON file or traditional experiment directory "
                "(default: ./experiments_tradition)"
            )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print("错误：输入不存在：%s" % input_path, file=sys.stderr)
        return 2
    reports = _core.discover_reports(
        input_path, args.include_partial, args.include_legacy, args.include_invalid
    )
    if not reports:
        print("错误：没有找到有效的 traditional 最终评测 JSON：%s" % input_path,
              file=sys.stderr)
        return 1
    schemas = sorted({int(report["data"].get("schema_version", 1)) for report in reports})
    if len(schemas) > 1 and not args.allow_mixed_schema:
        print("错误：发现混合评测 schema %s" % schemas, file=sys.stderr)
        return 2
    methodologies = {_core.methodology_signature(report) for report in reports}
    if len(methodologies) > 1 and not args.allow_mixed_configuration:
        print("错误：发现不同算法、地图或评测阈值；请分别汇总" , file=sys.stderr)
        return 2
    if any(
        report["data"].get("provenance", {}).get("algorithm_label")
        != "traditional_tracker"
        for report in reports
    ):
        print("错误：输入中包含非 traditional_tracker 数据", file=sys.stderr)
        return 2
    try:
        result = _core.aggregate(reports)
    except ValueError as error:
        print("错误：无法计算控制指标：%s" % error, file=sys.stderr)
        return 2
    single_run = input_path.is_file() or (
        input_path.is_dir() and (input_path / "metrics.json").is_file()
    )
    result["calculation_scope"] = "single_run" if single_run else "aggregate"
    traditionalize(result)
    prefix = _core.output_prefix(input_path, args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
