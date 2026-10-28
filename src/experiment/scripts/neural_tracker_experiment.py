#!/usr/bin/env python3
"""Run one isolated neural-tracker route experiment from clean ROS processes."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional


DEFAULT_POLICY = Path(
    "neural_tracker/models/"
    "v41_directional_recovery/policy.onnx"
)


def safe_label(value: str) -> str:
    label = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    if not label or label.lower() in {"summary", "all"}:
        raise ValueError("--run-label must be a non-empty label other than summary/all")
    return label


def resolve_file(workspace: Path, value: Path, option: str) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError("%s does not exist: %s" % (option, candidate))
    return candidate


def run_command(command: list[str], workspace: Path) -> None:
    completed = subprocess.run(command, cwd=workspace, check=False)
    if completed.returncode != 0:
        raise RuntimeError("command failed (%d): %s" % (
            completed.returncode, " ".join(command)
        ))


def ros_nodes(workspace: Path) -> list[str]:
    completed = subprocess.run(
        ["ros2", "node", "list"], cwd=workspace, check=False,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError("ros2 node list failed: %s" % completed.stderr.strip())
    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def wait_for_no_nodes(workspace: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    nodes = ros_nodes(workspace)
    while nodes and time.monotonic() < deadline:
        time.sleep(0.25)
        nodes = ros_nodes(workspace)
    if nodes:
        raise RuntimeError(
            "ROS graph is not clean after stopping the previous run: %s"
            % ", ".join(nodes)
        )


def stop_process(process: Optional[subprocess.Popen[str]]) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def open_log(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated simulator+tracker+metrics+route experiment and "
            "always stop all run-owned processes afterwards"
        )
    )
    parser.add_argument("goals", type=Path, help="goal sequence JSON")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--algorithm-label", default="neural_tracker_v41")
    parser.add_argument(
        "--direct-target",
        action="store_true",
        help="control tracker from target truth and make FOV loss observational",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--map-file", type=Path, default=Path("src/simulator/map/maze2.json"))
    parser.add_argument("--policy-file", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--route-timeout", type=float, default=900.0)
    parser.add_argument("--lost-timeout", type=float, default=3.0)
    parser.add_argument("--max-tracking-distance", type=float, default=4.0)
    parser.add_argument("--distance-timeout", type=float, default=2.0)
    parser.add_argument("--final-goal-tolerance", type=float, default=2.0)
    parser.add_argument("--waypoint-radius", type=float, default=2.0)
    parser.add_argument("--waypoint-timeout", type=float, default=120.0)
    parser.add_argument("--recovery-timeout", type=float, default=120.0)
    parser.add_argument("--sensor-fov-deg", type=float, default=87.0)
    parser.add_argument("--sensor-range", type=float, default=4.0)
    parser.add_argument("--pose-max-age", type=float, default=0.25)
    parser.add_argument("--max-fov-missing-rate", type=float, default=0.01)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        label = safe_label(args.run_label)
        workspace = args.workspace.expanduser().resolve()
        start_script = resolve_file(
            workspace, Path("start_tracking.sh"), "launcher"
        )
        goals = resolve_file(workspace, args.goals, "goals")
        map_file = resolve_file(workspace, args.map_file, "--map-file")
        policy_file = resolve_file(workspace, args.policy_file, "--policy-file")
    except ValueError as error:
        parser.error(str(error))

    if min(
        args.route_timeout,
        args.lost_timeout,
        args.max_tracking_distance,
        args.distance_timeout,
        args.final_goal_tolerance,
        args.waypoint_radius,
        args.waypoint_timeout,
        args.recovery_timeout,
        args.sensor_fov_deg,
        args.sensor_range,
        args.pose_max_age,
        args.startup_timeout,
    ) <= 0.0:
        parser.error("all distance, time, FOV and tolerance values must be positive")
    if not 0.0 <= args.max_fov_missing_rate <= 1.0:
        parser.error("--max-fov-missing-rate must be in [0, 1]")
    if not math.isclose(
        args.final_goal_tolerance,
        args.waypoint_radius,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        parser.error(
            "--final-goal-tolerance and --waypoint-radius must be identical"
        )

    experiment_root = args.experiment_root.expanduser()
    if not experiment_root.is_absolute():
        experiment_root = workspace / experiment_root
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    run_directory = experiment_root.resolve() / day / label
    metrics_path = run_directory / "metrics.json"
    if metrics_path.exists():
        parser.error(
            "%s already exists; every repetition needs a unique --run-label"
            % metrics_path
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    session = "nt_exp_" + re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:40]
    launcher = [str(start_script), "--backend", "neural", "--session", session]
    if args.direct_target:
        launcher.append("--direct-target")
    metrics_process: Optional[subprocess.Popen[str]] = None
    replay_process: Optional[subprocess.Popen[str]] = None
    metrics_log: Optional[IO[str]] = None
    replay_log: Optional[IO[str]] = None

    try:
        run_command([*launcher, "--stop"], workspace)
        wait_for_no_nodes(workspace, 15.0)
        run_command([
            *launcher,
            "--no-rviz",
            "--timing-interval", "0",
            "--truth-fallback-after", str(args.lost_timeout),
            "--map-json", str(map_file),
            "--onnx", str(policy_file),
        ], workspace)

        metrics_log = open_log(run_directory / "metrics.log")
        metrics_command = [
            "ros2", "run", "experiment", "neural_tracker_metrics",
            "--run-label", label,
            "--algorithm-label", args.algorithm_label,
            "--experiment-root", str(experiment_root),
            "--trial-mode", "per-goal",
            "--auto-stop",
            "--route-timeout", str(args.route_timeout),
            "--lost-timeout", str(args.lost_timeout),
            "--max-tracking-distance", str(args.max_tracking_distance),
            "--distance-timeout", str(args.distance_timeout),
            "--goal-tolerance", str(args.final_goal_tolerance),
            "--sensor-fov-deg", str(args.sensor_fov_deg),
            "--sensor-range", str(args.sensor_range),
            "--pose-max-age", str(args.pose_max_age),
            "--max-fov-missing-rate", str(args.max_fov_missing_rate),
            "--startup-timeout", str(args.startup_timeout),
            "--map-file", str(map_file),
            "--policy-file", str(policy_file),
        ]
        if args.direct_target:
            metrics_command.append("--direct-target")
        metrics_process = subprocess.Popen(
            metrics_command, cwd=workspace, text=True,
            stdout=metrics_log, stderr=subprocess.STDOUT,
        )

        replay_log = open_log(run_directory / "replay.log")
        replay_command = [
            "ros2", "run", "experiment", "goal_pose_sequence", "replay",
            str(goals),
            "--benchmark",
            "--timing", "reached",
            "--goal-tolerance", str(args.waypoint_radius),
            "--waypoint-timeout", str(args.waypoint_timeout),
            "--recovery-timeout", str(args.recovery_timeout),
            "--benchmark-ready-timeout", str(args.startup_timeout),
            "--start-delay", "3",
        ]
        replay_process = subprocess.Popen(
            replay_command, cwd=workspace, text=True,
            stdout=replay_log, stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + args.route_timeout + args.startup_timeout + 60.0
        while time.monotonic() < deadline:
            metrics_return = metrics_process.poll()
            replay_return = replay_process.poll()
            if metrics_return is not None and replay_return is not None:
                break
            if replay_return not in {None, 0}:
                stop_process(metrics_process)
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("experiment orchestration timeout")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("实验运行失败：%s" % error, file=sys.stderr)
    finally:
        stop_process(replay_process)
        stop_process(metrics_process)
        if replay_log is not None:
            replay_log.close()
        if metrics_log is not None:
            metrics_log.close()
        try:
            run_command([*launcher, "--stop"], workspace)
            wait_for_no_nodes(workspace, 15.0)
        except RuntimeError as error:
            print("清理警告：%s" % error, file=sys.stderr)

    if not metrics_path.is_file():
        print("没有生成 metrics.json；查看 %s" % run_directory, file=sys.stderr)
        return 2
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print("无法读取 metrics.json：%s" % error, file=sys.stderr)
        return 2
    success = report.get("success_rate", {})
    valid = bool(report.get("final")) and bool(
        report.get("validation", {}).get("valid")
    )
    print(
        "实验结束：valid=%s, 成功=%s, 失败=%s, 总试次=%s, K=%s\n结果目录：%s"
        % (
            valid,
            success.get("successful", "missing"),
            success.get("failed", "missing"),
            success.get("completed", "missing"),
            report.get("control", {}).get("samples_K", 0),
            run_directory,
        )
    )
    if not valid:
        return 2
    return 0 if int(success.get("failed", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
