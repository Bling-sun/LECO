from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_stop_matches_only_the_requested_session(tmp_path: Path) -> None:
    command_log = tmp_path / "tmux.jsonl"
    stub = tmp_path / "tmux"
    stub.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TMUX_TEST_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'list-panes': print('%123')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(SCRIPTS.parents[2] / "start_tracking.sh"),
         "--stop", "--session", "tracking_review"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.defpath,
             "TMUX_TEST_LOG": str(command_log)},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    assert ["has-session", "-t", "=tracking_review"] in commands
    assert ["kill-session", "-t", "=tracking_review"] in commands
    assert ["send-keys", "-t", "%123", "C-c"] in commands


def valid_report() -> dict:
    return {
        "final": True,
        "control": {"samples_K": 10},
        "visibility": {"N_all": 10},
        "planning_time_ms": {"samples": 10},
        "branch_usage": {"total_control_cycles": 10},
        "sampling": {
            "active_missed_sequence_samples": 0,
            "active_duplicate_sequence_samples": 0,
            "active_out_of_order_sequence_samples": 0,
        },
        "target_fov": {"samples": 10, "missing_or_stale_pose_samples": 0},
        "configuration": {"max_fov_missing_rate": 0.01},
        "validation": {
            "valid": True,
            "startup_ready": True,
            "required_inputs_received": {
                "metrics": True,
                "odom": True,
                "target_truth": True,
                "detection": True,
                "map": True,
            },
        },
        "trials": [{
            "status": "success",
            "goal_count": 80,
            "route_metadata": {"goal_count": 80},
        }],
    }


def test_benchmark_defaults_to_continuous_reached() -> None:
    replay = load_script("goal_pose_sequence")
    assert replay.resolve_timing(None, True) == "reached"
    assert replay.resolve_timing(None, False) == "recorded"
    assert replay.build_parser().parse_args([
        "replay", "goals.json", "--benchmark"
    ]).goal_tolerance == 2.0


def test_record_output_uses_selected_experiment_root(tmp_path: Path) -> None:
    replay = load_script("goal_pose_sequence")
    root = tmp_path / "LECO" / "experiments"
    args = replay.build_parser().parse_args([
        "record", "--experiment-root", str(root),
    ])
    output, label, redirected = replay.resolve_record_output(args)
    assert output == root / "maze2_goal1.json"
    assert label == "maze2_goal1"
    assert redirected
    output.write_text("{}", encoding="utf-8")
    next_output, _, _ = replay.resolve_record_output(args)
    assert next_output == root / "maze2_goal2.json"


def test_benchmark_rejects_non_reached_timing() -> None:
    replay = load_script("goal_pose_sequence")
    try:
        replay.resolve_timing("recorded", True)
    except ValueError:
        pass
    else:
        raise AssertionError("benchmark recorded timing must be rejected")


def test_exact_record_output_does_not_redirect_or_overwrite(tmp_path: Path) -> None:
    replay = load_script("goal_pose_sequence")
    root = tmp_path / "experiments"
    root.mkdir()
    requested = root / "manual.json"
    args = replay.build_parser().parse_args([
        "record", str(requested), "--experiment-root", str(root), "--exact-output",
    ])
    output, _, redirected = replay.resolve_record_output(args)
    assert output == requested
    assert not redirected
    requested.write_text("original", encoding="utf-8")
    try:
        replay.resolve_record_output(args)
    except ValueError:
        pass
    else:
        raise AssertionError("exact recording must reject an existing file")
    assert requested.read_text(encoding="utf-8") == "original"


def test_formal_report_validation() -> None:
    results = load_script("neural_tracker_results")
    report = valid_report()
    assert results.report_validation_errors(report) == []
    report["control"]["samples_K"] = 0
    assert "K=0" in results.report_validation_errors(report)


def test_formal_report_rejects_packet_loss_and_fov_missing() -> None:
    results = load_script("neural_tracker_results")
    report = valid_report()
    report["sampling"]["active_missed_sequence_samples"] = 1
    report["target_fov"] = {
        "samples": 9,
        "missing_or_stale_pose_samples": 1,
    }
    errors = results.report_validation_errors(report)
    assert "active_missed_sequence_samples != 0" in errors
    assert "FOV missing rate exceeds threshold" in errors


def test_results_run_directory_reads_only_direct_metrics(tmp_path: Path) -> None:
    results = load_script("neural_tracker_results")
    run_directory = tmp_path / "maze1_neural_run01"
    nested_directory = run_directory / "diagnostics"
    nested_directory.mkdir(parents=True)
    report = valid_report()
    report["schema_version"] = 4
    report["success_rate"] = {}
    (run_directory / "metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (nested_directory / "metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    discovered = results.discover_reports(
        run_directory,
        include_partial=False,
        include_legacy=False,
        include_invalid=False,
    )

    assert [item["path"] for item in discovered] == [
        (run_directory / "metrics.json").resolve()
    ]


def test_metrics_default_to_eighty_independent_goal_trials() -> None:
    metrics = load_script("neural_tracker_metrics")
    args = metrics.build_parser().parse_args([])
    assert args.trial_mode == "per-goal"
    assert args.lost_timeout == 3.0
    assert args.goal_tolerance == 2.0
    assert args.direct_target is False
    assert metrics.build_parser().parse_args(["--direct-target"]).direct_target is True

    experiment = load_script("neural_tracker_experiment")
    runner_args = experiment.build_parser().parse_args([
        "goals.json", "--run-label", "test_run",
    ])
    assert runner_args.final_goal_tolerance == 2.0
    assert runner_args.waypoint_radius == 2.0
    assert runner_args.lost_timeout == 3.0


def test_failed_fov_goal_is_marked_for_truth_assisted_continuation() -> None:
    replay = load_script("goal_pose_sequence")
    state = {
        "metrics_ready": False,
        "abort_reason": None,
        "recovery_required": False,
    }
    route_hash = "abc123"

    replay.apply_metrics_status(state, {
        "event": "trial_end",
        "trial_mode": "per-goal",
        "status": "failure",
        "reason": "target_lost_over_3s",
        "sequence_sha256": route_hash,
    }, route_hash)

    assert state["recovery_required"] is True
    assert state["abort_reason"] is None


def test_failed_fov_goal_notice_is_consumed_without_waiting_for_recovery() -> None:
    replay = load_script("goal_pose_sequence")
    state = {
        "metrics_ready": True,
        "abort_reason": None,
        "recovery_required": True,
    }

    assert replay.consume_recovery_notice(state) is True
    assert "recovery_required" not in state
    assert replay.consume_recovery_notice(state) is False


def test_legacy_fov_failure_cannot_abort_the_remaining_route() -> None:
    replay = load_script("goal_pose_sequence")
    state = {
        "metrics_ready": True,
        "abort_reason": "target_outside_fov_over_3s",
        "recovery_required": False,
    }
    route_hash = "abc123"

    # Older metrics publishers omitted trial_mode and used the 3-second reason.
    # The current replay must still treat it as a single-Goal failure.
    replay.apply_metrics_status(state, {
        "event": "trial_end",
        "status": "failure",
        "reason": "target_outside_fov_over_3s",
        "sequence_sha256": route_hash,
    }, route_hash)

    assert state["recovery_required"] is True
    assert state["abort_reason"] is None


def test_non_goal_terminal_status_still_aborts_replay() -> None:
    replay = load_script("goal_pose_sequence")
    state = {
        "metrics_ready": False,
        "abort_reason": None,
        "recovery_required": False,
    }

    replay.apply_metrics_status(state, {
        "event": "startup_failed",
        "status": "invalid",
        "reason": "duplicate publisher",
    }, "abc123")

    assert state["abort_reason"] == "duplicate publisher"


def test_runtime_input_failure_aborts_replay() -> None:
    replay = load_script("goal_pose_sequence")
    state = {
        "metrics_ready": True,
        "abort_reason": None,
        "recovery_required": False,
    }

    replay.apply_metrics_status(state, {
        "event": "runtime_failed",
        "status": "invalid",
        "reason": "publisher count != 1: metrics=0",
    }, "abc123")

    assert state["abort_reason"] == "publisher count != 1: metrics=0"


def test_random_goal_generator_satisfies_clearance_and_step_constraints(
    tmp_path: Path,
) -> None:
    generator = load_script("random_goal_sequence")
    replay = load_script("goal_pose_sequence")
    map_path = tmp_path / "map.json"
    output_path = tmp_path / "goals.json"
    map_payload = {
        "format": "neural-tracker-map",
        "version": 1,
        "map": {"width": 40.0, "height": 40.0, "resolution": 0.5},
        "boundary": {"enabled": True, "thickness": 0.5, "height": 2.0},
        "target": {"x": -10.0, "y": 0.0, "yaw": 0.0},
        "obstacles": [{
            "shape": "box",
            "x": 0.0,
            "y": 0.0,
            "angle": 0.0,
            "width": 4.0,
            "depth": 4.0,
            "height": 2.0,
        }],
    }
    map_path.write_text(json.dumps(map_payload), encoding="utf-8")

    sequence = generator.build_sequence(
        map_path=map_path,
        map_payload=map_payload,
        output=output_path,
        count=12,
        seed=1234,
        minimum_clearance=1.0,
        path_clearance=0.3,
        minimum_step=8.0,
        maximum_step=10.0,
        start=(-10.0, 0.0),
        nominal_speed=1.7,
        topic="/goal_pose",
        frame_id="world",
        run_label="test_route",
        max_restarts=50,
        coverage_grid=4,
        candidate_pool=512,
    )
    generator.atomic_write(output_path, sequence)

    loaded = replay.load_sequence(output_path)
    assert len(loaded["goals"]) == 12
    assert loaded["format"] == "neural-tracker-goal-sequence"
    assert loaded["version"] == 1
    assert loaded["topic"] == "/goal_pose"
    assert loaded["run_directory"] == str(output_path.parent)
    assert loaded["time_basis"] == "seconds since first received goal"
    metadata = loaded["generator"]
    assert metadata["achieved_minimum_clearance_m"] >= 1.0
    assert metadata["achieved_minimum_step_distance_m"] >= 8.0
    assert metadata["achieved_maximum_step_distance_m"] <= 10.0
    assert metadata["visited_coverage_cells"] > 1
    assert 0.0 < metadata["coverage_ratio_percent"] <= 100.0
    assert metadata["unique_goal_points"] == 12
    assert metadata["near_parallel_repeated_segments"] >= 0
    assert "non-adjacent" in metadata["repeated_segment_definition"]


def test_random_goal_generator_rasterizes_arch_and_frustum() -> None:
    generator = load_script("random_goal_sequence")
    payload = {
        "format": "neural-tracker-map",
        "version": 1,
        "map": {"width": 20.0, "height": 20.0, "resolution": 0.1},
        "boundary": {"enabled": False},
        "obstacles": [
            {
                "shape": "arch",
                "x": 0.0,
                "y": 0.0,
                "angle": 0.0,
                "width": 4.0,
                "depth": 1.0,
                "height": 3.0,
                "thickness": 0.5,
                "clearance": 2.0,
            },
            {
                "shape": "frustum",
                "x": 5.0,
                "y": 0.0,
                "angle": 0.0,
                "width": 2.0,
                "depth": 2.0,
                "height": 2.0,
                "topScale": 0.5,
            },
        ],
    }

    occupied, origin_x, origin_y, resolution = generator.rasterize_map(payload)

    def is_occupied(x: float, y: float) -> bool:
        row, column = generator.world_to_cell(
            x, y, origin_x, origin_y, resolution, occupied.shape
        )
        return bool(occupied[row, column])

    assert not is_occupied(0.0, 0.0)
    assert is_occupied(-1.75, 0.0)
    assert is_occupied(1.75, 0.0)
    assert is_occupied(5.0, 0.0)


def test_random_goal_generator_resolves_map_and_next_output_from_name(
    tmp_path: Path,
) -> None:
    generator = load_script("random_goal_sequence")
    map_directory = tmp_path / "src" / "simulator" / "map"
    map_directory.mkdir(parents=True)
    map_path = map_directory / "maze4.json"
    map_path.write_text("{}", encoding="utf-8")
    experiment_root = tmp_path / "experiments"
    experiment_root.mkdir()
    (experiment_root / "maze4_goal0.json").write_text("{}", encoding="utf-8")
    # A plural legacy filename is not part of the goal<N> sequence.
    (experiment_root / "maze4_goals1.json").write_text("{}", encoding="utf-8")

    assert generator.resolve_map_path(Path("maze4"), tmp_path) == map_path
    assert generator.next_automatic_output(
        experiment_root, "maze4"
    ) == experiment_root / "maze4_goal1.json"


def test_fov_recovery_is_counted_only_within_three_seconds() -> None:
    metrics = load_script("neural_tracker_metrics")

    def recorder() -> object:
        node = object.__new__(metrics.NeuralTrackerMetrics)
        node.visibility_count = 0
        node.invisible_count = 0
        node.visibility_rows = []
        node.visibility_state = True
        node.loss_events = 0
        node.recovery_events = 0
        node.lost_episode_open = False
        node.visibility_loss_started_at = None
        node.loss_failure_latched = False
        node.active_trial = {"lost_since": None, "loss_timeout_exceeded": False}
        node.args = SimpleNamespace(lost_timeout=3.0)
        node.elapsed = lambda now=None: float(now or 0.0)
        return node

    recovered = recorder()
    recovered.record_visibility(False, 10.0)
    recovered.record_visibility(True, 12.9)
    assert recovered.loss_events == 1
    assert recovered.recovery_events == 1
    assert not recovered.active_trial["loss_timeout_exceeded"]

    timed_out = recorder()
    timed_out.record_visibility(False, 10.0)
    timed_out.record_visibility(True, 13.1)
    assert timed_out.loss_events == 1
    assert timed_out.recovery_events == 0
    assert timed_out.active_trial["loss_timeout_exceeded"]

    # The same continuous loss cannot fail/count again until FOV recovery.
    timed_out.loss_failure_latched = True
    timed_out.lost_episode_open = False
    timed_out.active_trial = {"lost_since": None, "loss_timeout_exceeded": False}
    timed_out.record_visibility(False, 13.2)
    assert timed_out.loss_events == 1
    timed_out.record_visibility(True, 13.3)
    timed_out.record_visibility(False, 13.4)
    assert timed_out.loss_events == 2


def test_continuous_fov_loss_fails_current_goal_after_three_seconds() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    now = time.monotonic()
    node.active_trial = {
        "lost_since": now - 3.1,
        "loss_timeout_exceeded": False,
    }
    node.args = SimpleNamespace(lost_timeout=3.0)
    node.loss_failure_latched = False
    node.lost_episode_open = True
    node.visibility_loss_started_at = node.active_trial["lost_since"]
    terminal = []
    node.finish_trial = lambda status, reason, when: terminal.append((status, reason))

    node.evaluate_trial()

    assert terminal == [("failure", "target_lost_over_3s")]


def test_direct_target_makes_fov_loss_observational() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.args = SimpleNamespace(direct_target=True, lost_timeout=3.0)
    trial = {
        "lost_since": time.monotonic() - 30.0,
        "loss_timeout_exceeded": True,
    }

    assert node.trial_loss_timed_out(trial, time.monotonic()) is False


def test_branch_switches_and_atomic_timeline_are_recorded() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.branch_counts = metrics.Counter()
    node.branch_rows = []
    node.branch_switch_rows = []
    node.timeline_rows = []
    node.last_recorded_branch = None
    node.command_count = 8
    node.active_last_metrics_sequence = 108
    node.active_trial = {"id": 3}
    node.tracker_xy = (1.0, 2.0)
    node.tracker_yaw = 0.25
    node.tracker_pose_time = 9.9
    node.target_xy = (3.0, 4.0)
    node.target_yaw = -0.5
    node.target_pose_time = 9.8
    node.elapsed = lambda now=None: float(now or 0.0)

    assert node.record_branch(0, 10.0) is False
    assert node.record_branch(1, 10.1) is True
    node.record_timeline(
        108, 10.1, (0.6, -0.2), False, 4.5, 1, "traditional", True
    )

    assert node.branch_switch_rows == [
        (1, 8, 108, 3, 10.1, "neural", "traditional")
    ]
    row = node.timeline_rows[0]
    assert row[0:4] == (8, 108, 3, 10.1)
    assert row[4:10] == (1.0, 2.0, 0.25, 3.0, 4.0, -0.5)
    assert row[12:19] == (0.6, -0.2, 0, 4.5, 1, "traditional", 1)


def test_trial_evaluation_waits_for_first_atomic_sample() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.active_trial = {"lost_since": time.monotonic() - 30.0}
    node.active_last_metrics_sequence = None
    node.args = SimpleNamespace(direct_target=False)
    node.fail_active_trial_for_fov_timeout = lambda _now: (_ for _ in ()).throw(
        AssertionError("trial terminated before its first metrics sample")
    )

    node.evaluate_trial()


def test_truth_fallback_marks_goal_failed_until_raw_fov_recovers() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.truth_fallback_seen = False
    node.truth_fallback_active = False
    node.truth_fallback_trial_failure = False
    node.active_trial = {"truth_fallback_used": False}
    node.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    terminal = []
    node.fail_active_trial_for_fov_timeout = lambda when: terminal.append(when)

    node.on_observation_status(metrics.String(data="truth_fallback_started"))
    assert node.truth_fallback_seen is True
    assert node.truth_fallback_active is True
    assert node.active_trial["truth_fallback_used"] is True
    assert len(terminal) == 1

    node.on_observation_status(metrics.String(data="fov_detection_recovered"))
    assert node.truth_fallback_active is False


def test_goal_arrival_is_not_success_until_raw_fov_recovers() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    now = time.monotonic()
    node.active_trial = {
        "lost_since": now - 1.0,
        "loss_timeout_exceeded": False,
        "max_tracking_distance_m": None,
        "far_since": None,
        "near_since": None,
        "start_time": now - 2.0,
        "goal_x": 2.0,
        "goal_y": 0.0,
        "replay_finished": False,
    }
    node.tracker_xy = (0.0, 0.0)
    node.target_xy = (2.0, 0.0)
    node.latest_detection_visible = False
    node.args = SimpleNamespace(
        lost_timeout=3.0,
        max_tracking_distance=4.0,
        min_tracking_distance=0.5,
        distance_timeout=2.0,
        route_timeout=900.0,
        trial_timeout=60.0,
        trial_mode="per-goal",
        goal_tolerance=2.0,
    )
    node.tracker_in_collision = lambda: False
    terminal = []
    node.finish_trial = lambda status, reason, when: terminal.append((status, reason))

    node.evaluate_trial()
    assert terminal == []

    node.latest_detection_visible = True
    node.active_trial["lost_since"] = None
    node.evaluate_trial()
    assert terminal == [("success", "target_reached_goal")]


def test_per_goal_report_requires_all_route_goals() -> None:
    results = load_script("neural_tracker_results")
    report = valid_report()
    report["configuration"]["trial_mode"] = "per-goal"
    report["trials"] = [
        {
            "status": "success",
            "goal_count": 1,
            "route_metadata": {"goal_count": 2},
        },
        {
            "status": "failure",
            "goal_count": 1,
            "route_metadata": {"goal_count": 2},
        },
    ]
    assert results.report_validation_errors(report) == []
    report["trials"].pop()
    assert (
        "per-goal trial count 1 differs from expected 2"
        in results.report_validation_errors(report)
    )


def test_corrected_control_metrics_use_linear_velocity_rms_jerk(
    tmp_path: Path,
) -> None:
    results = load_script("neural_tracker_results")
    report_path = tmp_path / "metrics.json"
    command_path = tmp_path / "metrics_commands.csv"
    with command_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "elapsed_s", "v_mps", "omega_radps"))
        writer.writerow((1, 0.0, 0.00, 0.0))
        writer.writerow((2, 0.1, 0.01, 0.0))
        writer.writerow((3, 0.2, 0.04, 0.0))
    report = {
        "path": report_path,
        "data": {
            "configuration": {
                "control_dt_s": 0.1,
                "linear_control_scale_mps": 1.0,
                "angular_control_scale_radps": 1.0,
            },
            "control": {"samples_K": 3},
        },
    }

    metrics = results.control_quality_for_report(report)

    assert math.isclose(metrics["rms_jerk_ev_mps3"], 2.0)


def test_rms_jerk_does_not_difference_across_goal_trial_boundaries(
    tmp_path: Path,
) -> None:
    results = load_script("neural_tracker_results")
    report_path = tmp_path / "metrics.json"
    command_path = tmp_path / "metrics_commands.csv"
    with command_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "elapsed_s", "v_mps", "omega_radps"))
        for sample, velocity in enumerate((0, 0, 0, 10, 10, 10), 1):
            writer.writerow((sample, sample * 0.1, velocity, 0.0))
    report = {
        "path": report_path,
        "data": {
            "configuration": {"control_dt_s": 0.1},
            "control": {"samples_K": 6},
            "trials": [{"command_samples": 3}, {"command_samples": 3}],
        },
    }

    metrics = results.control_quality_for_report(report)

    assert metrics["energy_samples"] == 2
    assert metrics["rms_jerk_ev_mps3"] == 0.0


def test_run_label_is_sanitized() -> None:
    runner = load_script("neural_tracker_experiment")
    assert runner.safe_label(" maze2 neural/run02 ") == "maze2_neural_run02"


def test_trial_success_cannot_pass_with_zero_samples() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.command_count = 0
    node.visibility_count = 0
    node.planning_times_ms = []
    node.branch_rows = []
    node.target_fov_rows = []
    node.target_fov_missing_pose_samples = 0
    node.input_received = {name: True for name in (
        "metrics", "odom", "target_truth", "detection", "map"
    )}
    node.active_missed_metrics_samples = 0
    node.active_duplicate_metrics_samples = 0
    node.active_out_of_order_metrics_samples = 0
    node.args = SimpleNamespace(max_fov_missing_rate=0.01)
    trial = {
        "start_command_count": 0,
        "start_visibility_count": 0,
        "start_planning_count": 0,
        "start_branch_count": 0,
        "start_fov_count": 0,
        "start_fov_missing": 0,
        "goal_count": 80,
        "route_metadata": {"goal_count": 80},
    }
    assert "K=0" in node.trial_validation_errors(trial)


def test_tracking_distance_exceeded_does_not_terminate_route() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.active_trial = {
        "max_tracking_distance_m": None,
        "far_since": time.monotonic() - 10.0,
        "near_since": None,
        "start_time": time.monotonic(),
        "goal_x": 20.0,
        "goal_y": 0.0,
        "replay_finished": False,
    }
    node.tracker_xy = (0.0, 0.0)
    node.target_xy = (10.0, 0.0)
    node.truth_fallback_trial_failure = False
    node.args = SimpleNamespace(
        max_tracking_distance=4.0,
        min_tracking_distance=0.5,
        distance_timeout=2.0,
        route_timeout=900.0,
        trial_timeout=120.0,
        trial_mode="route",
        goal_tolerance=3.0,
    )
    node.tracker_in_collision = lambda: False
    terminal = []
    node.finish_trial = lambda status, reason, now: terminal.append((status, reason))

    node.evaluate_trial()

    assert terminal == []
    assert node.active_trial["far_since"] is not None
    assert node.active_trial["max_tracking_distance_m"] == 10.0


def test_traditional_metrics_defaults_to_auto_stop(monkeypatch) -> None:
    traditional = load_script("traditional_tracker_metrics")
    forwarded = []
    traditional.metrics_main = lambda argv: forwarded.extend(argv) or 0
    monkeypatch.setattr(
        traditional.sys,
        "argv",
        ["traditional_tracker_metrics", "--run-label", "test_run"],
    )

    assert traditional.main() == 0
    assert "--auto-stop" in forwarded
    assert forwarded[forwarded.index("--experiment-root") + 1] == "experiments_tradition"
    assert forwarded[forwarded.index("--algorithm-label") + 1] == "traditional_tracker"
    assert forwarded[forwarded.index("--tracker-family") + 1] == "traditional"


def test_runtime_publisher_loss_is_debounced_and_not_called_startup_failure() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.startup_ready = True
    node.startup_failure = None
    node.runtime_input_failure = None
    node.runtime_input_bad_since = None
    node.input_received = {
        "metrics": True,
        "odom": True,
        "target_truth": True,
        "detection": True,
        "map": True,
    }
    node.args = SimpleNamespace(startup_timeout=30.0)
    node.input_topics = lambda: {
        name: "/" + name for name in node.input_received
    }
    node.get_publishers_info_by_topic = lambda _topic: []
    node.elapsed = lambda: 100.0
    messages = []
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda message: messages.append(("warning", message)),
        error=lambda message: messages.append(("error", message)),
    )
    statuses = []
    node.publish_metrics_status = lambda event, **kwargs: statuses.append(
        (event, kwargs)
    )
    node.auto_stop_requested = False

    node.check_startup_ready()
    assert node.runtime_input_failure is None
    assert node.startup_failure is None
    assert node.auto_stop_requested is False

    node.runtime_input_bad_since -= 2.1
    node.check_startup_ready()
    assert node.runtime_input_failure is not None
    assert node.startup_failure is None
    assert node.auto_stop_requested is True
    assert statuses[-1][0] == "runtime_failed"


def test_metrics_close_flushes_without_ros_notifications() -> None:
    metrics = load_script("neural_tracker_metrics")
    node = object.__new__(metrics.NeuralTrackerMetrics)
    node.active_trial = {"id": 22}
    node.args = SimpleNamespace(output=Path("metrics.json"))
    events = []

    def finish_trial(status, reason, _now, *, notify_ros=True):
        events.append(("finish", status, reason, notify_ros))
        node.active_trial = None

    node.finish_trial = finish_trial
    node.write_report = lambda final=False: events.append(("report", final))
    node.write_raw_csv = lambda: events.append(("csv",))

    node.close()

    assert events == [
        ("finish", "incomplete", "recorder_stopped", False),
        ("report", True),
        ("csv",),
    ]


def test_traditional_results_remove_neural_branch_statistics() -> None:
    traditional = load_script("traditional_tracker_results")
    result = {
        "aggregation": {"branch_share": "test", "measurement_window": "active"},
        "planning_time_by_branch_ms": {"neural": {"mean": 1.0}},
        "planning_time_algorithm_excluding_hold_ms": {"mean": 1.0},
        "branch_usage": {"neural_share_percent_excluding_hold": 50.0},
        "across_run": {"algorithm_planning_mean_ms": {"mean": 1.0}},
        "per_run": [{
            "run": "traditional_run01",
            "neural_share_percent": 50.0,
            "traditional_share_percent": 50.0,
            "traditional_cycles": 10,
        }],
    }

    traditional.traditionalize(result)

    assert result["tracker_family"] == "traditional"
    assert "branch_usage" not in result
    assert "planning_time_by_branch_ms" not in result
    assert "branch_share" not in result["aggregation"]
    assert "neural_share_percent" not in result["per_run"][0]
    assert "traditional_cycles" not in result["per_run"][0]
