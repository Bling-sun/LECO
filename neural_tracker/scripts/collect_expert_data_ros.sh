#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NEURAL_DIR="${ROOT_DIR}/neural_tracker"
SIM_CONFIG="${ROOT_DIR}/src/simulator/config/expert_collection.yaml"

RVIZ=true
SEEDS=1
SAMPLES_PER_SEED=4
EPISODE_SECONDS=10.0
SAMPLE_HZ=10.0
EXPERT_HORIZON=30
EXPERT_PLAN_HORIZON=31
SAVE_DIR="./data/expert_ros"
READY_TIMEOUT=30.0
STARTUP_WAIT=5
START_SEED=""
MAP_STYLES="blocks,sparse,dense,pillars,mixed_shapes,polygons_s,l_shapes,corner_maze,loop_maze,corridors,rooms,mixed"
if [[ -z "${ROS_PYTHON_BIN:-}" ]]; then
    if [[ -x /usr/bin/python3 ]]; then
        ROS_PYTHON_BIN="/usr/bin/python3"
    else
        ROS_PYTHON_BIN="python3"
    fi
fi
LOG_DIR="${ROOT_DIR}/log/data_collection"
GOAL_PERIOD=1.0
GOAL_REPUBLISH_PERIOD=2.0
MIN_GOAL_DISTANCE=3.0
GOAL_CLEARANCE=0.6
GOAL_ADVANCE_DISTANCE=1.5
GOAL_REACHED_DISTANCE=0.75
MIN_FORWARD_DOT=0.25
ESCAPE_MIN_FORWARD_DOT=-1.0
STUCK_WINDOW=3.0
STUCK_DISTANCE=0.25
STUCK_REPLAN_COOLDOWN=3.0
NO_TRAJECTORY_TIMEOUT=2.0

SIM_PID=""
TRACK_PID=""
GOAL_PID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --seeds N               Number of map seeds to collect. Each seed runs all map styles. Default: ${SEEDS}
  --samples N             Alias for --seeds, kept for old commands.
  --samples-per-seed N    Episodes to record for each map style under each seed. Default: ${SAMPLES_PER_SEED}
  --episode-seconds SEC   Episode duration for each sample. Default: ${EPISODE_SECONDS}
  --sample-hz HZ          Recorder sampling frequency. Default: ${SAMPLE_HZ}
  --expert-horizon N      Future cmd_vel steps saved per frame as expert_cmd_seq. Default: ${EXPERT_HORIZON}
  --expert-plan-horizon N Number of /tracker/plan points saved per frame. Default: ${EXPERT_PLAN_HORIZON}
  --save-dir DIR          Base output dir relative to neural_tracker/. Samples go into DIR/<map_style>. Default: ${SAVE_DIR}
  --ready-timeout SEC     Recorder topic wait timeout. Default: ${READY_TIMEOUT}
  --startup-wait SEC      Seconds to wait after launching ROS stack. Default: ${STARTUP_WAIT}
  --start-seed N          First map.seed to use. Default: seed from --sim-config
  --map-styles LIST       Comma-separated map styles collected for every seed. Default: ${MAP_STYLES}
  --log-dir DIR           Directory for launch/recorder logs. Default: ${LOG_DIR}
  --ros-python-bin PATH   Python used for rclpy recorder/goal scripts. Default: ${ROS_PYTHON_BIN}
  --goal-period SEC       How often to check whether target needs a new goal. Default: ${GOAL_PERIOD}
  --goal-republish-period SEC
                          How often to republish the active goal. Default: ${GOAL_REPUBLISH_PERIOD}
  --min-goal-distance M   Minimum distance from current target to next goal. Default: ${MIN_GOAL_DISTANCE}
  --goal-clearance M      Required obstacle clearance around goal. Default: ${GOAL_CLEARANCE}
  --advance-distance M    Publish the next goal when remaining distance is below this. Default: ${GOAL_ADVANCE_DISTANCE}
  --reached-distance M    Distance treated as target reaching a goal. Default: ${GOAL_REACHED_DISTANCE}
  --min-forward-dot X     Minimum dot(previous_dir, next_dir); 0.25 favors smoother turns. Default: ${MIN_FORWARD_DOT}
  --escape-min-forward-dot X
                          Direction constraint used only when escaping a stuck target. Default: ${ESCAPE_MIN_FORWARD_DOT}
  --stuck-window SEC      Time window used to detect a stuck target. Default: ${STUCK_WINDOW}
  --stuck-distance M      Movement below this window distance is considered stuck. Default: ${STUCK_DISTANCE}
  --stuck-replan-cooldown SEC
                          Minimum time between stuck escape replans. Default: ${STUCK_REPLAN_COOLDOWN}
  --no-trajectory-timeout SEC
                          Replan if target has no fresh plan/trajectory for this long. Default: ${NO_TRAJECTORY_TIMEOUT}
  --sim-config FILE      Simulator YAML (default: 20 m expert_collection.yaml; never modified).
  --no-rviz              Disable RViz for headless collection.
  -h, --help              Show this help.

Example:
  $(basename "$0") --seeds 3 --samples-per-seed 4 --episode-seconds 10.0 --sample-hz 10.0 --save-dir ./data/expert_ros
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-rviz|-h|--help) ;;
        *) [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; } ;;
    esac
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --samples)
            SEEDS="$2"
            shift 2
            ;;
        --samples-per-seed)
            SAMPLES_PER_SEED="$2"
            shift 2
            ;;
        --episode-seconds)
            EPISODE_SECONDS="$2"
            shift 2
            ;;
        --sample-hz)
            SAMPLE_HZ="$2"
            shift 2
            ;;
        --expert-horizon)
            EXPERT_HORIZON="$2"
            shift 2
            ;;
        --expert-plan-horizon)
            EXPERT_PLAN_HORIZON="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --ready-timeout)
            READY_TIMEOUT="$2"
            shift 2
            ;;
        --startup-wait)
            STARTUP_WAIT="$2"
            shift 2
            ;;
        --start-seed)
            START_SEED="$2"
            shift 2
            ;;
        --map-styles)
            MAP_STYLES="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --ros-python-bin)
            ROS_PYTHON_BIN="$2"
            shift 2
            ;;
        --goal-period)
            GOAL_PERIOD="$2"
            shift 2
            ;;
        --goal-republish-period)
            GOAL_REPUBLISH_PERIOD="$2"
            shift 2
            ;;
        --min-goal-distance)
            MIN_GOAL_DISTANCE="$2"
            shift 2
            ;;
        --goal-clearance)
            GOAL_CLEARANCE="$2"
            shift 2
            ;;
        --advance-distance)
            GOAL_ADVANCE_DISTANCE="$2"
            shift 2
            ;;
        --reached-distance)
            GOAL_REACHED_DISTANCE="$2"
            shift 2
            ;;
        --min-forward-dot)
            MIN_FORWARD_DOT="$2"
            shift 2
            ;;
        --escape-min-forward-dot)
            ESCAPE_MIN_FORWARD_DOT="$2"
            shift 2
            ;;
        --stuck-window)
            STUCK_WINDOW="$2"
            shift 2
            ;;
        --stuck-distance)
            STUCK_DISTANCE="$2"
            shift 2
            ;;
        --stuck-replan-cooldown)
            STUCK_REPLAN_COOLDOWN="$2"
            shift 2
            ;;
        --no-trajectory-timeout)
            NO_TRAJECTORY_TIMEOUT="$2"
            shift 2
            ;;
        --sim-config)
            SIM_CONFIG="$(realpath "$2")"
            shift 2
            ;;
        --no-rviz)
            RVIZ=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

current_seed() {
    "$ROS_PYTHON_BIN" - "$SIM_CONFIG" <<'PYTHON'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    print(yaml.safe_load(stream)["simulation"]["ros__parameters"]["map"]["seed"])
PYTHON
}

source_ros() {
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/install/setup.bash"
    set -u
}

check_ros_python() {
    if ! "$ROS_PYTHON_BIN" - <<'PY'
import rclpy
import yaml
import numpy
PY
    then
        cat >&2 <<EOF
ROS Python dependency check failed for: ${ROS_PYTHON_BIN}

This script runs rclpy helper scripts with ROS_PYTHON_BIN. If you are inside a
virtualenv and see "No module named 'yaml'", use the system Python:

  bash neural_tracker/scripts/collect_expert_data_ros.sh --ros-python-bin /usr/bin/python3 ...

or install PyYAML/numpy into the same Python environment.
EOF
        exit 1
    fi
}

stop_stack() {
    set +e
    # Each child owns a process group. Never stop another user's ROS session.
    for pid in "${GOAL_PID}" "${TRACK_PID}" "${SIM_PID}"; do
        [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null
    done
    sleep 2
    for pid in "${GOAL_PID}" "${TRACK_PID}" "${SIM_PID}"; do
        if [[ -n "$pid" ]]; then
            kill -KILL -- "-$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done
    GOAL_PID=""
    SIM_PID=""
    TRACK_PID=""
    set -e
}

start_stack() {
    local seed="$1"
    local style="$2"
    local sim_log="${LOG_DIR}/simulation_${style}_seed_${seed}.log"
    local tracker_log="${LOG_DIR}/tracking_${style}_seed_${seed}.log"

    echo "[seed ${seed}, style ${style}] launching simulator"
    setsid ros2 launch simulator simulation.launch.py config:="$SIM_CONFIG" map.seed:="$seed" map.style:="$style" rviz:="$RVIZ" >"${sim_log}" 2>&1 &
    SIM_PID="$!"

    sleep 2

    echo "[seed ${seed}, style ${style}] launching tracker"
    setsid ros2 launch tracker tracking.launch.py >"${tracker_log}" 2>&1 &
    TRACK_PID="$!"

    echo "[seed ${seed}, style ${style}] waiting ${STARTUP_WAIT}s for ROS stack startup"
    sleep "${STARTUP_WAIT}"

    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "[seed ${seed}, style ${style}] simulator exited early. Log:"
        sed -n '1,160p' "${sim_log}" >&2
        exit 1
    fi
    if grep -Eq "\\[simulation-[0-9]+\\]: process has died|exit code -11" "${sim_log}"; then
        echo "[seed ${seed}, style ${style}] simulator process died during startup. Log:"
        sed -n '1,180p' "${sim_log}" >&2
        exit 1
    fi
    if ! kill -0 "$TRACK_PID" 2>/dev/null; then
        echo "[seed ${seed}, style ${style}] tracker launch exited early. Log:"
        sed -n '1,160p' "${tracker_log}" >&2
        exit 1
    fi
    if grep -Eq "\\[(prediction|tracker)-[0-9]+\\]: process has died|exit code -11" "${tracker_log}"; then
        echo "[seed ${seed}, style ${style}] tracker process died during startup. Log:"
        sed -n '1,180p' "${tracker_log}" >&2
        exit 1
    fi
}

start_safe_goal_publisher() {
    local seed="$1"
    local style="$2"
    local goal_log="${LOG_DIR}/safe_goals_${style}_seed_${seed}.log"

    echo "[seed ${seed}, style ${style}] starting safe target goal publisher"
    setsid "$ROS_PYTHON_BIN" "${NEURAL_DIR}/scripts/publish_safe_target_goals.py" \
        --seed "$seed" \
        --period "$GOAL_PERIOD" \
        --republish-period "$GOAL_REPUBLISH_PERIOD" \
        --min-goal-distance "$MIN_GOAL_DISTANCE" \
        --goal-clearance "$GOAL_CLEARANCE" \
        --advance-distance "$GOAL_ADVANCE_DISTANCE" \
        --reached-distance "$GOAL_REACHED_DISTANCE" \
        --min-forward-dot "$MIN_FORWARD_DOT" \
        --escape-min-forward-dot "$ESCAPE_MIN_FORWARD_DOT" \
        --stuck-window "$STUCK_WINDOW" \
        --stuck-distance "$STUCK_DISTANCE" \
        --stuck-replan-cooldown "$STUCK_REPLAN_COOLDOWN" \
        --no-trajectory-timeout "$NO_TRAJECTORY_TIMEOUT" \
        >"${goal_log}" 2>&1 &
    GOAL_PID="$!"

    sleep 2

    if ! kill -0 "$GOAL_PID" 2>/dev/null; then
        echo "[seed ${seed}, style ${style}] safe target goal publisher exited early. Log:"
        sed -n '1,120p' "${goal_log}" >&2
        exit 1
    fi
}

record_samples_for_seed() {
    local seed="$1"
    local style="$2"
    local style_save_dir="${SAVE_DIR%/}/${style}"
    local recorder_log="${LOG_DIR}/recorder_${style}_seed_${seed}.log"

    echo "[seed ${seed}, style ${style}] recording ${SAMPLES_PER_SEED} sample(s) into ${style_save_dir}"
    (
        cd "$NEURAL_DIR"
        "$ROS_PYTHON_BIN" scripts/generate_expert_data_ros.py \
            --num_samples "$SAMPLES_PER_SEED" \
            --episode_seconds "$EPISODE_SECONDS" \
            --sample_hz "$SAMPLE_HZ" \
            --expert_horizon "$EXPERT_HORIZON" \
            --expert_plan_horizon "$EXPERT_PLAN_HORIZON" \
            --scenario_type "$style" \
            --save_dir "$style_save_dir" \
            --ready_timeout "$READY_TIMEOUT"
    ) 2>&1 | tee "${recorder_log}"
}

main() {
    require_file "$SIM_CONFIG"
    require_file "${ROOT_DIR}/install/setup.bash"
    require_file "/opt/ros/jazzy/setup.bash"

    mkdir -p "$LOG_DIR"
    source_ros
    check_ros_python

    if [[ -z "$START_SEED" ]]; then
        START_SEED="$(current_seed)"
    fi
    [[ "$SEEDS" =~ ^[1-9][0-9]*$ && "$SAMPLES_PER_SEED" =~ ^[1-9][0-9]*$ ]] || {
        echo "--seeds and --samples-per-seed must be positive integers" >&2; exit 2;
    }
    [[ "$START_SEED" =~ ^[0-9]+$ ]] || { echo "--start-seed must be non-negative" >&2; exit 2; }
    IFS=',' read -r -a STYLES <<< "$MAP_STYLES"
    STYLE_COUNT="${#STYLES[@]}"
    [[ "$STYLE_COUNT" -gt 0 ]] || { echo "--map-styles cannot be empty" >&2; exit 2; }
    for style in "${STYLES[@]}"; do
        [[ "$style" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "Invalid map style: $style" >&2; exit 2; }
    done

    echo "LECO root: ${ROOT_DIR}"
    echo "Simulator config: ${SIM_CONFIG}"
    echo "Seeds: ${SEEDS}, samples per seed/style: ${SAMPLES_PER_SEED}, first seed: ${START_SEED}"
    echo "Map styles (${STYLE_COUNT}): ${MAP_STYLES}"
    echo "ROS Python: ${ROS_PYTHON_BIN}"
    echo "Goal constraints: min_distance=${MIN_GOAL_DISTANCE}m, clearance=${GOAL_CLEARANCE}m, advance_distance=${GOAL_ADVANCE_DISTANCE}m, min_forward_dot=${MIN_FORWARD_DOT}"
    echo "Stuck recovery: window=${STUCK_WINDOW}s, distance=${STUCK_DISTANCE}m, cooldown=${STUCK_REPLAN_COOLDOWN}s, no_trajectory_timeout=${NO_TRAJECTORY_TIMEOUT}s"
    echo "Save dir: ${SAVE_DIR}/<map_style> (relative to ${NEURAL_DIR} if relative)"
    echo "Logs: ${LOG_DIR}"

    trap 'echo "Stopping ROS stack..."; stop_stack' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    for ((i = 0; i < SEEDS; i++)); do
        seed=$((START_SEED + i))
        echo
        echo "========== seed $((i + 1))/${SEEDS}, map.seed=${seed} =========="


        for ((j = 0; j < STYLE_COUNT; j++)); do
            style="${STYLES[$j]}"
            echo
            echo "----- style $((j + 1))/${STYLE_COUNT}, map.style=${style} -----"

            start_stack "$seed" "$style"
            start_safe_goal_publisher "$seed" "$style"
            record_samples_for_seed "$seed" "$style"

            echo "[seed ${seed}, style ${style}] recording finished; stopping ROS stack"
            stop_stack
        done

    done

    echo
    echo "Done. Collected $((SEEDS * STYLE_COUNT * SAMPLES_PER_SEED)) sample(s) into per-style folders under ${SAVE_DIR}."
}

main "$@"
