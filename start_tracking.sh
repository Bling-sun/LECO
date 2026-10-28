#!/usr/bin/env bash
# Shared simulator launcher: C++/ONNX, traditional MPC, Python/PyTorch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP=/opt/ros/jazzy/setup.bash
WS_SETUP="$ROOT/install/setup.bash"
ROS2=/opt/ros/jazzy/bin/ros2
BACKEND=neural
SESSION=""
ONNX="$ROOT/neural_tracker/models/v41_directional_recovery/policy.onnx"
MODEL="$ROOT/neural_tracker/models/v41_directional_recovery_retrain_clean_20260717/best.pth"
MODEL_EXPLICIT=0
ONNX_EXPLICIT=0
DEVICE=cpu
MAP_JSON=""
SIM_CONFIG=""
GOALS=""
RECORD=""
REFERENCE=""
ONNX_THREADS=4
TIMING_INTERVAL=2.0
TRUTH_FALLBACK_AFTER=3.0
DIRECT_TARGET=0
TEMPORARY_OBSTACLES=0
OBSTACLE_SEED=-1
TARGET_EXTERNAL_CONTROL=0
RVIZ=true
ATTACH=0
ACTION=start
SIM_ARGS=()
TRACKER_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./start_tracking.sh [options]

  --backend neural|traditional|python   Controller (default: neural = C++/ONNX)
  --onnx FILE                          ONNX policy for neural
  --model FILE                         PyTorch checkpoint for python
  --device cpu|cuda                    Python inference device (default: cpu)
  --map-json FILE                      Map and actor poses (C++ default: maze2)
  --sim-config FILE                    Simulator YAML; no default map override
  --goals FILE                         Replay a Goal sequence after startup
  --record-waypoints FILE              Record RViz Goals to a new JSON file
  --reference FILE                     Replay external target poses
  --temporary-obstacles                Add persistent obstacles every 5 s
  --obstacle-seed N                     Obstacle seed; -1 means random (default)
  --direct-target                      Use target ground truth instead of FOV
  --truth-fallback-after SEC            C++ FOV recovery delay (default: 3)
  --onnx-threads N                      ONNX CPU threads (default: 4)
  --timing-interval SEC                 Timing log interval; <=0 disables it
  --sim-arg NAME:=VALUE                 Extra ROS simulator argument; repeatable
  --tracker-arg NAME:=VALUE             Extra C++ tracker argument; repeatable
  --target-external-control            Accept /target/reference_pose externally
  --no-rviz                            Run without a graphical interface
  --session NAME                       tmux session (default: tracking_BACKEND)
  --attach                             Attach after all nodes are ready
  --status                             Show this session's panes
  --stop                               Stop this session; option order is free
  -h, --help                           Show help

Input paths: current directory first, then repository root.
Recording output paths: relative to the repository root.
Examples:
  ./start_tracking.sh --onnx neural_tracker/models/demo_train/policy.onnx
  ./start_tracking.sh --backend traditional --no-rviz
  ./start_tracking.sh --backend python --model neural_tracker/models/demo_train/best.pth
  ./start_tracking.sh --map-json src/simulator/map/maze1.json --temporary-obstacles --direct-target
  ./start_tracking.sh --goals experiments/maze2_goal1.json --no-rviz
  ./start_tracking.sh --stop --session tracking_neural
EOF
}
fail() { echo "ERROR: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --backend|--onnx|--model|--device|--map-json|--sim-config|--goals|--record-waypoints|--reference|--obstacle-seed|--truth-fallback-after|--onnx-threads|--timing-interval|--session|--sim-arg|--tracker-arg)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
            case "$1" in
                --backend) BACKEND="$2" ;;
                --onnx) ONNX="$2"; ONNX_EXPLICIT=1 ;;
                --model) MODEL="$2"; MODEL_EXPLICIT=1 ;;
                --device) DEVICE="$2" ;;
                --map-json) MAP_JSON="$2" ;;
                --sim-config) SIM_CONFIG="$2" ;;
                --goals) GOALS="$2" ;;
                --record-waypoints) RECORD="$2" ;;
                --reference) REFERENCE="$2" ;;
                --obstacle-seed) OBSTACLE_SEED="$2" ;;
                --truth-fallback-after) TRUTH_FALLBACK_AFTER="$2" ;;
                --onnx-threads) ONNX_THREADS="$2" ;;
                --timing-interval) TIMING_INTERVAL="$2" ;;
                --session) SESSION="$2" ;;
                --sim-arg) SIM_ARGS+=("$2") ;;
                --tracker-arg) TRACKER_ARGS+=("$2") ;;
            esac
            shift 2 ;;
        --direct-target) DIRECT_TARGET=1; shift ;;
        --use-fov-detection|--use_fov_detection) DIRECT_TARGET=0; shift ;;
        --temporary-obstacles) TEMPORARY_OBSTACLES=1; shift ;;
        --target-external-control) TARGET_EXTERNAL_CONTROL=1; shift ;;
        --no-rviz) RVIZ=false; shift ;;
        --attach) ATTACH=1; shift ;;
        --status) ACTION=status; shift ;;
        --stop) ACTION=stop; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1 (see --help)" ;;
    esac
done
case "$BACKEND" in neural|traditional|python) ;; *) fail "invalid --backend: $BACKEND" ;; esac
SESSION="${SESSION:-tracking_$BACKEND}"
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || fail "--session accepts letters, digits, underscores and hyphens"
command -v tmux >/dev/null || fail "install tmux first"
stop_session() {
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        while read -r pane; do tmux send-keys -t "$pane" C-c 2>/dev/null || true; done \
            < <(tmux list-panes -s -t "=$SESSION" -F '#{pane_id}')
        sleep 0.5
        tmux kill-session -t "=$SESSION" 2>/dev/null || true
    fi
}
case "$ACTION" in
    stop) stop_session; echo "Stopped session '$SESSION'."; exit 0 ;;
    status) exec tmux list-panes -s -t "=$SESSION" -F '#S:#P #{pane_current_command} dead=#{pane_dead}' ;;
esac
[[ "$MODEL_EXPLICIT" == 0 || "$BACKEND" == python ]] || fail "--model requires --backend python; use --onnx for C++ inference"
[[ "$ONNX_EXPLICIT" == 0 || "$BACKEND" == neural ]] || fail "--onnx requires --backend neural"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    printf -v STOP_HINT '%q --session %q --stop' "$0" "$SESSION"
    fail "session '$SESSION' is already running.
  Stop: $STOP_HINT
  View: tmux attach -t $SESSION"
fi
[[ -f "$ROS_SETUP" && -f "$WS_SETUP" ]] || fail "build the workspace first (see README.md)"
[[ "$DEVICE" == cpu || "$DEVICE" == cuda ]] || fail "--device must be cpu or cuda"
[[ "$ONNX_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "--onnx-threads must be positive"
/usr/bin/python3 - "$OBSTACLE_SEED" "$TIMING_INTERVAL" "$TRUTH_FALLBACK_AFTER" <<'PY'
import math
import sys
try:
    seed = int(sys.argv[1])
    timing, fallback = map(float, sys.argv[2:])
    assert -1 <= seed < 2**63 and math.isfinite(timing)
    assert math.isfinite(fallback) and fallback >= 0
except (ValueError, AssertionError):
    raise SystemExit('ERROR: invalid obstacle seed, timing interval or fallback delay')
PY
[[ "$BACKEND" != python || ${#TRACKER_ARGS[@]} -eq 0 ]] || fail "--tracker-arg is for C++ backends"
[[ "$BACKEND" != python || "$TRUTH_FALLBACK_AFTER" == 3.0 ]] || fail "Python uses predictive recovery rather than the C++ truth fallback delay"
for argument in "${SIM_ARGS[@]}" "${TRACKER_ARGS[@]}"; do
    [[ "$argument" == ?*:=?* ]] || fail "expected NAME:=VALUE, got: $argument"
done
ROUTE_MODES=0
for route in "$GOALS" "$RECORD" "$REFERENCE"; do [[ -z "$route" ]] || ROUTE_MODES=$((ROUTE_MODES + 1)); done
[[ "$ROUTE_MODES" -le 1 ]] || fail "choose one of --goals, --record-waypoints, --reference"
resolve_file() {
    if [[ -f "$1" ]]; then realpath "$1"
    elif [[ -f "$ROOT/$1" ]]; then realpath "$ROOT/$1"
    else fail "file not found: $1"; fi
}
if [[ -z "$MAP_JSON" && -z "$SIM_CONFIG" ]]; then
    if [[ "$BACKEND" == python ]]; then SIM_CONFIG="$ROOT/src/simulator/config/expert_collection.yaml"
    else MAP_JSON="$ROOT/src/simulator/map/maze2.json"; fi
fi
[[ -z "$MAP_JSON" ]] || MAP_JSON="$(resolve_file "$MAP_JSON")"
[[ -z "$SIM_CONFIG" ]] || SIM_CONFIG="$(resolve_file "$SIM_CONFIG")"
[[ -z "$GOALS" ]] || GOALS="$(resolve_file "$GOALS")"
[[ -z "$REFERENCE" ]] || REFERENCE="$(resolve_file "$REFERENCE")"
if [[ -n "$RECORD" ]]; then
    [[ "$RECORD" == /* ]] || RECORD="$ROOT/$RECORD"
    [[ ! -e "$RECORD" ]] || fail "recording output already exists: $RECORD"
    mkdir -p "$(dirname "$RECORD")"
fi
case "$BACKEND" in
    neural) ONNX="$(resolve_file "$ONNX")"; BINARY=tracker_neural ;;
    traditional) BINARY=tracker ;;
    python)
        MODEL="$(resolve_file "$MODEL")"
        VENV_PY="$ROOT/neural_tracker/.venv/bin/python"
        [[ -x "$VENV_PY" ]] || fail "create neural_tracker/.venv first (see docs/training/README.md)"
        BINARY=prediction ;;
esac
[[ -x "$ROOT/install/tracker/lib/tracker/$BINARY" ]] || fail "$BINARY is not built"
[[ -x "$ROOT/install/simulator/lib/simulator/simulation" ]] || fail "simulator is not built"
/usr/bin/python3 - "$MAP_JSON" "$GOALS" "$REFERENCE" <<'PY'
import hashlib
import json
import sys
map_path, goals_path, reference_path = sys.argv[1:]
if map_path:
    payload = json.load(open(map_path, encoding='utf-8'))
    if payload.get('format') not in {'neural-tracker-map', 'neural-tracker-forest-map'}:
        raise SystemExit('ERROR: unsupported map format')
    if any(float(payload.get('map', {}).get(k, 0)) <= 0 for k in ('width', 'height', 'resolution')):
        raise SystemExit('ERROR: map dimensions must be positive')
    if not isinstance(payload.get('obstacles'), list):
        raise SystemExit('ERROR: map obstacles must be an array')
for path, format_name, key in ((goals_path, 'neural-tracker-goal-sequence', 'goals'),
                               (reference_path, 'elastic-target-reference-v1', 'samples')):
    if not path:
        continue
    payload = json.load(open(path, encoding='utf-8'))
    if payload.get('format') != format_name or not payload.get(key):
        raise SystemExit(f'ERROR: invalid or empty route: {path}')
    expected = (payload.get('map_sha256') or payload.get('experiment', {}).get('map_sha256')
                or payload.get('generator', {}).get('map_sha256'))
    if expected and (not map_path or hashlib.sha256(open(map_path, 'rb').read()).hexdigest() != expected):
        raise SystemExit('ERROR: route map hash mismatch; use its original map')
PY

# Restore the caller's discovery/display settings on every tmux pane.
printf -v ROS_Q '%q' "$ROS_SETUP"
printf -v WS_Q '%q' "$WS_SETUP"
ENV_PREFIX="unset VIRTUAL_ENV PYTHONHOME; source $ROS_Q && source $WS_Q"
for name in ROS_DOMAIN_ID ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE \
    ROS_STATIC_PEERS RMW_IMPLEMENTATION CYCLONEDDS_URI FASTRTPS_DEFAULT_PROFILES_FILE \
    DISPLAY WAYLAND_DISPLAY XAUTHORITY; do
    if [[ -v "$name" ]]; then
        printf -v value_q '%q' "${!name}"
        ENV_PREFIX+=" && export $name=$value_q"
    else ENV_PREFIX+=" && unset $name"; fi
done
CREATED=0
SUCCESS=0
cleanup() { if [[ "$CREATED" == 1 && "$SUCCESS" == 0 ]]; then stop_session; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_pane() {
    local pane="$1" command_q
    shift
    printf -v command_q '%q ' "$@"
    tmux respawn-pane -k -t "$pane" /usr/bin/bash --noprofile --norc -c "$ENV_PREFIX && exec $command_q"
}
wait_ready() {
    local pane="$1" marker="$2" output
    for ((attempt=0; attempt<120; attempt++)); do
        output="$(tmux capture-pane -p -S -200 -t "$pane" 2>/dev/null || true)"
        if [[ "$output" == *"$marker"* ]]; then return; fi
        if [[ "$(tmux display-message -p -t "$pane" '#{pane_dead}' 2>/dev/null || true)" == 1 ]] \
            || [[ "$output" =~ process\ has\ died|Traceback|terminate\ called ]]; then break; fi
        sleep 0.25
    done
    echo "$output" >&2
    fail "node did not become ready: $marker"
}
SIM_COMMAND=("$ROS2" launch simulator simulation.launch.py "rviz:=$RVIZ")
[[ -z "$MAP_JSON" ]] || SIM_COMMAND+=("map.file:=$MAP_JSON")
[[ -z "$SIM_CONFIG" ]] || SIM_COMMAND+=("config:=$SIM_CONFIG")
if [[ "$TEMPORARY_OBSTACLES" == 1 ]]; then
    SIM_COMMAND+=(temporary_obstacle.enabled:=true temporary_obstacle.auto_add:=true
        temporary_obstacle.interval:=5.0 temporary_obstacle.height:=4.0
        "temporary_obstacle.seed:=$OBSTACLE_SEED" rviz_clicked_point_topic:=/temporary_obstacle)
fi
if [[ -n "$REFERENCE" || "$TARGET_EXTERNAL_CONTROL" == 1 ]]; then SIM_COMMAND+=(target.external_control:=true); fi
SIM_COMMAND+=("${SIM_ARGS[@]}")
echo "Starting $BACKEND in tmux '$SESSION'"
echo "  map: ${MAP_JSON:-$SIM_CONFIG}"
tmux new-session -d -s "$SESSION" -n demo -c "$ROOT" -x 180 -y 60
CREATED=1
tmux set-window-option -t "$SESSION:demo" remain-on-exit on
SIM_PANE="$(tmux display-message -p -t "$SESSION:demo.0" '#{pane_id}')"
start_pane "$SIM_PANE" "${SIM_COMMAND[@]}"
wait_ready "$SIM_PANE" 'Simulator ready'
TARGET_TOPIC=/target/detection
TARGET_DIRECT=false
USE_FOV=true
if [[ "$DIRECT_TARGET" == 1 ]]; then TARGET_TOPIC=/target/ground_truth; TARGET_DIRECT=true; USE_FOV=false; fi
TRACKER_PANE="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$SIM_PANE" -c "$ROOT")"
case "$BACKEND" in
    neural)
        TRACK_COMMAND=("$ROS2" launch tracker tracking_neural.launch.py
            "onnx_path:=$ONNX" "onnx_threads:=$ONNX_THREADS"
            "timing_log_interval:=$TIMING_INTERVAL" "truth_fallback_after:=$TRUTH_FALLBACK_AFTER"
            "target_topic:=$TARGET_TOPIC" "target_direct:=$TARGET_DIRECT" "use_fov_detection:=$USE_FOV")
        READY_MARKER='Neural policy loaded:' ;;
    traditional)
        [[ "$DIRECT_TARGET" == 1 ]] || TARGET_TOPIC=/target/fov_or_truth
        TRACK_COMMAND=("$ROS2" launch tracker tracking.launch.py
            "timing_log_interval:=$TIMING_INTERVAL" "truth_fallback_after:=$TRUTH_FALLBACK_AFTER"
            "target_topic:=$TARGET_TOPIC" "target_direct:=$TARGET_DIRECT" "use_fov_detection:=$USE_FOV"
            max_linear_speed:=2.5 max_linear_accel:=3.0)
        READY_MARKER='Target input:' ;;
    python)
        start_pane "$TRACKER_PANE" "$ROS2" launch tracker prediction.launch.py \
            "target_topic:=$TARGET_TOPIC" "target_direct:=$TARGET_DIRECT"
        wait_ready "$TRACKER_PANE" 'Target input:'
        TRACKER_PANE="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$TRACKER_PANE" -c "$ROOT/neural_tracker")"
        TRACK_COMMAND=("$VENV_PY" "$ROOT/neural_tracker/scripts/deploy_neural_tracker_ros.py"
            --model "$MODEL" --device "$DEVICE" --cmd-topic /tracker/cmd_vel
            --target-topic "$TARGET_TOPIC" --timing-interval "$TIMING_INTERVAL" --safety-stop --lost-policy predict)
        READY_MARKER='Neural tracker ready:' ;;
esac
start_pane "$TRACKER_PANE" "${TRACK_COMMAND[@]}" "${TRACKER_ARGS[@]}"
wait_ready "$TRACKER_PANE" "$READY_MARKER"
if [[ "$BACKEND" == python ]]; then wait_ready "$TRACKER_PANE" 'Map received:'; fi
ROUTE_COMMAND=()
if [[ -n "$GOALS" ]]; then
    ROUTE_COMMAND=(/usr/bin/python3 "$ROOT/src/experiment/scripts/goal_pose_sequence.py"
        replay "$GOALS" --timing reached --goal-tolerance 1.0 --waypoint-timeout 300 --start-delay 2)
elif [[ -n "$RECORD" ]]; then
    ROUTE_COMMAND=(/usr/bin/python3 "$ROOT/src/experiment/scripts/goal_pose_sequence.py"
        record "$RECORD" --exact-output)
elif [[ -n "$REFERENCE" ]]; then
    ROUTE_COMMAND=(/usr/bin/python3 "$ROOT/src/experiment/scripts/elastic_target_reference_replay.py"
        "$REFERENCE" --start-delay 2 --publish-rate 100)
fi
if ((${#ROUTE_COMMAND[@]})); then
    ROUTE_PANE="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$TRACKER_PANE" -c "$ROOT")"
    start_pane "$ROUTE_PANE" "${ROUTE_COMMAND[@]}"
    if [[ -n "$GOALS" ]]; then wait_ready "$ROUTE_PANE" 'Replaying '
    elif [[ -n "$RECORD" ]]; then wait_ready "$ROUTE_PANE" 'Recording '
    else wait_ready "$ROUTE_PANE" 'Replaying exact'; fi
fi
tmux select-layout -t "$SESSION:demo" tiled >/dev/null
SUCCESS=1
echo "Ready. Attach: tmux attach -t $SESSION"
echo "Stop: $0 --session $SESSION --stop"
[[ -n "$GOALS$REFERENCE$RECORD" ]] || echo 'Send a 2D Goal Pose in RViz, or publish /goal_pose to move the target.'
if [[ "$ATTACH" == 1 ]]; then tmux attach -t "$SESSION"; fi
