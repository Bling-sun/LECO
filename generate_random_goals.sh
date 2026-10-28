#!/usr/bin/env bash
# Run the Goal generator without depending on which same-named ROS
# "experiment" package is currently first in AMENT_PREFIX_PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 src/experiment/scripts/random_goal_sequence.py "$@"
