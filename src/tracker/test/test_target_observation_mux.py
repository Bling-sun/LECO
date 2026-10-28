import importlib.util
from pathlib import Path

from geometry_msgs.msg import PoseStamped


SCRIPT = Path(__file__).parents[1] / "scripts" / "target_observation_mux.py"
SPEC = importlib.util.spec_from_file_location("target_observation_mux", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_detection_visibility_requires_valid_flag_and_finite_position():
    message = PoseStamped()
    message.pose.orientation.w = 1.0
    message.pose.position.x = 2.0
    message.pose.position.y = -1.0
    assert MODULE.detection_is_visible(message)

    message.pose.orientation.w = -1.0
    assert not MODULE.detection_is_visible(message)

    message.pose.orientation.w = 1.0
    message.pose.position.x = float("nan")
    assert not MODULE.detection_is_visible(message)


def test_truth_fallback_starts_only_after_full_timeout():
    assert not MODULE.fallback_due(None, 10.0, 3.0)
    assert not MODULE.fallback_due(10.0, 12.999, 3.0)
    assert MODULE.fallback_due(10.0, 13.0, 3.0)
