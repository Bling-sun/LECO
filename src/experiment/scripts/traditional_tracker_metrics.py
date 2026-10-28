#!/usr/bin/env python3
"""Traditional-tracker metrics entry point with isolated output defaults."""
import sys
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
_base = Path(__file__).resolve().parent
_metrics_path = _base / "neural_tracker_metrics.py"
if not _metrics_path.exists():
    _metrics_path = _base / "neural_tracker_metrics"
_loader = SourceFileLoader("experiment_metrics", str(_metrics_path))
_spec = importlib.util.spec_from_loader("experiment_metrics", _loader)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
metrics_main = _module.main


def main() -> int:
    args = sys.argv[1:]
    if "--experiment-root" not in args:
        args += ["--experiment-root", "experiments_tradition"]
    if "--algorithm-label" not in args:
        args += ["--algorithm-label", "traditional_tracker"]
    if "--tracker-family" not in args:
        args += ["--tracker-family", "traditional"]
    if "--auto-stop" not in args:
        args += ["--auto-stop"]
    # neural_tracker_metrics.main expects argv[0] to be the program name.
    return metrics_main([sys.argv[0], *args])


if __name__ == "__main__":
    raise SystemExit(main())
