#!/usr/bin/env python3
"""Traditional-tracker FOV heatmap entry point."""

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path


_base = Path(__file__).resolve().parent
_heatmap_path = _base / "neural_tracker_fov_heatmap.py"
if not _heatmap_path.exists():
    _heatmap_path = _base / "neural_tracker_fov_heatmap"
_loader = SourceFileLoader("experiment_fov_heatmap_core", str(_heatmap_path))
_spec = importlib.util.spec_from_loader("experiment_fov_heatmap_core", _loader)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


if __name__ == "__main__":
    raise SystemExit(_module.main())
