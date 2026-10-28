"""Export a trained TrackingPolicy checkpoint to ONNX and verify numerical parity.

The C++ tracker (tracker_neural) runs the policy through ONNX Runtime. This
script exports the exact forward the deploy node uses (six named inputs ->
controls/positions) at FP32 / opset 17, then loads the ONNX with onnxruntime and
asserts its `controls` match the original PyTorch output elementwise (< 1e-4).

Usage:
    MPLCONFIGDIR=/tmp .venv/bin/python scripts/export_onnx.py \
        --model models/v38_moving_dist_strong/best.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

# Resolve built-in paths from the neural_tracker package, independent of cwd.
TRACKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRACKER_DIR)

from config import Config, restore_config_from_checkpoint  # noqa: E402
from network import TrackingPolicy  # noqa: E402


def adapt_state_dict_for_target_dim(model, state_dict):
    """Mirror train.py: copy overlapping columns when TARGET_DIM changed."""
    key = "state_encoder.mlp.0.weight"
    model_state = model.state_dict()
    if key not in state_dict or key not in model_state:
        return state_dict
    old_w, new_w = state_dict[key], model_state[key]
    if old_w.shape == new_w.shape:
        return state_dict
    if old_w.ndim != 2 or new_w.ndim != 2 or old_w.shape[0] != new_w.shape[0]:
        return state_dict
    merged = new_w.clone()
    cols = min(old_w.shape[1], new_w.shape[1])
    merged[:, :cols] = old_w[:, :cols]
    if new_w.shape[1] > old_w.shape[1]:
        merged[:, old_w.shape[1]:] = 0.0
    adapted = dict(state_dict)
    adapted[key] = merged
    return adapted


class PolicyWrapper(torch.nn.Module):
    """Fix positional arg order for ONNX export and return only the tensors the
    C++ side consumes (controls, positions). esdf_map is dropped."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, esdf, robot_state, target_rel, history, visibility_ray, init_pose):
        out = self.policy(esdf, robot_state, target_rel, history, visibility_ray,
                          init_pose=init_pose)
        return out["controls"], out["positions"]


def make_dummy_inputs(cfg, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    esdf = torch.rand(1, 1, cfg.LOCAL_MAP_PIXELS, cfg.LOCAL_MAP_PIXELS, generator=g) * 2 - 1
    robot_state = torch.rand(1, cfg.STATE_DIM, generator=g) * 2 - 1
    target_rel = torch.rand(1, cfg.TARGET_DIM, generator=g) * 2 - 1
    history = torch.zeros(1, cfg.HISTORY_DIM)
    visibility_ray = torch.rand(1, cfg.VISIBILITY_SAMPLES, generator=g) * 2 - 1
    init_pose = torch.rand(1, 3, generator=g) * 2 - 1
    return tuple(t.to(device).float() for t in
                 (esdf, robot_state, target_rel, history, visibility_ray, init_pose))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(
            TRACKER_DIR, "models/g1_clearance_v2_20260727/best.pth"
        ),
    )
    parser.add_argument("--output", default=None,
                        help="ONNX path (default: alongside the checkpoint as policy.onnx)")
    parser.add_argument("--opset", type=int, default=None, help="override ONNX_OPSET")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cpu")

    model_path = args.model if os.path.isabs(args.model) else os.path.abspath(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    out_path = args.output or os.path.join(os.path.dirname(model_path), "policy.onnx")

    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = restore_config_from_checkpoint(ckpt, Config(), exclude={'DEVICE'})
    cfg.DEVICE = "cpu"
    opset = args.opset if args.opset is not None else getattr(cfg, "ONNX_OPSET", 17)
    policy = TrackingPolicy(cfg).to(device)
    state = adapt_state_dict_for_target_dim(policy, ckpt["model_state_dict"])
    policy.load_state_dict(state, strict=True)
    policy.eval()

    wrapper = PolicyWrapper(policy).eval()
    dummy = make_dummy_inputs(cfg, device, seed=0)

    input_names = ["esdf", "robot_state", "target_rel", "history",
                   "visibility_ray", "init_pose"]
    output_names = ["controls", "positions"]

    print(f"Exporting ONNX (opset {opset}, fp32) -> {out_path}")
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=input_names, output_names=output_names,
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )

    # --- Parity check: PyTorch vs onnxruntime on fresh inputs (different seed) ---
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    max_ctrl_err = 0.0
    max_pos_err = 0.0
    for seed in (1, 7, 42):
        ins = make_dummy_inputs(cfg, device, seed=seed)
        with torch.no_grad():
            pt_ctrl, pt_pos = wrapper(*ins)
        feed = {n: t.cpu().numpy().astype(np.float32) for n, t in zip(input_names, ins)}
        ox_ctrl, ox_pos = sess.run(output_names, feed)
        max_ctrl_err = max(max_ctrl_err, float(np.abs(pt_ctrl.numpy() - ox_ctrl).max()))
        max_pos_err = max(max_pos_err, float(np.abs(pt_pos.numpy() - ox_pos).max()))

    print(f"Parity over 3 seeds: max|Δcontrols|={max_ctrl_err:.3e}  "
          f"max|Δpositions|={max_pos_err:.3e}  (tol={args.tol:.1e})")
    if max_ctrl_err > args.tol:
        print("FAIL: controls parity exceeds tolerance.")
        sys.exit(1)
    print("OK: ONNX export is numerically faithful.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
