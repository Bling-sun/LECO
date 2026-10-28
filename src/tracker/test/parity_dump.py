"""Dump a synthetic scenario + Python-computed preprocessing/inference outputs
for the C++ tracker_neural parity test.

Reproduces the deploy node's _build_inputs on a synthetic occupancy grid and a
chosen robot/target state, then runs the ONNX policy. The C++ test
(parity_test.cpp) rebuilds the same inputs from the same raw scenario and
asserts its preprocessing + inference match these dumps.

Writes plain text files under this directory:
  scenario.txt      raw inputs (grid, robot, target)
  expect_esdf.txt   80*80 esdf patch
  expect_rel.txt    target_rel (6)
  expect_vis.txt    visibility_ray (8)
  expect_ctrl.txt   controls[0] = [v, omega]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "neural_tracker"))
sys.path.insert(0, NT)

os.chdir(NT)
from config import Config  # noqa: E402
from data_generator import ScenarioGenerator  # noqa: E402
import onnxruntime as ort  # noqa: E402


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def main():
    cfg = Config()
    gen = ScenarioGenerator(cfg)
    res = cfg.MAP_RESOLUTION
    N = int(round(cfg.MAP_SIZE / res))  # 400

    # Synthetic occupancy: a couple of rectangular obstacles (deterministic).
    occ = np.zeros((N, N), dtype=np.uint8)
    occ[150:180, 200:260] = 1     # a wall
    occ[240:300, 120:150] = 1     # another wall
    esdf = gen.compute_esdf(occ)  # (400,400) float32, map-centered convention

    # Robot + target in map-centered world meters.
    rx, ry, ryaw = 0.3, -0.5, 0.6
    tvx, tvy = 0.8, 0.2
    tx, ty = rx + 2.0, ry + 0.7
    robot_state = np.array([rx, ry, ryaw, 0.4, 0.1], dtype=np.float32)
    target_pos = np.array([tx, ty], dtype=np.float32)
    target_vel = np.array([tvx, tvy], dtype=np.float32)

    # --- esdf patch (extract_local_esdf) ---
    patch = gen.extract_local_esdf(esdf, rx, ry, ryaw).astype(np.float32)  # (80,80)

    # --- target_rel ---
    dx, dy = tx - rx, ty - ry
    rho = np.hypot(dx, dy)
    t_rel = wrap(np.arctan2(dy, dx) - ryaw)
    speed = np.hypot(tvx, tvy)
    t_v = wrap(np.arctan2(tvy, tvx) - ryaw) if speed > 1e-3 else 0.0
    target_rel = np.array([rho, np.sin(t_rel), np.cos(t_rel),
                           np.sin(t_v), np.cos(t_v), speed], dtype=np.float32)

    # --- visibility_ray ---
    K = cfg.VISIBILITY_SAMPLES
    vis = np.zeros(K, dtype=np.float32)
    alphas = np.linspace(0, 1, K)
    H, W = esdf.shape
    for i, a in enumerate(alphas):
        px_m = rx * (1 - a) + tx * a
        py_m = ry * (1 - a) + ty * a
        px = int((px_m + cfg.MAP_SIZE / 2) / res)
        py = int((py_m + cfg.MAP_SIZE / 2) / res)
        vis[i] = esdf[py, px] if (0 <= px < W and 0 <= py < H) else -1.0

    # --- ONNX inference ---
    onnx_path = os.path.join(
        NT, "models", "v41_directional_recovery", "policy.onnx"
    )
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {
        "esdf": patch[None, None].astype(np.float32),
        "robot_state": robot_state[3:][None].astype(np.float32),
        "target_rel": target_rel[None].astype(np.float32),
        "history": np.zeros((1, cfg.HISTORY_DIM), np.float32),
        "visibility_ray": vis[None].astype(np.float32),
        "init_pose": robot_state[:3][None].astype(np.float32),
    }
    controls, _ = sess.run(["controls", "positions"], feed)
    ctrl0 = controls[0, 0]  # [v, omega]

    # --- write dumps ---
    def dump(name, arr):
        np.savetxt(os.path.join(HERE, name), np.asarray(arr).ravel(), fmt="%.8e")

    with open(os.path.join(HERE, "scenario.txt"), "w") as f:
        f.write(f"{N} {res}\n")
        f.write(f"{rx} {ry} {ryaw} {robot_state[3]} {robot_state[4]}\n")
        f.write(f"{tx} {ty} {tvx} {tvy}\n")
        # occupancy as run-length of occupied rects is overkill; dump flat 0/1
        # (400*400=160k ints) is large but fine for a one-off test.
        f.write(" ".join(map(str, occ.ravel().tolist())) + "\n")

    dump("expect_esdf.txt", patch)
    dump("expect_rel.txt", target_rel)
    dump("expect_vis.txt", vis)
    dump("expect_ctrl.txt", ctrl0)
    print("Dumped parity scenario + expected outputs to", HERE)
    print("expected controls[0] =", ctrl0)


if __name__ == "__main__":
    main()
