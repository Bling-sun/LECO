"""Replay a TensorBoard training run into Weights & Biases.

This is useful for runs that were launched without ``--wandb``. It uploads
scalar curves from a TensorBoard event file and checkpoints from the matching
model directory as a W&B artifact.
"""
import argparse
import glob
import os
import sys

# Make repo root importable so pickled config.Config in checkpoints can load.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def config_to_dict(config_obj):
    cfg = {}
    for key in dir(config_obj):
        if key.startswith("__"):
            continue
        value = getattr(config_obj, key)
        if callable(value):
            continue
        cfg[key] = value
    return cfg


def load_checkpoint(model_dir):
    best_path = os.path.join(model_dir, "best.pth")
    latest_path = os.path.join(model_dir, "latest.pth")
    checkpoint_path = best_path if os.path.exists(best_path) else latest_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No best.pth/latest.pth found in {model_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, checkpoint


def pick_event_file(log_dir, checkpoint):
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    target_metrics = checkpoint.get("behavior_metrics") or checkpoint.get("best_behavior_metrics") or {}
    target_collision = target_metrics.get("collision_rate")
    if target_collision is None:
        event_file = event_files[-1]
        ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
        ea.Reload()
        return event_file, ea

    best = None
    for event_file in event_files:
        ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "val_eval_metric/collision_rate" not in tags:
            continue
        scalars = ea.Scalars("val_eval_metric/collision_rate")
        if not scalars:
            continue
        diff = abs(scalars[-1].value - target_collision)
        if best is None or diff < best[0]:
            best = (diff, event_file, ea)

    if best is not None:
        return best[1], best[2]

    event_file = event_files[-1]
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    return event_file, ea


def replay_scalars(event_accumulator):
    by_step = {}
    for tag in event_accumulator.Tags().get("scalars", []):
        for scalar in event_accumulator.Scalars(tag):
            by_step.setdefault(scalar.step, {})[tag] = scalar.value

    for step in sorted(by_step):
        wandb.log(by_step[step], step=step)
    return len(by_step)


def add_summary(run, checkpoint):
    for key, value in (checkpoint.get("best_behavior_metrics") or {}).items():
        run.summary[f"best/{key}"] = value
    for key, value in (checkpoint.get("behavior_metrics") or {}).items():
        run.summary[f"final/{key}"] = value
    if "best_val_loss" in checkpoint:
        run.summary["best_val_loss"] = checkpoint["best_val_loss"]
    if "epoch" in checkpoint:
        run.summary["epochs_completed"] = checkpoint["epoch"] + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="neural_tracker")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--job-type", default="train_replay")
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.join("models", args.run_name)
    log_dir = args.log_dir or os.path.join("logs", args.run_name)

    checkpoint_path, checkpoint = load_checkpoint(model_dir)
    event_file, event_accumulator = pick_event_file(log_dir, checkpoint)
    config = config_to_dict(checkpoint["config"]) if "config" in checkpoint else {}

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using event file: {event_file}")

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        config=config,
        job_type=args.job_type,
        reinit=True,
    )

    steps = replay_scalars(event_accumulator)
    add_summary(run, checkpoint)

    artifact = wandb.Artifact(
        args.run_name,
        type="model",
        description=f"Checkpoints replayed from {model_dir}",
    )
    for checkpoint_file in sorted(glob.glob(os.path.join(model_dir, "*.pth"))):
        artifact.add_file(checkpoint_file, name=os.path.basename(checkpoint_file))
    run.log_artifact(artifact)

    run.finish()
    print(f"Uploaded {steps} scalar steps.")
    print("Done. View at:", run.url)


if __name__ == "__main__":
    main()
