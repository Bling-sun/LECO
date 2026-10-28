"""Focused regression tests for training/validation correctness fixes."""
import os
import sys
import unittest

import numpy as np
import torch


TRACKER_DIR = os.path.dirname(os.path.abspath(__file__))
if TRACKER_DIR not in sys.path:
    sys.path.insert(0, TRACKER_DIR)

from config import (  # noqa: E402
    Config,
    config_to_dict,
    get_loss_weights,
    restore_config_from_checkpoint,
    validate_config,
)
from dataset import TrackingDataset  # noqa: E402
from loss import (  # noqa: E402
    FeasibilityLoss,
    MinTrackingDistanceLoss,
    SmoothLoss,
    TotalLoss,
    grid_sample_esdf,
)
from network import TrackingPolicy  # noqa: E402
from train import behavior_metrics, rescale_control_output_head  # noqa: E402
from validate import trajectory_quality  # noqa: E402


class DatasetLabelTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TrackingDataset.__new__(TrackingDataset)
        self.dataset.cfg = Config()

    def test_single_expert_command_only_supervises_first_step(self):
        cmd = np.array([0.7, -0.2], dtype=np.float32)
        seq, mask = self.dataset._extract_cmd_seq({'expert_cmd': cmd}, cmd)
        np.testing.assert_allclose(seq[0], cmd)
        np.testing.assert_array_equal(mask[:3], [1.0, 0.0, 0.0])

    def test_missing_expert_command_has_no_imitation_mask(self):
        seq, mask = self.dataset._extract_cmd_seq({}, np.zeros(2, dtype=np.float32))
        self.assertEqual(float(mask.sum()), 0.0)
        self.assertEqual(float(seq.sum()), 0.0)

    def test_short_mask_does_not_broadcast(self):
        sample = {
            'expert_cmd_seq': np.ones((3, 2), dtype=np.float32),
            'expert_cmd_seq_mask': np.array([1.0], dtype=np.float32),
        }
        _, mask = self.dataset._extract_cmd_seq(sample, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(mask[:3], [1.0, 0.0, 0.0])

    def test_control_history_is_left_padded_oldest_to_newest(self):
        history = self.dataset._extract_control_history({
            'control_history': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        })
        np.testing.assert_allclose(history[-4:], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(history[:-4], 0.0)


class LossCorrectnessTests(unittest.TestCase):
    def test_platform_limits_and_tracking_floor_defaults(self):
        cfg = validate_config(Config())
        self.assertEqual(cfg.MAX_VEL, 5.6)
        self.assertEqual(cfg.MAX_OMEGA, 4.0)
        self.assertEqual(cfg.MIN_TRACKING_DISTANCE, 0.7)

    def test_checkpoint_velocity_scaling_is_version_safe(self):
        legacy = Config()
        legacy.ROBOT_RADIUS = 0.5
        restored_legacy = restore_config_from_checkpoint({'config': legacy}, Config())
        self.assertEqual(restored_legacy.MAX_VEL, 2.0)
        self.assertEqual(restored_legacy.MAX_OMEGA, 1.2)
        self.assertEqual(restored_legacy.ROBOT_RADIUS, 0.5)

        current = Config()
        restored_current = restore_config_from_checkpoint(
            {'config_dict': config_to_dict(current)}, Config()
        )
        self.assertEqual(restored_current.MAX_VEL, 5.6)
        self.assertEqual(restored_current.MAX_OMEGA, 4.0)

    def test_finetune_rescales_legacy_control_head_logits(self):
        model = TrackingPolicy(Config())
        final_layer = model.control_head.control_head[-1]
        with torch.no_grad():
            final_layer.weight.fill_(1.0)
            final_layer.bias.fill_(1.0)
        ratios = rescale_control_output_head(model, 2.0, 1.2, 5.6, 4.0)
        self.assertAlmostEqual(ratios[0], 2.0 / 5.6)
        self.assertAlmostEqual(ratios[1], 1.2 / 4.0)
        torch.testing.assert_close(
            final_layer.weight[:, 0], torch.tensor([2.0 / 5.6, 1.2 / 4.0])
        )
        torch.testing.assert_close(
            final_layer.bias, torch.tensor([2.0 / 5.6, 1.2 / 4.0])
        )

    def test_out_of_bounds_esdf_is_occupied_and_has_gradient(self):
        cfg = Config()
        esdf = torch.ones(1, 400, 400)
        pos = torch.tensor([[[10.01, 0.0]]], requires_grad=True)
        values, _, in_bounds = grid_sample_esdf(
            pos, esdf, cfg.MAP_SIZE, cfg.MAP_RESOLUTION, return_in_bounds=True
        )
        self.assertFalse(bool(in_bounds.item()))
        self.assertLess(float(values.item()), 0.0)
        values.sum().backward()
        self.assertGreater(float(pos.grad.abs().sum()), 0.0)

    def test_feasibility_includes_current_to_first_command(self):
        cfg = Config()
        controls = torch.zeros(1, 3, 2)
        controls[:, :, 0] = cfg.MAX_VEL
        loss_fn = FeasibilityLoss(cfg)
        without_current = loss_fn(controls)
        with_current = loss_fn(controls, torch.zeros(1, 2))
        self.assertEqual(float(without_current), 0.0)
        self.assertGreater(float(with_current), 0.0)

    def test_smooth_yaw_wrap_is_not_two_pi_jump(self):
        cfg = Config()
        yaw = torch.tensor([3.0, 3.1, -3.0831853, -2.9831853])
        positions = torch.zeros(1, 4, 3)
        positions[0, :, 2] = yaw
        self.assertLess(float(SmoothLoss(cfg)(positions)), 1e-8)

    def test_loss_weights_use_instance_not_global_class(self):
        cfg = Config()
        cfg.USE_V43_OBJECTIVES = True
        self.assertGreater(get_loss_weights(0, cfg)['progress_speed'], 0.0)
        other = Config()
        other.USE_V43_OBJECTIVES = False
        self.assertEqual(get_loss_weights(0, other)['progress_speed'], 0.0)

    def test_min_tracking_distance_is_one_sided_and_differentiable(self):
        cfg = Config()
        loss_fn = MinTrackingDistanceLoss(cfg)
        unsafe = torch.zeros(1, 3, 3, requires_grad=True)
        unsafe.data[:, 1:, 0] = 0.5
        target = torch.zeros(1, 2)
        unsafe_loss = loss_fn(unsafe, target)
        self.assertGreater(float(unsafe_loss.detach()), 0.0)
        unsafe_loss.backward()
        self.assertGreater(float(unsafe.grad.abs().sum()), 0.0)

        safe = torch.zeros(1, 3, 3)
        safe[:, 1:, 0] = 0.8
        self.assertEqual(float(loss_fn(safe, target)), 0.0)


class ValidationMetricTests(unittest.TestCase):
    def test_segment_collision_detects_thin_obstacle_and_oob(self):
        cfg = Config()
        cfg.MAP_SIZE = 2.0
        cfg.MAP_RESOLUTION = 0.1
        cfg.ROBOT_RADIUS = 0.25
        esdf = np.ones((20, 20), dtype=np.float32)
        esdf[:, 10] = -0.1
        trajectory = np.array([
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ], dtype=np.float32)
        metrics = trajectory_quality(trajectory, np.array([0.5, 0.0]), esdf, cfg)
        self.assertGreater(metrics['collisions'], 0)

        trajectory[1, 0] = 1.1
        metrics = trajectory_quality(trajectory, np.array([0.5, 0.0]), esdf, cfg)
        self.assertGreater(metrics['map_violations'], 0)


class EndToEndSmokeTests(unittest.TestCase):
    def test_direct_control_forward_and_backward(self):
        torch.manual_seed(1)
        cfg = Config()
        cfg.DEVICE = 'cpu'
        model = TrackingPolicy(cfg)
        criterion = TotalLoss(cfg)
        batch_size = 1
        init_pose = torch.zeros(batch_size, 3)
        outputs = model(
            torch.ones(batch_size, 1, cfg.LOCAL_MAP_PIXELS, cfg.LOCAL_MAP_PIXELS),
            torch.zeros(batch_size, cfg.STATE_DIM),
            torch.tensor([[2.5, 0.0, 1.0, 0.0, 1.0, 0.5]]),
            torch.zeros(batch_size, cfg.HISTORY_DIM),
            torch.ones(batch_size, cfg.VISIBILITY_SAMPLES),
            init_pose=init_pose,
        )
        self.assertLessEqual(
            float(outputs['controls'][..., 0].abs().max().detach()), cfg.MAX_VEL
        )
        self.assertLessEqual(
            float(outputs['controls'][..., 1].abs().max().detach()), cfg.MAX_OMEGA
        )
        future = torch.zeros(batch_size, cfg.HORIZON, 2)
        future_mask = torch.zeros(batch_size, cfg.HORIZON)
        expert_seq = torch.zeros(batch_size, cfg.HORIZON, 2)
        expert_mask = torch.ones(batch_size, cfg.HORIZON)
        batch = {
            'init_pose': init_pose,
            'current_control': torch.zeros(batch_size, 2),
            'target_pos': torch.tensor([[2.5, 0.0]]),
            'target_vel': torch.tensor([[0.5, 0.0]]),
            'target_future_pos': future,
            'target_future_pos_mask': future_mask,
            'global_esdf': torch.ones(batch_size, 400, 400),
            'expert_cmd_seq': expert_seq,
            'expert_cmd_seq_mask': expert_mask,
            'sample_weight': torch.ones(batch_size),
        }
        weights = {
            'collision': 1.0,
            'feasibility': 1.0,
            'smooth': 0.1,
            'visibility': 1.0,
            'moving_dist': 1.0,
            'min_tracking_dist': 1.0,
            'expert_cmd_seq': 1.0,
        }
        total, losses = criterion(outputs, batch, weights)
        self.assertTrue(torch.isfinite(total))
        self.assertIn('feasibility', losses)
        total.backward()
        grad_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters() if parameter.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)
        metrics = behavior_metrics(
            outputs, batch['target_pos'], init_pose, batch['global_esdf'], cfg,
            target_vel=batch['target_vel'],
            expert_cmd_seq=expert_seq,
            expert_cmd_seq_mask=expert_mask,
            target_future_pos=future,
            target_future_pos_mask=future_mask,
        )
        self.assertEqual(metrics['samples'], batch_size)
        self.assertTrue(np.isfinite(metrics['moving_obs_err_sum']))


if __name__ == '__main__':
    unittest.main()
