"""CPU regression: python -m unittest discover -s tests -p test_flow_rgbd_fusion_v2.py -v"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.flow_rgbd_fusion_v2 import compute_flow_rgbd_gate_v2, tracking_loss_v2


class GateTests(unittest.TestCase):
    def setUp(self):
        self.rgb = torch.ones((3, 32, 32), dtype=torch.float32) * 0.5
        self.obs = self.rgb.clone()
        self.z = torch.ones((1, 32, 32), dtype=torch.float32) * 2
        self.obs_z = self.z.clone()
        self.alpha = torch.ones((1, 32, 32))
        self.static = torch.ones((32, 32), dtype=torch.bool)
        self.flow = torch.ones((32, 32)) * 0.2
        self.cfg = {}

    def gate(self):
        return compute_flow_rgbd_gate_v2(
            self.rgb, self.z, self.alpha, self.obs, self.obs_z,
            self.static, self.flow, self.cfg)

    def introduce_mover(self):
        self.obs[:, 8:24, 8:24] = 0.1
        self.obs_z[:, 8:24, 8:24] = 1.5
        self.flow[8:24, 8:24] = 3.2

    def test_perfect_static_returns_baseline(self):
        r = self.gate()
        self.assertFalse(r["applied"])
        self.assertEqual(r["reason"], "no_joint_candidates")
        self.assertTrue(bool(torch.all(r["weights"] == 1)))

    def test_matching_cues_downweight_candidates(self):
        self.introduce_mover()
        r = self.gate()
        self.assertTrue(r["applied"])
        self.assertLess(float(r["weights"].min()), 0.8)
        self.assertEqual(float(r["weights"][3, 3]), 1)

    def test_color_only_is_not_movement(self):
        self.obs[:, 8:24, 8:24] = 0.1
        self.assertFalse(self.gate()["applied"])

    def test_flow_only_is_not_movement(self):
        self.flow[8:24, 8:24] = 3.2
        self.assertFalse(self.gate()["applied"])

    def test_depth_behind_map_is_not_front_occlusion(self):
        self.introduce_mover()
        self.obs_z[:, 8:24, 8:24] = 2.6
        self.assertFalse(self.gate()["applied"])

    def test_missing_flow_fallback(self):
        self.introduce_mover()
        r = compute_flow_rgbd_gate_v2(
            self.rgb, self.z, self.alpha, self.obs, self.obs_z,
            self.static, None, {})
        self.assertEqual(r["reason"], "missing_flow")
        self.assertFalse(r["applied"])

    def test_low_static_coverage_fallback(self):
        self.introduce_mover()
        self.alpha.zero_()
        r = self.gate()
        self.assertEqual(r["reason"], "low_coverage")
        self.assertTrue(bool(torch.all(r["weights"] == 1)))

    def test_preserves_hard_static_mask(self):
        self.introduce_mover()
        self.static[8:24, 8:24] = False
        r = self.gate()
        self.assertFalse(r["applied"])
        self.assertTrue(bool(torch.all(r["weights"] == 1)))

    def test_fraction_budget(self):
        self.introduce_mover()
        self.cfg["max_downweight_fraction"] = 0.02
        r = self.gate()
        self.assertLessEqual(int((r["weights"] < 1).sum()),
                             int(r["valid"].sum()) * 0.02)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            compute_flow_rgbd_gate_v2(
                self.rgb, self.z, self.alpha, self.obs, self.obs_z,
                self.static, self.flow[:10], {})

    def test_no_gradient_through_gate(self):
        self.introduce_mover()
        x = self.rgb.clone().requires_grad_()
        r = compute_flow_rgbd_gate_v2(
            x, self.z, self.alpha, self.obs, self.obs_z,
            self.static, self.flow, {})
        self.assertFalse(r["weights"].requires_grad)
        ((x - self.obs).abs() * r["weights"]).mean().backward()
        self.assertIsNotNone(x.grad)


class WeightedLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.image = torch.rand((3, 16, 16), requires_grad=True)
        self.depth = (torch.rand((1, 16, 16)) * 2 + 1).requires_grad_()
        self.alpha_map = torch.ones((1, 16, 16))
        self.cam = SimpleNamespace(
            exposure_a=torch.zeros(()), exposure_b=torch.zeros(()),
            original_image=torch.rand((3, 16, 16)),
            depth=np.random.RandomState(2).uniform(1, 2, (16, 16)).astype("float32"),
            motion_mask=torch.ones((16, 16), dtype=torch.bool),
            grad_mask=torch.ones((1, 16, 16)), uid=10)
        self.config = {"Training": {"alpha": 0.9, "rgb_boundary_threshold": 0.01, "monocular": False}}

    def reference_original_loss(self):
        mask = ((self.cam.original_image.sum(0, keepdim=True) > .01).float()
                * self.cam.grad_mask * self.cam.motion_mask[None])
        z = torch.as_tensor(self.cam.depth)[None]
        zmask = ((z > 0.01) & (z < 1000) & (self.alpha_map > .95)
                 & self.cam.motion_mask[None])
        rgb = (self.alpha_map * (self.image - self.cam.original_image).abs() * mask).mean()
        dep = ((self.depth - z).abs() * zmask).mean()
        return .9 * rgb + .1 * dep

    def test_ones_match_original_loss(self):
        original = self.reference_original_loss()
        actual = tracking_loss_v2(
            self.config, self.image, self.depth, self.alpha_map,
            self.cam, torch.ones((16, 16)))
        self.assertTrue(torch.allclose(original, actual, rtol=1e-6, atol=1e-6))

    def test_uniform_scaling_does_not_change_loss(self):
        low = tracking_loss_v2(
            self.config, self.image, self.depth, self.alpha_map,
            self.cam, torch.full((16, 16), 0.7))
        high = tracking_loss_v2(
            self.config, self.image, self.depth, self.alpha_map,
            self.cam, torch.ones((16, 16)))
        self.assertTrue(torch.allclose(low, high, rtol=1e-6, atol=1e-6))

    def test_gradients_survive_and_weights_detached(self):
        weights = torch.ones((16, 16), requires_grad=True)
        loss = tracking_loss_v2(
            self.config, self.image, self.depth, self.alpha_map,
            self.cam, weights)
        loss.backward()
        self.assertIsNotNone(self.image.grad)
        self.assertIsNotNone(self.depth.grad)
        self.assertIsNone(weights.grad)

    def test_zero_static_returns_finite_zero(self):
        self.cam.motion_mask.zero_()
        loss = tracking_loss_v2(
            self.config, self.image, self.depth, self.alpha_map,
            self.cam, torch.ones((16, 16)))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
