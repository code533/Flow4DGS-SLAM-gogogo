import unittest
import torch
from utils.camera_motion_v31a import (candidate_initial_pose, compare_flow_residuals,
                                       relative_pose_metrics)
from utils.camera_motion_v31 import interaction_rows


class ShadowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.H, self.W = 40, 48
        self.d = 1.5 + .4 * torch.rand(self.H, self.W)
        self.K = torch.tensor([[175., 0., 24.], [0., 173., 20.], [0., 0., 1.]])
        self.mask = torch.ones((self.H, self.W), dtype=torch.bool)
        y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        self.xi = torch.tensor([.005, -.003, .007, .001, -.001, .002])
        rows = interaction_rows(self.d, self.K, y.flatten(), x.flatten())
        self.flow = torch.einsum("nij,j->ni", rows, self.xi).reshape(
            self.H, self.W, 2).permute(2, 0, 1)

    def test_equal_twists_equal_residuals(self):
        a = compare_flow_residuals(self.d, self.flow, self.K, self.mask, self.xi, self.xi)
        self.assertEqual(a["n_common"], self.H * self.W)
        self.assertGreater(a["n_holdout"], 0)
        self.assertAlmostEqual(a["legacy_common_mean_px"], a["geo_common_mean_px"], places=8)

    def test_wrong_twist_has_higher_residual(self):
        other = self.xi.clone()
        other[0] += .01
        a = compare_flow_residuals(self.d, self.flow, self.K, self.mask, self.xi, other)
        self.assertLess(a["legacy_common_rmse_px"], a["geo_common_rmse_px"])
        self.assertLess(a["legacy_holdout_rmse_px"], a["geo_holdout_rmse_px"])

    def test_empty_mask(self):
        self.mask.zero_()
        a = compare_flow_residuals(self.d, self.flow, self.K, self.mask, self.xi, self.xi)
        self.assertEqual(a["n_common"], 0)

    def test_no_mutation(self):
        prev = torch.eye(4)
        xi = self.xi.clone()
        candidate_initial_pose(prev, xi, .01, .02)
        self.assertTrue(torch.equal(prev, torch.eye(4)))
        self.assertTrue(torch.equal(xi, self.xi))

    def test_zero_step(self):
        T = candidate_initial_pose(torch.eye(4), torch.zeros(6), .01, .02)
        self.assertTrue(torch.allclose(T, torch.eye(4), atol=1e-6))

    def test_capped_vs_legacy(self):
        xi = torch.tensor([.001, 0., 0., 0., 0., 0.])
        a = candidate_initial_pose(torch.eye(4), xi, .01, .1, "legacy")
        b = candidate_initial_pose(torch.eye(4), xi, .01, .1, "capped")
        self.assertGreater(torch.norm(a[:3, 3]), torch.norm(b[:3, 3]))

    def test_relative_equal_pose(self):
        a = candidate_initial_pose(torch.eye(4), self.xi, .01, .04)
        t, r = relative_pose_metrics(torch.eye(4), a, torch.eye(4), a)
        self.assertLess(t, 1e-9)
        self.assertLess(r, 1e-4)

    def test_relative_translation_error(self):
        bad = torch.eye(4)
        bad[:3, 3] = torch.tensor([.1, 0., 0.])
        t, r = relative_pose_metrics(torch.eye(4), bad, torch.eye(4), torch.eye(4))
        self.assertAlmostEqual(t, .1)
        self.assertLess(r, 1e-5)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            compare_flow_residuals(self.d, self.flow, self.K, self.mask, self.xi[:5], self.xi)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            candidate_initial_pose(torch.eye(4), self.xi, .01, .02, "unknown")

    def test_detached(self):
        xi = self.xi.clone().requires_grad_()
        T = candidate_initial_pose(torch.eye(4), xi, .01, .02)
        self.assertFalse(T.requires_grad)


if __name__ == "__main__":
    unittest.main()
