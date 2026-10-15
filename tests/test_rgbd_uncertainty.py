"""CPU regression tests: python -m unittest discover -s tests -p 'test_rgbd_uncertainty.py' -v"""
import unittest
import torch
from utils.uncertainty_fusion import compute_rgbd_uncertainty


class RGBDUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.pred = torch.full((3, 16, 16), 0.5, requires_grad=True)
        self.depth = torch.full((1, 16, 16), 2.0, requires_grad=True)
        self.alpha = torch.ones((1, 16, 16))
        self.rgb = torch.full((3, 16, 16), 0.5)
        self.z = torch.full((16, 16), 2.0)
        self.static = torch.ones((16, 16), dtype=torch.bool)

    def run_gate(self):
        return compute_rgbd_uncertainty(self.pred, self.depth, self.alpha,
                                       self.rgb, self.z, self.static,
                                       {"smoothing_kernel": 1, "min_coverage": 0.08})

    def test_static_match_has_full_weight(self):
        result = self.run_gate()
        self.assertTrue(result["applied"])
        self.assertGreater(float(result["weights"].min()), 0.98)

    def test_dynamic_like_residual_reduces_weight(self):
        self.rgb[:, 4:12, 4:12] = 0
        self.z[4:12, 4:12] = 1
        result = self.run_gate()
        self.assertLess(float(result["weights"][6, 6]), 0.2)
        self.assertGreater(float(result["weights"][1, 1]), 0.98)

    def test_uncovered_pixels_preserve_baseline(self):
        self.alpha[:, :8, :] = 0
        self.rgb[:, :8, :] = 0
        result = self.run_gate()
        self.assertEqual(float(result["weights"][2, 2]), 1.0)

    def test_original_motion_exclusions_are_not_changed(self):
        self.static[2:8, 2:8] = False
        self.rgb[:, 2:8, 2:8] = 0
        result = self.run_gate()
        self.assertEqual(float(result["weights"][4, 4]), 1.0)
        self.assertFalse(bool(result["new_dynamic_candidate"][4, 4]))

    def test_low_coverage_falls_back(self):
        self.alpha.zero_()
        result = self.run_gate()
        self.assertFalse(result["applied"])
        self.assertTrue(bool((result["weights"] == 1).all()))

    def test_invalid_depth_is_ignored(self):
        self.z[3:6, 3:6] = 0
        result = self.run_gate()
        self.assertFalse(bool(result["valid"][4, 4]))

    def test_gate_is_detached(self):
        self.rgb[:, 3:8, 3:8] = 0
        result = self.run_gate()
        self.assertFalse(result["weights"].requires_grad)
        ((self.pred - self.rgb).abs() * result["weights"]).mean().backward()
        self.assertIsNotNone(self.pred.grad)
        self.assertIsNone(self.depth.grad)


if __name__ == "__main__":
    unittest.main()
