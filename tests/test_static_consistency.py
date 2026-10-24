import unittest

import torch

from utils.static_consistency import (
    ConsistencyConfig, DYNAMIC, STATIC, UNKNOWN, compute_evidence, select_tracking_mask,
)


class StaticConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.rgb = torch.full((3, 4, 5), 0.5)
        self.z = torch.full((4, 5), 2.0)
        self.alpha = torch.ones((1, 4, 5))
        self.cfg = ConsistencyConfig(min_tracking_pixels=4)

    def evidence(self, observed=None, raw=None, alpha=None, rendered=None, **kwargs):
        return compute_evidence(
            self.rgb, self.z if observed is None else observed,
            self.rgb if rendered is None else rendered,
            self.z[None] if raw is None else raw,
            self.alpha if alpha is None else alpha, self.cfg, **kwargs,
        )

    def test_matching_surface_is_static(self):
        self.assertTrue((self.evidence()["state"] == STATIC).all())

    def test_accumulated_depth_is_normalized(self):
        alpha = self.alpha * 0.96
        evidence = self.evidence(raw=self.z[None] * alpha, alpha=alpha)
        self.assertTrue((evidence["state"] == STATIC).all())
        self.assertTrue(torch.allclose(evidence["depth_score"], torch.zeros_like(self.z)))

    def test_unmapped_surface_is_unknown_not_dynamic(self):
        evidence = self.evidence(alpha=self.alpha * 0.2)
        self.assertTrue((evidence["state"] == UNKNOWN).all())
        self.assertFalse(evidence["map_inconsistent"].any())

    def test_foreground_alone_is_not_confirmed_motion(self):
        evidence = self.evidence(observed=self.z - 1)
        self.assertTrue(evidence["foreground_candidate"].all())
        self.assertTrue((evidence["state"] == UNKNOWN).all())

    def test_motion_and_foreground_support_dynamic_candidate(self):
        ones = torch.ones_like(self.z, dtype=torch.bool)
        evidence = self.evidence(observed=self.z - 1, flow_dynamic=ones, flow_valid=ones)
        self.assertTrue((evidence["state"] == DYNAMIC).all())

    def test_revealed_background_stays_unknown_even_with_flow_outlier(self):
        ones = torch.ones_like(self.z, dtype=torch.bool)
        evidence = self.evidence(observed=self.z + 1, flow_dynamic=ones, flow_valid=ones)
        self.assertTrue(evidence["revealed_candidate"].all())
        self.assertTrue((evidence["state"] == UNKNOWN).all())

    def test_flow_map_conflict_stays_unknown(self):
        ones = torch.ones_like(self.z, dtype=torch.bool)
        evidence = self.evidence(flow_dynamic=ones, flow_valid=ones)
        self.assertTrue((evidence["state"] == UNKNOWN).all())

    def test_invalid_flow_is_not_motion_evidence(self):
        ones = torch.ones_like(self.z, dtype=torch.bool)
        evidence = self.evidence(flow_dynamic=ones, flow_valid=~ones)
        self.assertTrue((evidence["state"] == STATIC).all())

    def test_invalid_observations_never_become_static_or_dynamic(self):
        z = self.z.clone()
        z[0, :4] = torch.tensor([0.0, -1.0, float("nan"), float("inf")])
        evidence = self.evidence(observed=z)
        self.assertTrue((evidence["state"][0, :4] == UNKNOWN).all())
        self.assertTrue(torch.isnan(evidence["depth_score"][0, :4]).all())

    def test_nan_render_is_unknown(self):
        rendered = self.rgb.clone()
        rendered[:, 0, 0] = float("nan")
        self.assertEqual(self.evidence(rendered=rendered)["state"][0, 0], UNKNOWN)

    def test_photometric_outlier_excluded_without_dynamic_assignment(self):
        rendered = self.rgb.clone()
        rendered[:, 0, 0] = 1.0
        evidence = self.evidence(rendered=rendered)
        support = torch.ones_like(self.z, dtype=torch.bool)
        keep, info = select_tracking_mask(evidence, support, self.cfg)
        self.assertTrue(info["mask_applied"])
        self.assertFalse(keep[0, 0])
        self.assertEqual(evidence["state"][0, 0], UNKNOWN)
        self.assertTrue(support.all())  # no mutation of the baseline mask

    def test_global_mismatch_falls_back_to_baseline(self):
        support = torch.ones_like(self.z, dtype=torch.bool)
        keep, info = select_tracking_mask(self.evidence(observed=self.z + 1), support, self.cfg)
        self.assertIsNone(keep)
        self.assertFalse(info["mask_applied"])

    def test_minimum_relative_support_is_enforced(self):
        rendered = self.rgb.clone()
        rendered[:, :3, :] = 1.0
        cfg = ConsistencyConfig(min_tracking_pixels=1, min_tracking_fraction=0.5)
        keep, _ = select_tracking_mask(self.evidence(rendered=rendered), self.alpha[0].bool(), cfg)
        self.assertIsNone(keep)

    def test_outputs_are_detached(self):
        evidence = self.evidence(rendered=self.rgb.clone().requires_grad_())
        self.assertTrue(all(not value.requires_grad for value in evidence.values()))

    def test_bad_configuration_and_partial_flow_fail(self):
        with self.assertRaises(ValueError):
            ConsistencyConfig(opacity_min=0)
        with self.assertRaises(ValueError):
            ConsistencyConfig(depth_abs_tolerance=float("nan"))
        with self.assertRaises(ValueError):
            ConsistencyConfig(mode="typo")
        with self.assertRaises(ValueError):
            self.evidence(flow_dynamic=self.alpha[0].bool())


if __name__ == "__main__":
    unittest.main()
