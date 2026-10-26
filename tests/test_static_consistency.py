import unittest

import torch

from utils.static_consistency import (
    ConsistencyConfig, DYNAMIC, STATIC, UNKNOWN, compute_evidence, select_tracking_mask,
    tracking_supports, tracking_support_statistics, save_tracking_decision, select_tracking_weights,
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

    def test_definition_switch_changes_only_selected_evidence(self):
        raw = torch.full_like(self.z[None], 3.0)
        observed = torch.full_like(self.z, 3.0)
        alpha = self.alpha * 0.96
        norm = self.evidence(raw=raw, observed=observed, alpha=alpha)
        self.cfg = ConsistencyConfig(depth_definition="accumulated")
        acc = self.evidence(raw=raw, observed=observed, alpha=alpha)
        self.assertTrue(norm["map_inconsistent"].all())
        self.assertFalse(acc["map_inconsistent"].any())
        self.assertTrue(torch.equal(norm["map_valid"], acc["map_valid"]))
        for key in ["observed_depth", "accumulated_depth", "normalized_depth",
                    "render_opacity", "signed_depth_accumulated", "signed_depth_normalized"]:
            self.assertTrue(torch.equal(norm[key], acc[key]), key)
        with self.assertRaises(ValueError):
            ConsistencyConfig(depth_definition="typo")

    def test_supports_match_distinct_loss_masks(self):
        grad = torch.ones_like(self.z, dtype=torch.bool)
        grad[0, 0] = False
        alpha = self.alpha.clone()
        alpha[0, 0, 1] = 0.5
        motion = torch.ones_like(grad)
        motion[0, 2] = False
        supports = tracking_supports(self.rgb, self.z, alpha, grad, motion, 0.01, 1)
        self.assertFalse(supports["rgb"][0, 0])
        self.assertTrue(supports["depth"][0, 0])
        self.assertTrue(supports["rgb"][0, 1])
        self.assertFalse(supports["depth"][0, 1])
        self.assertFalse(supports["union"][0, 2])
        self.assertEqual(int(supports["union"].sum()), 19)

    def test_fallback_reports_actual_retention(self):
        evidence = self.evidence(observed=self.z + 1)
        support = torch.ones_like(self.z, dtype=torch.bool)
        keep, info = select_tracking_mask(evidence, support, self.cfg)
        self.assertIsNone(keep)
        self.assertEqual(info["candidate_excluded_pixels"], 20)
        self.assertEqual(info["applied_excluded_pixels"], 0)
        stats = tracking_support_statistics(evidence, {"union": support}, keep)
        self.assertEqual(stats["union_applied_retained_pixels"], 20)
        self.assertEqual(stats["union_candidate_retained_pixels"], 0)
        empty = torch.zeros_like(support)
        _, info = select_tracking_mask(evidence, empty, self.cfg)
        self.assertIsNone(info["candidate_excluded_fraction"])
        self.assertEqual(info["fallback_reason"], "empty_support")

    def test_decision_files_capture_applied_and_fallback_masks(self):
        import tempfile
        import json
        import numpy as np
        from pathlib import Path
        cfg = ConsistencyConfig(mode="tracking", save_tracking_decisions=True,
                                save_every=10, min_tracking_pixels=1)
        rendered = self.rgb.clone()
        rendered[:, 0, 0] = 1.0
        evidence = self.evidence(rendered=rendered)
        original = {k: v.clone() for k, v in evidence.items()}
        support = torch.ones_like(self.z, dtype=torch.bool)
        supports = {k: support for k in ["rgb", "depth", "union"]}
        keep, info = select_tracking_mask(evidence, support, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            for frame, applied in [(10, keep), (11, None)]:
                local_info = dict(info, mask_applied=applied is not None)
                save_tracking_decision(tmp, frame, 10, evidence, supports, applied,
                                       local_info, cfg, self.rgb, torch.eye(3), torch.zeros(3))
            with np.load(Path(tmp)/"000010.npz", allow_pickle=False) as data:
                self.assertEqual(data["union_applied_excluded_mask"].sum(), 1)
                self.assertTrue(np.array_equal(data["union_applied_excluded_mask"],
                                               data["union_candidate_excluded_mask"]))
                self.assertIn("signed_depth_accumulated", data.files)
                self.assertIn("observed_rgb", data.files)
                self.assertEqual(data["mask_iteration"], 10)
            with np.load(Path(tmp)/"000011.npz", allow_pickle=False) as data:
                self.assertTrue(data["applied_keep_mask"].all())
                self.assertEqual(data["union_applied_excluded_mask"].sum(), 0)
                self.assertEqual(data["union_candidate_excluded_mask"].sum(), 1)
                self.assertNotIn("observed_rgb", data.files)
            rows = [json.loads(l) for l in (Path(tmp)/"summary.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["evaluation_pose"], "before_mask_iteration_update")
            disabled = Path(tmp)/"disabled"
            save_tracking_decision(disabled, 10, 10, evidence, supports, keep, info,
                                   ConsistencyConfig(), self.rgb, torch.eye(3), torch.zeros(3))
            self.assertFalse(disabled.exists())
        for k in evidence:
            self.assertTrue(torch.allclose(evidence[k], original[k], equal_nan=True), k)

    def test_soft_weights_endpoints_and_fallback(self):
        rendered = self.rgb.clone()
        rendered[:, 0, 0] = 1.0
        evidence = self.evidence(rendered=rendered)
        supports = {k: torch.ones_like(self.z, dtype=torch.bool) for k in ["rgb", "depth", "union"]}
        for weight in [0.0, 0.5, 1.0]:
            cfg = ConsistencyConfig(inconsistent_weight=weight, min_tracking_pixels=1)
            weights, info = select_tracking_weights(evidence, supports, cfg)
            self.assertEqual(weights[0, 0].item(), weight)
            self.assertEqual(weights[1, 0].item(), 1)
            self.assertEqual(info["applied_excluded_pixels"], int(weight == 0))
            self.assertEqual(info["union_weight_sum"], 19 + weight)
            self.assertEqual(info["mask_applied"], weight < 1)
            weights, info = select_tracking_weights(self.evidence(observed=self.z+1), supports, cfg)
            self.assertTrue(weights.eq(1).all())
            self.assertTrue(info["fallback"])
        for weight in [-1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                ConsistencyConfig(inconsistent_weight=weight)

    def test_existing_rgbd_loss_weights_linearly_and_preserves_gradient(self):
        # Execute the actual loss function definitions on CPU, without importing
        # optional plotting dependencies or changing production CUDA handling.
        import ast
        import tempfile
        import os
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        source = Path(__file__).resolve().parents[1] / "utils/slam_utils.py"
        module = ast.parse(source.read_text())
        funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)
                 and n.name in ["get_loss_tracking_rgb", "get_loss_tracking_rgbd"]]
        scope = {"torch": torch, "os": os}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), scope)
        camera = SimpleNamespace(original_image=self.rgb, depth=self.z.numpy(),
                                 grad_mask=torch.ones_like(self.alpha, dtype=torch.bool),
                                 motion_mask=torch.ones_like(self.z, dtype=torch.bool), uid=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.Tensor, "cuda", lambda x: x):
            cfg = {"Results": {"save_dir": tmp}, "Training": {"alpha": 0.9, "rgb_boundary_threshold": 0.01}}
            losses, grads = [], []
            for weight in [0.0, 0.5, 1.0]:
                image = (self.rgb + .2).requires_grad_()
                depth = (self.z[None] + .3).requires_grad_()
                loss = scope["get_loss_tracking_rgbd"](cfg, image, depth, self.alpha,
                    camera, rm_dynamic=True, mask=torch.full_like(self.z, weight))
                loss.backward()
                losses.append(loss.item()); grads.append(image.grad.clone())
            self.assertEqual(losses[0], 0)
            self.assertAlmostEqual(losses[1], losses[2] * .5, places=6)
            self.assertTrue(torch.allclose(grads[1], grads[2] * .5))

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
