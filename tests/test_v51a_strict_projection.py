import tempfile
import unittest
from pathlib import Path

import torch
import pandas as pd

from utils.v51a_strict_projection import (
    strict_ndc_valid,strict_indices,compare_candidates,append_comparison,
    validate_mode,
)


class TestV51aStrictFOV(unittest.TestCase):
    def test_source_fov_and_clip_w(self):
        # In-range, outside, negative clip-w, nonfinite, behind.
        xyz=torch.tensor([[0.,0.,0.],[1.01,0.,0.],[0.,0.,0.],
                          [float("nan"),0.,0.],[-1.,1.,0.]])
        w=torch.tensor([1.,1.,-1.,1.,1.])
        self.assertEqual(strict_ndc_valid(xyz,clip_w=w).tolist(),
                         [True,False,False,False,True])

    def test_warped_fov_filters_source_and_target(self):
        source=strict_ndc_valid(torch.tensor([[0.,0.],[2.,0.],[0.,0.],[.9,0.]]))
        target=strict_ndc_valid(torch.tensor([[0.,0.],[0.,0.],[2.,0.],[.9,0.]]))
        ix=torch.tensor([3,0,1,2])
        selected,keep=strict_indices(ix,source,target)
        self.assertEqual(selected.tolist(),[3,0])
        self.assertEqual(keep.tolist(),[True,True,False,False])

    def test_candidate_compare_and_shadow_remains_read_only(self):
        src=torch.tensor([True,False,True,True])
        dest=torch.tensor([True,True,False,True])
        ix=torch.tensor([0,1,2,3])
        raw=torch.tensor([[.1,0,0],[.2,0,0],[.1,0,0],[.5,0,0.]])
        g0=torch.tensor([[.1,0,0],[.2,0,0],[.1,0,0],[.25,0,0.]])
        old=raw.clone()
        strict=torch.tensor([[.15,0,0],[.20,0,0]])
        row,keep,sel=compare_candidates(
            frame_id=357,prior_keyframe_id=352,mode="shadow",
            source_valid=src,target_valid=dest,
            legacy_indices=ix,legacy_raw=raw,legacy_g0=g0,
            strict_g0=strict)
        self.assertEqual(row["n_removed_candidates"],2)
        self.assertEqual(row["n_removed_that_pass_legacy_0p3m_gate"],2)
        self.assertEqual(row["n_common_gate_changed"],0)
        self.assertEqual(sel.tolist(),[0,3])
        self.assertAlmostEqual(row["shared_strict_vs_legacy_g0_mean_m"],.05,places=6)
        torch.testing.assert_close(raw,old)
        with tempfile.TemporaryDirectory() as td:
            append_comparison(td,row,details={"legacy_indices":ix})
            p=Path(td)
            self.assertTrue((p/"v51a_00357.npz").exists())
            self.assertEqual(len(pd.read_csv(p/"v51a_per_keyframe.csv")),1)

    def test_all_removed_and_guards(self):
        source=torch.tensor([False,False])
        target=torch.tensor([True,True])
        legacy=torch.tensor([0,1])
        idx,keep=strict_indices(legacy,source,target)
        self.assertEqual(len(idx),0)
        self.assertEqual(keep.tolist(),[False,False])
        row,_,_=compare_candidates(
            frame_id=1,prior_keyframe_id=0,mode="strict",
            source_valid=source,target_valid=target,legacy_indices=legacy,
            legacy_raw=torch.zeros((2,3)),legacy_g0=torch.zeros((2,3)),
            strict_g0=torch.zeros((0,3)))
        self.assertEqual(row["n_strict_candidates"],0)
        self.assertEqual(row["n_legacy_gate_pass"],2)
        self.assertEqual(row["n_removed_that_pass_legacy_0p3m_gate"],2)

    def test_mode_dependencies(self):
        validate_mode("off",online_mode="off",flow_ds=4,
                      use_knn=False,flow_offset=False,dynamic_model="mlp")
        validate_mode("shadow",online_mode="g0",flow_ds=1,
                      use_knn=True,flow_offset=True,dynamic_model="offset")
        with self.assertRaises(ValueError):
            validate_mode("strict",online_mode="g0",flow_ds=4,
                          use_knn=True,flow_offset=True,dynamic_model="offset")
        with self.assertRaises(ValueError):
            validate_mode("shadow",online_mode="g0",flow_ds=1,
                          use_knn=True,flow_offset=True,
                          dynamic_model="offset",v50_enabled=True)


if __name__=="__main__":
    unittest.main()
