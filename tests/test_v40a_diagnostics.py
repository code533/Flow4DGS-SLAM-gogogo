import tempfile
import unittest
import numpy as np
import torch
from pathlib import Path
from utils.v40a_diagnostics import (
    full_flow_residual_px, assemble_frame, analyze_frame, save_frame)


class TestV40a(unittest.TestCase):
    def test_full_flow_uses_prior_dynamic_and_nan(self):
        observed = torch.ones(2, 3, 4)
        rigid = torch.zeros_like(observed)
        depth = torch.ones(3, 4)
        depth[0, 0] = 0
        residual, valid = full_flow_residual_px(observed, rigid, depth)
        self.assertFalse(valid[0, 0])
        self.assertTrue(torch.isnan(residual[0, 0]))
        self.assertAlmostEqual(residual[1, 2].item(), 2**0.5, places=5)

    def test_common_region_excludes_new_map_coverage(self):
        h, w = 3, 4
        snap = lambda: dict(rgb_residual=np.full((h,w),0.2,np.float32),
                            depth_residual=np.full((h,w),0.1,np.float32),
                            map_covered=np.ones((h,w),bool),
                            depth_valid=np.ones((h,w),bool),
                            opacity=np.ones((h,w),np.float32))
        pre, post = snap(), snap()
        pre["map_covered"][0, 0] = False
        post["rgb_residual"][0, 0] = 20
        post["rgb_residual"][1, 1] = .1
        common = np.ones((h,w),bool)
        f = np.ones((h,w),np.float32)
        maps = assemble_frame(30, pre, post, f, common, f, common, 
                              np.zeros((h,w),bool), common,
                              np.ones((h,w),np.float32),
                              np.ones((h,w),np.float32),
                              np.ones((h,w),np.float32),
                              np.ones((h,w,3),np.uint8),
                              np.ones((h,w,3),np.uint8))
        row = analyze_frame(maps)
        self.assertAlmostEqual(row["common_valid_fraction"],11/12)
        self.assertLess(row["rgb_post_common"],0.2)
        with tempfile.TemporaryDirectory() as folder:
            save_frame(folder, maps)
            self.assertTrue((Path(folder)/"v40a_00030.npz").exists())
            self.assertTrue((Path(folder)/"v40a_per_frame.csv").exists())

    def test_uncovered_depth_not_trusted_as_static(self):
        h,w = 2,3
        snap=lambda: dict(rgb_residual=np.ones((h,w)),depth_residual=np.ones((h,w)),
                          map_covered=np.zeros((h,w),bool),
                          depth_valid=np.ones((h,w),bool),
                          opacity=np.zeros((h,w)))
        maps=assemble_frame(10,snap(),snap(),
                            np.ones((h,w)),np.ones((h,w),bool),
                            np.zeros((h,w)),np.ones((h,w),bool),
                            np.zeros((h,w),bool),np.ones((h,w),bool),
                            np.ones((h,w)),np.zeros((h,w)),np.zeros((h,w)),
                            np.zeros((h,w,3),np.uint8),np.zeros((h,w,3),np.uint8))
        self.assertTrue(np.isnan(analyze_frame(maps)["depth_raw_uncovered"]))
        self.assertEqual(analyze_frame(maps)["common_valid_fraction"],0)


if __name__ == "__main__":
    unittest.main()
