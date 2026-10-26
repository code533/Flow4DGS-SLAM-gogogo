import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from utils.v40_diagnostics import flow_cycle_error_px, summarize_maps, save_frame


class TestV40(unittest.TestCase):
    def test_flow_cycle(self):
        h, w = 6, 7
        back = torch.zeros(h, w, 2)
        fwd = torch.zeros_like(back)
        back[..., 0] = 1
        fwd[..., 0] = -1
        error, valid = flow_cycle_error_px(back, fwd)
        self.assertTrue(torch.all(error[valid] < 1e-5))
        self.assertFalse(valid[:, -1].any())

    def test_masks_and_nan(self):
        h, w = 3, 4
        s = {"rgb_residual": np.ones((h, w), dtype=np.float32),
             "depth_residual": np.ones((h, w), dtype=np.float32),
             "map_covered": np.ones((h, w), bool),
             "depth_valid": np.ones((h, w), bool),
             "opacity": np.ones((h, w), dtype=np.float32)}
        f = np.ones((h, w), dtype=np.float32)
        flow_dynamic = np.zeros((h,w), bool)
        flow_dynamic[1, 1] = True
        prior = np.ones((h,w), bool)
        final = prior & ~flow_dynamic
        with tempfile.TemporaryDirectory() as tmp:
            row = save_frame(tmp, 2, s, s, f, flow_dynamic, np.ones((h,w),bool),
                             prior, final)
            self.assertAlmostEqual(row["final_dynamic_fraction"], 1.0/(h*w))
            self.assertTrue((Path(tmp) / "v40_00002.npz").exists())
            self.assertTrue((Path(tmp) / "v40_per_frame.csv").exists())
            self.assertTrue(np.isnan(row["fb_median_px"]))

    def test_empty_depth(self):
        h, w = 4, 4
        s = {"rgb_residual": np.zeros((h,w)), "depth_residual": np.full((h,w), np.nan),
             "map_covered": np.zeros((h,w), bool), "depth_valid": np.zeros((h,w), bool)}
        m = {"pre_"+k:v for k,v in s.items()}
        m.update({"post_"+k:v for k,v in s.items()})
        m.update(flow_residual_px=np.full((h,w), np.nan),
                 flow_dynamic=np.zeros((h,w),bool), flow_valid=np.zeros((h,w),bool),
                 prior_static=np.ones((h,w),bool), final_static=np.ones((h,w),bool))
        row = summarize_maps(1,m)
        self.assertTrue(np.isnan(row["depth_pre_mean_covered"]))


if __name__ == "__main__":
    unittest.main()
