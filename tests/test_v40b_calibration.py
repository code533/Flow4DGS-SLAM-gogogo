import tempfile
import unittest
import numpy as np
import torch
from pathlib import Path
from utils.v40b_calibration import (
    normalized_tracking_flow_to_native_px, depth_boundary_mask,
    visibility_proxies, summarize_calibration)


class TestV40b(unittest.TestCase):
    def test_native_scaling(self):
        h, w = 10, 20
        raw = torch.zeros(h, w, 2)
        raw[..., 0] = 8.0
        raw[..., 1] = -4.0
        normalized = raw * raw.new_tensor([2.0/w, 2.0/h])
        recovered = normalized_tracking_flow_to_native_px(normalized, h, w)
        self.assertTrue(torch.allclose(raw, recovered, atol=1e-6))
        legacy = normalized * normalized.new_tensor([(w-1)/2, (h-1)/2])
        self.assertGreater((legacy - recovered).abs().max().item(), 0.0)

    def test_depth_step(self):
        depth = torch.ones(10, 12)
        depth[:, 6:] = 2
        edge = depth_boundary_mask(depth, .08, .06, 0)
        self.assertTrue(edge[:, 5:7].all().item())
        self.assertFalse(edge[:, 1:3].any().item())

    def test_visibility_static_and_out_of_bounds(self):
        d = torch.ones(8, 9)
        f = torch.zeros(8, 9, 2)
        fb = torch.zeros(8, 9)
        valid = torch.ones(8, 9, dtype=torch.bool)
        a = visibility_proxies(d, d, f, fb, valid, dilate_edge=0)
        self.assertTrue(a["interior_correspondence"].all().item())
        f[..., 0] = 3.
        b = visibility_proxies(d, d, f, fb, valid, dilate_edge=0)
        self.assertTrue((~b["flow_in_bounds"][:, -3:]).all().item())
        self.assertTrue(b["possible_occlusion"][:, -3:].all().item())

    def test_groups_use_identical_interior_pixels(self):
        shape=(5, 6)
        z=np.zeros(shape)
        o=np.ones(shape)
        maps={
          "frame_id":np.array(10),
          "flow_full_residual_px":np.full(shape, 8.0),
          "flow_full_valid":np.ones(shape,bool),
          "fb_error_px":np.full(shape, .1),
          "fb_valid":np.ones(shape,bool),
          "prior_static":np.ones(shape,bool),
          "flow_dynamic":np.zeros(shape,bool),
          "pre_depth_valid":np.ones(shape,bool),
          "pre_map_covered":np.ones(shape,bool),
          "post_map_covered":np.ones(shape,bool),
          "pre_render_depth_valid":np.ones(shape,bool),
          "pre_rgb_residual":z,"post_rgb_residual":z,
          "pre_depth_raw":z,"post_depth_raw":z,
          "boundary_near":np.zeros(shape,bool),
          "possible_occlusion":np.zeros(shape,bool),
          "flow_in_bounds":np.ones(shape,bool),
          "interior_correspondence":np.ones(shape,bool),
        }
        maps["prior_static"][0,0] = False
        row=summarize_calibration(maps, 5.0, .5)
        self.assertEqual(row["prior_dynamic_n_common_interior"],1)
        self.assertEqual(row["static_candidate_n_common_interior"],29)
        self.assertEqual(row["prior_dynamic_low_fb_given_high_motion"],1.0)


if __name__=="__main__":
    unittest.main()
