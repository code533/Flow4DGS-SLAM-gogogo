"""Synthetic no-GPU checks; no claim of dynamic accuracy."""
import unittest
import numpy as np
from utils.v41a_shadow import ShadowConfig, infer_shadow, summarize_shadow


def fake_frame(h=6,w=8):
    sh=(h,w)
    f=lambda v:np.full(sh,v,np.float32)
    b=lambda v:np.full(sh,v,bool)
    return dict(
        frame_id=np.array(140), flow_full_residual_px=f(.1),
        flow_full_valid=b(True), fb_error_px=f(.1), fb_valid=b(True),
        pre_opacity=f(1.), pre_map_covered=b(True),
        pre_depth_valid=b(True), pre_render_depth_valid=b(True),
        pre_rgb_residual=f(.01), pre_depth_raw=f(.01),
        boundary_near=b(False), possible_occlusion=b(False),
        temporal_depth_support=b(True), prior_static=b(True),
        flow_dynamic=b(False), final_static=b(True))


class TestShadow(unittest.TestCase):
    def test_stable_static(self):
        m=fake_frame()
        o=infer_shadow(m)
        self.assertTrue(np.all(o["state_decoupled_3state"]==0))
        self.assertTrue(np.isfinite(o["p_dynamic"]).all())
        self.assertLess(float(o["p_dynamic"].mean()),.15)

    def test_fast_motion_with_good_correspondence_not_downweighted(self):
        m=fake_frame()
        m["flow_full_residual_px"][2:4,2:5] = 20
        m["fb_error_px"][2:4,2:5] = .01
        o=infer_shadow(m)
        roi=np.s_[2:4,2:5]
        self.assertGreater(float(o["q_flow"][roi].mean()), .9)
        self.assertGreater(float(o["motion_score"][roi].mean()), .95)
        self.assertGreater(float(o["p_dynamic"][roi].mean()), .6)

    def test_no_flow_or_map_becomes_unknown(self):
        m=fake_frame()
        m["flow_full_valid"][:,:]=False
        m["fb_valid"][:,:]=False
        m["pre_opacity"][:,:]=0
        m["pre_map_covered"][:,:]=False
        m["pre_render_depth_valid"][:,:]=False
        m["pre_depth_raw"][:,:]=np.nan
        o=infer_shadow(m)
        self.assertTrue(np.all(o["state_decoupled_3state"]==2))
        self.assertTrue(np.all(np.isfinite(o["p_dynamic"])))

    def test_map_not_used_without_support(self):
        m=fake_frame()
        m["pre_opacity"][:]=0
        m["pre_map_covered"][:]=False
        m["pre_render_depth_valid"][:]=False
        m["pre_depth_raw"][:]=1e4
        m["pre_rgb_residual"][:]=1e4
        o=infer_shadow(m)
        self.assertTrue(np.all(o["q_map"]==0))
        self.assertTrue(np.all(o["map_score"]==0))

    def test_legacy_masks_not_used_as_model_inputs(self):
        m=fake_frame()
        a=infer_shadow(m)
        m["prior_static"][:]=False
        m["flow_dynamic"][:]=True
        m["final_static"][:]=False
        b=infer_shadow(m)
        for key in ("p_dynamic","q_flow","q_map","state_decoupled_3state"):
            np.testing.assert_array_equal(a[key],b[key])
        row=summarize_shadow(m,b)
        self.assertIn("agreement_with_legacy_decided_NOT_accuracy", row)

    def test_all_modes_shapes_and_bounds(self):
        m=fake_frame()
        m["boundary_near"][0,0]=True
        m["possible_occlusion"][0,1]=True
        o=infer_shadow(m)
        for k in ("p_dynamic","q_flow","q_map","evidence"):
            self.assertEqual(o[k].shape,(6,8))
            self.assertGreaterEqual(float(o[k].min()),0)
            self.assertLessEqual(float(o[k].max()),1)
        for k in ("state_flow_only","state_naive_rgbd",
                  "state_reliability_binary","state_decoupled_3state"):
            self.assertTrue(set(np.unique(o[k])).issubset({0,1,2}))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ShadowConfig(fb_scale_px=0)
        with self.assertRaises(ValueError):
            ShadowConfig(static_threshold=.9,dynamic_threshold=.4)

if __name__=="__main__":
    unittest.main()
