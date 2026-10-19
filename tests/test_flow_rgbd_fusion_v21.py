import unittest
import torch
from utils.flow_rgbd_fusion_v21 import compute_flow_rgbd_gate_v21

class TestV21(unittest.TestCase):
    def setUp(self):
        self.pred=torch.full((3,32,32),0.5,requires_grad=True)
        self.obs=self.pred.detach().clone()
        self.pd=torch.full((1,32,32),2.0)
        self.od=self.pd.clone()
        self.opacity=torch.ones((1,32,32))
        self.static=torch.ones((32,32),dtype=torch.bool)
        self.flow=torch.full((32,32),0.2)
        self.cfg={"min_candidate_pixels":1}
    def gate(self):
        return compute_flow_rgbd_gate_v21(self.pred,self.pd,self.opacity,self.obs,self.od,self.static,self.flow,self.cfg)
    def motion(self):
        self.obs[:,8:24,8:24]=0.1
        self.od[:,8:24,8:24]=1.5
        self.flow[8:24,8:24]=3.2
    def test_static_baseline(self):
        r=self.gate()
        self.assertFalse(r["applied"])
        self.assertTrue(bool((r["weights"]==1).all()))
    def test_joint_cue(self):
        self.motion()
        r=self.gate()
        self.assertGreater(r["count_both_priors"],0)
        self.assertTrue(r["applied"])
        self.assertLess(float(r["weights"].min()),1.0)
    def test_rgb_only_not_dynamic(self):
        self.obs[:,8:24,8:24]=0.1
        self.assertEqual(self.gate()["count_raw"],0)
    def test_flow_only_not_dynamic(self):
        self.flow[8:24,8:24]=3.2
        self.assertEqual(self.gate()["count_raw"],0)
    def test_product_ablation(self):
        self.motion()
        self.cfg["fusion_mode"]="product"
        r=self.gate()
        self.assertEqual(r["mode"],"product")
        self.assertGreater(float(r["score_product"].max()),0.55)
    def test_diagnostic_only(self):
        self.motion()
        self.cfg["diagnostic_only"]=True
        r=self.gate()
        self.assertEqual(r["reason"],"diagnostic_only")
        self.assertTrue(bool((r["weights"]==1).all()))
        self.assertGreater(r["count_selected"],0)
    def test_min_candidates_protects_tracking(self):
        self.motion()
        self.cfg["min_candidate_pixels"]=500
        r=self.gate()
        self.assertFalse(r["applied"])
        self.assertTrue(bool((r["weights"]==1).all()))
    def test_budget(self):
        self.motion()
        self.cfg["max_downweight_fraction"]=0.01
        r=self.gate()
        self.assertLessEqual(r["count_selected"],int(r["valid"].sum().item())//100)
    def test_missing_flow(self):
        r=compute_flow_rgbd_gate_v21(self.pred,self.pd,self.opacity,self.obs,self.od,self.static,None,self.cfg)
        self.assertEqual(r["reason"],"missing_flow")
    def test_static_mask_protection(self):
        self.motion()
        self.static[8:24,8:24]=False
        r=self.gate()
        self.assertFalse(bool(r["candidate"].any()))
    def test_detached(self):
        self.motion()
        r=self.gate()
        self.assertFalse(r["weights"].requires_grad)
    def test_all_cues_saved(self):
        self.motion()
        r=self.gate()
        for key in ("p_flow","p_rgb","p_front","p_map","score_product","score_layered","score_selected"):
            self.assertEqual(tuple(r[key].shape),(32,32))
    def test_no_coverage(self):
        self.opacity.zero_()
        self.assertEqual(self.gate()["reason"],"low_coverage")
    def test_invalid_mode(self):
        self.cfg["fusion_mode"]="unknown"
        with self.assertRaises(ValueError): self.gate()
if __name__=="__main__": unittest.main()
