"""V2.2 regression tests; run: python -m unittest discover -s tests -p test_flow_rgbd_fusion_v22.py -v"""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from utils.flow_rgbd_fusion_v22 import compute_decoupled_weights,tracking_loss_decoupled

class V22Test(unittest.TestCase):
    def setUp(self):
        self.flow=torch.full((24,24),.1)
        self.rgb=torch.full((24,24),.1)
        self.front=torch.full((24,24),.1)
        self.flow[2:16,2:16]=.9
        self.rgb[2:16,2:16]=.9
        self.front[2:16,2:16]=.9
        self.valid=torch.ones((24,24),dtype=torch.bool)
        self.cfg=dict(rgb_min_candidate_pixels=1,depth_min_candidate_pixels=1)
    def gate(self,**params):
        cfg=dict(self.cfg,**params)
        return compute_decoupled_weights(dict(valid=self.valid,p_flow=self.flow,p_rgb=self.rgb,
                                               p_front=self.front,coverage=.7),cfg)
    def test_both_branches(self):
        g=self.gate()
        self.assertTrue(g['applied'])
        self.assertLess(float(g['rgb_weights'].min()),1)
        self.assertLess(float(g['depth_weights'].min()),1)
    def test_rgb_only_cue(self):
        self.front[:]=.1
        g=self.gate()
        self.assertTrue(bool((g['depth_weights']==1).all()))
        self.assertLess(float(g['rgb_weights'].min()),1)
    def test_depth_only_cue(self):
        self.rgb[:]=.1
        g=self.gate()
        self.assertTrue(bool((g['rgb_weights']==1).all()))
        self.assertLess(float(g['depth_weights'].min()),1)
    def test_modes(self):
        for mode,k in [('rgb_only','depth_weights'),('depth_only','rgb_weights')]:
            self.assertTrue(bool((self.gate(mode=mode)[k]==1).all()))
    def test_all_ones_is_active_control(self):
        g=self.gate(mode='all_ones')
        self.assertTrue(g['applied'])
        self.assertTrue(bool((g['rgb_weights']==1).all()))
        self.assertTrue(bool((g['depth_weights']==1).all()))
    def test_debug_does_not_change_weights(self):
        g=self.gate(diagnostic_only=True)
        self.assertFalse(g['applied'])
        self.assertTrue(bool((g['rgb_weights']==1).all()))
    def test_original_motion_mask_protected(self):
        self.valid[2:16,2:16]=False
        self.assertFalse(self.gate()['applied'])
    def test_budget(self):
        g=self.gate(rgb_max_fraction=.01,depth_max_fraction=.02)
        self.assertLessEqual(g['rgb_selected_count'],int(.01*self.valid.sum()))
        self.assertLessEqual(g['depth_selected_count'],int(.02*self.valid.sum()))
    def test_bad_mode(self):
        with self.assertRaises(ValueError):self.gate(mode='invalid')

class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.im=torch.rand((3,12,12),requires_grad=True)
        self.z=torch.rand((1,12,12),requires_grad=True)+1
        self.op=torch.rand((1,12,12))*.12+.88
        self.camera=SimpleNamespace(exposure_a=torch.tensor(.1),exposure_b=torch.tensor(.02),
             original_image=torch.rand((3,12,12)),depth=np.ones((12,12),dtype='float32')*1.5,
             grad_mask=torch.randint(0,2,(1,12,12)).float(),
             motion_mask=torch.randint(0,2,(12,12)).bool(),uid=6)
        self.conf={'Training':{'monocular':False,'alpha':.9,'rgb_boundary_threshold':.01}}
        self.one=torch.ones((12,12))
    def loss(self,r,d,normalization='mass_preserving'):
        return tracking_loss_decoupled(self.conf,self.im,self.z,self.op,self.camera,r,d,normalization)
    def native(self):
        ob=self.camera.original_image
        pr=torch.exp(self.camera.exposure_a)*self.im+self.camera.exposure_b
        mask=self.camera.motion_mask[None]
        rm=(ob.sum(0,keepdim=True)>.01)*self.camera.grad_mask*mask
        zm=(self.op>.95)*mask
        return .9*(self.op*(pr-ob).abs()*rm).mean()+.1*((self.z-1.5).abs()*zm).mean()
    def test_unit_loss_and_gradient_matches_original(self):
        ref=self.native()
        loss=self.loss(self.one,self.one)
        self.assertTrue(torch.allclose(ref,loss,atol=1e-7,rtol=1e-6))
        g1=torch.autograd.grad(ref,self.im,retain_graph=True)[0]
        g2=torch.autograd.grad(loss,self.im)[0]
        self.assertTrue(torch.allclose(g1,g2,atol=1e-7,rtol=1e-6))
    def test_unit_original_mean(self):
        self.assertTrue(torch.allclose(self.native(),self.loss(self.one,self.one,'original_mean'),atol=1e-7))
    def test_rgb_weights_not_affect_depth_gradient(self):
        baseline=torch.autograd.grad(self.loss(self.one,self.one),self.z,retain_graph=True)[0]
        altered=torch.autograd.grad(self.loss(self.one*.7,self.one),self.z)[0]
        self.assertTrue(torch.allclose(baseline,altered,atol=1e-7))
    def test_depth_weights_not_affect_rgb_gradient(self):
        baseline=torch.autograd.grad(self.loss(self.one,self.one),self.im,retain_graph=True)[0]
        altered=torch.autograd.grad(self.loss(self.one,self.one*.7),self.im)[0]
        self.assertTrue(torch.allclose(baseline,altered,atol=1e-7))
    def test_uniform_weights_invariant(self):
        self.assertTrue(torch.allclose(self.loss(self.one,self.one),self.loss(self.one*.6,self.one*.8),atol=1e-7))
    def test_no_weight_grad(self):
        w=self.one.clone().requires_grad_()
        self.loss(w,w).backward()
        self.assertIsNone(w.grad)
    def test_bad_shapes(self):
        with self.assertRaises(ValueError):self.loss(torch.ones(7,7),self.one)
    def test_bad_normalization(self):
        with self.assertRaises(ValueError):self.loss(self.one,self.one,'bad')
if __name__=='__main__':unittest.main()
