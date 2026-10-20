import unittest
import torch
from utils.camera_motion_v30 import translation_step,rigid_transform,condition_number_normalized_jacobian
class V30Tests(unittest.TestCase):
    def test_legacy_amplification(self):
        out,r,s=translation_step(torch.tensor([.001,0.,0.]),.007,"legacy")
        self.assertAlmostEqual(float(out.norm()),.007,places=6)
        self.assertGreater(float(s),1.)
    def test_cap_small(self):
        x=torch.tensor([.001,0.,0.])
        out,_,s=translation_step(x,.007,"capped")
        self.assertTrue(torch.allclose(out,x,atol=1e-8));self.assertEqual(float(s),1.)
    def test_cap_large(self):
        o,_,s=translation_step(torch.tensor([.1,0.,0.]),.007,"capped")
        self.assertAlmostEqual(float(o.norm()),.007,places=6)
        self.assertLess(float(s),1.)
    def test_zero(self):
        o,_,_=translation_step(torch.zeros(3),.01,"capped")
        self.assertEqual(float(o.norm()),0.)
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):translation_step(torch.ones(3),1.,"bad")
    def test_negative_limit(self):
        with self.assertRaises(ValueError):translation_step(torch.ones(3),-1.)
    def test_finite(self):
        with self.assertRaises(ValueError):translation_step(torch.tensor([float("nan"),0.,0.]),1.)
    def test_grad_detach(self):
        o,_,_=translation_step(torch.ones(3,requires_grad=True),.5,"capped")
        self.assertFalse(o.requires_grad)
    def test_matrix(self):
        m=rigid_transform(torch.eye(3),torch.tensor([1.,2.,3.]))
        self.assertTrue(torch.equal(m[:3,3],torch.tensor([1.,2.,3.])))
    def test_condition_number(self):
        torch.manual_seed(6)
        z=1+torch.rand(48,64)
        K=torch.tensor([[525.,0.,32.],[0.,525.,24.],[0.,0.,1.]])
        c,n=condition_number_normalized_jacobian(z,K,torch.ones_like(z,dtype=torch.bool),stride=4)
        self.assertGreater(c,1.);self.assertEqual(n,192)
    def test_condition_empty(self):
        z=torch.ones(48,64);K=torch.eye(3)
        c,n=condition_number_normalized_jacobian(z,K,torch.zeros_like(z,dtype=torch.bool))
        self.assertEqual(c,float("inf"));self.assertEqual(n,0)
if __name__=="__main__":unittest.main()
