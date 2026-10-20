import unittest
import torch
from utils.camera_motion_v31 import fit_geometry_irls, interaction_rows

class GeometryIRLSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        H,W=96,128
        self.d=torch.full((H,W),1.8)
        self.d+=(torch.rand(H,W)-.5)*.4
        self.K=torch.tensor([[220.,0.,64.],[0.,220.,48.],[0.,0.,1.]])
        ys,xs=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
        L=interaction_rows(self.d,self.K,ys.flatten(),xs.flatten())
        self.true=torch.tensor([.002,-.003,.0015,.0012,-.0008,.002])
        self.flow=torch.einsum('nij,j->ni',L,self.true).reshape(H,W,2).permute(2,0,1)
        self.valid=torch.ones((H,W),dtype=torch.bool)
        self.cfg=dict(sampling_stride=2,iterations=8,min_points=200,max_condition=1e10)
    def fit(self): return fit_geometry_irls(self.d,self.flow,self.K,self.valid,self.cfg)
    def test_exact_motion(self):
        fit,diag=self.fit()
        self.assertIsNotNone(fit);self.assertEqual(diag['status'],'ok')
        self.assertTrue(torch.allclose(fit,self.true,atol=1e-4))
    def test_shared_vector_outlier_robustness(self):
        self.flow[:,20:55,26:73]+=torch.tensor([6.,-9.])[:,None,None]
        fit,diag=self.fit()
        self.assertIsNotNone(fit)
        self.assertLess(float(torch.norm(fit-self.true)),.01)
    def test_low_support_fallback(self):
        self.valid[:]=False;self.valid[:6,:6]=True
        fit,d=self.fit();self.assertIsNone(fit);self.assertEqual(d['status'],'low_points')
    def test_bad_geometry(self):
        with self.assertRaises(ValueError):
            fit_geometry_irls(self.d,self.flow[:,:10,:],self.K,self.valid,self.cfg)
    def test_low_grid_coverage(self):
        self.valid[:]=False;self.valid[:32,:32]=True
        self.cfg.update(min_points=25,min_cells=30,sampling_stride=2)
        fit,d=self.fit();self.assertIsNone(fit);self.assertEqual(d['status'],'low_grid_coverage')
    def test_mask_excludes_invalid_depth(self):
        self.d[:]=float('nan')
        fit,d=self.fit();self.assertIsNone(fit);self.assertEqual(d['status'],'low_points')
    def test_flow_discontinuities(self):
        self.flow[:,15:35,15:35]+=10
        _,d=self.fit();self.assertGreater(d['flow_edge_fraction'],0)
    def test_depth_edges(self):
        self.d[20:40,20:40]+=.6
        _,d=self.fit();self.assertGreater(d['depth_edge_fraction'],0)
    def test_grid_balance_off(self):
        self.cfg['grid_balance']=0
        fit,d=self.fit();self.assertEqual(d['status'],'ok');self.assertIsNotNone(fit)
    def test_illconditioned_fallback(self):
        self.cfg['max_condition']=2
        fit,d=self.fit();self.assertIsNone(fit);self.assertEqual(d['status'],'ill_conditioned')
    def test_no_grad(self):
        self.flow.requires_grad_()
        fit,d=self.fit();self.assertIsNotNone(fit);self.assertFalse(fit.requires_grad)
    def test_bad_weight(self):
        self.cfg['depth_edge_weight']=-.1
        with self.assertRaises(ValueError):self.fit()
if __name__=='__main__':unittest.main()
