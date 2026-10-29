import unittest
import torch
from utils.v42a_propagation import (
    bidirectional_flow_reliability,gather_prior_gaussian_confidence,
    knn_candidates_from_precomputed)


class TestV42a(unittest.TestCase):
    def test_bidirectional_cycle_consistency_and_oob(self):
        h,w=8,12
        f=torch.zeros(h,w,2);b=torch.zeros_like(f)
        f[...,0]=2*2/w;b[...,0]=-2*2/w
        d=torch.ones(h,w)
        q,diag=bidirectional_flow_reliability(f,b,d,d,edge_penalty=1)
        self.assertTrue(torch.all(q[:,:-2]>.99))
        self.assertTrue(torch.all(q[:,-2:]==0))
        self.assertFalse(diag["correspondence_valid"][:,-2:].any())

    def test_projected_gaussian_confidence(self):
        q=torch.ones(6,8)
        q[0,0]=.3
        uv=torch.tensor([[-1.,-1.,.2],[0.,0.,.3],[2.,0.,.1]])
        result=gather_prior_gaussian_confidence(uv,q)
        self.assertAlmostEqual(float(result[0]),.3,places=5)
        self.assertAlmostEqual(float(result[1]),1.,places=5)
        self.assertEqual(float(result[2]),0.)

    def test_constant_confidence_preserves_uniformly_weighted_knn(self):
        raw=torch.tensor([[1.,0.,0.],[3.,0.,0.]])
        idx=torch.tensor([[0,1],[1,0]])
        d2=torch.tensor([[0.,.01],[0.,.01]])
        weights=torch.exp(-d2/(2*.05**2+1e-8))
        g0=(weights[...,None]*raw[idx]).sum(1)/(weights.sum(1)[:,None]+1e-8)
        g1,g2,aux=knn_candidates_from_precomputed(
            raw,torch.ones(2),d2,idx,g0,radius=.2,
            g2_min_neighbors=1,neighbor_q_min=.1)
        self.assertTrue(torch.allclose(g1,g0,atol=1e-5))
        self.assertTrue(torch.allclose(g2,g0,atol=1e-5))

    def test_zero_reliability_falls_back_to_legacy(self):
        raw=torch.tensor([[1.,0.,0.],[3.,0.,0.]])
        idx=torch.tensor([[0,1],[1,0]])
        d2=torch.tensor([[0.,.01],[0.,.01]])
        g0=raw*.5
        g1,g2,aux=knn_candidates_from_precomputed(
            raw,torch.zeros(2),d2,idx,g0,g2_min_neighbors=2)
        self.assertTrue(torch.equal(g1,g0))
        self.assertTrue(torch.equal(g2,g0))
        self.assertTrue(aux["g2_fallback"].all())

    def test_insufficient_support_uses_g0_in_g2(self):
        raw=torch.tensor([[1.,0.,0.],[3.,0.,0.]])
        idx=torch.tensor([[0,1],[1,0]])
        d2=torch.tensor([[0.,.01],[0.,.01]])
        g0=raw*.5
        g1,g2,aux=knn_candidates_from_precomputed(
            raw,torch.tensor([1.,0.]),d2,idx,g0,
            g2_min_support=.8,g2_min_neighbors=2)
        self.assertTrue(torch.equal(g2,g0))
        self.assertTrue(aux["g2_fallback"].all())


if __name__=="__main__":
    unittest.main()
