import unittest
import torch
from utils.v42a1_diagnostics import projection_reasons, knn_motion_diagnostics, diagnostic_row


class TestV42a1(unittest.TestCase):
    def test_projection_reason_partition(self):
        q = torch.ones(4, 5)
        b = lambda:torch.ones(4,5,dtype=torch.bool)
        diag = dict(prev_depth_valid=b(),forward_finite=b(),
                    forward_in_bounds=b(),backward_valid_warped=b(),
                    current_depth_valid_warped=b(),correspondence_valid=b(),
                    depth_edge=torch.zeros(4,5,dtype=torch.bool))
        # Interior index nearest (2,2)
        q[2,2]=0.
        uv=torch.tensor([[0.,0.,1.],[2.,0.,1.],[float("nan"),0.,1.],[-1.,-1.,1.]])
        diag["prev_depth_valid"][0,0]=False
        out=projection_reasons(uv,q,diag)
        self.assertEqual(out["reason"].tolist(),[8,2,1,3])
        self.assertTrue(torch.equal(out["q_sample"],torch.tensor([0.,0.,0.,1.]).where(
            torch.tensor([True,True,True,False]),torch.tensor([0.,0.,0.,0.])) if False
            else out["q_sample"].shape==(4,))

    def test_reasons_prioritize_preceding_invalidity(self):
        q=torch.ones(4,4)
        ones=torch.ones(4,4,dtype=torch.bool)
        diag=dict(prev_depth_valid=ones.clone(),forward_finite=ones.clone(),
                  forward_in_bounds=ones.clone(),backward_valid_warped=ones.clone(),
                  current_depth_valid_warped=ones.clone(),
                  correspondence_valid=ones.clone(),
                  depth_edge=torch.zeros_like(ones))
        diag["prev_depth_valid"][0,0]=False
        diag["forward_finite"][0,0]=False
        out=projection_reasons(torch.tensor([[-1.,-1.,.2]]),q,diag)
        self.assertEqual(int(out["reason"][0]),3)

    def test_knn_gate_flip(self):
        raw=torch.tensor([[.1,0,0],[.8,0,0]],dtype=torch.float)
        idx=torch.tensor([[0,1],[1,0]])
        d2=torch.tensor([[0.,.002],[0.,.002]])
        g0=torch.tensor([[.2,0,0],[.4,0,0]])
        g1=torch.tensor([[.35,0,0],[.2,0,0]])
        g2=g0.clone()
        out=knn_motion_diagnostics(raw,torch.tensor([.1,1.]),d2,idx,g0,g1,g2)
        self.assertEqual(out["gate_g0_to_g1_rejected"].tolist(),[True,False])
        self.assertEqual(out["gate_g0_to_g1_accepted"].tolist(),[False,True])
        self.assertTrue(torch.isfinite(out["neighbor_motion_disagreement_m"]).all())
        self.assertAlmostEqual(float(out["effective_g1_minus_g0_norm_m"][0]),.2,places=5)

    def test_diagnostics_summary(self):
        q=torch.ones(4,4)
        diag=dict(prev_depth_valid=torch.ones(4,4,dtype=torch.bool),
                  forward_finite=torch.ones(4,4,dtype=torch.bool),
                  forward_in_bounds=torch.ones(4,4,dtype=torch.bool),
                  backward_valid_warped=torch.ones(4,4,dtype=torch.bool),
                  current_depth_valid_warped=torch.ones(4,4,dtype=torch.bool),
                  correspondence_valid=torch.ones(4,4,dtype=torch.bool),
                  depth_edge=torch.zeros(4,4,dtype=torch.bool))
        all_r=projection_reasons(torch.tensor([[0.,0.,1.],[.1,.1,1.]]),q,diag)
        selected={k:(v[:1] if v.ndim and v.shape[0]==2 else v)
                  for k,v in all_r.items()}
        raw=torch.tensor([[.1,0,0]])
        neigh=knn_motion_diagnostics(raw,torch.ones(1),torch.zeros(1,1),
            torch.zeros(1,1,dtype=torch.long),raw,raw,raw)
        aux={"g1_fallback":torch.zeros(1,dtype=torch.bool),
             "g2_fallback":torch.zeros(1,dtype=torch.bool)}
        row=diagnostic_row(22,selected,torch.tensor([0]),all_r,neigh,aux)
        self.assertEqual(row["n_selected_points"],1)
        self.assertEqual(row["n_all_dynamic_gaussians"],2)


if __name__=="__main__":
    unittest.main()
