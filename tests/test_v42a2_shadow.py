import unittest
import torch
from utils.v42a2_propagation import shadow_g3_g4, g34_summary


class TestG34Shadow(unittest.TestCase):
    def setup_data(self):
        raw=torch.tensor([[.1,0.,0.],[.11,0.,0.],[.12,0.,0.],[.8,0.,0.]])
        q=torch.tensor([.9,.9,.9,.8])
        idx=torch.tensor([[0,1,2,3],[1,0,2,3],[2,1,0,3],[3,2,1,0]])
        d2=torch.tensor([[0.,.001,.002,.003],[0.,.001,.002,.003],
                         [0.,.001,.002,.003],[0.,.001,.002,.003]])
        g0=raw*.5
        g1=raw*.9
        return raw,q,idx,d2,g0,g1

    def test_invisible_point_uses_exact_g0(self):
        raw,q,idx,d2,g0,g1=self.setup_data()
        reason=torch.tensor([0,0,0,2])
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reason,
            min_trusted_neighbors=2,min_support=.1)
        torch.testing.assert_close(g3[3],g0[3],rtol=0,atol=0)
        torch.testing.assert_close(g4[3],g0[3],rtol=0,atol=0)
        self.assertTrue(diag["g3_fallback_visibility"][3])
        self.assertTrue(diag["g4_fallback_visibility"][3])

    def test_near_zero_q_is_valid_observation_not_invisible(self):
        raw,q,idx,d2,g0,g1=self.setup_data()
        reason=torch.tensor([8,0,0,0])
        q[0]=1e-9
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reason,
            min_trusted_neighbors=2,min_support=.1)
        torch.testing.assert_close(g3[0],g1[0],rtol=0,atol=0)
        self.assertTrue(diag["visible"][0])

    def test_low_confidence_target_uses_consensus(self):
        raw,q,idx,d2,g0,g1=self.setup_data()
        reason=torch.zeros(4,dtype=torch.int16)
        q[0]=.01
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reason,
            min_trusted_neighbors=2,min_support=.1,
            max_consensus_disagreement_m=1.)
        self.assertTrue(diag["g4_accepted"][0])
        self.assertLess(float(g4[0,0]),.2)
        self.assertLess(float(diag["consensus_dispersion_m"][0]),.3)

    def test_unreliable_multimodal_neighbors_fallback(self):
        raw,q,idx,d2,g0,g1=self.setup_data()
        reason=torch.zeros(4,dtype=torch.int16)
        raw[:]=torch.tensor([[0.,0.,0.],[.5,0.,0.],[-.5,0.,0.],[1.0,0.,0.]])
        q[0]=.01
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reason,
            min_trusted_neighbors=2,min_support=.1,
            max_consensus_disagreement_m=.01)
        self.assertFalse(diag["g4_accepted"][0])
        torch.testing.assert_close(g4[0],g0[0],rtol=0,atol=0)

    def test_gate_and_shape_checks(self):
        raw,q,idx,d2,g0,g1=self.setup_data()
        reason=torch.zeros(4,dtype=torch.int16)
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reason,
            min_trusted_neighbors=2,min_support=.1)
        row=g34_summary(212,"shadow",g0,g1,g3,g4,reason,diag)
        self.assertEqual(row["n_points"],4)
        self.assertEqual(g3.shape,raw.shape)
        with self.assertRaises(ValueError):
            shadow_g3_g4(raw,q,d2,idx,g0,g1,torch.zeros(3))


if __name__=="__main__":
    unittest.main()
