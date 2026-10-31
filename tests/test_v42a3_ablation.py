import unittest
import torch
from utils.v42a2_propagation import shadow_g3_g4
from utils.v42a3_ablation import decompose_g4,summarize_ablation,STAGES

class TestV42a3(unittest.TestCase):
    def inputs(self):
        raw=torch.tensor([[.04,0.,0.],[.08,0.,0.],[.12,0.,0.],[.40,0.,0.]],dtype=torch.float32)
        q=torch.tensor([.05,.9,.9,.9])
        idx=torch.tensor([[0,1,2,3],[1,0,2,3],[2,1,0,3],[3,2,1,0]])
        d2=torch.tensor([[0.,.001,.002,.003]]*4)
        g0=raw*.5
        g1=raw*.8
        reasons=torch.zeros(4,dtype=torch.int16)
        return raw,q,idx,d2,g0,g1,reasons

    def compute(self,**options):
        raw,q,idx,d2,g0,g1,reasons=self.inputs()
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reasons,
            min_trusted_neighbors=2,min_support=.05,
            max_consensus_disagreement_m=options.get("dispersion",.12))
        stages,gates,extra=decompose_g4(raw,q,d2,idx,reasons,g0,g3,g4,
            {**diag,"g0":g0},trusted_q=.3)
        return raw,g0,g3,g4,stages,gates,extra

    def test_reconstruct_full_exactly(self):
        _,g0,g3,g4,s,g,e=self.compute()
        torch.testing.assert_close(s["g4"],g4,rtol=0,atol=0)
        torch.testing.assert_close(s["g3"],g3,rtol=0,atol=0)
        self.assertTrue(torch.all(g["g4"]<=g["soft_no_hard_dispersion"]))
        self.assertTrue(torch.all(g["soft_no_hard_dispersion"]<=g["support_only"]))

    def test_hard_dispersion_only_changes_acceptance(self):
        _,g0,g3,g4,s,g,e=self.compute(dispersion=.001)
        self.assertTrue((g["soft_no_hard_dispersion"] & ~g["g4"]).any())
        rejected=g["soft_no_hard_dispersion"] & ~g["g4"]
        torch.testing.assert_close(s["g4"][rejected],g0[rejected],rtol=0,atol=0)

    def test_shared_soft_and_support_difference(self):
        _,g0,g3,g4,s,g,e=self.compute()
        rows=summarize_ablation(212,s,g,{"g0":g0},e)
        self.assertEqual([r["stage"] for r in rows],list(STAGES))
        self.assertEqual({r["n_points"] for r in rows},{4})
        self.assertGreaterEqual(rows[1]["accepted_fraction"],rows[2]["accepted_fraction"])

    def test_invalid_shape(self):
        raw,q,idx,d2,g0,g1,reasons=self.inputs()
        g3,g4,diag=shadow_g3_g4(raw,q,d2,idx,g0,g1,reasons,
            min_trusted_neighbors=2,min_support=.05)
        with self.assertRaises(ValueError):
            decompose_g4(raw,q[:2],d2,idx,reasons,g0,g3,g4,diag)

if __name__=="__main__":unittest.main()
