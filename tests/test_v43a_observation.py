import unittest
from types import SimpleNamespace
import numpy as np
import torch
from utils.v43a_observation_consistency import compose_offsets, score_rgbd, evaluate_hypotheses

class TestV43a(unittest.TestCase):
    def test_strict_gate_preserves_unselected_and_rejected(self):
        prev=torch.tensor([[1.,0,0],[2.,0,0],[3.,0,0]])
        idx=torch.tensor([0,2])
        new,passed=compose_offsets(prev,idx,torch.tensor([[.2,0,0],[.3,0,0]]))
        self.assertEqual(passed.tolist(),[True,False])
        torch.testing.assert_close(new,torch.tensor([[1.2,0,0],[2.,0,0],[3.,0,0]]))
        torch.testing.assert_close(prev,torch.tensor([[1.,0,0],[2.,0,0],[3.,0,0]]))
        with self.assertRaises(ValueError):
            compose_offsets(prev,torch.tensor([0,0]),torch.ones(2,3))

    def test_missing_depth_does_not_win_by_hiding(self):
        gt=torch.ones(3,2,2); z=torch.ones(2,2); dynamic=torch.tensor([[1,0],[1,0]],dtype=torch.bool)
        good,_=score_rgbd(gt,z,torch.ones(2,2),gt,z,dynamic)
        absent,_=score_rgbd(gt,torch.zeros(2,2),torch.zeros(2,2),gt,z,dynamic)
        self.assertEqual(good["dynamic_depth_penalized_mae_m"],0.)
        self.assertAlmostEqual(absent["dynamic_depth_penalized_mae_m"],.5)
        self.assertEqual(good["dynamic_depth_coverage"],1.)
        self.assertEqual(absent["dynamic_depth_coverage"],0.)
        self.assertEqual(good["dynamic_depth_n"],absent["dynamic_depth_n"])

    def test_fixed_map_candidate_render_nonmutation(self):
        prev=torch.zeros(3,3); candidates={k:torch.full((2,3),i*.01)
            for i,k in enumerate(("g0","g3_support","g4"))}
        camera=SimpleNamespace(uid=22,original_image=torch.ones(3,2,2),depth=np.ones((2,2),dtype=np.float32),
                               motion_mask=torch.tensor([[True,False],[False,True]]))
        model=SimpleNamespace(init_deform="offset",dygs=torch.ones(3,dtype=torch.bool),_delta_xyz=[prev])
        seen=[]
        def mock_render(viewpoint,pc,pipe,bg,**kwargs):
            seen.append(kwargs["dx"].detach().clone())
            x=kwargs["dx"][0,0]
            return {"render":torch.ones(3,2,2)-x,"depth":torch.ones(1,2,2)+x,
                    "opacity":torch.ones(1,2,2)}
        rows,_=evaluate_hypotheses(frame_id=22,pc=model,viewpoint=camera,prior_keyframe_id=17,
            prev_delta=prev,selected_indices=torch.tensor([0,1]),candidates=candidates,
            render_fn=mock_render,pipeline=None,background=None)
        self.assertEqual([r["hypothesis"] for r in rows],["g0","g3_support","g4"])
        self.assertEqual(rows[0]["dynamic_rgb_n"],2)
        self.assertEqual(rows[0]["dynamic_depth_n"],2)
        self.assertGreater(rows[2]["dynamic_rgb_mae"],rows[1]["dynamic_rgb_mae"])
        torch.testing.assert_close(prev,torch.zeros_like(prev))
        self.assertEqual(len(seen),3)

if __name__=="__main__":unittest.main()
