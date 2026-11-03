import unittest
from types import SimpleNamespace
import numpy as np
import torch
from utils.v43a2_temporal import (
    validate_future_settings,capture_frozen_source,evaluate_frozen_future)
from utils.v43a_observation_consistency import HYPOTHESES

class TemporalTest(unittest.TestCase):
    def test_settings(self):
        validate_future_settings(True,[207,357],12)
        with self.assertRaises(ValueError):validate_future_settings(True,[],12)
        with self.assertRaises(ValueError):validate_future_settings(True,[207],0)

    def test_frozen_next_keyframe_paired(self):
        delta=torch.zeros((2,3))
        model=SimpleNamespace(
            init_deform="offset",dygs=torch.ones(2,dtype=torch.bool),
            _delta_xyz=[delta])
        # Clone with distinct Gaussian state.
        clone=lambda obj:SimpleNamespace(
            init_deform=obj.init_deform,dygs=obj.dygs.clone(),
            _delta_xyz=[t.clone() for t in obj._delta_xyz])
        candidates={
            "g0":torch.zeros((1,3)),
            "g3_support":torch.tensor([[.1,0.,0.]]),
            "g4":torch.tensor([[.2,0.,0.]])}
        current=[
            dict(hypothesis=h,dynamic_rgb_mae=.1,
                 dynamic_depth_penalized_mae_m=.2,dynamic_rgb_n=2,
                 dynamic_depth_n=2,dynamic_depth_coverage=1.)
            for h in HYPOTHESES]
        pending=capture_frozen_source(
            model,source_uid=22,prior_uid=17,event_index=4,
            prev_delta=delta,indices=torch.tensor([0]),
            candidates=candidates,current_rows=current,clone_fn=clone)
        cam=SimpleNamespace(uid=27,original_image=torch.ones(3,2,2),
            depth=np.ones((2,2),np.float32),
            motion_mask=torch.tensor([[True,False],[False,True]]))
        def fake_render(viewpoint,pc,pipe,bg,**kwargs):
            x=kwargs["dx"][0,0]
            return dict(render=torch.ones(3,2,2)-x,
                        depth=torch.ones(1,2,2)+x,
                        opacity=torch.ones(1,2,2))
        rows=evaluate_frozen_future(
            pending,cam,render_fn=fake_render,pipeline=None,
            background=None,max_frame_gap=12)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["future_keyframe_id"],27)
        self.assertEqual(rows[0]["temporal_model"],"zero_order_hold_offsets")
        rows_with_maps,future_maps=evaluate_frozen_future(
            pending,cam,render_fn=fake_render,pipeline=None,
            background=None,max_frame_gap=12,return_maps=True)
        self.assertEqual(rows_with_maps,rows)
        self.assertEqual(set(future_maps),set(HYPOTHESES))
        self.assertEqual(future_maps["g0"]["depth_penalized_abs"].shape,(2,2))
        torch.testing.assert_close(model._delta_xyz[0],delta)
        with self.assertRaises(ValueError):
            evaluate_frozen_future(pending,SimpleNamespace(uid=40),
                render_fn=fake_render,pipeline=None,background=None,max_frame_gap=12)

if __name__=="__main__":unittest.main()
