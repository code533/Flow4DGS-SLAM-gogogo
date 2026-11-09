import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from utils.v51c_controlled_attribution import (
    compose_controlled,evaluate_controlled,should_capture,append_rows)
from scripts.analyze_v51c_attribution import frozen_summary,cross_run_confounding


class TestV51c(unittest.TestCase):
    def test_sampler(self):
        self.assertEqual([n for n in range(1,23)
                          if should_capture(n,10,12,0)],[1,11,21])
        self.assertFalse(should_capture(11,10,1,1))

    def test_exact_offset_gate_and_exclusion(self):
        prev=torch.zeros((3,3))
        idx=torch.tensor([0,1,2])
        legacy=torch.tensor([[.1,0,0],[.31,0,0],[.2,0,0]])
        strict_idx=torch.tensor([0,2])
        strict=torch.tensor([[.05,0,0],[.2,0,0]])
        state,counts=compose_controlled(prev,idx,legacy,strict_idx,strict)
        self.assertEqual(counts["legacy_gate_pass_n"],2)
        self.assertEqual(counts["strict_gate_pass_n"],2)
        self.assertAlmostEqual(float(state["legacy"][0,0]),.1,places=5)
        self.assertAlmostEqual(float(state["legacy"][1,0]),0.,places=5)
        self.assertAlmostEqual(float(state["strict"][0,0]),.05,places=5)
        torch.testing.assert_close(prev,torch.zeros_like(prev))

    def test_equal_map_render_without_mutation(self):
        h=w=16
        prev=torch.tensor([[.02,0,0],[.03,0,0.]])
        model=SimpleNamespace(init_deform="offset",
           dygs=torch.tensor([True,True]),
           kf_list=[5,10],
           _delta_xyz=[torch.zeros((2,3)),prev])
        model.get_delta_xyz=lambda ix:model._delta_xyz[ix]
        camera=SimpleNamespace(
            uid=15,original_image=torch.ones((3,h,w)),
            depth=np.ones((h,w),dtype=np.float32),
            motion_mask=torch.zeros((h,w),dtype=torch.bool))
        def renderer(cam,pc,pipe,bg,**kwargs):
            dx=kwargs["dx"]
            total=dx[:,0].sum()
            rgb=torch.ones((3,h,w))*total
            depth=torch.ones((1,h,w))*total
            return dict(render=rgb,depth=depth,opacity=torch.ones((1,h,w)))
        before=prev.clone()
        rows=evaluate_controlled(
            model=model,viewpoint=camera,prior_uid=10,frame_id=15,event_index=1,
            legacy_indices=torch.tensor([0,1]),
            legacy_g0=torch.tensor([[.1,0,0],[.1,0,0]]),
            strict_indices=torch.tensor([0]),
            strict_g0=torch.tensor([[.15,0,0]]),
            render_fn=renderer,pipeline=None,background=None)
        self.assertEqual([x["mode"] for x in rows],["hold","legacy","strict"])
        self.assertTrue(all(x["dynamic_depth_n"]==h*w for x in rows))
        self.assertEqual(rows[0]["n_excluded"],1)
        self.assertEqual(rows[0]["n_actual_moved_from_hold"],0)
        self.assertEqual(rows[1]["n_actual_moved_from_hold"],2)
        self.assertEqual(rows[2]["n_actual_moved_from_hold"],1)
        torch.testing.assert_close(model.get_delta_xyz(1),before)
        with tempfile.TemporaryDirectory() as tmp:
            append_rows(tmp,rows)
            df=pd.read_csv(Path(tmp)/"v51c_frozen_per_keyframe.csv")
            summary,hist=frozen_summary(Path(tmp)/"v51c_frozen_per_keyframe.csv")
            self.assertEqual(len(df),3)
            self.assertEqual(len(hist),1)
            self.assertEqual(len(summary),len(("dynamic_rgb_mae","dynamic_depth_penalized_mae_m",
              "interior_depth_penalized_mae_m","boundary_depth_penalized_mae_m",
              "dynamic_psnr_db"))*3)

    def test_confounding_counts_not_causal(self):
        df=pd.DataFrame([
            dict(frame_idx=0,frame_uid=5,is_keyframe_legacy=1,is_keyframe_strict=1,
                 delta_strict_minus_legacy_dynamic_depth_penalized_mae_m=.01,
                 delta_strict_minus_legacy_dynamic_rgb_mae=.02),
            dict(frame_idx=1,frame_uid=10,is_keyframe_legacy=1,is_keyframe_strict=0,
                 delta_strict_minus_legacy_dynamic_depth_penalized_mae_m=-.01,
                 delta_strict_minus_legacy_dynamic_rgb_mae=-.02)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"matched.csv"
            df.to_csv(path,index=False)
            res=cross_run_confounding(path)
            self.assertEqual(set(res.group),{"both_keyframe","legacy_only_keyframe"})


if __name__=="__main__":
    unittest.main()
