import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from utils.v52a_conditioned_history import (
    validate_history_cfg,predict_conditioned_history,extend_source,
    evaluate_four_future,append_source_rows,append_future_rows)
from utils.v51c_controlled_attribution import evaluate_controlled
from utils.v51d_temporal_persistence import capture_source,log_event
from scripts.analyze_v52a_history import pairs,analyze


def model():
    # Strict rejection index 2, coherent historical motion +0.01 per frame.
    t0=torch.zeros((4,3))
    t1=t0.clone()
    t2=t0.clone()
    t1[2,0]=.05
    t2[2,0]=.10
    m=SimpleNamespace(
        init_deform="offset",dygs=torch.tensor([True,True,True,True]),
        unique_kfIDs=torch.tensor([0,0,0,0]),kf_list=[0,5,10],
        _delta_xyz=[t0,t1,t2])
    m.get_delta_xyz=lambda ix:m._delta_xyz[ix]
    return m


def camera(uid):
    h=w=16
    mask=torch.ones((h,w),dtype=torch.bool)
    mask[2:14,2:14]=False
    return SimpleNamespace(uid=uid,
        original_image=torch.ones((3,h,w)),
        depth=np.ones((h,w),dtype=np.float32),
        motion_mask=mask)


def render(cam,pc,pipe,bg,**kwargs):
    z=kwargs["dx"][2,0]
    return dict(render=torch.ones((3,16,16))*(.8+z*.5),
                depth=torch.ones((1,16,16))*(.8+z*.5),
                opacity=torch.ones((1,16,16)))


def clone(o):
    m=model()
    m._delta_xyz=[x.clone() for x in o._delta_xyz]
    return m


class V52aTests(unittest.TestCase):
    def candidate(self,mod=None,**kw):
        m=mod or model()
        return predict_conditioned_history(
            model=m,source_uid=15,prior_uid=10,
            legacy_indices=torch.tensor([1,2]),
            legacy_g0=torch.tensor([[.03,0,0],[.04,0,0]]),
            strict_indices=torch.tensor([1]),
            strict_g0=torch.tensor([[.03,0,0]]),**kw)

    def test_history_only_on_fov_rejected_candidate(self):
        m=model()
        original=m._delta_xyz[-1].clone()
        predicted,stats=self.candidate()
        self.assertEqual(stats["n_fov_excluded"],1)
        self.assertEqual(stats["n_history_predicted"],1)
        self.assertAlmostEqual(float(predicted[2,0]),.15,places=5)
        self.assertAlmostEqual(float(predicted[1,0]),.03,places=5)
        self.assertAlmostEqual(float(predicted[0,0]),0,places=5)
        torch.testing.assert_close(original,m._delta_xyz[-1])

    def test_history_must_preexist_oldest_keyframe(self):
        m=model()
        m.unique_kfIDs[2]=5
        predicted,stats=self.candidate(mod=m)
        self.assertEqual(stats["n_history_predicted"],0)
        self.assertAlmostEqual(float(predicted[2,0]),.1,places=5)

    def test_direction_and_speed_checks(self):
        m=model()
        m._delta_xyz[0][2,0]=.2
        _,stats=self.candidate(mod=m)
        self.assertEqual(stats["n_direction_coherent"],0)
        self.assertEqual(stats["n_history_predicted"],0)
        m=model()
        predicted,stats=self.candidate(mod=m,max_predicted_increment_m=.025)
        self.assertEqual(stats["n_history_clipped"],1)
        self.assertAlmostEqual(float(predicted[2,0]),.125,places=5)
        with self.assertRaisesRegex(ValueError,"threshold"):
            validate_history_cfg(min_step_m=.2,max_speed_m_per_frame=.04,
               min_direction_cos=.5,max_speed_ratio=2.,
               max_predicted_increment_m=.1,history_frame_gap_cap=20)

    def test_four_way_snapshot_and_future_csv(self):
        m=model()
        old=m._delta_xyz[-1].clone()
        candidate,stats=self.candidate(mod=m)
        idx=torch.tensor([1,2])
        l=torch.tensor([[.03,0,0],[.04,0,0]])
        sidx=torch.tensor([1])
        smotion=torch.tensor([[.03,0,0]])
        source=evaluate_controlled(model=m,viewpoint=camera(15),
           prior_uid=10,frame_id=15,event_index=1,
           legacy_indices=idx,legacy_g0=l,strict_indices=sidx,
           strict_g0=smotion,render_fn=render,pipeline=None,background=None)
        pending=capture_source(model=m,source_uid=15,prior_uid=10,
           event_index=1,source_kind="systematic",
           legacy_indices=idx,legacy_g0=l,strict_indices=sidx,
           strict_g0=smotion,source_rows=source,clone_fn=clone)
        four=extend_source(pending=pending,
           source_viewpoint=camera(15),history_offset=candidate,
           diagnostics=stats,render_fn=render,pipeline=None,background=None)
        self.assertEqual([a["mode"] for a in four],
                         ["hold","legacy","strict","strict_history"])
        self.assertEqual(len(set(r["dynamic_mask_hash"] for r in four)),1)
        future_rows=[]
        with tempfile.TemporaryDirectory() as temp:
            append_source_rows(temp,four)
            log_event(temp,source_uid=15,source_kind="systematic",
                event_index=1,action="source_captured",reason="pre_mapping_frozen")
            for order,uid in [(1,20),(3,30),(5,40)]:
                pending["seen_future_keyframes"]=order
                v,status=evaluate_four_future(
                  pending=pending,viewpoint=camera(uid),future_keyframe_order=order,
                  render_fn=render,pipeline=None,background=None,
                  min_dynamic_depth_pixels=16)
                self.assertEqual(status,"evaluated")
                self.assertEqual(len(v),4)
                append_future_rows(temp,v)
                future_rows+=v
            torch.testing.assert_close(old,m._delta_xyz[-1])
            df=pairs(Path(temp)/"v52a_future_per_horizon.csv",
                     Path(temp)/"v52a_source_per_keyframe.csv")
            self.assertEqual(len(df),3)
            self.assertTrue(df.history_eligible.all())
            parsed,summary=analyze(
                Path(temp)/"v52a_future_per_horizon.csv",
                Path(temp)/"v52a_source_per_keyframe.csv",
                Path(temp)/"v51d_event_log.csv",
                Path(temp)/"summary")
            self.assertEqual(len(parsed),3)
            self.assertTrue((Path(temp)/"summary"/"v52a_report.md").exists())

    def test_incomplete_future_rejected(self):
        m=model()
        candidate,stats=self.candidate(mod=m)
        source=evaluate_controlled(model=m,viewpoint=camera(15),
           prior_uid=10,frame_id=15,event_index=1,
           legacy_indices=torch.tensor([1,2]),
           legacy_g0=torch.tensor([[.03,0,0],[.04,0,0]]),
           strict_indices=torch.tensor([1]),
           strict_g0=torch.tensor([[.03,0,0]]),
           render_fn=render,pipeline=None,background=None)
        pending=capture_source(model=m,source_uid=15,prior_uid=10,
           event_index=1,source_kind="focused",
           legacy_indices=torch.tensor([1,2]),
           legacy_g0=torch.tensor([[.03,0,0],[.04,0,0]]),
           strict_indices=torch.tensor([1]),
           strict_g0=torch.tensor([[.03,0,0]]),
           source_rows=source,clone_fn=clone)
        four=extend_source(pending=pending,source_viewpoint=camera(15),
           history_offset=candidate,diagnostics=stats,
           render_fn=render,pipeline=None,background=None)
        pending["seen_future_keyframes"]=1
        fut,_=evaluate_four_future(pending=pending,viewpoint=camera(20),
           future_keyframe_order=1,render_fn=render,
           pipeline=None,background=None,min_dynamic_depth_pixels=16)
        with tempfile.TemporaryDirectory() as temp:
            append_source_rows(temp,four)
            p=Path(temp)/"v52a_future_per_horizon.csv"
            pd.DataFrame(fut[:3]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"Missing Hold"):
                pairs(p,Path(temp)/"v52a_source_per_keyframe.csv")


if __name__=="__main__":
    unittest.main()
