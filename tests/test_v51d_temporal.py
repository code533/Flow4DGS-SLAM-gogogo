import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from utils.v51c_controlled_attribution import evaluate_controlled
from utils.v51d_temporal_persistence import (
    validate_protocol,selection_kind,capture_source,evaluate_future,
    append_future_rows,log_event)
from scripts.analyze_v51d_temporal import paired_deltas,analyze


class FrozenV51dTests(unittest.TestCase):
    @staticmethod
    def make_model():
        prev=torch.tensor([[.02,0,0],[.03,0,0]],dtype=torch.float32)
        result=SimpleNamespace(init_deform="offset",
                 dygs=torch.tensor([True,True]),
                 kf_list=[5,10],
                 _delta_xyz=[torch.zeros((2,3)),prev])
        result.get_delta_xyz=lambda ix:result._delta_xyz[ix]
        return result

    @staticmethod
    def camera(uid):
        shape=(16,16)
        stat=torch.ones(shape,dtype=torch.bool)
        stat[2:14,2:14]=False
        return SimpleNamespace(uid=uid,
             original_image=torch.ones((3,*shape),dtype=torch.float32),
             depth=np.ones(shape,dtype=np.float32),motion_mask=stat)

    @staticmethod
    def fake_render(cam,pc,pipeline,bg,**kwargs):
        # Make each candidate a distinguishable but common-state score.
        offset=kwargs["dx"][:,0].sum()
        return dict(render=torch.ones((3,16,16))*(.9+offset*.1),
             depth=torch.ones((1,16,16))*(1.+offset*.2),
             opacity=torch.ones((1,16,16)))

    def test_protocol_and_selection(self):
        params=dict(every_n_event=10,max_sources=12,horizons=(1,3,5),
              max_frame_gap=45,focused_source_frames=[342,347],
              min_dynamic_depth_pixels=256,opacity_threshold=.1,
              uncovered_penalty_m=.5,boundary_width_px=5)
        validate_protocol(**params)
        self.assertEqual(selection_kind(1,347,every_n_event=10,
                                         focus_ids=(347,)),"systematic_and_focused")
        self.assertEqual(selection_kind(2,347,every_n_event=10,
                                         focus_ids=(347,)),"focused")
        self.assertEqual(selection_kind(11,99,every_n_event=10,
                                         focus_ids=(347,)),"systematic")
        self.assertIsNone(selection_kind(2,99,every_n_event=10,
                                         focus_ids=(347,)))
        with self.assertRaises(ValueError):
            validate_protocol(**{**params,"horizons":(3,1,5)})

    def make_pending(self):
        model=self.make_model()
        src=self.camera(15)
        legacy_idx=torch.tensor([0,1])
        legacy_disp=torch.tensor([[.1,0,0],[.2,0,0]])
        strict_idx=torch.tensor([0])
        strict_disp=torch.tensor([[.15,0,0]])
        source_rows=evaluate_controlled(
            model=model,viewpoint=src,prior_uid=10,frame_id=15,event_index=1,
            legacy_indices=legacy_idx,legacy_g0=legacy_disp,
            strict_indices=strict_idx,strict_g0=strict_disp,
            render_fn=self.fake_render,pipeline=None,background=None)
        # Clone objects with separate Gaussian offset storage.
        def clone(o):
            result=self.make_model()
            result._delta_xyz=[t.clone() for t in o._delta_xyz]
            return result
        pending=capture_source(model=model,source_uid=15,prior_uid=10,
                   event_index=1,source_kind="focused",
                   legacy_indices=legacy_idx,legacy_g0=legacy_disp,
                   strict_indices=strict_idx,strict_g0=strict_disp,
                   source_rows=source_rows,clone_fn=clone)
        return pending,model

    def test_future_one_three_five_and_map_invariance(self):
        pending,live=self.make_pending()
        initial=live._delta_xyz[-1].clone()
        triples=[]
        for order,uid in [(1,20),(3,30),(5,40)]:
            pending["seen_future_keyframes"]=order
            rows,status=evaluate_future(pending=pending,viewpoint=self.camera(uid),
                future_keyframe_order=order,render_fn=self.fake_render,
                pipeline=None,background=None,min_dynamic_depth_pixels=16)
            self.assertEqual(status,"evaluated")
            self.assertEqual([r["mode"] for r in rows],
                             ["hold","legacy","strict"])
            self.assertEqual(rows[0]["elapsed_image_frames"],uid-15)
            self.assertEqual(rows[0]["future_keyframe_order"],order)
            self.assertEqual(len({r["dynamic_mask_hash"] for r in rows}),1)
            triples+=rows
        torch.testing.assert_close(live._delta_xyz[-1],initial)
        self.assertIsNot(pending["model"],live)
        self.assertEqual(len(triples),9)
        with tempfile.TemporaryDirectory() as temp:
            for i in range(0,9,3):
                append_future_rows(temp,triples[i:i+3])
            log_event(temp,source_uid=15,source_kind="focused",event_index=1,
                      action="source_captured",reason="pre_mapping_frozen")
            log_event(temp,source_uid=15,source_kind="focused",event_index=1,
                      action="snapshot_closed",reason="all_horizons_consumed")
            csv=Path(temp)/"v51d_future_per_horizon.csv"
            paired=paired_deltas(pd.read_csv(csv))
            self.assertEqual(set(paired.future_keyframe_order),{1,3,5})
            result,summary=analyze(csv,Path(temp)/"v51d_event_log.csv",
                                   Path(temp)/"analysis")
            self.assertEqual(result.has_all_requested_horizons.sum(),3)
            self.assertTrue((Path(temp)/"analysis"/"v51d_report.md").exists())

    def test_invalid_future_and_few_pixels(self):
        pending,_=self.make_pending()
        pending["seen_future_keyframes"]=1
        with self.assertRaisesRegex(ValueError,"must follow source"):
            evaluate_future(pending=pending,viewpoint=self.camera(14),
               future_keyframe_order=1,render_fn=self.fake_render,
               pipeline=None,background=None)
        rows,status=evaluate_future(pending=pending,viewpoint=self.camera(20),
              future_keyframe_order=1,render_fn=self.fake_render,
              pipeline=None,background=None,min_dynamic_depth_pixels=10000)
        self.assertEqual(rows,[])
        self.assertEqual(status,"too_few_future_dynamic_depth_pixels")

    def test_incomplete_pair_fails(self):
        pending,_=self.make_pending()
        pending["seen_future_keyframes"]=1
        rows,_=evaluate_future(pending=pending,viewpoint=self.camera(20),
            future_keyframe_order=1,render_fn=self.fake_render,
            pipeline=None,background=None,min_dynamic_depth_pixels=16)
        with self.assertRaisesRegex(ValueError,"Expected exactly|Incomplete future triplet"):
            paired_deltas(pd.DataFrame(rows[:-1]))


if __name__=="__main__":unittest.main()
