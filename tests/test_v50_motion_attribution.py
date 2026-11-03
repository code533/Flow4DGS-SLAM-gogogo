import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.v50_motion_attribution import (
    GROUPS, REASONS, GROUP_CODES, eligible_event, classify_selected,
    summarize_v50, append_v50)
from scripts.analyze_v50_attribution import summarize


class TestV50MotionAttribution(unittest.TestCase):
    def test_event_count(self):
        self.assertEqual([i for i in range(1,14) if eligible_event(i,5)],
                         [1,6,11])
        with self.assertRaises(ValueError):
            eligible_event(1,0)

    def test_reason_and_original_depth_partition(self):
        reasons=torch.tensor([0,1,2,3,4,5,6,7,8,0])
        q=torch.tensor([.9,.9,.9,.9,.9,.9,.9,.9,.01,.2])
        original_depth=torch.ones(10,dtype=torch.bool)
        original_depth[0]=False
        actual=classify_selected(reasons,q,original_depth,.3)
        expected=[
            "target_observation_missing",
            "source_projection_invalid","source_projection_invalid",
            "previous_depth_missing",
            "flow_correspondence_invalid","flow_correspondence_invalid",
            "flow_correspondence_invalid",
            "target_observation_missing",
            "low_flow_confidence","low_flow_confidence"]
        self.assertEqual(actual.tolist(),[GROUP_CODES[x] for x in expected])
        self.assertEqual(len(GROUPS),6)
        self.assertEqual(len(REASONS),9)

    def test_summary_preserves_motion_and_groups(self):
        raw=torch.tensor([[0.1,0.,0.],[.4,0,0],[.02,0,0]])
        g0=torch.tensor([[.1,0,0],[.1,0,0],[.02,0,0]])
        original=raw.clone()
        reasons=torch.tensor([0,3,2,8,7],dtype=torch.int16)
        indices=torch.tensor([0,1,3])
        stages=dict(source_projected=4,source_depth_consistent=4,
          target_projection_valid=3,target_filled_depth_valid=3,
          raw_motion_under_1m=3)
        row,groups,detail=summarize_v50(
           frame_id=22,prior_keyframe_id=17,event_index=2,
           all_reasons=reasons,selected_indices=indices,
           selected_target_observed=torch.tensor([True,True,False]),
           raw_disp=raw,g0_disp=g0,
           selected_q=torch.tensor([.8,.1,.01]),stage_counts=stages)
        self.assertEqual(row["n_all_dynamic_gaussians"],5)
        self.assertEqual(row["n_selected_movements"],3)
        self.assertEqual(sum(g["n_selected_group"] for g in groups),3)
        torch.testing.assert_close(raw,original)
        self.assertEqual(row["n_actual_g0_update_gate_pass"],3)
        self.assertEqual(detail["selected_gaussian_index"].tolist(),[0,1,3])
        with tempfile.TemporaryDirectory() as folder:
           append_v50(folder,row,groups,detail)
           self.assertEqual(len(pd.read_csv(Path(folder)/"v50_per_keyframe.csv")),1)
           self.assertEqual(len(pd.read_csv(Path(folder)/"v50_per_group.csv")),6)
           self.assertTrue((Path(folder)/"v50_00022.npz").exists())
           summarize(folder)
           self.assertTrue((Path(folder)/"v50_summary.txt").exists())
           self.assertTrue((Path(folder)/"v50_group_summary.csv").exists())

    def test_invalid_indices_rejected(self):
        with self.assertRaises(IndexError):
            summarize_v50(frame_id=1,prior_keyframe_id=0,event_index=1,
                all_reasons=torch.zeros(2,dtype=torch.int16),
                selected_indices=torch.tensor([5]),
                selected_target_observed=torch.ones(1,dtype=torch.bool),
                raw_disp=torch.zeros((1,3)),g0_disp=torch.zeros((1,3)),
                selected_q=torch.ones(1),stage_counts={})

if __name__=="__main__":
    unittest.main()
