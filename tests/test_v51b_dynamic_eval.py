import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.v51b_dynamic_metrics import measure_dynamic_reconstruction,write_frame_metrics
from scripts.compare_v51b_dynamic import compare


def make(frame_idx,mode="legacy",bad=False):
    h,w=16,16
    target=torch.ones(3,h,w)
    pred=target.clone()
    pred[:,4:12,4:12]=.8 if bad else .9
    depth=torch.ones(1,h,w)
    if bad:depth[:,:,4:12]=1.2
    alpha=torch.ones(1,h,w)
    static=torch.ones(h,w,dtype=torch.bool)
    static[3:13,3:13]=False
    return measure_dynamic_reconstruction(
        frame_idx=frame_idx,frame_uid=frame_idx,phase="before_opt",
        mode=mode,is_keyframe=False,render_rgb=pred,
        render_depth=depth,render_opacity=alpha,
        gt_rgb=target,gt_depth=np.ones((h,w)),
        static_mask=static,boundary_width_px=2)


class TestV51b(unittest.TestCase):
    def test_no_prediction_can_hide_depth(self):
        h,w=16,16
        image=torch.ones(3,h,w)
        gt=torch.ones(3,h,w)
        depth=torch.ones(1,h,w)
        alpha=torch.zeros(1,h,w)
        stat=torch.zeros(h,w,dtype=torch.bool)
        x=measure_dynamic_reconstruction(
            frame_idx=0,frame_uid=0,phase="before_opt",
            mode="legacy",is_keyframe=True,
            render_rgb=image,render_depth=depth,render_opacity=alpha,
            gt_rgb=gt,gt_depth=np.ones((h,w)),static_mask=stat)
        self.assertEqual(x["dynamic_depth_n"],h*w)
        self.assertEqual(x["dynamic_depth_coverage"],0)
        self.assertAlmostEqual(x["dynamic_depth_penalized_mae_m"],.5)
        self.assertTrue(np.isnan(x["static_depth_penalized_mae_m"]))

    def test_boundary_and_interior_split(self):
        a=make(10)
        self.assertEqual(a["dynamic_depth_n"],100)
        self.assertEqual(a["interior_depth_n"]+a["boundary_depth_n"],
                         a["dynamic_depth_n"])
        self.assertGreater(a["interior_depth_n"],0)
        self.assertGreater(a["boundary_depth_n"],0)
        self.assertEqual(a["dynamic_mask_hash"],make(10,"strict")["dynamic_mask_hash"])

    def test_per_frame_rgb_and_depth_worse(self):
        better=make(1,"legacy")
        worse=make(1,"strict",bad=True)
        self.assertLess(better["dynamic_rgb_mae"],worse["dynamic_rgb_mae"])
        self.assertLess(better["dynamic_depth_penalized_mae_m"],
                        worse["dynamic_depth_penalized_mae_m"])
        self.assertLess(better["dynamic_psnr_db"],120)
        self.assertGreater(better["dynamic_psnr_db"],worse["dynamic_psnr_db"])

    def test_match_and_reject_inconsistent_masks(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            l=write_frame_metrics(root/"a","before_opt",[make(0),make(1)])
            r=write_frame_metrics(root/"b","before_opt",[
                make(0,"strict",bad=True),make(1,"strict")])
            summary,joined=compare(l,r,root/"out")
            self.assertEqual(len(joined),2)
            score=summary[(summary.window=="all")&
                (summary.region=="dynamic")&
                (summary.metric=="depth_penalized_mae_m")].iloc[0]
            self.assertGreater(score.mean_delta,0)
            self.assertTrue((root/"out"/"v51b_comparison_report.md").exists())
            corrupted=pd.read_csv(r)
            corrupted.loc[0,"dynamic_mask_hash"]="different"
            corrupted.to_csv(r,index=False)
            with self.assertRaisesRegex(ValueError,"Incomparable"):
                compare(l,r,root/"out2")

    def test_reject_missing_frames(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            l=write_frame_metrics(root/"a","before_opt",[make(0),make(1)])
            r=write_frame_metrics(root/"b","before_opt",[make(1,"strict")])
            with self.assertRaisesRegex(ValueError,"Frame sets"):
                compare(l,r,root/"out")


if __name__=="__main__":
    unittest.main()
