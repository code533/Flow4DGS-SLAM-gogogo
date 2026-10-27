import unittest
import numpy as np
from utils.v41a_shadow import ShadowConfig,infer_shadow
from utils.v41b_evaluation import (
 attribute_unknown,summarize_reasons,confusion,metrics_from_counts,
 combine_confusions,selective_curve,validate_labels,eval_frame)
from scripts.v41b_make_gt import rasterize


def sample(h=6,w=8):
    sh=(h,w)
    f=lambda v: np.full(sh,v,np.float32)
    b=lambda v: np.full(sh,v,bool)
    return dict(frame_id=np.array(140),flow_full_residual_px=f(.1),
      flow_full_valid=b(True),fb_error_px=f(.1),fb_valid=b(True),
      pre_opacity=f(1),pre_map_covered=b(True),pre_depth_valid=b(True),
      pre_render_depth_valid=b(True),pre_rgb_residual=f(.01),
      pre_depth_raw=f(.01),boundary_near=b(False),
      possible_occlusion=b(False),temporal_depth_support=b(True),
      prior_static=b(True),flow_dynamic=b(False),final_static=b(True))


class TestV41b(unittest.TestCase):
    def test_reasons_partition_unknown(self):
        m=sample();m["flow_full_valid"][0,0]=False
        o=infer_shadow(m)
        reason,cues=attribute_unknown(m,o,ShadowConfig())
        self.assertEqual(reason[0,0],1)
        self.assertTrue(np.array_equal(reason!=0,o["state_decoupled_3state"]==2))
        self.assertEqual(summarize_reasons(reason,cues)["unknown_n"],
                         int((reason!=0).sum()))

    def test_confusion_abstention_not_free(self):
        gt=np.array([[1,1,0,0],[1,0,0,255]],np.uint8)
        pred=np.array([[1,2,0,2],[0,1,0,1]],np.uint8)
        v=metrics_from_counts(confusion(gt,pred))
        self.assertAlmostEqual(v["coverage"],5/7)
        self.assertAlmostEqual(v["recall_all_labeled_dynamic"],1/3)
        self.assertAlmostEqual(v["risk_abstention_cost_1"],4/7)
        self.assertGreater(v["f1_decided"],0)

    def test_all_ignore_rejected(self):
        with self.assertRaises(ValueError):
            validate_labels(np.full((2,3),255,np.uint8),(2,3))
        with self.assertRaises(ValueError):
            validate_labels(np.zeros((2,3,3),np.uint8),(2,3))

    def test_rasterizer_ignore_default(self):
        x=rasterize((10,10,3),[dict(label="dynamic",points=[[2,2],[5,2],[5,5],[2,5]])])
        self.assertEqual(int(x[3,3]),1)
        self.assertEqual(int(x[0,0]),255)
        with self.assertRaises(ValueError):
            rasterize((10,10),[dict(label="static",points=[[0,0],[10,1],[3,3]])])

    def test_curve_does_not_increase_acceptance(self):
        m=sample();o=infer_shadow(m)
        gt=np.zeros((6,8),np.uint8)
        curve=selective_curve([(m,o,gt)],thresholds=[0.,.5,.95])
        filtered=[r for r in curve if r["method"]=="state_decoupled_3state"]
        self.assertGreaterEqual(filtered[0]["coverage"],filtered[-1]["coverage"])

if __name__=="__main__":unittest.main()
