import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.analyze_v43a3_robust import (
    paired_robust_score, huber_magnitude, evaluate_npz, evaluate_tile_csv
)


class TestV43a3RobustAudit(unittest.TestCase):
    def test_huber_has_linear_tail(self):
        self.assertAlmostEqual(float(huber_magnitude([.2],.02)[0]),.19)
        self.assertAlmostEqual(float(huber_magnitude([.01],.02)[0]),.0025)

    def test_separate_clipping_vs_clip_paired_delta(self):
        base=np.array([[.50,.01], [.50,.01]],dtype=float)
        candidate=np.array([[.10,.03], [.10,.03]],dtype=float)
        mask=np.ones_like(base,dtype=bool)
        row=paired_robust_score(base,candidate,mask)
        # Raw: -190 mm, but clipped candidate residuals give +10 mm.
        self.assertAlmostEqual(row["mean_depth_delta_mm"],-190)
        self.assertAlmostEqual(row["candidate_clip50mm_delta_mm"],10)
        self.assertAlmostEqual(row["paired_clip50_depth_delta_mm"],-15)
        self.assertNotEqual(row["candidate_clip50mm_delta_mm"],
                            row["paired_clip50_depth_delta_mm"])

    def test_fixed_observation_masks(self):
        H,W=12,16
        dyn=np.ones((H,W),dtype=bool)
        depth=np.ones((H,W),dtype=bool)
        rgb=np.ones((H,W),dtype=bool)
        def fields(mul):
            return dict(
                depth_penalized_abs=np.ones((H,W),np.float32)*mul,
                rgb_abs_mean=np.zeros((H,W),np.float32),
                dynamic_region=dyn.copy(),
                gt_depth_valid=depth.copy(),
                gt_rgb_valid=rgb.copy())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"v43a_00212.npz"
            data={}
            for name,score in [("g0",.2),("g3_support",.1),("g4",.3)]:
                data.update({f"{name}__{key}":value for key,value in fields(score).items()})
            np.savez_compressed(path,**data)
            rows=evaluate_npz(path)
            whole=[r for r in rows if r["region"]=="dynamic_all"]
            self.assertEqual(len(whole),2)
            self.assertLess(whole[0]["mean_depth_delta_mm"],0)
            data["g4__gt_depth_valid"][0,0]=False
            np.savez_compressed(path,**data)
            with self.assertRaises(ValueError):evaluate_npz(path)

    def test_tile_csv_incomplete_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"tiles.csv"
            pd.DataFrame([
                dict(keyframe_id=22,tile_index=0,hypothesis="g0",depth_penalized_mae_m=.2),
                dict(keyframe_id=22,tile_index=0,hypothesis="g4",depth_penalized_mae_m=.1)
            ]).to_csv(source,index=False)
            with self.assertRaises(ValueError):evaluate_tile_csv(source)


if __name__=="__main__":
    unittest.main()
