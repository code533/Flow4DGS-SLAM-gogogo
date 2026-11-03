import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from utils.v43a4_predictive import (is_systematic_source,score_from_maps,
    choose_by_source,audit_source_future,save_policy_results)
from scripts.analyze_v43a4_predictive import summarize

class TestV43a4(unittest.TestCase):
    def make_maps(self,base=.10,good=.08,bad=.12,cover=1.):
        shape=(20,20)
        dyn=np.ones(shape,dtype=bool)
        obs=np.ones(shape,dtype=bool)
        data={}
        for name,v in zip(("g0","g3_support","g4"),(base,good,bad)):
            data[name]=dict(dynamic_region=dyn.copy(),gt_depth_valid=obs.copy(),
              gt_rgb_valid=obs.copy(),
              prediction_covered=np.ones(shape,dtype=bool),
              depth_penalized_abs=np.full(shape,v),
              rgb_abs_mean=np.full(shape,.1))
        return data
    def test_sampling(self):
        self.assertEqual([i for i in range(1,32) if is_systematic_source(i,10)],[1,11,21,31])
        self.assertFalse(is_systematic_source(1,0))
    def test_scores_and_selection(self):
        source=score_from_maps(self.make_maps())
        self.assertAlmostEqual(source["g0"]["mae"],.1)
        self.assertEqual(choose_by_source(source,"always_g0"),"g0")
        self.assertEqual(choose_by_source(source,"mae"),"g3_support")
        self.assertEqual(choose_by_source(source,"clipped50"),"g0") # all saturated
        self.assertTrue(np.isfinite(source["g0"]["interior_clipped50"]))
        with self.assertRaises(ValueError):
            maps=self.make_maps()
            maps["g4"]["gt_depth_valid"][1,1]=False
            score_from_maps(maps)
    def test_paired_fixed_future_and_summary(self):
        source=score_from_maps(self.make_maps(base=.1,good=.08,bad=.12))
        future=score_from_maps(self.make_maps(base=.1,good=.13,bad=.09))
        rows=audit_source_future(source,future,source_uid=22,future_uid=27,event_index=4)
        self.assertEqual(len(rows),5)
        m={r["policy"]:r for r in rows}
        self.assertEqual(m["mae"]["selected"],"g3_support")
        self.assertTrue(m["mae"]["future_depth_loss_vs_g0"])
        self.assertTrue(m["mae"]["future_regret_m"]>0)
        with tempfile.TemporaryDirectory() as folder:
            save_policy_results(folder,rows,source,future)
            saved=pd.read_csv(Path(folder)/"v43a4_policy_per_pair.csv")
            self.assertEqual(len(summarize(saved)),5)
            self.assertEqual(len(pd.read_csv(Path(folder)/"v43a4_candidate_scores.csv")),6)
    def test_mask_empty(self):
        x=self.make_maps()
        for v in x.values():v["dynamic_region"][:]=False
        with self.assertRaises(ValueError):score_from_maps(x)

if __name__=="__main__":unittest.main()
