"""Lock final one-shot validation protocol and reject accidental balloon2 reuse."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.v43a4_final_controlled import (
    validate_resolved_config, read_scores, select, evaluate_pairs,
    aggregate, contrast, decision, run_control, POLICIES)


def fake_config(scene):
    return {"Dataset":{"dataset_path":scene},
        "Training":{"flow_ds":1},
        "model_params":{
            "dynamic_model":"offset","flow_offset":True,"use_knn":True,
            "v42a_propagation":{"mode":"g0"},
            "v43a_observation":{
                "enabled":True,"temporal_enabled":True,
                "predictive_enabled":True,
                "predictive_decision_margin_m":0.0,
                "predictive_source_every_n_event":4,
                "predictive_max_sources":25,
                "temporal_max_frame_gap":12,
                "predictive_min_dynamic_depth_pixels":256,
                "predictive_min_coverage_fraction":.95,
                "predictive_clip_m":.05,
                "predictive_huber_delta_m":.02,
                "predictive_interior_erosion_px":5}}}


def fake_pair(src):
    rows=[]
    for when in ("source","future"):
        for name,mae,clipped in (
            ("g0",.10,.045),
            ("g3_support",.09 if when=="source" else .08,.043),
            ("g4",.11,.048)):
            rows.append(dict(
                source_keyframe_id=src,future_keyframe_id=src+5,
                timepoint=when,hypothesis=name,
                mae=mae,clipped50=clipped,huber20=mae-.01,
                interior_clipped50=clipped,
                rgb_mae=.1,depth_coverage=1.,
                n_dynamic_depth=500,n_interior_depth=350,n_dynamic_rgb=500))
    return rows


class TestFinalControl(unittest.TestCase):
    def test_config_lock(self):
        with tempfile.TemporaryDirectory() as td:
            c=Path(td)/"config.yml"
            c.write_text(yaml.safe_dump(fake_config(
                "datasets/bonn/rgbd_bonn_balloon")),encoding="utf-8")
            self.assertTrue(validate_resolved_config(c).endswith("rgbd_bonn_balloon"))
            c.write_text(yaml.safe_dump(fake_config(
                "datasets/bonn/rgbd_bonn_balloon2")),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Heldout scene"):
                validate_resolved_config(c)

    def test_locked_comparison_without_future_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"scores.csv"
            pd.DataFrame(fake_pair(5)+fake_pair(25)).to_csv(file,index=False)
            groups=read_scores(file)
            self.assertEqual(len(groups),2)
            source=groups[0][2]["source"]
            self.assertEqual(select(source,"mae",0.),"g3_support")
            # After changing FUTURE, source decision should never change.
            groups[0][2]["future"]["g3_support"]["mae"]=.50
            self.assertEqual(select(groups[0][2]["source"],"mae",0.),"g3_support")
            rows=evaluate_pairs(groups,0.,"heldout")
            self.assertEqual(len(rows),len(groups)*len(POLICIES))
            summary=aggregate(rows)
            self.assertEqual(len(summary),len(POLICIES))
            contrasts,stats=contrast(rows)
            self.assertEqual(stats["n_pairs"],2)
            self.assertEqual(len(contrasts),2)

    def test_strict_pair_validation(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"scores.csv"
            items=fake_pair(5)
            pd.DataFrame(items[:-1]).to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError,"Unpaired"):
                read_scores(f)

    def test_insufficient_data(self):
        result,reasons=decision(dict(n_pairs=10))
        self.assertEqual(result,"INSUFFICIENT_DATA_NO_GO")
        self.assertTrue(reasons)

    def test_end_to_end_writes_sha_and_decision(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            scores=td/"scores.csv"
            config=td/"config.yml"
            folder=td/"out"
            pd.DataFrame(sum((fake_pair(5+i*20) for i in range(13)),[])).to_csv(
                scores,index=False)
            config.write_text(yaml.safe_dump(fake_config(
                "datasets/bonn/rgbd_bonn_balloon")),encoding="utf-8")
            result=run_control(scores,config,folder)
            self.assertEqual(result["primary_contrast"]["n_pairs"],13)
            self.assertTrue((folder/"v43a4_final_protocol_and_decision.json").exists())
            self.assertTrue((folder/"v43a4_final_decision.md").exists())
            self.assertTrue((folder/"v43a4_final_mae_vs_clipped50.csv").exists())
            self.assertEqual(len(pd.read_csv(folder/"v43a4_final_policy_per_pair.csv")),
                             13*len(POLICIES)*2)


if __name__=="__main__":
    unittest.main()
