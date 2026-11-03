"""Validate standalone V4.3a.4 Balloon2 YAML presets without a GPU.

Usage:
  python scripts/check_v43a4_config.py
  python scripts/check_v43a4_config.py --config configs/rgbd/bonn/v43a4_balloon2_full.yaml

Fails fast on missing inheritance, accidental early-stage diagnostics,
off-mode propagation, wrong scene, or inconsistent source sampling.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.config_utils import load_config

SMOKE="configs/rgbd/bonn/v43a4_balloon2_smoke.yaml"
FULL="configs/rgbd/bonn/v43a4_balloon2_full.yaml"


def assert_preset(config_path):
    config_path=Path(config_path)
    if not config_path.is_absolute():
        config_path=ROOT/config_path
    # Loader uses file-relative-to-CWD inherit_from paths; run from repo root.
    import os
    prior=os.getcwd()
    try:
        os.chdir(ROOT)
        cfg=load_config(str(config_path))
    finally:
        os.chdir(prior)
    model=cfg["model_params"]
    propagation=model["v42a_propagation"]
    obs=model["v43a_observation"]
    for key in ("rgbd_uncertainty","flow_rgbd_v2","flow_rgbd_v21",
                "flow_rgbd_v22","flow_cam_v30","flow_cam_v31",
                "flow_cam_v31a","v40_diagnostics","v40a_diagnostics",
                "v40b_calibration"):
        if key in model:
            raise AssertionError("Legacy experiment still active in resolved config: "+key)
    assert "rgbd_bonn_balloon2" in cfg["Dataset"]["dataset_path"]
    assert cfg["Training"]["flow_ds"] == 1
    assert model["flow_offset"] and model["use_knn"]
    assert model["dynamic_model"]=="offset"
    assert propagation["mode"]=="g0"
    assert not propagation.get("g34_shadow",False)
    assert not propagation.get("g4_ablation",False)
    assert not propagation.get("attribute_failures",False)
    assert obs["enabled"] and obs["temporal_enabled"] and obs["predictive_enabled"]
    assert obs["predictive_source_every_n_event"]>=1
    assert obs["predictive_max_sources"]>=1
    assert obs["temporal_max_frame_gap"]>=1
    assert not obs.get("region_analysis",False)
    assert not obs.get("save_maps",False)
    assert not obs.get("predictive_save_future_maps",False)
    assert cfg["Training"]["mask_loss"]==0
    return cfg


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",nargs="*",default=[SMOKE,FULL])
    opts=ap.parse_args()
    for file in opts.config:
        cfg=assert_preset(file)
        v=cfg["model_params"]["v43a_observation"]
        print("OK",file,
              "scene=",cfg["Dataset"]["dataset_path"],
              "mode=",cfg["model_params"]["v42a_propagation"]["mode"],
              "source_every=",v["predictive_source_every_n_event"],
              "max_sources=",v["predictive_max_sources"])
    print("Both YAML inheritance and V4.3a.4 critical flags validated.")


if __name__=="__main__":main()
