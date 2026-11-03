"""Preflight V5.1a experimental YAMLs: same core, only FOV mode differs."""
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.config_utils import load_config
from utils.v51a_strict_projection import validate_mode

PREFIX="configs/rgbd/bonn/v51a_balloon2_"
CONFIGS={name:PREFIX+name+".yaml" for name in ("legacy","shadow","strict")}


def read_config(path):
    import os
    p=Path(path)
    if not p.is_absolute():
        p=ROOT/p
    cwd=Path.cwd()
    try:
        os.chdir(ROOT)
        cfg=load_config(str(p))
    finally:
        os.chdir(cwd)
    return cfg


def validate_config(path,mode):
    cfg=read_config(path)
    param=cfg["model_params"]
    if not cfg["Dataset"]["dataset_path"].rstrip("/").endswith("rgbd_bonn_balloon2"):
        raise ValueError("Unexpected dataset: need balloon2")
    if cfg["Training"]["flow_ds"]!=1:
        raise ValueError("V5.1a needs full-res optical flow")
    if param["v51a_projection"]["mode"]!=mode:
        raise ValueError("Unexpected Strict-FOV mode")
    if cfg["Training"]["mask_loss"]!=0:
        raise ValueError("Different balloon2 Training baseline")
    if any(bool(param.get(k,{}).get("enabled",False)) for k in (
        "v43a_observation","v50_attribution","flow_cam_v30","flow_cam_v31",
        "flow_cam_v31a","rgbd_uncertainty","flow_rgbd_v2","flow_rgbd_v21",
        "flow_rgbd_v22")):
        raise ValueError("Legacy experiment enabled")
    validate_mode(mode,online_mode=param["v42a_propagation"]["mode"],
                  flow_ds=cfg["Training"]["flow_ds"],use_knn=param["use_knn"],
                  flow_offset=param["flow_offset"],dynamic_model=param["dynamic_model"])
    return cfg


def check_all():
    configs={name:validate_config(path,name) for name,path in CONFIGS.items()}
    stripped=[]
    for name in CONFIGS:
        x=copy.deepcopy(configs[name])
        x["model_params"]["v51a_projection"].pop("mode")
        # YAML inheritance metadata differs across preset files; it
        # does not alter runtime Mapping behavior.
        x.pop("inherit_from",None)
        stripped.append(x)
    if not all(x==stripped[0] for x in stripped[1:]):
        raise AssertionError("Legacy/Shadow/Strict configs differ beyond V5.1a mode")
    return configs


if __name__=="__main__":
    data=check_all()
    for mode,cfg in data.items():
        print("OK",mode,CONFIGS[mode],cfg["Dataset"]["dataset_path"])
    print("All three YAMLs have the same resolved SLAM settings apart from V5.1a.mode.")
