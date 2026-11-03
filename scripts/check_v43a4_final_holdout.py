"""Preflight the locked V4.3a.4 Bonn balloon heldout experiment.

Use before running SLAM, independently of GPU availability.
"""
import argparse
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.config_utils import load_config
from scripts.v43a4_final_controlled import validate_resolved_config

DEFAULT="configs/rgbd/bonn/v43a4_final_balloon_holdout.yaml"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=DEFAULT)
    parser.add_argument("--check-dataset-path",action="store_true",
                        help="Require local Bonn balloon data directory to exist")
    args=parser.parse_args()
    import os
    prior=os.getcwd()
    try:
        os.chdir(ROOT)
        cfg=load_config(args.config)
    finally:
        os.chdir(prior)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/"resolved_config.yml"
        path.write_text(yaml.safe_dump(cfg),encoding="utf-8")
        scene=validate_resolved_config(path)
    if args.check_dataset_path and not (ROOT/scene).exists():
        raise SystemExit("Bonn balloon data not present: "+str(ROOT/scene))
    exp=cfg["model_params"]["v43a_observation"]
    print("LOCKED HOLDOUT CONFIG OK")
    print(" scene:",scene)
    print(" online propagation:",cfg["model_params"]["v42a_propagation"]["mode"])
    print(" source event interval:",exp["predictive_source_every_n_event"])
    print(" source cap:",exp["predictive_max_sources"])
    print(" switch margin:",exp["predictive_decision_margin_m"])
    print(" source and future metrics will be compared offline using raw future depth MAE")


if __name__=="__main__":
    main()
