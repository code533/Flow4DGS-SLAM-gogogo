"""Summarize V2 debug npz files: python scripts/analyze_flow_rgbd_v2.py results"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("result_root", help="the directory containing run result directories")
a = p.parse_args()
files = sorted(glob.glob(os.path.join(a.result_root, "**", "flow_rgbd_v2_*.npz"), recursive=True))
counts = Counter()
effect = []
if not files:
    print("No files found. Check flow_rgbd_v2.enabled=true, save_debug=true, and run config.yml.")
for f in files:
    with np.load(f, allow_pickle=False) as z:
        reason = str(z["reason"].item())
        counts[reason] += 1
        if bool(z["applied"].item()):
            effect.append(float(z["effective_ratio"].item()))
print("Files:", len(files))
print("Reason counts:", dict(counts))
if effect:
    print("Mean downweighted fraction (all image pixels, applied samples):", np.mean(effect))
