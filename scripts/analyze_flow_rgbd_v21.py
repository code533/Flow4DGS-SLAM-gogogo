"""summarize V2.1 debug npz: python scripts/analyze_flow_rgbd_v21.py results"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

p=argparse.ArgumentParser()
p.add_argument("root")
a=p.parse_args()
paths=sorted(glob.glob(os.path.join(a.root,"**","flow_rgbd_v21_*.npz"),recursive=True))
reasons=Counter()
counters={k:[] for k in ("count_flow_prior","count_map_prior","count_both_priors","count_raw","count_selected","effective_ratio")}
modes=Counter()
for f in paths:
    with np.load(f,allow_pickle=False) as z:
        reasons[str(z["reason"].item())]+=1
        modes[str(z["mode"].item())]+=1
        for k in counters: counters[k].append(float(z[k].item()))
print("frames",len(paths),"modes",dict(modes),"reasons",dict(reasons))
for k,values in counters.items():
    if values:print(k,"mean",round(float(np.mean(values)),4),"max",round(float(np.max(values)),4))
