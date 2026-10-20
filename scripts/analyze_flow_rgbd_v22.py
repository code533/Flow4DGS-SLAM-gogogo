"""Summarize V2.2 npz archives: python scripts/analyze_flow_rgbd_v22.py results"""
import argparse,glob,os
from collections import Counter
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("root")
a=p.parse_args()
files=glob.glob(os.path.join(a.root,"**","flow_rgbd_v22_*.npz"),recursive=True)
counts=Counter()
keys=("rgb_selected_count","depth_selected_count","rgb_changed_fraction","depth_changed_fraction")
data={k:[] for k in keys}
for f in files:
    with np.load(f,allow_pickle=False) as z:
        counts[(str(z["mode"].item()),str(z["reason"].item()))]+=1
        for k in keys:data[k].append(float(z[k].item()))
print("frames",len(files),"modes/reasons",dict(counts))
for k,v in data.items():
    if v:print(k,"mean",round(float(np.mean(v)),6),"max",round(float(np.max(v)),6))
