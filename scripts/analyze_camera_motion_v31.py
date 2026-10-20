"""Summarize camera_motion_v31_*.npz. Example: python scripts/analyze_camera_motion_v31.py results"""
import argparse
import glob
import os
from collections import Counter
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('results_root')
args = p.parse_args()
files = sorted(glob.glob(os.path.join(args.results_root, '**', 'camera_motion_v31_*.npz'), recursive=True))
statuses = {'first':Counter(),'refit':Counter()}
fallbacks = Counter()
values = {k:[] for k in ('first_samples','refit_samples','first_condition','refit_condition',
                          'first_median_residual_px','refit_median_residual_px')}
for path in files:
    with np.load(path, allow_pickle=False) as z:
        for label in ('first','refit'):
            status = str(z[label+'_status'].item())
            statuses[label][status]+=1
            fallbacks[(label, bool(z[label+'_fallback'].item()))]+=1
        for k in values:values[k].append(float(z[k].item()))
print('Files:',len(files))
print('Statuses:',{k:dict(v) for k,v in statuses.items()})
print('Fallback counts:',dict(fallbacks))
for k,v in values.items():
    good=np.asarray(v)[np.isfinite(v)]
    if good.size:print(k,'median:',round(float(np.median(good)),4),
                       'mean:',round(float(np.mean(good)),4))
