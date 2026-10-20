"""Summarize V3.0 motion diagnostics without GT leaking into estimation."""
import argparse,glob,os,collections
import numpy as np
p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args()
fs=sorted(glob.glob(os.path.join(a.root,"**","camera_motion_v30_*.npz"),recursive=True))
m=collections.defaultdict(list)
for f in fs:
    with np.load(f,allow_pickle=False) as z:
        m["mode"].append(str(z["translation_mode"].item()))
        for k in ("translation_raw","translation_applied","translation_max",
                  "scale_translation_unclamped","scale_translation_applied",
                  "rotation_raw_rad","rotation_applied_rad","static_fraction",
                  "residual_first_median","residual_second_median","condition_normalized"):
            m[k].append(float(z[k].item()))
print("files:",len(fs),"modes:",dict(collections.Counter(m["mode"])))
if fs:
    amp=sum(x>1.001 for x in m["scale_translation_applied"])
    print("frames with translation amplification:",amp)
    for k,v in m.items():
        if k=="mode":continue
        arr=np.asarray(v);good=arr[np.isfinite(arr)]
        print(k,"mean",np.mean(good) if len(good) else float("nan"),
              "median",np.median(good) if len(good) else float("nan"),
              "nonfinite",len(arr)-len(good))
