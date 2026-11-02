"""V4.3a.3 offline robust RGB-D audit of paired G0/Support/G4 renders.

Input: V4.3a detailed v43a_XXXXX.npz and optional V4.3a.1 tile CSV.
No Gaussian state mutation. Negative candidate-minus-G0 score is better.
Candidate-wise clipped residual and Huber scores differ from clipping the
paired residual difference; both are reported, never conflated.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

METHODS = ("g3_support", "g4")
CAPS_M = (.02, .05, .10, .20)
HUBER_M = (.01, .02, .05, .10)


def huber_magnitude(r, delta_m):
    """Huber loss / delta in meters, with a linear tail (not bounded)."""
    r = np.asarray(r, dtype=np.float64)
    return np.where(r <= delta_m, r*r/(2*delta_m), r-delta_m/2)


def paired_robust_score(g0_residual, candidate_residual, mask):
    """Same pixels for both candidates; no coverage-dependent denominators."""
    b = np.asarray(g0_residual)[mask].astype(np.float64)
    c = np.asarray(candidate_residual)[mask].astype(np.float64)
    if not len(b):
        return None
    if not (np.isfinite(b).all() and np.isfinite(c).all()):
        raise FloatingPointError("Nonfinite residual inside evaluation mask")
    d = c-b
    sorted_d = np.sort(d)
    k = int(len(d)*.05)
    row = {
        "n":len(d),
        "mean_depth_delta_mm":1000*float(d.mean()),
        "median_paired_depth_delta_mm":1000*float(np.median(d)),
        "paired_clip50_depth_delta_mm":1000*float(np.clip(d,-.05,.05).mean()),
        "paired_trim5_depth_delta_mm":1000*float(sorted_d[k:len(d)-k].mean()),
        "fraction_improve_gt_0p1mm":float((d<-.0001).mean()),
        "fraction_degrade_gt_0p1mm":float((d>.0001).mean()),
    }
    for cap in CAPS_M:
        row[f"candidate_clip{int(cap*1000)}mm_delta_mm"] = (
            1000*float((np.minimum(c,cap)-np.minimum(b,cap)).mean()))
    for h in HUBER_M:
        row[f"candidate_huber{int(h*1000)}mm_delta_mm"] = (
            1000*float((huber_magnitude(c,h)-huber_magnitude(b,h)).mean()))
    return row


def evaluate_npz(path):
    path=Path(path)
    fid=int(path.stem.split("_")[-1])
    with np.load(path,allow_pickle=False) as file:
        data={k:file[k] for k in file.files}
    for method in METHODS:
        for mask_name in ("dynamic_region","gt_depth_valid","gt_rgb_valid"):
            if not np.array_equal(data["g0__"+mask_name],
                                  data[method+"__"+mask_name]):
                raise ValueError("Candidate-dependent observed target mask")
    dyn=data["g0__dynamic_region"].astype(bool)
    dep=data["g0__gt_depth_valid"].astype(bool)
    observed=dyn & dep
    masks={"dynamic_all":observed}
    for erosion in (3,5,10):
        inside=binary_erosion(dyn,iterations=erosion)
        masks[f"edge{erosion}"]=observed&~inside
        masks[f"interior{erosion}"]=observed&inside
    rows=[]
    for method in METHODS:
        for label,mask in masks.items():
            score=paired_robust_score(data["g0__depth_penalized_abs"],
                                     data[method+"__depth_penalized_abs"],mask)
            if score is not None:
                rows.append({"keyframe_id":fid,"hypothesis":method,
                             "region":label,**score})
    return rows


def evaluate_tile_csv(path):
    """Only clip DELTAS of already-aggregated tile means, not pixel errors."""
    t=pd.read_csv(path)
    if t.duplicated(["keyframe_id","tile_index","hypothesis"]).any():
        raise ValueError("Duplicate keyframe/hypothesis/tile rows")
    pivot=t.pivot(index=["keyframe_id","tile_index"],columns="hypothesis",
                  values="depth_penalized_mae_m")
    if pivot.isna().any().any() or set(pivot.columns)!={"g0",*METHODS}:
        raise ValueError("Incomplete tile candidate pairs")
    out=[]
    for name in METHODS:
        d=pivot[name]-pivot["g0"]
        for cap in (None,.0005,.001,.002,.005,.01):
            x=d if cap is None else d.clip(-cap,cap)
            frame=x.groupby(level=0).mean()
            out.append(dict(hypothesis=name,
                tile_delta_cap_mm="none" if cap is None else cap*1000,
                n_frames=len(frame),n_tiles=len(x),
                mean_tile_delta_mm=float(x.mean()*1000),
                mean_frame_delta_mm=float(frame.mean()*1000),
                frames_better_than_minus0p1mm=int((frame<-.0001).sum()),
                frames_worse_than_plus0p1mm=int((frame>.0001).sum())))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--npz",nargs="+",required=True,type=Path)
    p.add_argument("--regions",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    pixels=[row for fn in args.npz for row in evaluate_npz(fn)]
    tiles=evaluate_tile_csv(args.regions)
    pd.DataFrame(pixels).to_csv(args.output/"v43a3_pixel_robust.csv",index=False)
    pd.DataFrame(tiles).to_csv(args.output/"v43a3_tile_sensitivity.csv",index=False)
    lines=["V4.3a.3 OFFLINE only. Negative delta means lower measured residual.",
           "Pixel-wise robust scoring is possible ONLY for detailed NPZ frames.",
           "Tile CSV contains mean residuals; tile-level clipping is not pixel-level loss.",
           "Huber has a linear (unbounded) tail; check bounded loss sensitivity too.",
           "No future-frame robust prediction test without paired full-res future observations.",
           "No depth GT gradients in NPZ; boundary here is estimated dynamic-mask edge."]
    for row in pixels:
        if row["region"] in ("dynamic_all","edge5","interior5"):
            lines.append(
                f"KF{row['keyframe_id']} {row['hypothesis']} {row['region']} "
                f"mean={row['mean_depth_delta_mm']:+.4f}mm "
                f"clip50={row['candidate_clip50mm_delta_mm']:+.4f}mm "
                f"pairedWinsor50={row['paired_clip50_depth_delta_mm']:+.4f}mm "
                f"Huber20={row['candidate_huber20mm_delta_mm']:+.4f}mm")
    (args.output/"v43a3_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
