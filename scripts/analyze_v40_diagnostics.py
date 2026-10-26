"""Analyze V4.0 diagnostic NPZs without accessing SLAM or ground truth.

Outputs: per-frame CSV (generated online), v40_groups.csv, v40_summary.txt
and v40_overview.png. These are residual associations, NOT motion accuracy.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def safe_mean(arr):
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Results v40_diagnostics directory")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    folder = Path(args.input)
    out = Path(args.output) if args.output else folder
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob("v40_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit("No v40_NNNNN.npz found. Set v40_diagnostics.save_maps: true.")
    rows = []
    series = {"frame": [], "coverage": [], "rgb": [], "flow": []}
    for file in files:
        with np.load(file) as z:
            f = z["flow_residual_px"].astype(float)
            rgb = z["pre_rgb_residual"].astype(float)
            d = z["pre_depth_residual"].astype(float)
            prior = z["prior_static"].astype(bool)
            m = z["flow_dynamic"].astype(bool)
            covered = z["pre_map_covered"].astype(bool) & z["pre_depth_valid"].astype(bool)
            valid = z["flow_valid"].astype(bool)
            # Four explanatory quadrants with no GT label assumptions.
            groups = {
                "candidate_static_covered": prior & ~m & covered & valid,
                "candidate_dynamic_covered": prior & m & covered & valid,
                "prior_dynamic_covered": ~prior & covered,
                "map_uncovered_valid_flow": ~covered & valid,
            }
            fid = int(file.stem.split("_")[-1])
            for name, sel in groups.items():
                rows.append({
                    "frame_id": fid, "group": name,
                    "n": int(sel.sum()), "fraction": float(sel.mean()),
                    "flow_median_px": float(np.median(f[sel & np.isfinite(f)])) if np.any(sel & np.isfinite(f)) else float("nan"),
                    "rgb_mean": safe_mean(rgb[sel]),
                    "depth_mean": safe_mean(d[sel]),
                })
            series["frame"].append(fid)
            series["coverage"].append(float(covered.mean()))
            series["rgb"].append(safe_mean(rgb[covered]))
            series["flow"].append(safe_mean(f[valid]))
    with open(out / "v40_groups.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for group in sorted({r["group"] for r in rows}):
        subset = [r for r in rows if r["group"] == group]
        summary.append(f"{group}: mean_fraction={safe_mean([r['fraction'] for r in subset]):.4f}, "
                       f"mean_rgb={safe_mean([r['rgb_mean'] for r in subset]):.4f}, "
                       f"mean_depth={safe_mean([r['depth_mean'] for r in subset]):.4f}, "
                       f"mean_flow_median_px={safe_mean([r['flow_median_px'] for r in subset]):.4f}")
    report = (f"V4.0: {len(files)} observed frames, NOT labeled motion evaluation.\n"
              "Caution: map coverage is a proxy based on opacity; no human GT.\n"
              "Missing flow-cycle statistics: the online diagnostic does not run a second RAFT pass.\n"
              + "\n".join(summary) + "\n")
    (out / "v40_summary.txt").write_text(report, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, key, ylabel in zip(axes, ("coverage", "rgb", "flow"),
                                    ("Map coverage", "RGB residual", "Flow residual (px)")):
            ax.plot(series["frame"], series[key], marker=".", lw=1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(out / "v40_overview.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(report)
    print("Wrote", out / "v40_groups.csv", "and", out / "v40_summary.txt")


if __name__ == "__main__":
    main()
