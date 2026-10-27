"""Offline V4.0a replay: aggregate diagnostics and render per-frame heatmaps.

No annotation or GT assumed. Run after the corresponding SLAM experiment.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.v40a_diagnostics import analyze_frame


def finite_display(a, upper=98):
    a = np.asarray(a, float)
    values = a[np.isfinite(a)]
    scale = np.percentile(values, upper) if values.size else 1.0
    return np.clip(np.nan_to_num(a, nan=0, posinf=0, neginf=0), 0, max(scale, 1e-7))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--frames", default="60,140,220,230,310,390",
                   help="Comma-separated ids to visualize; use 'all' for all")
    args = p.parse_args()
    folder = Path(args.input)
    out = Path(args.output) if args.output else folder
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob("v40a_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit("No v40a_XXXXX.npz files. Enable v40a_diagnostics.save_maps.")
    requested = None if args.frames == "all" else set(int(x) for x in args.frames.split(",") if x.strip())
    records = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for path in files:
        with np.load(path) as data:
            m = {key: data[key] for key in data.files}
        row = analyze_frame(m)
        records.append(row)
        idx = row["frame_id"]
        if requested is not None and idx not in requested:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        items = [
            ("Observed RGB", "rgb_observed", None), ("Static-only RGB pre", "rgb_static_pre", None),
            ("Flow full residual (px)", "flow_full_residual_px", "magma"),
            ("Flow F/B cycle (px)", "fb_error_px", "magma"),
            ("Prior dynamic (not GT)", None, "gray"), ("Flow dynamic (not GT)", "flow_dynamic", "gray"),
            ("Static map covered pre", "pre_map_covered", "gray"),
            ("Static map covered post", "post_map_covered", "gray"),
            ("RGB residual pre", "pre_rgb_residual", "magma"),
            ("Raw depth residual pre (m)", "pre_depth_raw", "magma"),
            ("RGB residual post", "post_rgb_residual", "magma"),
            ("Common valid pixels", None, "gray"),
        ]
        for ax, (title, key, cmap) in zip(axes.flat, items):
            if key is None and title.startswith("Prior"):
                img = ~m["prior_static"].astype(bool)
            elif key is None:
                img = (m["pre_map_covered"].astype(bool) &
                       m["post_map_covered"].astype(bool) &
                       m["pre_depth_valid"].astype(bool))
            elif key not in m:
                img = np.zeros_like(m["pre_map_covered"], dtype=float)
                title += " (disabled)"
            else:
                img = m[key]
            if img.ndim == 2 and img.dtype.kind == "f":
                img = finite_display(img)
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle("V4.0a diagnostic (not labeled motion ground truth), frame %05d" % idx)
        fig.tight_layout()
        fig.savefig(out / ("v40a_visual_%05d.png" % idx), dpi=130)
        plt.close(fig)
    with (out / "v40a_per_frame_offline.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    metric_keys = ["map_covered_pre_fraction", "flow_prior_dynamic_median_px",
                   "flow_prior_static_median_px", "rgb_pre_common_static",
                   "rgb_post_common_static", "depth_pre_common_static",
                   "depth_post_common_static", "fb_median_px"]
    def avg(k):
        a = np.asarray([v[k] for v in records], float)
        return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")
    report = "V4.0a observational only; no GT dynamic masks. %d frames.\n" % len(records)
    report += "\n".join("%s: %.6f" % (k, avg(k)) for k in metric_keys)
    (out / "v40a_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
