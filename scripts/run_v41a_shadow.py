"""V4.1a OFFLINE Shadow inference on V4.0b per-frame NPZs.

No SLAM calls and no ground truth inferred from baseline masks.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.v41a_shadow import ShadowConfig, infer_shadow, summarize_shadow, config_as_dict


def _average(rows, key):
    xs = np.asarray([r[key] for r in rows], dtype=float)
    v = np.isfinite(xs)
    return float(xs[v].mean()) if v.any() else float("nan")


def _plot(m, o, name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    state_colors = ListedColormap(["#328a67", "#df6c41", "#8b90a7"])
    state_norm = BoundaryNorm([-.5, .5, 1.5, 2.5], state_colors.N)

    items = [
        ("RGB", m.get("rgb_observed"), None, None),
        ("Independent motion score", o["motion_score"], "magma", (0, 1)),
        ("Correspondence score Q_F", o["q_flow"], "viridis", (0, 1)),
        ("Map support score Q_M", o["q_map"], "viridis", (0, 1)),
        ("RGB-D discrepancy score", o["map_score"], "magma", (0, 1)),
        ("Dynamic evidence score P_D*", o["p_dynamic"], "magma", (0, 1)),
        ("Original mask (not GT)", (~m["final_static"].astype(bool)).astype(np.uint8),
         state_colors, "binary"),
        ("Flow-only ablation", o["state_flow_only"], state_colors, "state"),
        ("Naive RGB-D fusion", o["state_naive_rgbd"], state_colors, "state"),
        ("Reliability-weighted binary", o["state_reliability_binary"], state_colors, "state"),
        ("Decoupled three-state", o["state_decoupled_3state"], state_colors, "state"),
        ("Map covered (opacity proxy)", o["map_covered"], "gray", None),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    for ax, (title, data, cmap, bounds) in zip(axes.flat, items):
        if data is None:
            ax.text(.5, .5, "Not saved", ha="center", va="center")
        elif bounds == "state":
            ax.imshow(data, cmap=state_colors, norm=state_norm, interpolation="nearest")
        elif bounds == "binary":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        elif bounds == (0, 1):
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("V4.1a frame %s | green=static, orange=dynamic, grey=unknown\n"
                 "Scores are UNCALIBRATED; baseline mask is NOT ground truth." % name)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(out / ("v41a_visual_%s.png" % name), dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Directory of v40b_XXXXX.npz")
    p.add_argument("--output", default=None,
                   help="Output dir (default: sibling v41a_shadow)")
    p.add_argument("--frames", default="60,140,220,230,310,390",
                   help="Comma-separated frames to visualize or 'all'")
    p.add_argument("--save-maps", choices=("none", "selected", "all"), default="selected")
    p.add_argument("--motion-scale-px", type=float, default=2.5)
    p.add_argument("--fb-scale-px", type=float, default=1.)
    p.add_argument("--flow-weight", type=float, default=.65)
    p.add_argument("--map-weight", type=float, default=.35)
    p.add_argument("--static-threshold", type=float, default=.35)
    p.add_argument("--dynamic-threshold", type=float, default=.65)
    args = p.parse_args()
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        p.error("Input is not a directory: %s" % input_dir)
    out = Path(args.output).resolve() if args.output else input_dir.parent / "v41a_shadow"
    if out == input_dir:
        p.error("Output must differ from input; preserve raw diagnostic NPZs")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("v40b_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        p.error("No v40b_XXXXX.npz files found (V4.0b save_maps must be true)")
    selected = None if args.frames.lower() == "all" else set(
        int(x.strip()) for x in args.frames.split(",") if x.strip())
    cfg = ShadowConfig(motion_scale_px=args.motion_scale_px,
                       fb_scale_px=args.fb_scale_px,
                       flow_weight=args.flow_weight,
                       map_weight=args.map_weight,
                       static_threshold=args.static_threshold,
                       dynamic_threshold=args.dynamic_threshold)
    (out / "v41a_config.json").write_text(
        json.dumps({"model": config_as_dict(cfg),
                    "input_dir": str(input_dir),
                    "note": "Uncalibrated evidence scores; NOT GT"},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            m = {k: z[k] for k in z.files}
        o = infer_shadow(m, cfg)
        row = summarize_shadow(m, o)
        rows.append(row)
        frame_name = "%05d" % row["frame_id"]
        is_selected = selected is None or row["frame_id"] in selected
        if args.save_maps == "all" or (args.save_maps == "selected" and is_selected):
            to_save = {k: v.astype(np.float16) if v.dtype.kind == "f" else v
                       for k, v in o.items()}
            to_save["frame_id"] = np.asarray(row["frame_id"], np.int32)
            np.savez_compressed(out / ("v41a_%s.npz" % frame_name), **to_save)
        if is_selected:
            _plot(m, o, frame_name, out)
        print("[V4.1a] frame={} Unknown={:.3f} map={:.3f} agreement(non-GT)={:.3f}".format(
            row["frame_id"], row["three_state_unknown_fraction"],
            row["map_covered_fraction"],
            row["agreement_with_legacy_decided_NOT_accuracy"]))
    with (out / "v41a_per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    keys = ("three_state_static_fraction", "three_state_dynamic_fraction",
            "three_state_unknown_fraction", "unknown_in_map_uncovered",
            "unknown_in_map_covered", "mean_q_flow_prior_dynamic",
            "mean_q_flow_prior_static", "mean_q_map_covered",
            "agreement_with_legacy_decided_NOT_accuracy")
    report = ["V4.1a: %d frames, offline SHADOW only." % len(rows),
              "P_D*, Q_F and Q_M are heuristic, NOT calibrated probabilities.",
              "Baseline mask is not ground truth. Agreement is not IoU/F1.",
              "Threshold sensitivity + human GT needed before an efficacy claim."]
    report += ["%s: mean_over_frames=%.6f" % (key, _average(rows, key))
               for key in keys]
    for key in ("state_flow_only", "state_naive_rgbd",
                "state_reliability_binary", "state_decoupled_3state"):
        report.append("%s coverage=%.4f dynamic=%.4f agreement(NOT accuracy)=%.4f" %
                      (key, _average(rows, key + "_coverage_fraction"),
                       _average(rows, key + "_dynamic_fraction"),
                       _average(rows, key + "_agreement_with_legacy_NOT_accuracy")))
    (out / "v41a_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("Saved diagnostics to", out)


if __name__ == "__main__":
    main()
