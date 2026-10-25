#!/usr/bin/env python3
"""Offline B1 visualizer. No torch, CUDA, SLAM imports, or model weights needed.

Example:
  python visualize_consistency.py --results /data/results/B1/run \
      --dataset /data/tum/rgbd_dataset_freiburg3_walking_xyz --frames 10 50 80 90

Uses the saved, resolved config.yml, not the original inherited experiment YAML.
Reproduces this repository's TUM/Bonn association (including pose skiprows=1),
32 Hz subsampling, Calibration.start/end, and RGB undistortion.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import yaml


MODALITIES = [
    ("depth_only", "Depth only", (0, 0.65, 1)),
    ("rgb_only", "RGB only", (1, 0.6, 0)),
    ("both", "Depth + RGB", (0.9, 0.1, 0.8)),
]


def decompose_residuals(arrays):
    """Partition saved map mismatches without changing the detector thresholds."""
    valid = arrays["map_valid"].astype(bool)
    for name in ("depth_score", "rgb_score", "baseline_static_mask", "map_inconsistent"):
        if arrays[name].shape != valid.shape:
            raise ValueError("Decomposition array shapes differ: " + name)
    finite = np.isfinite(arrays["depth_score"]) & np.isfinite(arrays["rgb_score"])
    if np.any(valid & ~finite):
        raise ValueError("Nonfinite residual in map_valid; inspect the original NPZ")
    depth_bad = valid & (arrays["depth_score"] > 1)
    rgb_bad = valid & (arrays["rgb_score"] > 1)
    mismatch = depth_bad | rgb_bad
    if not np.array_equal(mismatch, arrays["map_inconsistent"].astype(bool)):
        raise ValueError("Saved mismatch differs from score > 1; detector version/threshold mismatch")
    masks = {"depth_only": depth_bad & ~rgb_bad,
             "rgb_only": rgb_bad & ~depth_bad,
             "both": depth_bad & rgb_bad}
    kept = arrays["baseline_static_mask"].astype(bool)
    scopes = {"all_valid": valid, "baseline_kept_valid": valid & kept,
              "baseline_excluded_valid": valid & ~kept}
    rows = []
    for scope_name, scope in scopes.items():
        scope_n = int(scope.sum())
        mismatch_n = int((mismatch & scope).sum())
        for key, _, _ in MODALITIES:
            count = int((masks[key] & scope).sum())
            rows.append({"scope": scope_name, "category": key,
                         "image_pixels": valid.size, "scope_pixels": scope_n,
                         "mismatch_pixels": mismatch_n, "category_pixels": count,
                         "percent_image": 100 * count / valid.size,
                         "percent_scope": 100 * count / scope_n if scope_n else "",
                         "percent_mismatch": 100 * count / mismatch_n if mismatch_n else ""})
    return masks, scopes, rows


def render_decomposition(npz_path, rgb, output, timestamp):
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ("map_valid", "map_inconsistent", "depth_score",
                                      "rgb_score", "baseline_static_mask")}
    masks, scopes, rows = decompose_residuals(arrays)
    keep = scopes["baseline_kept_valid"]
    if keep.shape != rgb.shape[:2]:
        raise ValueError("RGB/decomposition shape mismatch")
    candidate = keep & arrays["map_inconsistent"].astype(bool)
    candidate_n = int(candidate.sum())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("Observed RGB")
    ax[1].imshow(overlay(rgb, candidate, (1, 0, 1)))
    ax[1].set_title(f"Baseline kept + map mismatch: {100 * candidate.mean():.2f}% of image")
    combined = rgb.copy()
    for key, label, color in MODALITIES:
        combined = overlay(combined, masks[key] & keep, color)
    ax[2].imshow(combined)
    ax[2].set_title("Exclusive residual categories (baseline-kept region)")
    ax[2].legend(handles=[Patch(color=color, label=label) for _, label, color in MODALITIES],
                 loc="lower right", fontsize=9)
    for panel, (key, label, color) in zip(ax[3:], MODALITIES):
        mask = masks[key] & keep
        percent = f"{100 * mask.sum() / candidate_n:.1f}%" if candidate_n else "N/A"
        panel.imshow(overlay(rgb, mask, color))
        panel.set_title(f"{label}: {100 * mask.mean():.2f}% of image\n"
                        f"{percent} of baseline-kept mismatches")
    for panel in ax:
        panel.axis("off")
    fig.suptitle(f"Frame {int(npz_path.stem)} | RGB timestamp {timestamp}\n"
                 "Original score > 1 thresholds; only map-valid, baseline-kept pixels are highlighted.\n"
                 "Residual categories are not motion labels. No temporal correspondence is assumed.",
                 fontsize=12)
    fig.savefig(str(output), dpi=130)
    plt.close(fig)
    return [{"frame": int(npz_path.stem), "rgb_timestamp": timestamp, **row} for row in rows]


def render_decomposition_trends(rows, output):
    frames = sorted({row["frame"] for row in rows})
    lookup = {(r["frame"], r["scope"], r["category"]): r for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, scope, title in zip(axes,
            ["baseline_kept_valid", "baseline_excluded_valid"],
            ["Baseline-kept mismatches (new evidence)", "Baseline-excluded mismatches (already excluded)"]):
        bottom = np.zeros(len(frames))
        for key, label, color in MODALITIES:
            values = [lookup[(f, scope, key)]["percent_image"] for f in frames]
            axis.bar(np.arange(len(frames)), values, bottom=bottom, color=color, label=label)
            bottom += values
        axis.set_title(title)
        axis.set_ylabel("Percent of full image")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    limit = max(axis.get_ylim()[1] for axis in axes)
    for axis in axes:
        axis.set_ylim(0, max(limit, 1))
    positions = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 20), dtype=int))
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([str(frames[i]) for i in positions], rotation=45)
    axes[-1].set_xlabel("Saved frame IDs only (no interpolation across unsaved frames)")
    fig.savefig(str(output), dpi=130)
    plt.close(fig)


DEPTH_REGIONS = [
    ("near_invalid", "Near invalid depth", (1, 0.65, 0)),
    ("near_edge", "Near depth edge", (0, 0.65, 1)),
    ("interior", "Away from edges/holes", (0.25, 0.8, 0.35)),
]


def dilate_square(mask, radius):
    """Chebyshev-radius neighborhood; do not treat image exterior as a hole."""
    if radius == 0:
        return mask.copy()
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * radius + 1))) > 0


def split_depth_regions(depth, depth_min, edge_abs, edge_rel, radius):
    """Use observed depth only, never the residual itself, to define strata.

    Mark both endpoints of a valid horizontal/vertical pair when
    abs(z1-z2) > edge_abs + edge_rel * min(z1,z2). Invalid neighborhoods
    take priority over edge neighborhoods; the remaining valid pixels are interior.
    Interior does not imply a planar, correct, or static surface.
    """
    valid = np.isfinite(depth) & (depth > depth_min)
    seeds = np.zeros_like(valid)
    for left, right in [(np.s_[:, :-1], np.s_[:, 1:]),
                        (np.s_[:-1, :], np.s_[1:, :])]:
        pair = valid[left] & valid[right]
        with np.errstate(invalid="ignore"):
            jump = pair & (np.abs(depth[left] - depth[right]) >
                           edge_abs + edge_rel * np.minimum(depth[left], depth[right]))
        seeds[left] |= jump
        seeds[right] |= jump
    near_invalid = valid & dilate_square(~valid, radius)
    near_edge = valid & ~near_invalid & dilate_square(seeds, radius)
    interior = valid & ~near_invalid & ~near_edge
    return {"near_invalid": near_invalid, "near_edge": near_edge,
            "interior": interior}, valid, seeds


def depth_region_rates(counts):
    n, bad = counts["support_pixels"], counts["depth_bad_pixels"]
    pct = lambda value, denominator: 100 * value / denominator if denominator else ""
    return {
        "support_percent_scope": pct(n, counts["scope_pixels"]),
        "depth_bad_percent_support": pct(bad, n),
        "depth_only_percent_support": pct(counts["depth_only_pixels"], n),
        "closer_percent_depth_bad": pct(counts["closer_pixels"], bad),
        "farther_percent_depth_bad": pct(counts["farther_pixels"], bad),
        "direction_unknown_percent_depth_bad": pct(counts["direction_unknown_pixels"], bad),
    }


def analyze_depth_regions(arrays, depth, depth_min, edge_abs, edge_rel, radii):
    masks, scopes, _ = decompose_residuals(arrays)
    if depth.shape != arrays["map_valid"].shape:
        raise ValueError("Depth/NPZ shapes differ: check dataset and saved config")
    obs_valid = np.isfinite(depth) & (depth > depth_min)
    valid = arrays["map_valid"].astype(bool)
    if np.any(valid & ~obs_valid):
        raise ValueError("NPZ map_valid includes invalid observed depth; check frame association/config")
    bad = masks["depth_only"] | masks["both"]
    fg = arrays["foreground_candidate"].astype(bool)
    bg = arrays["revealed_candidate"].astype(bool)
    if fg.shape != depth.shape or bg.shape != depth.shape or np.any(fg & bg):
        raise ValueError("Inconsistent saved depth-direction masks")
    # Sign is recoverable only for threshold-exceeding residuals in current NPZs.
    closer, farther = bad & fg, bad & bg
    unknown_direction = bad & ~fg & ~bg
    rows = []
    for radius in radii:
        regions, _, _ = split_depth_regions(depth, depth_min, edge_abs, edge_rel, radius)
        for scope_name, scope in scopes.items():
            for region_name, region in {"all_regions": obs_valid, **regions}.items():
                support = scope & region
                counts = {"image_pixels": depth.size, "scope_pixels": int(scope.sum()),
                          "support_pixels": int(support.sum()),
                          "depth_bad_pixels": int((bad & support).sum()),
                          "depth_only_pixels": int((masks["depth_only"] & support).sum()),
                          "closer_pixels": int((closer & support).sum()),
                          "farther_pixels": int((farther & support).sum()),
                          "direction_unknown_pixels": int((unknown_direction & support).sum())}
                scores = arrays["depth_score"][support]
                quantiles = {f"depth_score_p{q}": float(np.percentile(scores, q)) if scores.size else ""
                             for q in (50, 90, 95)}
                rows.append({"radius_px": radius, "scope": scope_name, "region": region_name,
                             **counts, **depth_region_rates(counts), **quantiles})
    return rows


def render_depth_regions(npz_path, rgb, depth, output, timestamp,
                         depth_min, edge_abs, edge_rel, radii):
    keys = ["map_valid", "map_inconsistent", "depth_score", "rgb_score",
            "baseline_static_mask", "foreground_candidate", "revealed_candidate"]
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in keys}
    rows = analyze_depth_regions(arrays, depth, depth_min, edge_abs, edge_rel, radii)
    radius = radii[len(radii) // 2]
    regions, obs_valid, _ = split_depth_regions(depth, depth_min, edge_abs, edge_rel, radius)
    keep = arrays["map_valid"].astype(bool) & arrays["baseline_static_mask"].astype(bool)
    bad = keep & (arrays["depth_score"] > 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("Observed RGB")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#555555")
    im = ax[1].imshow(np.ma.masked_where(~obs_valid, depth), cmap=cmap)
    fig.colorbar(im, ax=ax[1], shrink=0.7, label="Observed depth (m)")
    ax[1].set_title("Observed depth (gray = invalid)")
    strata = np.full(depth.shape + (3,), 0.35)
    anomaly_overlay = rgb.copy()
    for key, label, color in DEPTH_REGIONS:
        strata[regions[key]] = color
        anomaly_overlay = overlay(anomaly_overlay, bad & regions[key], color)
    ax[2].imshow(strata)
    ax[2].set_title(f"Disjoint observed-depth regions (radius={radius}px)")
    ax[2].legend(handles=[Patch(color=c, label=l) for _, l, c in DEPTH_REGIONS],
                 loc="lower right", fontsize=8)
    ax[3].imshow(anomaly_overlay)
    ax[3].set_title("Baseline-kept depth outliers colored by region")
    direction = overlay(rgb, bad & arrays["foreground_candidate"].astype(bool), (1, 0.45, 0))
    direction = overlay(direction, bad & arrays["revealed_candidate"].astype(bool), (0, 0.65, 1))
    ax[4].imshow(direction)
    ax[4].set_title("Outliers only: orange=observed closer; blue=farther")
    selected = [next(r for r in rows if r["radius_px"] == radius and
                     r["scope"] == "baseline_kept_valid" and r["region"] == key)
                for key, _, _ in DEPTH_REGIONS]
    rates = [r["depth_bad_percent_support"] if r["support_pixels"] else 0 for r in selected]
    ax[5].bar(range(3), rates, color=[c for _, _, c in DEPTH_REGIONS])
    ax[5].set_xticks(range(3))
    ax[5].set_xticklabels(["Near invalid", "Near edge", "Interior"])
    ax[5].set_ylim(0, 110)
    ax[5].set_ylabel("Depth outliers / comparable pixels in this region (%)")
    ax[5].set_title("Baseline-kept region-conditioned rates")
    ax[5].grid(axis="y", alpha=0.25)
    for i, (r, rate) in enumerate(zip(selected, rates)):
        label = f"{rate:.1f}%\nn={r['support_pixels']}" if r["support_pixels"] else "N/A\nn=0"
        ax[5].text(i, rate + 2, label, ha="center", fontsize=10)
    for panel in ax[:5]:
        panel.axis("off")
    fig.suptitle(f"Frame {int(npz_path.stem)} | RGB timestamp {timestamp}\n"
                 f"Depth edge: adjacent jump > {edge_abs:g} m + {edge_rel:g} * min(depth pair). "
                 "Invalid-depth neighborhoods take priority.\n"
                 "Observed-depth strata only; interior does not mean static or correct. "
                 "No change to saved residual thresholds.", fontsize=11)
    fig.savefig(str(output), dpi=130)
    plt.close(fig)
    return [{"frame": int(npz_path.stem), "rgb_timestamp": timestamp, **r} for r in rows]


def pool_depth_region_rows(rows):
    """Pool pixel counts, not frame rates or quantiles; never pool across radii."""
    grouped = {}
    count_keys = ["image_pixels", "scope_pixels", "support_pixels", "depth_bad_pixels",
                  "depth_only_pixels", "closer_pixels", "farther_pixels", "direction_unknown_pixels"]
    for row in rows:
        key = (row["radius_px"], row["scope"], row["region"])
        if key not in grouped:
            grouped[key] = {"radius_px": key[0], "scope": key[1], "region": key[2],
                            "saved_frames": 0, **{k: 0 for k in count_keys}}
        group = grouped[key]
        group["saved_frames"] += 1
        for k in count_keys:
            group[k] += row[k]
    return [{**group, **depth_region_rates(group)} for group in grouped.values()]


def render_depth_region_summary(pooled, output, radii):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    x = np.arange(len(radii))
    for i, (key, label, color) in enumerate(DEPTH_REGIONS):
        selected = [next(r for r in pooled if r["radius_px"] == radius and
                         r["scope"] == "baseline_kept_valid" and r["region"] == key)
                    for radius in radii]
        for axis, metric in zip(axes[:2], ["depth_bad_percent_support", "support_percent_scope"]):
            vals = [float(r[metric]) if r[metric] != "" else np.nan for r in selected]
            axis.bar(x + (i - 1) * 0.25, vals, width=0.25, color=color, label=label)
    axes[0].set_title("Pooled depth-outlier rate within each region (baseline-kept only)")
    axes[0].set_ylabel("Outliers / region support (%)")
    axes[1].set_title("Region support size: check denominators before interpreting high rates")
    axes[1].set_ylabel("Region / comparable kept pixels (%)")
    primary = radii[len(radii) // 2]
    selected = [next(r for r in pooled if r["radius_px"] == primary and
                     r["scope"] == "baseline_kept_valid" and r["region"] == key)
                for key, _, _ in DEPTH_REGIONS]
    bottom = np.zeros(3)
    for metric, label, color in [("closer_percent_depth_bad", "Observed closer", "#ff7300"),
                                 ("farther_percent_depth_bad", "Observed farther", "#00a5ff"),
                                 ("direction_unknown_percent_depth_bad", "Unknown sign", "#888888")]:
        vals = np.array([float(r[metric]) if r[metric] != "" else 0 for r in selected])
        axes[2].bar(np.arange(3), vals, bottom=bottom, color=color, label=label)
        bottom += vals
    axes[2].set_title(f"Direction among depth outliers only (radius={primary}px)")
    axes[2].set_ylabel("Percent of region's depth outliers")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([f"{label}\nn={r['depth_bad_pixels']}" for (_, label, _), r
                            in zip(DEPTH_REGIONS, selected)])
    for axis in axes[:2]:
        axis.set_xticks(x)
        axis.set_xticklabels([f"{r}px" for r in radii])
        axis.set_xlabel("Neighborhood radius (independent sensitivity settings)")
    for axis in axes:
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Saved frames only; repeated observations are not independent samples.\n"
                 "Empty support / zero-outlier regions have undefined rates (see CSV).", fontsize=11)
    fig.savefig(str(output), dpi=130)
    plt.close(fig)


def read_table(path, skiprows=0):
    rows = np.loadtxt(str(path), dtype=str, comments="#", skiprows=skiprows, ndmin=2)
    if rows.size == 0:
        raise ValueError("Empty timestamp table: " + str(path))
    return rows


def associate_rgbd(dataset, calibration):
    """Return associated RGB/depth records indexed by Camera.uid."""
    rgb = read_table(dataset / "rgb.txt")
    depth = read_table(dataset / "depth.txt")
    pose_path = dataset / "groundtruth.txt"
    if not pose_path.exists():
        pose_path = dataset / "pose.txt"
    poses = read_table(pose_path, skiprows=1)
    rgb_t, depth_t, pose_t = (a[:, 0].astype(np.float64) for a in (rgb, depth, poses))
    associated = []
    for i, t in enumerate(rgb_t):
        j = np.argmin(np.abs(depth_t - t))
        k = np.argmin(np.abs(pose_t - t))
        if abs(depth_t[j] - t) < 0.08 and abs(pose_t[k] - t) < 0.08:
            associated.append((i, j))
    if not associated:
        raise ValueError("No RGB/depth/pose associations within 0.08 seconds")
    selected = [associated[0]]
    for i, j in associated[1:]:
        if rgb_t[i] - rgb_t[selected[-1][0]] > 1.0 / 32:
            selected.append((i, j))
    start = int(calibration.get("start", 0))
    end = int(calibration.get("end", -1))
    end = len(selected) if end == -1 else end
    return [{"rgb_timestamp": str(rgb[i, 0]), "rgb_file": str(rgb[i, 1]),
             "depth_timestamp": str(depth[j, 0]), "depth_file": str(depth[j, 1])}
            for i, j in selected[start:end]]


def associate_rgb(dataset, calibration):
    """Backward-compatible RGB-only view of the association."""
    return [(r["rgb_timestamp"], r["rgb_file"]) for r in associate_rgbd(dataset, calibration)]


def load_depth(path, calibration):
    scale = float(calibration["depth_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration.depth_scale must be finite and positive")
    depth = np.asarray(Image.open(path)).astype(np.float32) / scale
    if depth.ndim != 2:
        raise ValueError("Expected a single-channel depth image: " + str(path))
    # Match MonocularDataset: RGB may be undistorted, depth is NOT remapped.
    return depth


def load_rgb(path, calibration):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if calibration.get("distorted", False):
        import cv2
        k = np.array([[calibration["fx"], 0, calibration["cx"]],
                      [0, calibration["fy"], calibration["cy"]], [0, 0, 1]], dtype=float)
        distortion = np.array([calibration.get(x, 0) for x in ("k1", "k2", "p1", "p2", "k3")])
        size = (int(calibration["width"]), int(calibration["height"]))
        mx, my = cv2.initUndistortRectifyMap(k, distortion, np.eye(3), k, size, cv2.CV_32FC1)
        rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255


def overlay(rgb, mask, color):
    out = rgb.copy()
    out[mask] = 0.45 * out[mask] + 0.55 * np.asarray(color)
    return out


def render_frame(npz_path, rgb, output, timestamp, score_max):
    required = ["state", "map_valid", "map_inconsistent", "foreground_candidate",
                "revealed_candidate", "depth_score", "rgb_score", "baseline_static_mask"]
    with np.load(npz_path, allow_pickle=False) as data:
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{npz_path.name}: missing arrays {sorted(missing)}")
        arrays = {key: data[key] for key in required}
    shape = rgb.shape[:2]
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError(f"{npz_path.name}: RGB/array shapes differ; check dataset and config")
    if not np.isin(arrays["state"], [0, 1, 2]).all():
        raise ValueError(f"{npz_path.name}: unsupported state labels")
    valid = arrays["map_valid"].astype(bool)
    inconsistent = arrays["map_inconsistent"].astype(bool)
    baseline_static = arrays["baseline_static_mask"].astype(bool)
    candidate = baseline_static & valid & inconsistent
    pct = lambda m: 100 * float(np.mean(m))
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("Observed RGB (dataset image)")
    ax[1].imshow(overlay(rgb, ~baseline_static, (1, 0.1, 0.1)))
    ax[1].set_title(f"Baseline excluded (red): {pct(~baseline_static):.2f}%")
    ax[2].imshow(valid, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"Map comparison valid (white): {pct(valid):.2f}%")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#607080")
    for panel, key, name in [(3, "depth_score", "Depth"), (4, "rgb_score", "RGB")]:
        values = np.ma.masked_where(~valid | ~np.isfinite(arrays[key]), arrays[key])
        im = ax[panel].imshow(values, cmap=cmap, vmin=0, vmax=score_max)
        ax[panel].set_title(f"{name} residual / threshold (>1 = mismatch)")
        fig.colorbar(im, ax=ax[panel], shrink=0.7, extend="max")
    state_cmap = ListedColormap(["#85929e", "#2ecc71", "#e74c3c"])
    im = ax[5].imshow(arrays["state"], cmap=state_cmap,
                      norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    ax[5].set_title("Diagnostic states (not ground truth)")
    cb = fig.colorbar(im, ax=ax[5], ticks=[0, 1, 2], shrink=0.7)
    cb.ax.set_yticklabels(["Unknown", "Static", "Dynamic candidate"])
    for panel, key, color, title in [
        (6, "foreground_candidate", (1, 0.55, 0), "Observed surface closer"),
        (7, "revealed_candidate", (0, 0.7, 1), "Observed surface farther"),
    ]:
        mask = arrays[key].astype(bool)
        ax[panel].imshow(overlay(rgb, mask, color))
        ax[panel].set_title(f"{title}: {pct(mask):.2f}%")
    ax[8].imshow(overlay(rgb, candidate, (1, 0, 1)))
    ax[8].set_title(f"Baseline kept + map mismatch: {pct(candidate):.2f}%")
    for panel in ax:
        panel.axis("off")
    fig.suptitle(f"Frame {int(npz_path.stem)} | RGB timestamp {timestamp}\n"
                 "All percentages use the full image; gray-blue heatmap pixels are invalid.\n"
                 "Mismatch / closer surface is not proof of motion.", fontsize=12)
    fig.savefig(str(output), dpi=130)
    plt.close(fig)
    return {"map_valid_percent": pct(valid), "map_mismatch_percent": pct(inconsistent),
            "baseline_kept_mismatch_percent": pct(candidate),
            "mismatch_among_valid_percent": (100 * inconsistent.sum() / valid.sum()
                                              if valid.any() else "")}


def render_summary(path, output):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        return
    rows.sort(key=lambda r: r["frame"])
    x = [r["frame"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for key in ["map_valid_fraction", "map_inconsistent_fraction"]:
        axes[0].plot(x, [100 * r[key] for r in rows], label=key)
    for key in ["static_fraction", "unknown_fraction", "dynamic_candidate_fraction"]:
        axes[1].plot(x, [100 * r[key] for r in rows], label=key)
    ratio = [100 * r["map_inconsistent_fraction"] / r["map_valid_fraction"]
             if r["map_valid_fraction"] > 0 else np.nan for r in rows]
    axes[2].plot(x, ratio, label="Mismatch / map-valid pixels")
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Processed frame ID")
    fig.savefig(str(output), dpi=130)
    plt.close(fig)


def compare_depth_definitions(args, config, paths, output):
    """Paired diagnostics on saved support; no dataset or rerender required."""
    required = ['observed_depth', 'accumulated_depth', 'normalized_depth',
                'render_opacity', 'depth_tolerance', 'signed_depth_accumulated',
                'signed_depth_normalized', 'map_valid', 'baseline_static_mask',
                'rgb_score', 'depth_score', 'map_inconsistent']
    depth_min = float(config['StaticConsistency'].get('depth_min', 0.01))
    radius = args.edge_radii[len(args.edge_radii) // 2]
    rows, totals = [], {}
    colors = ['#202020', '#ff8900', '#00b8ef', '#bb55cc']
    labels = ['Neither', 'Normalized only', 'Accumulated only', 'Both']
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = set(required) - set(data.files)
            if missing:
                raise ValueError(f'{path.name}: missing diagnostics {sorted(missing)}; rerun recording')
            a = {k: data[k] for k in required}
        shape = a['observed_depth'].shape
        if len(shape) != 2 or any(x.shape != shape for x in a.values()):
            raise ValueError(f'{path.name}: inconsistent shapes')
        z, raw, norm, alpha, tol = [a[k] for k in required[:5]]
        ra, rn = a['signed_depth_accumulated'], a['signed_depth_normalized']
        for actual, expected in [(norm, raw / alpha.clip(min=1e-6)),
                                 (ra, raw-z), (rn, norm-z)]:
            if not np.allclose(actual, expected, equal_nan=True):
                raise ValueError(f'{path.name}: inconsistent recorded depth definitions')
        valid = a['map_valid'].astype(bool)
        if any(not np.isfinite(x[valid]).all() for x in [z, raw, norm, alpha, tol, ra, rn, a['rgb_score']]) or np.any(tol[valid] <= 0):
            raise ValueError(f'{path.name}: invalid values on map_valid')
        ba, bn = abs(ra) > tol, abs(rn) > tol
        rgb = a['rgb_score'] > 1
        definition = config.get('StaticConsistency', {}).get('depth_definition', 'normalized')
        if definition not in ('normalized', 'accumulated'):
            raise ValueError('Unknown depth_definition: ' + str(definition))
        chosen = ra if definition == 'accumulated' else rn
        chosen_bad = ba if definition == 'accumulated' else bn
        if not np.allclose(a['depth_score'][valid], abs(chosen[valid])/tol[valid]):
            raise ValueError(f'{path.name}: saved selected depth score disagrees')
        if not np.array_equal(a['map_inconsistent'].astype(bool), valid & (chosen_bad | rgb)):
            raise ValueError(f'{path.name}: saved RGB-D mask disagrees')
        regions, _, _ = split_depth_regions(z, depth_min, args.edge_abs, args.edge_rel, radius)
        scopes = {'all_valid': valid,
                  'baseline_kept_valid': valid & a['baseline_static_mask'].astype(bool),
                  'baseline_excluded_valid': valid & ~a['baseline_static_mask'].astype(bool)}
        for scope, support in scopes.items():
            for region, region_mask in {'all_regions': np.ones(shape, bool), **regions}.items():
                m = support & region_mask
                for modality, x, y in [('depth', ba, bn), ('rgbd', ba | rgb, bn | rgb)]:
                    counts = dict(support_pixels=int(m.sum()),
                                  accumulated_bad_pixels=int((m & x).sum()),
                                  normalized_bad_pixels=int((m & y).sum()),
                                  normalized_only_pixels=int((m & ~x & y).sum()),
                                  accumulated_only_pixels=int((m & x & ~y).sum()),
                                  both_pixels=int((m & x & y).sum()),
                                  neither_pixels=int((m & ~x & ~y).sum()))
                    key = (scope, region, modality)
                    total = totals.setdefault(key, {k: 0 for k in counts})
                    for k, v in counts.items(): total[k] += v
                    row = dict(frame=int(path.stem), scope=scope, region=region, modality=modality, **counts)
                    for name, residual in [('accumulated', ra), ('normalized', rn)]:
                        for q in [10, 50, 90]:
                            row[f'{name}_signed_p{q}_m'] = float(np.percentile(residual[m], q)) if m.any() else ''
                    rows.append(row)
        keep = scopes['baseline_kept_valid']
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        obs = np.ma.masked_where(~np.isfinite(z) | (z <= depth_min), z)
        im = axes[0, 0].imshow(obs, cmap='viridis')
        fig.colorbar(im, ax=axes[0, 0], label='metres')
        axes[0, 0].set_title('Observed depth (background reference)')
        for ax, residual, title in [(axes[0, 1], ra, 'Accumulated signed residual'),
                                    (axes[0, 2], rn, 'Normalized signed residual')]:
            im = ax.imshow(np.ma.masked_where(~keep, residual), cmap='coolwarm',
                           vmin=-args.residual_max, vmax=args.residual_max)
            fig.colorbar(im, ax=ax, label='metres; positive = rendered farther', extend='both')
            ax.set_title(title)
        for ax, x, y, title in [(axes[1, 0], ba, bn, 'Depth threshold comparison'),
                                (axes[1, 1], ba | rgb, bn | rgb, 'Final RGB-D comparison')]:
            ax.imshow(obs, cmap='gray')
            classes = np.where(x & y, 3, np.where(x, 2, np.where(y, 1, 0)))
            ax.imshow(np.ma.masked_where(~keep | (classes == 0), classes),
                      cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5, alpha=0.85)
            ax.set_title(title + '\nGray: no highlighted mismatch / outside support')
        im = axes[1, 2].imshow(np.ma.masked_where(~keep, alpha), vmin=0.95, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=axes[1, 2], label='accumulated opacity')
        axes[1, 2].set_title('Opacity on baseline-kept valid support')
        for ax in axes.flat: ax.axis('off')
        fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors[1:], labels[1:])],
                   loc='lower center', ncol=3)
        fig.suptitle(f'Frame {int(path.stem)} | paired definitions, identical saved support | not motion labels')
        fig.savefig(output / f'{path.stem}_depth_comparison.png', dpi=130)
        plt.close(fig)
    def rates(row):
        n = row['support_pixels']
        for key in ['accumulated_bad', 'normalized_bad', 'normalized_only', 'accumulated_only', 'both', 'neither']:
            row[key + '_percent_support'] = 100 * row[key + '_pixels'] / n if n else ''
        d = row['normalized_bad_pixels']
        row['normalized_only_percent_normalized_bad'] = 100 * row['normalized_only_pixels'] / d if d else ''
        return row
    pooled = [rates(dict(scope=k[0], region=k[1], modality=k[2], **v)) for k, v in totals.items()]
    rows = [rates(r) for r in rows]
    for name, data in [('depth_comparison_statistics.csv', rows), ('depth_comparison_summary.csv', pooled)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, region in enumerate(['all_regions', 'interior']):
        for j, modality in enumerate(['depth', 'rgbd']):
            ax = axes[i, j]
            selected = [r for r in rows if r['scope'] == 'baseline_kept_valid' and r['region'] == region and r['modality'] == modality]
            for key, label in [('accumulated_bad', 'Accumulated'), ('normalized_bad', 'Normalized')]:
                ax.plot([r['frame'] for r in selected],
                        [r[key+'_percent_support'] if r[key+'_percent_support'] != '' else np.nan for r in selected],
                        'o-', label=label)
            ax.set(title=f'{region}: {modality}', xlabel='Saved frame ID', ylabel='Mismatch / support (%)')
            ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(output / 'depth_comparison_trends.png', dpi=130); plt.close(fig)
    (output / 'depth_comparison_settings.json').write_text(json.dumps({
        'frames': [int(p.stem) for p in paths], 'region_radius_px': radius,
        'edge_abs_m': args.edge_abs, 'edge_rel': args.edge_rel, 'depth_min_m': depth_min,
        'residual_color_limit_m': args.residual_max, 'support': 'saved map_valid shared by both definitions',
        'config': config}, indent=2))
    (output / 'depth_comparison_README.txt').write_text(
        'Paired comparisons on the SAME saved map_valid support; thresholds are saved depth_tolerance.\n'
        'RGB-D = depth bad OR saved RGB score > 1. These are candidate exclusions, not actual applied masks.\n'
        'Scopes overlap: all_valid = kept + excluded. Regions overlap only with all_regions total.\n'
        'Do not sum scopes, total and regions, or depth and RGB-D rows together.\n'
        'Interior is away from observed-depth edges/holes, not ground-truth static.\n'
        'Summary pools counts over saved frames; repeated pixel observations are not independent.\n'
        'Signed quantiles include all support pixels, not just outliers; pooled quantiles are not reported.\n'
        'Empty percentages mean zero denominator. No accuracy, pose improvement or motion is inferred.\n'
        'Spatial plots show baseline-kept valid support; residuals share a fixed symmetric colour scale.\n'
        'No dataset needed: the background and depth strata use observed_depth saved in NPZ.\n')
    print('Depth comparison complete:', output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path,
                        help="Run directory containing config.yml and static_consistency/")
    parser.add_argument("--dataset", type=Path, help="Actual TUM/Bonn dataset root")
    parser.add_argument("--config", type=Path, help="Saved resolved config.yml (default: results/config.yml)")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--frames", type=int, nargs="+", help="Saved frame IDs; omit to render all NPZs")
    parser.add_argument("--score-max", type=float, default=4.0, help="Fixed heatmap maximum (>1)")
    parser.add_argument("--decompose", action="store_true",
                        help="Also split depth-only, RGB-only and joint mismatches; export plots and CSV")
    parser.add_argument("--depth-regions", action="store_true",
                        help="Stratify saved depth residuals by observed-depth edges and invalid neighborhoods")
    parser.add_argument("--edge-radii", type=int, nargs="+", default=[1, 3, 5],
                        help="Independent neighborhood radii in pixels (default: 1 3 5); middle radius is plotted per frame")
    parser.add_argument("--edge-abs", type=float, default=0.05,
                        help="Absolute neighboring-depth jump threshold in metres (default: 0.05)")
    parser.add_argument("--edge-rel", type=float, default=0.02,
                        help="Relative neighboring-depth jump threshold (default: 0.02)")
    parser.add_argument("--compare-depth", action="store_true", help="Standalone paired depth/RGB-D analysis from NPZ; no dataset required")
    parser.add_argument("--residual-max", type=float, default=0.2, help="Symmetric signed residual colour limit in metres")
    args = parser.parse_args()
    if not np.isfinite(args.residual_max) or args.residual_max <= 0:
        parser.error("--residual-max must be finite and positive")
    if args.compare_depth and (args.decompose or args.depth_regions):
        parser.error("--compare-depth is standalone; run other analysis modes separately")
    if not np.isfinite(args.score_max) or args.score_max <= 1:
        parser.error("--score-max must be finite and greater than 1")
    args.edge_radii = sorted(set(args.edge_radii))
    if any(r < 0 or r > 30 for r in args.edge_radii):
        parser.error("--edge-radii must be between 0 and 30 pixels")
    if not np.isfinite(args.edge_abs) or args.edge_abs <= 0:
        parser.error("--edge-abs must be finite and positive")
    if not np.isfinite(args.edge_rel) or args.edge_rel < 0:
        parser.error("--edge-rel must be finite and nonnegative")
    run = args.results.resolve()
    if run.name == "static_consistency":
        run = run.parent
    diagnostics = run / "static_consistency"
    if args.compare_depth and not diagnostics.is_dir():
        diagnostics = run  # Also accept flattened diagnostic exports.
    config_path = args.config or run / "config.yml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if config.get("Dataset", {}).get("type") != "tum":
        parser.error("Only repository TUM/Bonn Dataset.type=tum is supported")
    calibration = config["Dataset"]["Calibration"]
    # Saved config.yml includes resolved fields even when inherit_from remains.
    if not all(k in calibration for k in ("width", "height", "fx", "fy", "cx", "cy")):
        parser.error("Use the saved resolved config.yml, not a partial experiment YAML")
    available = sorted((p for p in diagnostics.glob("*.npz") if p.stem.isdigit()),
                       key=lambda p: int(p.stem))
    if args.frames is not None:
        requested = set(args.frames)
        missing = requested - {int(p.stem) for p in available}
        if missing:
            parser.error(f"No NPZ for frames {sorted(missing)}; saved frames: "
                         + str([int(p.stem) for p in available]))
        available = [p for p in available if int(p.stem) in requested]
    if not available:
        parser.error("No numeric NPZ files found; summary.jsonl alone cannot reconstruct spatial maps")
    if args.compare_depth:
        output = args.output or run / "depth_comparison"
        output.mkdir(parents=True, exist_ok=False)
        compare_depth_definitions(args, config, available, output)
        return
    if args.dataset is None:
        parser.error("--dataset is required unless --compare-depth is used")
    dataset = args.dataset.resolve()
    mapping = associate_rgbd(dataset, calibration)
    depth_min = float(config.get("StaticConsistency", {}).get("depth_min", 0.01))
    if not np.isfinite(depth_min) or depth_min <= 0:
        parser.error("StaticConsistency.depth_min must be finite and positive")
    if args.depth_regions and "depth_scale" not in calibration:
        parser.error("Depth stratification requires Calibration.depth_scale")
    for path in available:
        if int(path.stem) >= len(mapping):
            parser.error(f"Frame {path.stem} exceeds dataset slice length {len(mapping)}")
        rgb_path = dataset / mapping[int(path.stem)]["rgb_file"]
        if not rgb_path.is_file():
            parser.error("Missing RGB image: " + str(rgb_path))
        if args.depth_regions and not (dataset / mapping[int(path.stem)]["depth_file"]).is_file():
            parser.error("Missing associated depth image: " + mapping[int(path.stem)]["depth_file"])
    output = args.output or run / "consistency_visualizations"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    modality_rows = []
    depth_rows = []
    for path in available:
        association = mapping[int(path.stem)]
        timestamp, relative = association["rgb_timestamp"], association["rgb_file"]
        rgb = load_rgb(dataset / relative, calibration)
        target = output / (path.stem + ".png")
        stats = render_frame(path, rgb, target, timestamp, args.score_max)
        records.append({"frame": int(path.stem), **association,
                        "npz_file": path.name, **stats})
        print("Saved", target)
        if args.decompose:
            target = output / (path.stem + "_modalities.png")
            modality_rows.extend(render_decomposition(path, rgb, target, timestamp))
            print("Saved", target)
        if args.depth_regions:
            depth = load_depth(dataset / association["depth_file"], calibration)
            target = output / (path.stem + "_depth_regions.png")
            depth_rows.extend(render_depth_regions(path, rgb, depth, target, timestamp,
                              depth_min, args.edge_abs, args.edge_rel, args.edge_radii))
            print("Saved", target)
    with (output / "frame_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.decompose:
        with (output / "modality_statistics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(modality_rows[0]))
            writer.writeheader()
            writer.writerows(modality_rows)
        render_decomposition_trends(modality_rows, output / "modality_trends.png")
        (output / "modality_README.txt").write_text(
            "Categories use the configured depth-definition residuals: score > 1 is abnormal.\n"
            "depth_only: depth > 1 and RGB <= 1; rgb_only: reverse; both: both > 1.\n"
            "Only map_valid pixels count. Categories are disjoint and sum to map_inconsistent.\n"
            "Scopes: all_valid; baseline_kept_valid; baseline_excluded_valid.\n"
            "The latter two partition all_valid. Baseline kept does not mean ground-truth static.\n"
            "percent_image divides by all image pixels; percent_scope divides by valid pixels\n"
            "in that scope; percent_mismatch divides by mismatch pixels in that scope.\n"
            "Empty cells mean zero denominator, not zero error. Counts allow pooled statistics.\n"
            "Trends cover saved NPZ frames only. No full-sequence inference or temporal warping.\n"
            "Joint residuals do NOT prove motion; pose and map errors can affect both modalities.\n",
            encoding="utf-8")
    if args.depth_regions:
        pooled = pool_depth_region_rows(depth_rows)
        for filename, rows in [("depth_region_statistics.csv", depth_rows),
                               ("depth_region_summary.csv", pooled)]:
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        render_depth_region_summary(pooled, output / "depth_region_summary.png", args.edge_radii)
        settings = {"config_file": str(config_path.resolve()), "dataset": str(dataset),
                    "depth_scale": calibration["depth_scale"], "depth_min": depth_min,
                    "edge_abs_m": args.edge_abs, "edge_rel": args.edge_rel,
                    "edge_radii_px": args.edge_radii,
                    "plotted_radius_px": args.edge_radii[len(args.edge_radii) // 2],
                    "frame_ids": [int(p.stem) for p in available],
                    "neighborhood": "square / Chebyshev radius",
                    "edges": "4-neighbor valid depth pairs; both endpoints",
                    "priority": "invalid neighborhood, then edge neighborhood, then interior",
                    "depth_remapped": False}
        (output / "depth_region_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (output / "depth_region_README.txt").write_text(
            "Depth-region analysis uses observed sensor depth, NOT the residual, to define regions.\n"
            "Valid depth: finite and > the experiment's depth_min. No RGB-depth resizing or new warping.\n"
            "The repository loader undistorts RGB if configured, but does not remap depth; we match it.\n"
            "Edges: abs(z1-z2) > edge_abs + edge_rel * min(z1,z2), among valid 4-neighbor pairs.\n"
            "Both endpoints are seeds; square neighborhoods use the requested pixel radius.\n"
            "Near-invalid takes priority over near-edge; interior is the remaining observed-valid area.\n"
            "Invalid pixels themselves are not comparable. Image exterior is not an invalid-depth hole.\n"
            "These groups partition valid observed depth. Interior does NOT mean a planar/static surface.\n"
            "Scopes use saved map_valid and baseline_static_mask; baseline-kept is not ground truth.\n"
            "Each scope includes all_regions and three disjoint region rows; do NOT add all_regions again.\n"
            "depth_bad uses saved depth_score > 1, including both depth-only and joint RGB-depth errors.\n"
            "depth_bad_percent_support = depth_bad_pixels / support_pixels * 100 (region-conditioned).\n"
            "depth_only_percent_support excludes joint RGB-depth outliers.\n"
            "support_percent_scope = region support / all comparable pixels in that scope * 100.\n"
            "closer/farther percentages divide by DEPTH OUTLIERS in the region, not all region pixels.\n"
            "Direction comes from saved foreground_candidate/revealed_candidate, not motion labels.\n"
            "Signs below the depth threshold are unavailable; do not infer a signed mean from this analysis.\n"
            "depth_score_p50/p90/p95 include ALL comparable pixels in the region, not just anomalies.\n"
            "They are dimensionless residual/tolerance quantiles; >1 exceeds the saved detector threshold.\n"
            "Empty CSV cells mean undefined (zero denominator); n=0 is not evidence of zero error.\n"
            "Summary pools pixel counts across saved frames separately for each radius, never quantiles.\n"
            "Do not sum different radii: they are sensitivity settings on the SAME observations.\n"
            "No rendered-depth edges, temporal correspondence, pose diagnosis, or causality is inferred.\n"
            "Parameters are diagnostic starting values, not calibrated sensor noise or tuned thresholds.\n",
            encoding="utf-8")
    if (diagnostics / "summary.jsonl").is_file():
        render_summary(diagnostics / "summary.jsonl", output / "summary_trends.png")
    print("Done. Outputs:", output.resolve())
    print("Frame correspondence is reconstructed from this repository's loader rules.")
    print("If the server loader was modified, verify frame_index.csv before interpreting overlays.")


if __name__ == "__main__":
    main()
