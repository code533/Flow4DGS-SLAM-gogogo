"""V4.0a: non-interventional motion / map consistency diagnostics.

This module never writes to SLAM state. No pseudo-label is assumed to be GT.
Routines are intentionally CPU-testable except optional RAFT inference.
"""
import csv
from pathlib import Path
import numpy as np
import torch

from utils.v40_diagnostics import flow_cycle_error_px, static_render_snapshot


@torch.no_grad()
def full_flow_residual_px(observed_px, rigid_px, observed_depth):
    """Flow and prediction are [2,H,W] in the SAME pixel convention.

    Unlike segment_motion_parametric's legacy residual, retain prior-dynamic
    pixels. NaN means invalid depth/flow, never zero-fill invalid pixels.
    """
    if observed_px.shape != rigid_px.shape or observed_px.ndim != 3 or observed_px.shape[0] != 2:
        raise ValueError("Expected observed_px and rigid_px with shape [2,H,W]")
    depth = observed_depth.squeeze()
    if depth.shape != observed_px.shape[1:]:
        raise ValueError("depth shape mismatch")
    valid = ((depth > 0) & torch.isfinite(depth) &
             torch.isfinite(observed_px).all(0) & torch.isfinite(rigid_px).all(0))
    diff = (observed_px - rigid_px).norm(dim=0)
    return torch.where(valid, diff, torch.full_like(diff, float("nan"))), valid


@torch.no_grad()
def forward_raft_flow_px(viewpoint, previous):
    """Previous -> current RAFT flow, native pixel units, no cache mutation.

    Uses exactly the same model, input scaling, padding, and RAFT settings as
    Camera.generate_flow. Only execute for sampled V4.0a frames.
    """
    from RAFT.utils.utils import InputPadder
    device = next(viewpoint.model.parameters()).device
    prev = (previous.original_image.detach().to(device) * 255.0)[None]
    curr = (viewpoint.original_image.detach().to(device) * 255.0)[None]
    padder = InputPadder(prev.shape)
    prev, curr = padder.pad(prev, curr)
    _, fwd = viewpoint.model(prev, curr, iters=20, test_mode=True)
    return padder.unpad(fwd[0]).permute(1, 2, 0).detach()


def _stat(data, selected, reducer="mean"):
    data = np.asarray(data, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool) & np.isfinite(data)
    if not selected.any():
        return float("nan")
    vals = data[selected]
    if reducer == "median":
        return float(np.median(vals))
    return float(np.mean(vals))


def _corr(x, y, selected):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    selected = np.asarray(selected, bool) & np.isfinite(x) & np.isfinite(y)
    if selected.sum() < 10:
        return float("nan")
    a, b = x[selected], y[selected]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_frame(m):
    """Summaries from arrays with explicitly named validity and coverage masks."""
    valid_d = m["pre_depth_valid"].astype(bool)
    pre = m["pre_map_covered"].astype(bool)
    post = m["post_map_covered"].astype(bool)
    fvalid = m["flow_full_valid"].astype(bool)
    prior = m["prior_static"].astype(bool)
    dynamic = m["flow_dynamic"].astype(bool)
    common = valid_d & pre & post
    static_common = common & prior & ~dynamic
    raw_depth_valid = m["pre_render_depth_valid"].astype(bool) & valid_d
    flow = m["flow_full_residual_px"]
    rgb_pre, rgb_post = m["pre_rgb_residual"], m["post_rgb_residual"]
    depth_pre, depth_post = m["pre_depth_raw"], m["post_depth_raw"]
    row = dict(
        frame_id=int(np.asarray(m["frame_id"]).item()),
        n_total=int(pre.size),
        depth_valid_fraction=float(valid_d.mean()),
        map_covered_pre_fraction=float(pre.mean()),
        map_covered_post_fraction=float(post.mean()),
        map_uncovered_depth_valid_fraction=float((~pre & valid_d).mean()),
        map_covered_but_depth_missing_fraction=float((pre & ~valid_d).mean()),
        common_valid_fraction=float(common.mean()),
        common_static_valid_fraction=float(static_common.mean()),
        prior_dynamic_fraction=float((~prior).mean()),
        flow_dynamic_fraction=float(dynamic.mean()),
        flow_full_valid_fraction=float(fvalid.mean()),
        flow_full_median_px=_stat(flow, fvalid, "median"),
        flow_prior_dynamic_median_px=_stat(flow, fvalid & ~prior, "median"),
        flow_prior_static_median_px=_stat(flow, fvalid & prior, "median"),
        flow_dynamic_candidate_median_px=_stat(flow, fvalid & dynamic, "median"),
        depth_raw_defined_uncovered_fraction=float((raw_depth_valid & ~pre).mean()),
        rgb_pre_common=_stat(rgb_pre, common),
        rgb_post_common=_stat(rgb_post, common),
        depth_pre_common=_stat(depth_pre, common),
        depth_post_common=_stat(depth_post, common),
        rgb_pre_common_static=_stat(rgb_pre, static_common),
        rgb_post_common_static=_stat(rgb_post, static_common),
        depth_pre_common_static=_stat(depth_pre, static_common),
        depth_post_common_static=_stat(depth_post, static_common),
        rgb_pre_uncovered=_stat(rgb_pre, ~pre & valid_d),
        depth_raw_uncovered=_stat(depth_pre, ~pre & raw_depth_valid),
        flow_rgb_corr_covered=_corr(flow, rgb_pre, fvalid & valid_d & pre),
        flow_depth_corr_covered=_corr(flow, depth_pre, fvalid & valid_d & pre),
        flow_rgb_corr_prior_dynamic=_corr(flow, rgb_pre, fvalid & ~prior & pre),
    )
    if "fb_error_px" in m:
        fv = m["fb_valid"].astype(bool)
        fb = m["fb_error_px"]
        row["fb_valid_fraction"] = float(fv.mean())
        row["fb_median_px"] = _stat(fb, fv, "median")
        row["fb_flow_corr"] = _corr(fb, flow, fv & fvalid)
        row["fb_prior_dynamic_median_px"] = _stat(fb, fv & ~prior, "median")
        row["fb_prior_static_median_px"] = _stat(fb, fv & prior, "median")
    else:
        row.update(fb_valid_fraction=float("nan"), fb_median_px=float("nan"),
                   fb_flow_corr=float("nan"), fb_prior_dynamic_median_px=float("nan"),
                   fb_prior_static_median_px=float("nan"))
    return row


def assemble_frame(frame_id, pre, post, full_flow, full_valid, legacy_residual,
                   prior_static, flow_dynamic, final_static, observed_depth,
                   rendered_depth_pre, rendered_depth_post, rgb_observed,
                   rgb_rendered_pre, fb_error=None, fb_valid=None):
    """Compose diagnostic maps. Raw depth requires positive render depth.

    "raw" indicates *unthresholded by opacity*, not that unrendered background
    depth is geometrically valid.
    """
    m = {"frame_id": np.asarray(frame_id, dtype=np.int32)}
    for prefix, snap in (("pre", pre), ("post", post)):
        for key, value in snap.items():
            m[prefix + "_" + key] = np.asarray(value)
    odepth = np.asarray(observed_depth, np.float32)
    for prefix, pred in (("pre", rendered_depth_pre), ("post", rendered_depth_post)):
        d = np.asarray(pred, np.float32)
        valid = np.isfinite(d) & (d > 0)
        mask = valid & np.asarray(m[prefix + "_depth_valid"], bool)
        raw = np.full_like(d, np.nan)
        raw[mask] = np.abs(d[mask] - odepth[mask])
        m[prefix + "_render_depth_valid"] = valid
        m[prefix + "_depth_raw"] = raw
    m.update(flow_full_residual_px=np.asarray(full_flow, np.float32),
             flow_full_valid=np.asarray(full_valid, bool),
             flow_legacy_residual_px=np.asarray(legacy_residual, np.float32),
             prior_static=np.asarray(prior_static, bool),
             flow_dynamic=np.asarray(flow_dynamic, bool),
             final_static=np.asarray(final_static, bool),
             rgb_observed=np.asarray(rgb_observed, np.uint8),
             rgb_static_pre=np.asarray(rgb_rendered_pre, np.uint8),
             observed_depth=odepth)
    if fb_error is not None and fb_valid is not None:
        m["fb_error_px"] = np.asarray(fb_error, np.float32)
        m["fb_valid"] = np.asarray(fb_valid, bool)
    expected = odepth.shape
    for k, v in m.items():
        if k in ("frame_id", "rgb_observed", "rgb_static_pre"):
            continue
        if np.asarray(v).shape != expected:
            raise ValueError("V4.0a shape mismatch for %s" % k)
    return m


def save_frame(out_dir, maps, save_maps=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = analyze_frame(maps)
    csv_path = out / "v40a_per_frame.csv"
    present = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(row.keys()))
        if not present:
            w.writeheader()
        w.writerow(row)
    if save_maps:
        compact = {}
        for k, arr in maps.items():
            a = np.asarray(arr)
            # float32 diagnostic residuals preserve useful pixel precision;
            # RGB already uint8, masks bool. NPZ uses lossless compression.
            compact[k] = a
        np.savez_compressed(out / ("v40a_%05d.npz" % row["frame_id"]), **compact)
    return row
