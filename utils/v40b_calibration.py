"""V4.0b: non-invasive optical-flow unit/visibility diagnostics.

Motion residual, forward-backward agreement, and visibility are DISTINCT
observables. None of these masks is a GT dynamic or occlusion annotation.
"""
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F

from utils.v40a_diagnostics import analyze_frame as analyze_v40a_frame


def normalized_tracking_flow_to_native_px(flow, height, width):
    """Invert Camera.generate_flow's normalization (flow_native / [W,H] * 2).

    Input [H,W,2] current->previous normalized displacement; output native
    RAFT displacement in PIXELS. Do not use the (W-1)/2 grid_sample scaling.
    The existing SLAM motion fitter retains its original convention.
    """
    if flow.ndim != 3 or tuple(flow.shape) != (height, width, 2):
        raise ValueError("Expected [H,W,2] flow")
    factors = flow.new_tensor([width / 2.0, height / 2.0])
    return flow * factors


@torch.no_grad()
def depth_boundary_mask(depth, abs_m=0.08, rel=0.06, dilate=2):
    """Depth-discontinuity *proximity* mask, not an occlusion ground truth."""
    if depth.ndim != 2:
        raise ValueError("depth must be [H,W]")
    finite = torch.isfinite(depth) & (depth > 0)
    d = torch.where(finite, depth, torch.zeros_like(depth))
    neg_inf = torch.full_like(d, float("-inf"))
    pos_inf = torch.full_like(d, float("inf"))
    dmax = F.max_pool2d(torch.where(finite, d, neg_inf)[None, None], 3, 1, 1)[0, 0]
    dmin = -F.max_pool2d(-torch.where(finite, d, pos_inf)[None, None], 3, 1, 1)[0, 0]
    minimum = torch.clamp_min(dmin, 0)
    valid_neighbors = torch.isfinite(dmin) & torch.isfinite(dmax)
    threshold = torch.maximum(torch.full_like(d, float(abs_m)), minimum * float(rel))
    edge = finite & valid_neighbors & ((dmax - dmin) > threshold)
    if dilate > 0:
        radius = int(dilate)
        edge = F.max_pool2d(edge.float()[None, None], 2 * radius + 1,
                            stride=1, padding=radius)[0, 0] > 0
    return edge


@torch.no_grad()
def visibility_proxies(current_depth, previous_depth, backward_native_px,
                       fb_error_px=None, fb_valid=None,
                       abs_edge_m=0.08, rel_edge=0.06,
                       dilate_edge=2, fb_bad_px=1.0):
    """Compute per-pixel validity, border/edge and *possible* occlusion cues.

    Warps previous-frame validity/edges using current->previous optical flow.
    Does NOT compare current/previous depths directly (camera and objects may
    move). Suspected occlusions are heuristic and must never be used as GT.
    """
    d = current_depth.squeeze()
    prev = previous_depth.squeeze()
    if d.shape != prev.shape or d.ndim != 2:
        raise ValueError("Current/previous depth must be matching [H,W]")
    h, w = d.shape
    if h < 2 or w < 2 or backward_native_px.shape != (h, w, 2):
        raise ValueError("Flow/depth dimensions mismatch")
    yy, xx = torch.meshgrid(torch.arange(h, device=d.device, dtype=d.dtype),
                            torch.arange(w, device=d.device, dtype=d.dtype),
                            indexing="ij")
    f = backward_native_px.to(d.device, dtype=d.dtype)
    x = xx + f[..., 0]
    y = yy + f[..., 1]
    valid_curr = torch.isfinite(d) & (d > 0)
    finite_flow = torch.isfinite(f).all(-1)
    inside = finite_flow & torch.isfinite(x) & torch.isfinite(y) & (
        x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x_safe = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
    y_safe = torch.nan_to_num(y, nan=0, posinf=0, neginf=0)
    grid = torch.stack([2.0 * x_safe / (w - 1) - 1.0,
                        2.0 * y_safe / (h - 1) - 1.0], -1)[None]
    prev_valid = (torch.isfinite(prev) & (prev > 0)).to(d.dtype)
    prev_valid_at_flow = F.grid_sample(
        prev_valid[None, None], grid, mode="nearest",
        padding_mode="zeros", align_corners=True)[0, 0] > .5
    curr_edge = depth_boundary_mask(d, abs_edge_m, rel_edge, dilate_edge)
    prev_edge = depth_boundary_mask(prev, abs_edge_m, rel_edge, dilate_edge)
    prev_edge_at_flow = F.grid_sample(
        prev_edge.to(d.dtype)[None, None], grid, mode="nearest",
        padding_mode="zeros", align_corners=True)[0, 0] > .5
    edge_near = curr_edge | (inside & prev_edge_at_flow)
    temporal_depth_support = valid_curr & inside & prev_valid_at_flow
    if fb_error_px is None:
        fb_error_px = torch.full_like(d, float("nan"))
    if fb_valid is None:
        fb_valid = torch.zeros_like(valid_curr)
    fb_bad = fb_valid & torch.isfinite(fb_error_px) & (fb_error_px > fb_bad_px)
    occlusion_suspect = valid_curr & finite_flow & (
        ~inside | ~prev_valid_at_flow | (edge_near & fb_bad))
    interior = temporal_depth_support & fb_valid & (~edge_near) & (~occlusion_suspect)
    return {
        "boundary_near": edge_near,
        "boundary_current": curr_edge,
        "boundary_previous_warped": prev_edge_at_flow & inside,
        "flow_in_bounds": inside,
        "previous_depth_valid_warped": prev_valid_at_flow & inside,
        "temporal_depth_support": temporal_depth_support,
        "possible_occlusion": occlusion_suspect,
        "interior_correspondence": interior,
    }


def _median(values, m):
    v = np.asarray(values, np.float64)
    valid = np.asarray(m, bool) & np.isfinite(v)
    return float(np.median(v[valid])) if valid.any() else float("nan")


def _fraction(numerator, denominator):
    den = int(np.asarray(denominator, bool).sum())
    return float(np.asarray(numerator, bool).sum() / den) if den else float("nan")


def summarize_calibration(m, high_motion_px=5.0, low_fb_px=0.5):
    """Aggregate disjoint prior/flow groups on VALID comparable pixels."""
    row = dict(analyze_v40a_frame(m))
    flow = np.asarray(m["flow_full_residual_px"], float)
    fb = np.asarray(m["fb_error_px"], float)
    full_valid = np.asarray(m["flow_full_valid"], bool)
    fb_valid = np.asarray(m["fb_valid"], bool)
    interior = np.asarray(m["interior_correspondence"], bool)
    prior = np.asarray(m["prior_static"], bool)
    motion = np.asarray(m["flow_dynamic"], bool)
    groups = {
        "prior_dynamic": ~prior,
        "flow_dynamic": prior & motion,
        "static_candidate": prior & ~motion,
    }
    row["boundary_near_fraction"] = float(np.asarray(m["boundary_near"],bool).mean())
    row["possible_occlusion_fraction"] = float(np.asarray(m["possible_occlusion"],bool).mean())
    row["interior_fraction"] = float(interior.mean())
    row["out_of_bounds_fraction"] = float((~np.asarray(m["flow_in_bounds"],bool) & full_valid).mean())
    for name, mask in groups.items():
        comparable = mask & full_valid & fb_valid & interior
        large = comparable & (flow > high_motion_px)
        stable = large & (fb < low_fb_px)
        row[name + "_n_common_interior"] = int(comparable.sum())
        row[name + "_n_high_motion_interior"] = int(large.sum())
        row[name + "_flow_med_interior_px"] = _median(flow, comparable)
        row[name + "_fb_med_interior_px"] = _median(fb, comparable)
        row[name + "_high_motion_fraction_interior"] = _fraction(large, comparable)
        row[name + "_low_fb_given_high_motion"] = _fraction(stable, large)
    row["high_motion_threshold_px"] = float(high_motion_px)
    row["low_fb_threshold_px"] = float(low_fb_px)
    return row


def save_v40b(outdir, m, save_maps=True, high_motion_px=5.0, low_fb_px=0.5):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    row = summarize_calibration(m, high_motion_px, low_fb_px)
    csv_path = out / "v40b_per_frame.csv"
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    if save_maps:
        np.savez_compressed(out / ("v40b_%05d.npz" % row["frame_id"]), **m)
    return row
