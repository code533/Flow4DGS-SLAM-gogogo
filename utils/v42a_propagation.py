"""V4.2a: confidence-guided dynamic Gaussian propagation.

A G0/G1/G2 ablation with an optional *shadow* mode: calculate all candidates
from identical observations, but commit ONLY the original G0 propagation.

This module makes no claims that FB agreement equals ground-truth flow error.
Only full-resolution mapping RAFT flow is supported: [H,W,2] normalized by
flow_native_px / [W,H] * 2; do not confuse it with renderer NDC coordinates.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def bidirectional_flow_reliability(flow_prev_to_curr, flow_curr_to_prev,
                                   depth_prev, depth_curr, fb_scale_px=1.0,
                                   edge_abs_m=0.08, edge_rel=0.06,
                                   edge_penalty=0.65):
    """Return Q_F on the PREVIOUS frame grid, plus diagnostic masks.

    This intentionally uses Mapping's already computed RAFT flow in BOTH
    directions and does not invoke RAFT again. Invalid correspondence Q=0.
    Nearest-depth consistency is a validity check, not a geometric
    displacement/occlusion ground truth.
    """
    forward = flow_prev_to_curr
    backward = flow_curr_to_prev
    if forward.ndim != 3 or forward.shape[-1] != 2 or forward.shape != backward.shape:
        raise ValueError("Expected matching full-res normalized [H,W,2] flows")
    h, w, _ = forward.shape
    if h < 2 or w < 2 or tuple(depth_prev.squeeze().shape) != (h, w) or tuple(depth_curr.squeeze().shape) != (h, w):
        raise ValueError("Flow/depth grid mismatch or dimensions <2")
    if fb_scale_px <= 0:
        raise ValueError("fb_scale_px must be > 0")
    d0 = depth_prev.squeeze().to(device=forward.device, dtype=forward.dtype)
    d1 = depth_curr.squeeze().to(device=forward.device, dtype=forward.dtype)
    sf = forward.new_tensor([w / 2.0, h / 2.0])
    fwd = forward * sf
    bwd = backward * sf
    yy, xx = torch.meshgrid(torch.arange(h, device=forward.device, dtype=forward.dtype),
                            torch.arange(w, device=forward.device, dtype=forward.dtype),
                            indexing="ij")
    sx, sy = xx + fwd[..., 0], yy + fwd[..., 1]
    finite = torch.isfinite(fwd).all(-1)
    in_bounds = finite & torch.isfinite(sx) & torch.isfinite(sy) & (
        sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    safe_x = torch.nan_to_num(sx, nan=0., posinf=0., neginf=0.)
    safe_y = torch.nan_to_num(sy, nan=0., posinf=0., neginf=0.)
    grid = torch.stack([2 * safe_x / (w - 1) - 1,
                        2 * safe_y / (h - 1) - 1], -1)[None]
    backward_valid = torch.isfinite(bwd).all(-1)
    bwd_safe = torch.nan_to_num(bwd).permute(2, 0, 1)[None]
    bwd_warped = F.grid_sample(bwd_safe, grid, align_corners=True,
                               mode="bilinear", padding_mode="zeros")[0].permute(1, 2, 0)
    bwd_valid_sample = F.grid_sample(
        backward_valid.to(d0.dtype)[None, None], grid,
        align_corners=True, mode="bilinear", padding_mode="zeros")[0, 0] > .999
    curr_depth_valid = torch.isfinite(d1) & (d1 > .01)
    curr_depth_warped = F.grid_sample(
        curr_depth_valid.to(d0.dtype)[None, None], grid,
        align_corners=True, mode="nearest", padding_mode="zeros")[0, 0] > .5
    prev_depth_valid = torch.isfinite(d0) & (d0 > .01)
    valid = in_bounds & bwd_valid_sample & prev_depth_valid & curr_depth_warped
    cycle = torch.linalg.vector_norm(fwd + bwd_warped, dim=-1)
    cycle = torch.where(valid, cycle, torch.full_like(cycle, float("nan")))
    q = torch.zeros((h, w), device=forward.device, dtype=forward.dtype)
    q[valid] = torch.exp(-cycle[valid].clamp(min=0) / float(fb_scale_px))
    # Penalize depth discontinuities (local edges, not an occlusion GT).
    d_safe = torch.where(prev_depth_valid, d0, torch.zeros_like(d0))
    d_min = -F.max_pool2d(-torch.where(prev_depth_valid, d_safe, torch.full_like(d0, float("inf")))[None, None],
                          3, 1, 1)[0, 0]
    d_max = F.max_pool2d(torch.where(prev_depth_valid, d_safe, torch.full_like(d0, float("-inf")))[None, None],
                         3, 1, 1)[0, 0]
    depth_step = (torch.isfinite(d_min) & torch.isfinite(d_max) &
                  (d_max - d_min > torch.maximum(
                      torch.full_like(d0, float(edge_abs_m)),
                      d_min.clamp_min(0) * float(edge_rel))))
    if edge_penalty < 0 or edge_penalty > 1:
        raise ValueError("edge_penalty must be in [0,1]")
    q *= torch.where(depth_step, float(edge_penalty), 1.)
    return torch.clamp(q, 0, 1), {
        "cycle_error_px": cycle,
        "correspondence_valid": valid,
        "depth_edge": depth_step,
        "out_of_bounds": ~in_bounds,
    }


@torch.no_grad()
def gather_prior_gaussian_confidence(uvz, q_map):
    """Nearest-neighbor Q_F from Gaussian projection in previous frame.

    uvz [N,3] is the existing perspective-divided NDC projection.
    IMPORTANT: clamp only AFTER requiring coordinates inside [-1,1].
    """
    if uvz.ndim != 2 or uvz.shape[1] < 2 or q_map.ndim != 2:
        raise ValueError("Unexpected projection/Q shape")
    h, w = q_map.shape
    u, v = uvz[:, 0], uvz[:, 1]
    supported = (torch.isfinite(u) & torch.isfinite(v) &
                 (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1))
    x = torch.nan_to_num((u + 1) * (w - 1) / 2, nan=0).round().long().clamp(0, w-1)
    y = torch.nan_to_num((v + 1) * (h - 1) / 2, nan=0).round().long().clamp(0, h-1)
    confidence = q_map[y, x]
    return torch.where(supported, confidence, torch.zeros_like(confidence)).detach()


def knn_candidates_from_precomputed(deformation, confidence, dists2, indices,
                                    g0, radius=0.2, sigma=0.05, eps=1e-8,
                                    g2_min_support=.45, g2_min_neighbors=3,
                                    neighbor_q_min=.3):
    """Compute G1, G2 from SAME KNN graph and G0, with no map update.

    G0 is supplied by the original smoother, guaranteeing exact legacy
    fallback and apples-to-apples shadow comparison. G1 is reliability-
    weighted with fallback ONLY when no weighted support exists. G2 uses
    G0 when local reliable support/neighbor count is insufficient.
    """
    n = deformation.shape[0]
    if (deformation.shape != (n, 3) or confidence.shape != (n,) or
            dists2.ndim != 2 or indices.shape != dists2.shape or
            dists2.shape[0] != n or g0.shape != deformation.shape):
        raise ValueError("Inconsistent KNN/point data shapes")
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive")
    if sigma <= 0 or g2_min_neighbors < 1:
        raise ValueError("Invalid sigma/neighbors")
    q = torch.nan_to_num(confidence.detach().to(deformation.dtype), nan=0., posinf=0., neginf=0.).clamp(0,1)
    idx = indices.long()
    distance_ok = torch.isfinite(dists2) & (dists2 >= 0)
    if radius is not None:
        distance_ok = distance_ok & (dists2 <= float(radius)**2)
    base_weights = torch.exp(-torch.clamp_min(dists2, 0) / (2*float(sigma)**2 + eps))
    base_weights = torch.where(distance_ok, base_weights, torch.zeros_like(base_weights))
    # Preserve original KNN smoothing behavior of retaining self.
    base_weights[:, 0] = torch.clamp_min(base_weights[:, 0], eps)
    neighbors = deformation[idx]
    nq = q[idx]
    weighted = base_weights * nq
    weight_mass = weighted.sum(dim=1, keepdim=True)
    weighted_displacement = (weighted[...,None] * neighbors).sum(1) / (weight_mass + eps)
    sufficient = weight_mass[:, 0] > 10 * eps
    g1 = torch.where(sufficient[:,None], weighted_displacement, g0)
    support = weight_mass[:,0] / (base_weights.sum(1) + eps)
    num_good = ((nq >= neighbor_q_min) & distance_ok).sum(dim=1)
    g2_keep = sufficient & (support >= g2_min_support) & (num_good >= g2_min_neighbors)
    g2 = torch.where(g2_keep[:,None], g1, g0)
    return g1, g2, {
        "support": support.detach(),
        "reliable_neighbor_count": num_good.detach(),
        "g1_fallback": (~sufficient).detach(),
        "g2_fallback": (~g2_keep).detach(),
    }


def displacement_stats(keyframe_id, mode, raw, g0, g1, g2, q, aux,
                       diag):
    """Summaries, NOT dynamic reconstruction metrics."""
    n = int(raw.shape[0])
    def avg(x):
        if x.numel()==0: return float("nan")
        return float(x.detach().float().mean().cpu())
    def median(x):
        if x.numel()==0: return float("nan")
        return float(x.detach().float().median().cpu())
    return dict(
        keyframe_id=int(keyframe_id), mode=str(mode), n_points=n,
        q_mean=avg(q), q_median=median(q),
        q_zero_fraction=avg((q <= 1e-6).float()),
        q_valid_image_fraction=avg(diag["correspondence_valid"].float()),
        fb_median_px=median(diag["cycle_error_px"][diag["correspondence_valid"]]),
        g0_disp_mean_m=avg(torch.linalg.vector_norm(g0, dim=-1)),
        g1_vs_g0_mean_m=avg(torch.linalg.vector_norm(g1-g0, dim=-1)),
        g2_vs_g0_mean_m=avg(torch.linalg.vector_norm(g2-g0, dim=-1)),
        g1_fallback_fraction=avg(aux["g1_fallback"].float()),
        g2_fallback_fraction=avg(aux["g2_fallback"].float()),
        support_median=median(aux["support"]),
        neighbor_count_mean=avg(aux["reliable_neighbor_count"].float()),
    )


def save_diagnostics(directory, row, raw=None, g0=None, g1=None, g2=None,
                     confidence=None, save_points=False):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    csv_path=directory/"v42a_per_keyframe.csv"
    is_new=not csv_path.exists() or csv_path.stat().st_size==0
    with csv_path.open("a", encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(row))
        if is_new:writer.writeheader()
        writer.writerow(row)
    if save_points and raw is not None:
        np.savez_compressed(directory/("v42a_%05d.npz"%row["keyframe_id"]),
                            raw=raw.detach().cpu().numpy(),
                            g0=g0.detach().cpu().numpy(),
                            g1=g1.detach().cpu().numpy(),
                            g2=g2.detach().cpu().numpy(),
                            q=confidence.detach().cpu().numpy())
