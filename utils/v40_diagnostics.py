"""V4.0 observation-only diagnostics for Flow4DGS-SLAM.

These helpers NEVER modify a camera, Gaussian, optimizer, or mask.  All
snapshots are detached.  A high map residual is NOT a ground-truth dynamic
label; a high flow residual is NOT a flow-confidence estimate.
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def flow_cycle_error_px(backward_px, forward_px):
    """Current->previous->current cycle error in pixels.

    Both fields have shape [H,W,2], in pixel displacement units on the
    SAME image grid. Returns (error[H,W], valid[H,W]); invalid errors = NaN.
    """
    if backward_px.shape != forward_px.shape or backward_px.ndim != 3 or backward_px.shape[-1] != 2:
        raise ValueError("Expected matching [H,W,2] pixel-flow tensors")
    h, w, _ = backward_px.shape
    if h < 2 or w < 2:
        raise ValueError("Flow-cycle diagnostic requires H,W >= 2")
    device, dtype = backward_px.device, backward_px.dtype
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    x_prev = xx + backward_px[..., 0]
    y_prev = yy + backward_px[..., 1]
    coord_valid = (torch.isfinite(x_prev) & torch.isfinite(y_prev) &
                   torch.isfinite(backward_px).all(-1) &
                   (x_prev >= 0) & (x_prev <= w - 1) &
                   (y_prev >= 0) & (y_prev <= h - 1))
    sampling_grid = torch.stack((2 * x_prev.nan_to_num() / (w - 1) - 1,
                                 2 * y_prev.nan_to_num() / (h - 1) - 1), -1)[None]
    forward_finite = torch.isfinite(forward_px).all(-1)
    safe_forward = torch.nan_to_num(forward_px).permute(2, 0, 1)[None]
    sampled_forward = F.grid_sample(safe_forward, sampling_grid, align_corners=True,
                                    mode="bilinear", padding_mode="zeros")[0].permute(1, 2, 0)
    forward_coverage = F.grid_sample(forward_finite.to(dtype)[None, None],
                                     sampling_grid, align_corners=True,
                                     mode="bilinear", padding_mode="zeros")[0, 0]
    valid = coord_valid & (forward_coverage > 0.999)
    error = (backward_px + sampled_forward).norm(dim=-1)
    error = torch.where(valid, error, torch.full_like(error, float("nan")))
    return error, valid


@torch.no_grad()
def static_render_snapshot(viewpoint, static_render, opacity_min=0.95):
    """Create detached CPU arrays using *static-only* rendering.

    RGB residual uses the same exposure compensation as tracking loss.
    'map_covered' means opacity/depth support, NOT model confidence.
    """
    image = static_render["render"].detach()
    depth = static_render["depth"].detach().squeeze()
    opacity = static_render["opacity"].detach().squeeze()
    obs_color = viewpoint.original_image.to(image.device).detach()
    obs_depth = torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype).squeeze()
    rgb_adjusted = image * torch.exp(viewpoint.exposure_a.detach()) + viewpoint.exposure_b.detach()
    rgb_resid = (rgb_adjusted - obs_color).abs().mean(0)
    depth_resid = (depth - obs_depth).abs()
    valid_depth = torch.isfinite(obs_depth) & (obs_depth > 0.01) & (obs_depth < 1000)
    coverage = ((opacity >= float(opacity_min)) & torch.isfinite(depth) &
                (depth > 0) & torch.isfinite(opacity))
    valid_rgb = torch.isfinite(rgb_resid)
    rgb_resid = torch.where(valid_rgb, rgb_resid, torch.full_like(rgb_resid, float("nan")))
    depth_resid = torch.where(valid_depth & coverage, depth_resid,
                              torch.full_like(depth_resid, float("nan")))
    return {
        "rgb_residual": rgb_resid.float().cpu().numpy(),
        "depth_residual": depth_resid.float().cpu().numpy(),
        "map_covered": coverage.bool().cpu().numpy(),
        "depth_valid": valid_depth.bool().cpu().numpy(),
        "opacity": opacity.float().cpu().numpy(),
    }


def _stat(values, selector, kind="mean"):
    a = np.asarray(values, dtype=np.float64)
    mask = np.asarray(selector, dtype=bool) & np.isfinite(a)
    if not np.any(mask):
        return float("nan")
    if kind == "median":
        return float(np.median(a[mask]))
    return float(np.mean(a[mask]))


def _correlation(x, y, mask):
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask, bool)
    v = mask & np.isfinite(x) & np.isfinite(y)
    if int(v.sum()) < 10:
        return float("nan")
    a, b = x[v].astype(np.float64), y[v].astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_maps(frame_id, maps):
    """Summary statistics. Mask labels are *pseudo labels*, not annotated GT."""
    prior_static = maps["prior_static"].astype(bool)
    final_static = maps["final_static"].astype(bool)
    flow_dynamic = maps["flow_dynamic"].astype(bool)
    flow_valid = maps["flow_valid"].astype(bool)
    pre = maps["pre_map_covered"].astype(bool)
    post = maps["post_map_covered"].astype(bool)
    valid = maps["pre_depth_valid"].astype(bool)
    covered = pre & valid
    flow_resid = maps["flow_residual_px"]
    rgb = maps["pre_rgb_residual"]
    depth = maps["pre_depth_residual"]
    flow_good = covered & flow_valid
    rows = {
        "frame_id": int(frame_id),
        "depth_valid_fraction": float(valid.mean()),
        "map_covered_pre_fraction": float(pre.mean()),
        "map_covered_post_fraction": float(post.mean()),
        "map_covered_depth_valid_fraction": float(covered.mean()),
        "semantic_dynamic_fraction": float((~prior_static).mean()),
        "flow_dynamic_fraction": float(flow_dynamic.mean()),
        "final_dynamic_fraction": float((~final_static).mean()),
        "flow_residual_median_px": _stat(flow_resid, flow_valid, "median"),
        "flow_residual_mean_px": _stat(flow_resid, flow_valid),
        "rgb_pre_mean_covered": _stat(rgb, covered),
        "depth_pre_mean_covered": _stat(depth, covered),
        "rgb_post_mean_covered": _stat(maps["post_rgb_residual"], post & valid),
        "depth_post_mean_covered": _stat(maps["post_depth_residual"], post & valid),
        "rgb_pre_flow_dynamic_covered": _stat(rgb, covered & flow_dynamic),
        "rgb_pre_flow_static_covered": _stat(rgb, covered & prior_static & ~flow_dynamic),
        "depth_pre_flow_dynamic_covered": _stat(depth, covered & flow_dynamic),
        "depth_pre_flow_static_covered": _stat(depth, covered & prior_static & ~flow_dynamic),
        "flow_rgb_corr_covered": _correlation(flow_resid, rgb, flow_good),
        "flow_depth_corr_covered": _correlation(flow_resid, depth, flow_good),
        "flow_dynamic_map_uncovered_fraction": float((flow_dynamic & ~pre).sum() /
                                                        max(int(flow_dynamic.sum()), 1)),
        "map_uncovered_fraction": float((~pre).mean()),
        "flow_valid_fraction": float(flow_valid.mean()),
    }
    if "fb_error_px" in maps:
        fb = maps["fb_error_px"]
        fbvalid = maps["fb_valid"].astype(bool)
        rows["fb_valid_fraction"] = float(fbvalid.mean())
        rows["fb_median_px"] = _stat(fb, fbvalid, "median")
        rows["fb_mean_px"] = _stat(fb, fbvalid)
        rows["fb_flow_resid_corr"] = _correlation(fb, flow_resid, fbvalid & flow_valid)
    else:
        rows.update(fb_valid_fraction=float("nan"), fb_median_px=float("nan"),
                    fb_mean_px=float("nan"), fb_flow_resid_corr=float("nan"))
    return rows


def save_frame(output_dir, frame_id, pre, post, flow_residual_px, flow_dynamic,
               flow_valid, prior_static, final_static, fb_error_px=None,
               fb_valid=None, save_maps=True):
    """Append one frame CSV record and optionally archive diagnostic maps."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps = {}
    for prefix, snap in (("pre", pre), ("post", post)):
        for key, arr in snap.items():
            maps[prefix + "_" + key] = np.asarray(arr)
    maps.update(flow_residual_px=np.asarray(flow_residual_px, dtype=np.float32),
                flow_dynamic=np.asarray(flow_dynamic, dtype=bool),
                flow_valid=np.asarray(flow_valid, dtype=bool),
                prior_static=np.asarray(prior_static, dtype=bool),
                final_static=np.asarray(final_static, dtype=bool))
    if fb_error_px is not None and fb_valid is not None:
        maps["fb_error_px"] = np.asarray(fb_error_px, dtype=np.float32)
        maps["fb_valid"] = np.asarray(fb_valid, dtype=bool)
    row = summarize_maps(frame_id, maps)
    csv_path = out / "v40_per_frame.csv"
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    if save_maps:
        compact = {}
        for k, v in maps.items():
            a = np.asarray(v)
            compact[k] = a.astype(np.float16) if a.dtype.kind == "f" else a
        np.savez_compressed(out / ("v40_%05d.npz" % int(frame_id)), **compact)
    return row
