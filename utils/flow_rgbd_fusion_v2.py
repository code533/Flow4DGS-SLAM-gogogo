"""Conservative, geometry-gated flow + RGB-D tracking reliability (experimental V2).

Inspired by UP-SLAM's static-map inconsistency; no DINO features or learned
uncertainty are used. 'candidate' is NOT a calibrated dynamic probability.
In Flow4DGS-SLAM `viewpoint.motion_mask=True` means static/inlier.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _hw(x, name):
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError("{} must be [H,W] or [1,H,W], got {}".format(name, tuple(x.shape)))


def _depth_edge_mask(depth, tolerance):
    """Exclude depth discontinuities plus one pixel around them."""
    z = depth
    scale = z.clamp_min(0.5)
    diffs = torch.zeros_like(z)
    dx = (z[:, 1:] - z[:, :-1]).abs() / torch.minimum(scale[:, 1:], scale[:, :-1])
    dy = (z[1:, :] - z[:-1, :]).abs() / torch.minimum(scale[1:, :], scale[:-1, :])
    diffs[:, 1:] = torch.maximum(diffs[:, 1:], dx)
    diffs[:, :-1] = torch.maximum(diffs[:, :-1], dx)
    diffs[1:, :] = torch.maximum(diffs[1:, :], dy)
    diffs[:-1, :] = torch.maximum(diffs[:-1, :], dy)
    return F.max_pool2d((diffs > tolerance).float()[None, None], 3, stride=1, padding=1)[0, 0] > 0


def compute_flow_rgbd_gate_v2(
    rendered_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_opacity: torch.Tensor,
    observed_rgb: torch.Tensor,
    observed_depth: torch.Tensor,
    original_static_mask: torch.Tensor,
    flow_residual_px: Optional[torch.Tensor],
    cfg: Dict,
) -> Dict:
    """Detached conservative weights. No input tensors are changed in place.

    Requires agreement between residual flow, front-surface depth occlusion,
    and appearance mismatch. Can ignore genuine dynamic pixels if one cue is
    missing, by design: first priority is retaining reliable static tracking.
    """
    with torch.no_grad():
        if rendered_rgb.ndim != 3 or rendered_rgb.shape[0] != 3:
            raise ValueError("rendered_rgb must be [3,H,W]")
        if observed_rgb.shape != rendered_rgb.shape:
            raise ValueError("observed_rgb shape mismatch")
        H, W = rendered_rgb.shape[-2:]
        device, dtype = rendered_rgb.device, rendered_rgb.dtype
        gt = observed_rgb.detach().to(device=device, dtype=dtype)
        pred = rendered_rgb.detach()
        pd = _hw(rendered_depth.detach(), "rendered_depth").to(device=device, dtype=dtype)
        gd = _hw(observed_depth.detach(), "observed_depth").to(device=device, dtype=dtype)
        op = _hw(rendered_opacity.detach(), "rendered_opacity").to(device=device, dtype=dtype)
        sta = _hw(original_static_mask.detach(), "original_static_mask").to(device=device, dtype=torch.bool)
        if any(x.shape != (H, W) for x in (pd, gd, op, sta)):
            raise ValueError("spatial shape mismatch")

        min_weight = float(cfg.get("min_weight", 0.70))
        min_coverage = float(cfg.get("min_coverage", 0.25))
        max_fraction = float(cfg.get("max_downweight_fraction", 0.12))
        if not (0 < min_weight <= 1 and 0 <= min_coverage <= 1 and 0 <= max_fraction <= 1):
            raise ValueError("invalid confidence limits")
        empty = torch.zeros((H, W), dtype=dtype, device=device)
        weights = torch.ones_like(empty)
        valid = torch.zeros((H, W), dtype=torch.bool, device=device)
        candidate = torch.zeros_like(valid)
        def out(reason, coverage=0.0, score=None):
            return {"weights": weights.detach(), "joint_score": (score if score is not None else empty).detach(),
                    "candidate": candidate.detach(), "valid": valid.detach(),
                    "coverage": float(coverage), "candidate_ratio": float(candidate.float().mean().item()),
                    "effective_ratio": float((weights < 0.999).float().mean().item()),
                    "applied": bool(reason == "applied"), "reason": reason}

        if flow_residual_px is None:
            return out("missing_flow")
        flow = _hw(flow_residual_px.detach(), "flow_residual_px").to(device=device, dtype=dtype)
        if flow.shape != (H, W):
            raise ValueError("flow spatial shape mismatch")
        lo, hi = float(cfg.get("min_depth", 0.01)), float(cfg.get("max_depth", 1000.0))
        valid = (sta & torch.isfinite(pd) & torch.isfinite(gd)
                 & (pd > lo) & (gd > lo) & (pd < hi) & (gd < hi)
                 & torch.isfinite(op) & (op >= float(cfg.get("opacity_min", 0.95)))
                 & torch.isfinite(gt).all(0) & torch.isfinite(pred).all(0)
                 & torch.isfinite(flow) & (flow >= 0))
        if bool(cfg.get("reject_depth_edges", True)):
            edge_thr = float(cfg.get("edge_threshold", 0.08))
            valid = valid & ~(_depth_edge_mask(gd, edge_thr) | _depth_edge_mask(pd, edge_thr))
        coverage = float(valid.sum().item()) / max(int(sta.sum().item()), 1)
        if coverage < min_coverage or not bool(valid.any()):
            return out("low_coverage", coverage)

        # Robust reference distribution excludes high-residual movers.
        fr = flow[valid]
        med = fr.median()
        robust_scale = (1.4826 * (fr - med).abs().median()).clamp_min(float(cfg.get("flow_scale_floor_px", 0.3)))
        z = ((flow - med) / robust_scale).clamp(0.0, 50.0)
        tz = max(float(cfg.get("flow_temperature", 0.5)), 1e-4)
        z_flow = float(cfg.get("flow_z_threshold", 2.0))
        p_flow = torch.sigmoid((z - z_flow) / tz)
        # Front occlusion: measured surface is *in front of* predicted static map.
        front = ((pd - gd) / gd.clamp_min(0.5)).clamp(-5.0, 5.0)
        td = max(float(cfg.get("front_temperature", 0.015)), 1e-5)
        p_front = torch.sigmoid((front - float(cfg.get("front_min_rel", 0.035))) / td)
        rgb_error = (pred - gt).abs().mean(0)
        tc = max(float(cfg.get("rgb_temperature", 0.035)), 1e-5)
        p_rgb = torch.sigmoid((rgb_error - float(cfg.get("rgb_min_error", 0.08))) / tc)
        score = (p_flow * p_front * p_rgb).clamp(0.0, 1.0)
        score = torch.where(valid, score, empty)
        candidate = valid & (score >= float(cfg.get("candidate_threshold", 0.55)))
        if not bool(candidate.any()):
            return out("no_joint_candidates", coverage, score)

        # Rate-limit downweighting, but retain the most confident candidates.
        indices = candidate.reshape(-1).nonzero(as_tuple=False).flatten()
        limit = max(0, int(math.floor(max_fraction * int(valid.sum().item()))))
        if limit == 0:
            return out("fraction_budget_zero", coverage, score)
        if indices.numel() > limit:
            best = torch.topk(score.reshape(-1)[indices], k=limit, sorted=False).indices
            selected = torch.zeros_like(candidate).view(-1)
            selected[indices[best]] = True
            candidate = selected.view(H, W)
        weights = torch.where(candidate, (1.0 - (1.0 - min_weight) * score).clamp_min(min_weight), weights)
        return out("applied", coverage, score)


def tracking_loss_v2(config, image, depth, opacity, viewpoint, weights):
    """Weighted Tracking loss preserving the original mean-loss scale.

    The normalization factors sum(M)/sum(M*W) apply separately to RGB and
    depth, giving bit-equivalent baseline loss for W==1 (up to FP roundoff).
    Called ONLY when the V2 gate is active; otherwise use original function.
    """
    if config["Training"]["monocular"]:
        raise ValueError("V2 only supports RGB-D")
    img = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
    gt = viewpoint.original_image.to(device=image.device, dtype=image.dtype)
    gt_z = torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype).reshape_as(depth)
    op = opacity.reshape_as(depth)
    w = _hw(weights.detach().to(device=image.device, dtype=image.dtype), "weights")[None]
    if w.shape != depth.shape:
        raise ValueError("weight image shape mismatch")
    static = viewpoint.motion_mask.to(device=image.device, dtype=torch.bool)[None] if viewpoint.uid > 0 else torch.ones_like(w, dtype=torch.bool)
    grad = viewpoint.grad_mask.to(device=image.device).reshape_as(w)
    rgb_m = ((gt.sum(dim=0, keepdim=True) > config["Training"]["rgb_boundary_threshold"]).to(image.dtype)
             * grad.to(image.dtype) * static.to(image.dtype))
    z_m = ((gt_z > 0.01) & (gt_z < 1000.0) & (op > 0.95) & static).to(image.dtype)
    def mass_normalized_loss(err, m, weights):
        numerator = m.sum()
        den = (m * weights).sum().clamp_min(1e-8)
        return ((err * m * weights).mean()) * numerator / den
    rgb_err = op * (img - gt).abs()
    depth_err = (depth - gt_z).abs()
    l_rgb = mass_normalized_loss(rgb_err, rgb_m, w)
    l_depth = mass_normalized_loss(depth_err, z_m, w)
    alpha = float(config["Training"].get("alpha", 0.95))
    return alpha * l_rgb + (1.0 - alpha) * l_depth
