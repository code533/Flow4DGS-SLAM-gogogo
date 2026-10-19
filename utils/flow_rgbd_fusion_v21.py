"""V2.1: interpretable, conservative flow/RGB-D two-branch tracking fusion.

Research prototype inspired by UP-SLAM's static-map inconsistency approach.
Scores are *not* calibrated dynamic probabilities. In this SLAM implementation
motion_mask=True means static/inlier; it is never modified here.
"""
import math

import torch
import torch.nn.functional as F


def _hw(x, name):
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError("{} must be [H,W] or [1,H,W]".format(name))


def _edges(depth, threshold):
    """Depth-discontinuity exclusion; expanded one pixel to both sides."""
    d = depth
    s = d.clamp_min(0.5)
    dx = (d[:, 1:] - d[:, :-1]).abs() / torch.minimum(s[:, 1:], s[:, :-1])
    dy = (d[1:, :] - d[:-1, :]).abs() / torch.minimum(s[1:, :], s[:-1, :])
    e = torch.zeros_like(d)
    e[:, 1:] = torch.maximum(e[:, 1:], dx)
    e[:, :-1] = torch.maximum(e[:, :-1], dx)
    e[1:, :] = torch.maximum(e[1:, :], dy)
    e[:-1, :] = torch.maximum(e[:-1, :], dy)
    return F.max_pool2d((e > threshold).float()[None, None], 3, 1, 1)[0, 0].bool()


@torch.no_grad()
def compute_flow_rgbd_gate_v21(
    rendered_rgb, rendered_depth, rendered_opacity,
    observed_rgb, observed_depth, original_static_mask,
    flow_residual_px, cfg,
):
    """Produce a detached reliability map without touching SLAM's hard mask.

    'product' reproduces V2's three-score product (+ V2 threshold).
    'layered' uses a flow prior and RGB-D appearance/depth union, followed by
    soft additive ranking with independent minimum priors. Both modes are
    subject to coverage and budget safeguards.
    """
    if rendered_rgb.ndim != 3 or rendered_rgb.shape[0] != 3:
        raise ValueError("rendered_rgb must be [3,H,W]")
    if observed_rgb.shape != rendered_rgb.shape:
        raise ValueError("RGB shape mismatch")
    h, w = rendered_rgb.shape[-2:]
    device, dtype = rendered_rgb.device, rendered_rgb.dtype

    def asmap(x, name, dt=dtype):
        if x is None:
            return None
        z = _hw(x.detach(), name).to(device=device, dtype=dt)
        if z.shape != (h, w):
            raise ValueError("{} spatial shape mismatch".format(name))
        return z

    pred = rendered_rgb.detach()
    obs = observed_rgb.detach().to(device=device, dtype=dtype)
    zd = asmap(rendered_depth, "rendered_depth")
    zo = asmap(observed_depth, "observed_depth")
    op = asmap(rendered_opacity, "rendered_opacity")
    static = asmap(original_static_mask, "original_static_mask", torch.bool)
    flow = asmap(flow_residual_px, "flow_residual_px") if flow_residual_px is not None else None

    mode = str(cfg.get("fusion_mode", "layered"))
    if mode not in ("layered", "product"):
        raise ValueError("fusion_mode must be layered or product")
    min_w = float(cfg.get("min_weight", 0.8))
    coverage_min = float(cfg.get("min_coverage", 0.25))
    budget = float(cfg.get("max_downweight_fraction", 0.05))
    min_candidates = int(cfg.get("min_candidate_pixels", 32))
    rgb_share = float(cfg.get("rgb_share", 0.55))
    flow_share = float(cfg.get("flow_share", 0.55))
    diagnostic_only = bool(cfg.get("diagnostic_only", False))
    if not (0 < min_w <= 1 and 0 <= coverage_min <= 1 and
            0 <= budget <= 1 and 0 <= rgb_share <= 1 and
            0 <= flow_share <= 1 and min_candidates >= 0):
        raise ValueError("invalid fusion configuration")

    zero = torch.zeros((h, w), device=device, dtype=dtype)
    empty = torch.zeros((h, w), device=device, dtype=torch.bool)
    ones = torch.ones_like(zero)
    maps = {k: zero.clone() for k in
            ("p_flow", "p_rgb", "p_front", "p_map", "score_product",
             "score_layered", "score_selected")}
    valid = empty.clone()
    raw_candidates = empty.clone()
    candidates = empty.clone()
    weights = ones.clone()
    counters = {"flow_prior": 0, "map_prior": 0, "both_priors": 0,
                "raw_candidates": 0, "selected": 0}

    def result(reason, coverage):
        active = bool(reason == "applied")
        out = dict(maps)
        out.update({
            "weights": weights.detach(), "valid": valid.detach(),
            "candidate_raw": raw_candidates.detach(),
            "candidate": candidates.detach(),
            "original_static": static.detach(),
            "coverage": float(coverage),
            "reason": reason, "applied": active,
            "mode": mode, "diagnostic_only": diagnostic_only,
            "count_flow_prior": counters["flow_prior"],
            "count_map_prior": counters["map_prior"],
            "count_both_priors": counters["both_priors"],
            "count_raw": counters["raw_candidates"],
            "count_selected": counters["selected"],
            "effective_ratio": (float((weights < 0.999).sum().item()) / (h*w)),
        })
        return out

    if flow is None:
        return result("missing_flow", 0.0)
    valid = (static & torch.isfinite(zd) & torch.isfinite(zo)
             & (zd > float(cfg.get("min_depth", 0.01)))
             & (zo > float(cfg.get("min_depth", 0.01)))
             & (zd < float(cfg.get("max_depth", 1000.0)))
             & (zo < float(cfg.get("max_depth", 1000.0)))
             & torch.isfinite(op) & (op >= float(cfg.get("opacity_min", 0.95)))
             & torch.isfinite(pred).all(0) & torch.isfinite(obs).all(0)
             & torch.isfinite(flow) & (flow >= 0))
    if cfg.get("reject_depth_edges", True):
        tol = float(cfg.get("edge_threshold", 0.08))
        valid &= ~(_edges(zo, tol) | _edges(zd, tol))
    coverage = valid.sum().item() / max(static.sum().item(), 1)
    if coverage < coverage_min or not bool(valid.any()):
        return result("low_coverage", coverage)

    # Flow motion score is robustly standardized within already-static pixels,
    # not a new 6DoF fit. Reuse flow residuals supplied by the original system.
    fr = flow[valid]
    med = fr.median()
    mad = (fr - med).abs().median()
    scale = (1.4826 * mad).clamp_min(float(cfg.get("flow_scale_floor_px", 0.3)))
    z = ((flow - med) / scale).clamp(0, 50)
    p_flow = torch.sigmoid(
        (z - float(cfg.get("flow_z_threshold", 2.0))) /
        max(float(cfg.get("flow_temperature", 0.5)), 1e-5))
    front_delta = ((zd - zo) / zo.clamp_min(0.5)).clamp(-5, 5)
    p_front = torch.sigmoid(
        (front_delta - float(cfg.get("front_min_rel", 0.035))) /
        max(float(cfg.get("front_temperature", 0.015)), 1e-5))
    rgb_err = torch.nan_to_num((pred - obs).abs().mean(0),
                               nan=0.0, posinf=0.0, neginf=0.0)
    p_rgb = torch.sigmoid(
        (rgb_err - float(cfg.get("rgb_min_error", 0.08))) /
        max(float(cfg.get("rgb_temperature", 0.035)), 1e-5))

    # Two complementary branches. RGB and front depth are alternative
    # static-map inconsistency cues; they no longer both have to be strong.
    p_map = (rgb_share * p_rgb + (1 - rgb_share) * p_front)
    product = p_flow * p_rgb * p_front
    layered = flow_share * p_flow + (1 - flow_share) * p_map
    for key, val in [
        ("p_flow", p_flow), ("p_rgb", p_rgb), ("p_front", p_front),
        ("p_map", p_map), ("score_product", product),
        ("score_layered", layered),
        ("score_selected", layered if mode == "layered" else product),
    ]:
        maps[key] = torch.where(valid, val, zero).detach()

    flow_prior = valid & (p_flow >= float(cfg.get("flow_prior_min", 0.35)))
    map_prior = valid & (p_map >= float(cfg.get("map_prior_min", 0.35)))
    counters["flow_prior"] = int(flow_prior.sum().item())
    counters["map_prior"] = int(map_prior.sum().item())
    counters["both_priors"] = int((flow_prior & map_prior).sum().item())
    if mode == "layered":
        raw_candidates = flow_prior & map_prior & (
            layered >= float(cfg.get("layered_threshold", 0.55)))
    else:
        raw_candidates = valid & (
            product >= float(cfg.get("product_threshold", 0.55)))
    counters["raw_candidates"] = int(raw_candidates.sum().item())
    if counters["raw_candidates"] == 0:
        return result("no_candidates", coverage)

    # Use only the most confident candidates; never exceed the pixel budget.
    indices = raw_candidates.flatten().nonzero(as_tuple=False).flatten()
    limit = int(math.floor(budget * valid.sum().item()))
    if limit < min_candidates:
        return result("budget_below_minimum", coverage)
    if indices.numel() > limit:
        rank = maps["score_selected"].flatten()[indices]
        chosen = torch.topk(rank, k=limit, sorted=False).indices
        keep = indices[chosen]
    else:
        keep = indices
    counters["selected"] = int(keep.numel())
    if counters["selected"] < min_candidates:
        return result("too_few_candidates", coverage)
    candidates = empty.clone()
    candidates.view(-1)[keep] = True

    if diagnostic_only:
        return result("diagnostic_only", coverage)

    s = maps["score_selected"]
    # With high but imperfect confidence, reduce rather than delete pixels.
    weights = torch.where(candidates,
                          (1 - (1 - min_w) * s).clamp(min=min_w),
                          ones).detach()
    return result("applied", coverage)
