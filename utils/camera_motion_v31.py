"""Experimental V3.1 geometry-aware IRLS with safe caller-side fallback.

Uses the SAME linear interaction matrix as slam_backend.py (pixel units).
Returned scores are observation reliabilities, not dynamic probabilities.
No ground-truth information enters the fitting algorithm.
"""
import math
import torch
import torch.nn.functional as F


def interaction_rows(depth, K, ys, xs):
    """Return interaction matrices for individual pixels [N,2,6]."""
    Z = depth[ys, xs].clamp_min(1e-6)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = xs.to(Z.dtype) - cx
    v = ys.to(Z.dtype) - cy
    zero = torch.zeros_like(Z)
    du = torch.stack([-fx / Z, zero, u / Z, u*v/fy, -(fx + u*u/fx), v], -1)
    dv = torch.stack([zero, -fy / Z, v / Z, fy + v*v/fy, -u*v/fx, -u], -1)
    return torch.stack([du, dv], dim=1)


def _edge_strength(depth, flow):
    """Symmetric relative depth and optical-flow neighbor discontinuities."""
    H, W = depth.shape
    d = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    f = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    depth_grad = torch.zeros_like(d)
    flow_grad = torch.zeros_like(d)
    if W > 1:
        e = (d[:, 1:] - d[:, :-1]).abs() / torch.minimum(d[:, 1:].clamp_min(.5), d[:, :-1].clamp_min(.5))
        g = (f[:, :, 1:] - f[:, :, :-1]).square().sum(0).sqrt()
        depth_grad[:, 1:] = torch.maximum(depth_grad[:, 1:], e)
        depth_grad[:, :-1] = torch.maximum(depth_grad[:, :-1], e)
        flow_grad[:, 1:] = torch.maximum(flow_grad[:, 1:], g)
        flow_grad[:, :-1] = torch.maximum(flow_grad[:, :-1], g)
    if H > 1:
        e = (d[1:, :] - d[:-1, :]).abs() / torch.minimum(d[1:, :].clamp_min(.5), d[:-1, :].clamp_min(.5))
        g = (f[:, 1:, :] - f[:, :-1, :]).square().sum(0).sqrt()
        depth_grad[1:, :] = torch.maximum(depth_grad[1:, :], e)
        depth_grad[:-1, :] = torch.maximum(depth_grad[:-1, :], e)
        flow_grad[1:, :] = torch.maximum(flow_grad[1:, :], g)
        flow_grad[:-1, :] = torch.maximum(flow_grad[:-1, :], g)
    return depth_grad, flow_grad


@torch.no_grad()
def fit_geometry_irls(depth, flow_px, K, inlier_mask, cfg):
    """Fit 6DoF with shared 2D Cauchy weights, edge reliability and grid balance.

    Returns (xi | None, diagnostic dict). Caller MUST use legacy fallback on None.
    All inputs on a single device. A deterministic spatial stride controls cost.
    """
    if depth.ndim != 2 or flow_px.shape != (2, *depth.shape) or K.shape != (3, 3) or inlier_mask.shape != depth.shape:
        raise ValueError("depth [H,W], flow [2,H,W], K [3,3], mask [H,W] required")
    stride = int(cfg.get('sampling_stride', 4))
    gh, gw = int(cfg.get('grid_rows', 6)), int(cfg.get('grid_cols', 8))
    iterations = int(cfg.get('iterations', 12))
    min_points = int(cfg.get('min_points', 600))
    min_cells = int(cfg.get('min_cells', 12))
    max_condition = float(cfg.get('max_condition', 1e7))
    min_depth, max_depth = float(cfg.get('min_depth', .1)), float(cfg.get('max_depth', 15.))
    if min(stride, gh, gw, iterations, min_points, min_cells) < 1 or max_condition <= 1 or max_depth <= min_depth:
        raise ValueError("invalid V3.1 configuration")
    H, W = depth.shape
    valid = (inlier_mask.bool() & torch.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
             & torch.isfinite(flow_px).all(0))
    ys, xs = torch.nonzero(valid[::stride, ::stride], as_tuple=True)
    ys, xs = ys * stride, xs * stride
    n = int(xs.numel())
    diag = dict(status='uninitialized', n_points=n, occupied_cells=0, condition=float('inf'),
                scale_px=float('nan'), median_residual_px=float('nan'),
                depth_edge_fraction=float('nan'), flow_edge_fraction=float('nan'))
    if n < min_points:
        diag['status'] = 'low_points'
        return None, diag
    # Valid samples are uniform image-grid samples, not data-dependent top-k.
    cid = ((ys * gh // max(H, 1)).clamp_max(gh-1) * gw + (xs * gw // max(W, 1)).clamp_max(gw-1)).long()
    counts = torch.bincount(cid, minlength=gh*gw).float()
    occupied = int((counts > 0).sum().item())
    diag['occupied_cells'] = occupied
    if occupied < min_cells:
        diag['status'] = 'low_grid_coverage'
        return None, diag
    dg, fg = _edge_strength(depth, flow_px)
    ed = float(cfg.get('depth_edge_relative', .06))
    ef = float(cfg.get('flow_edge_px', 3.))
    wd = float(cfg.get('depth_edge_weight', .35))
    wf = float(cfg.get('flow_edge_weight', .60))
    grid_mix = float(cfg.get('grid_balance', .5))
    if ed <= 0 or ef <= 0 or not (0 < wd <= 1 and 0 < wf <= 1 and 0 <= grid_mix <= 1):
        raise ValueError('invalid edge / spatial weighting')
    d_edge = dg[ys, xs] > ed
    f_edge = fg[ys, xs] > ef
    diag['depth_edge_fraction'] = float(d_edge.float().mean().item())
    diag['flow_edge_fraction'] = float(f_edge.float().mean().item())
    base = torch.where(d_edge, wd, 1.).float() * torch.where(f_edge, wf, 1.).float()
    # Grid balancing blended with uniform sampling to limit cell dominance.
    nonzero_counts = counts[counts > 0]
    target = nonzero_counts.mean().clamp_min(1.)
    area = (target / counts[cid].clamp_min(1.)).clamp(.25, 4.)
    base = base * ((1. - grid_mix) + grid_mix * area)
    base = base / base.mean().clamp_min(1e-6)
    L = interaction_rows(depth, K, ys, xs).to(torch.float64)
    f = flow_px[:, ys, xs].permute(1, 0).to(torch.float64)
    prior = base.to(torch.float64)
    if not bool(torch.isfinite(L).all()) or not bool(torch.isfinite(f).all()):
        diag['status'] = 'invalid_jacobian'
        return None, diag
    # Normalize Jacobian columns to make the condition threshold meaningful.
    col_norm = torch.linalg.vector_norm((L * prior[:,None,None].sqrt()).reshape(-1,6), dim=0).clamp_min(1e-12)
    N = (L / col_norm).reshape(-1,6)
    x = torch.zeros(6, device=depth.device, dtype=torch.float64)
    robust = torch.ones(n, device=depth.device, dtype=torch.float64)
    min_scale = float(cfg.get('robust_scale_floor_px', .25))
    if min_scale <= 0:
        raise ValueError('invalid robust floor')
    for _ in range(iterations):
        weight = prior * robust
        row_weights = weight[:,None].expand(n,2).reshape(-1)
        # Weighted normal equations are small 6x6 and solved in float64.
        info = N.T @ (N * row_weights[:,None])
        b = N.T @ (f.reshape(-1) * row_weights)
        vals = torch.linalg.eigvalsh(info)
        if not bool(torch.isfinite(vals).all()) or float(vals[0]) <= 1e-12:
            diag['status'] = 'singular'
            return None, diag
        cond = float((vals[-1] / vals[0]).item())
        diag['condition'] = cond
        if not math.isfinite(cond) or cond > max_condition:
            diag['status'] = 'ill_conditioned'
            return None, diag
        try:
            x = torch.linalg.solve(info, b)
        except RuntimeError:
            diag['status'] = 'solve_failed'
            return None, diag
        xi = x / col_norm
        residual = (f - torch.einsum('nij,j->ni', L, xi)).square().sum(1).sqrt()
        center = residual.median()
        spread = (1.4826 * (residual - center).abs().median()).clamp_min(min_scale)
        diag['scale_px'] = float(spread.item())
        diag['median_residual_px'] = float(center.item())
        robust = 1. / (1. + (residual / (2. * spread)).square())
    if not bool(torch.isfinite(xi).all()):
        diag['status'] = 'non_finite_solution'
        return None, diag
    diag['status'] = 'ok'
    return xi.to(dtype=depth.dtype), diag
