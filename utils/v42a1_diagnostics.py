"""V4.2a.1 read-only failure attribution for Gaussian motion propagation.

No prediction or update rule is altered. All masks and statistics are
diagnostics, not ground-truth flow error or dynamic-entity labels.
"""
import csv
from pathlib import Path

import numpy as np
import torch


REASON_NAMES = {
    0: "valid_nonzero_q",
    1: "nonfinite_projection",
    2: "projection_outside_image",
    3: "previous_depth_invalid",
    4: "forward_flow_invalid",
    5: "forward_correspondence_outside_image",
    6: "backward_flow_invalid_at_warp",
    7: "current_depth_invalid_at_warp",
    8: "q_near_zero_despite_valid_correspondence",
}


@torch.no_grad()
def projection_reasons(uvz, q_map, flow_diag, near_zero=1e-6):
    """Assign one primary reason to EVERY original dynamic Gaussian.

    Projection is the SAME NDC -> nearest-pixel mapping used by
    gather_prior_gaussian_confidence, except that reasons distinguish the
    out-of-image points from pixel-level invalidity.

    Reasons 0 and 8 denote pixels that passed all validity checks.
    Near-zero is a NUMERIC criterion, not verified flow error.
    """
    if uvz.ndim != 2 or uvz.shape[1] < 2 or q_map.ndim != 2:
        raise ValueError("Expected uvz [N,3], q_map [H,W]")
    h, w = q_map.shape
    required = (
        "prev_depth_valid", "forward_finite", "forward_in_bounds",
        "backward_valid_warped", "current_depth_valid_warped",
        "correspondence_valid", "depth_edge")
    for key in required:
        if key not in flow_diag or flow_diag[key].shape != (h, w):
            raise ValueError("Missing or mismatched flow diagnostic: " + key)
    u, v = uvz[:, 0], uvz[:, 1]
    finite = torch.isfinite(u) & torch.isfinite(v)
    inside = finite & (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)
    safe_x = torch.nan_to_num((u + 1) * (w - 1) / 2, nan=0., posinf=0., neginf=0.)
    safe_y = torch.nan_to_num((v + 1) * (h - 1) / 2, nan=0., posinf=0., neginf=0.)
    px = safe_x.round().long().clamp(0, w-1)
    py = safe_y.round().long().clamp(0, h-1)
    reason = torch.full((uvz.shape[0],), -1, dtype=torch.int16, device=uvz.device)

    def assign(predicate, code):
        take = (reason < 0) & predicate
        reason[take] = code

    assign(~finite, 1)
    assign(~inside, 2)
    sample = lambda key: flow_diag[key][py, px].bool()
    assign(~sample("prev_depth_valid"), 3)
    assign(~sample("forward_finite"), 4)
    assign(~sample("forward_in_bounds"), 5)
    assign(~sample("backward_valid_warped"), 6)
    assign(~sample("current_depth_valid_warped"), 7)
    q_sample = q_map[py, px]
    # This check guards against diagnostics drifting from the actual Q map.
    unresolved = inside & (reason < 0)
    if (unresolved & ~sample("correspondence_valid")).any():
        raise AssertionError("Unattributed invalid pixel (diag mismatch)")
    assign(q_sample <= float(near_zero), 8)
    assign(reason < 0, 0)
    if (reason < 0).any():
        raise AssertionError("Some Gaussian reliability failures unclassified")

    # Match the original sampler exactly, including outside-zero behavior.
    effective_q = torch.where(inside, q_sample, torch.zeros_like(q_sample))
    return {
        "reason": reason,
        "projection_finite": finite,
        "projection_inside": inside,
        "projected_pixel_xy": torch.stack([px, py], dim=1),
        "q_sample": effective_q.detach(),
        "at_depth_edge": inside & sample("depth_edge"),
    }


@torch.no_grad()
def knn_motion_diagnostics(raw, q, dists2, indices, g0, g1, g2,
                           radius=.2, sigma=.05, neighbor_q_min=.3,
                           motion_mismatch_m=.10, gate_m=.3, eps=1e-8):
    """Per-Gaussian movement, neighbor influence, G0/G1/G2 0.3m gating.

    Uses the ALREADY sampled KNN graph. No changes to G0/G1/G2.
    Effective increment is zero where update_delta_xyz excludes a point.
    """
    n = raw.shape[0]
    if (raw.shape != (n, 3) or q.shape != (n,) or
            g0.shape != raw.shape or g1.shape != raw.shape or g2.shape != raw.shape or
            dists2.shape != indices.shape or dists2.shape[0] != n):
        raise ValueError("KNN input shapes inconsistent")
    if radius is not None and radius <= 0 or sigma <= 0 or gate_m <= 0:
        raise ValueError("Invalid KNN radius/sigma/gate")
    if n == 0:
        raise ValueError("Expected at least one Gaussian")
    d = dists2.detach()
    idx = indices.long().detach()
    if idx.numel() and (idx.min() < 0 or idx.max() >= n):
        raise ValueError("KNN neighbor index outside selected Gaussian list")
    inside = torch.isfinite(d) & (d >= 0)
    if radius is not None:
        inside &= d <= radius**2
    base = torch.exp(-d.clamp_min(0) / (2*sigma**2 + eps))
    base = torch.where(inside, base, torch.zeros_like(base))
    base[:, 0] = base[:, 0].clamp_min(eps)   # original KNN self safeguard
    confidence = torch.nan_to_num(q.detach(), nan=0., posinf=0., neginf=0.).clamp(0,1)
    neighbor_q = confidence[idx]
    weight_q = base * neighbor_q
    mass = weight_q.sum(-1)
    mass_fraction = mass / (base.sum(-1) + eps)
    relative_motion = torch.linalg.vector_norm(raw[idx] - raw[:,None,:], dim=-1)
    local_motion_disagreement = (
        (base * relative_motion).sum(-1) / (base.sum(-1) + eps))
    trusted_motion_disagreement = (
        (weight_q * relative_motion).sum(-1) / (mass + eps))
    dominant_slot = weight_q.argmax(-1)
    dominant_id = idx.gather(1, dominant_slot[:,None])[:,0]
    dominant_weight = weight_q.gather(1, dominant_slot[:,None])[:,0] / (mass + eps)
    row = torch.arange(n, device=raw.device)
    dominant_motion_diff = relative_motion[row, dominant_slot]
    dominates_other = (dominant_id != row) & (dominant_weight > .5)
    cross_motion_risk = (
        (confidence < neighbor_q_min) & dominates_other &
        (dominant_motion_diff >= motion_mismatch_m))
    strong_mismatch_neighbor = (
        (relative_motion >= motion_mismatch_m) & (neighbor_q >= .7) & inside
    ).any(-1)
    norms = torch.stack([torch.linalg.vector_norm(x, dim=-1) for x in
                         (raw, g0, g1, g2)])
    passes = norms < gate_m  # exact condition in update_delta_xyz
    effective = torch.stack([torch.where(ok[:,None], v, torch.zeros_like(v))
                              for ok,v in zip(passes, (raw,g0,g1,g2))])
    return {
        "neighbor_count_in_radius": inside.sum(-1),
        "trusted_neighbor_count": (inside & (neighbor_q >= neighbor_q_min)).sum(-1),
        "neighbor_motion_disagreement_m": local_motion_disagreement,
        "trusted_motion_disagreement_m": trusted_motion_disagreement,
        "support": mass_fraction,
        "dominant_neighbor_index": dominant_id,
        "dominant_neighbor_weight_fraction": dominant_weight,
        "dominant_neighbor_motion_diff_m": dominant_motion_diff,
        "cross_motion_risk_proxy": cross_motion_risk,
        "has_strong_mismatched_neighbor": strong_mismatch_neighbor,
        "gate_pass_raw": passes[0],
        "gate_pass_g0": passes[1],
        "gate_pass_g1": passes[2],
        "gate_pass_g2": passes[3],
        "gate_g0_to_g1_rejected": passes[1] & ~passes[2],
        "gate_g0_to_g1_accepted": ~passes[1] & passes[2],
        "gate_g0_to_g2_rejected": passes[1] & ~passes[3],
        "gate_g0_to_g2_accepted": ~passes[1] & passes[3],
        "g1_minus_g0_norm_m": torch.linalg.vector_norm(g1-g0, dim=-1),
        "g2_minus_g0_norm_m": torch.linalg.vector_norm(g2-g0, dim=-1),
        "effective_g1_minus_g0_norm_m": torch.linalg.vector_norm(
            effective[2]-effective[1],dim=-1),
        "effective_g2_minus_g0_norm_m": torch.linalg.vector_norm(
            effective[3]-effective[1],dim=-1),
    }


@torch.no_grad()
def diagnostic_row(frame_id, sampled, selected_indices, all_reasons, neighborhood,
                   aux):
    reason = sampled["reason"]
    n = int(reason.numel())
    if n != neighborhood["gate_pass_g0"].numel():
        raise ValueError("Point-order misalignment between reasons and KNN")
    all_codes = all_reasons["reason"]
    def pct(x):
        return float(x.float().mean().cpu()) if x.numel() else float("nan")
    def avg(x):
        return float(x.float().mean().cpu()) if x.numel() else float("nan")
    def quantile(x, p):
        return float(torch.quantile(x.float(), p).cpu()) if x.numel() else float("nan")
    out = {
        "keyframe_id": int(frame_id),
        "n_selected_points": n,
        "n_all_dynamic_gaussians": int(all_codes.numel()),
        "q_zero_exact_fraction": pct(sampled["q_sample"] == 0),
        "q_near_zero_fraction": pct(sampled["q_sample"] <= 1e-6),
        "all_dynamic_projection_outside_fraction": pct(all_codes == 2),
        "all_dynamic_invalid_projection_fraction": pct(all_codes == 1),
        "selected_at_depth_edge_fraction": pct(sampled["at_depth_edge"]),
        "selected_cross_motion_risk_proxy_fraction": pct(
            neighborhood["cross_motion_risk_proxy"]),
        "selected_strong_mismatched_neighbor_fraction": pct(
            neighborhood["has_strong_mismatched_neighbor"]),
        "neighbor_motion_disagreement_mean_m": avg(
            neighborhood["neighbor_motion_disagreement_m"]),
        "trusted_motion_disagreement_mean_m": avg(
            neighborhood["trusted_motion_disagreement_m"]),
        "g1_diff_p95_m": quantile(neighborhood["g1_minus_g0_norm_m"], .95),
        "g1_diff_p99_m": quantile(neighborhood["g1_minus_g0_norm_m"], .99),
        "g1_diff_max_m": float(neighborhood["g1_minus_g0_norm_m"].max().cpu()),
        "g1_diff_gt_0p1m_fraction": pct(
            neighborhood["g1_minus_g0_norm_m"] > .1),
        "g0_gate_pass_fraction": pct(neighborhood["gate_pass_g0"]),
        "g1_gate_pass_fraction": pct(neighborhood["gate_pass_g1"]),
        "g2_gate_pass_fraction": pct(neighborhood["gate_pass_g2"]),
        "g0_to_g1_new_rejections_n": int(neighborhood["gate_g0_to_g1_rejected"].sum().cpu()),
        "g0_to_g1_new_acceptances_n": int(neighborhood["gate_g0_to_g1_accepted"].sum().cpu()),
        "g0_to_g2_new_rejections_n": int(neighborhood["gate_g0_to_g2_rejected"].sum().cpu()),
        "g0_to_g2_new_acceptances_n": int(neighborhood["gate_g0_to_g2_accepted"].sum().cpu()),
        "effective_g1_g0_mean_m": avg(
            neighborhood["effective_g1_minus_g0_norm_m"]),
        "effective_g2_g0_mean_m": avg(
            neighborhood["effective_g2_minus_g0_norm_m"]),
        "g1_fallback_fraction": pct(aux["g1_fallback"]),
        "g2_fallback_fraction": pct(aux["g2_fallback"]),
    }
    for scope, codes in (("selected", reason), ("all_dynamic", all_codes)):
        for key, title in REASON_NAMES.items():
            out["%s_reason_%s_fraction" % (scope,title)] = pct(codes == key)
    if sum(int((reason == code).sum()) for code in REASON_NAMES) != n:
        raise AssertionError("Reasons must be mutually exclusive and exhaustive")
    return out


def save_diag(directory, row, selected, neighborhood, points_xyz, raw,
              g0, g1, g2, gaussian_indices, save_points=False):
    """CSV every eligible KF; optional detailed selected-KF NPZ."""
    dest = Path(directory); dest.mkdir(parents=True, exist_ok=True)
    csvfile=dest/"v42a1_per_keyframe.csv"
    new=not csvfile.exists() or csvfile.stat().st_size == 0
    with csvfile.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,row.keys())
        if new: writer.writeheader()
        writer.writerow(row)
    if save_points:
        maps={
            "keyframe_id": np.asarray(row["keyframe_id"], np.int32),
            "selected_gaussian_index": gaussian_indices,
            "xyz_at_prev": points_xyz,
            "raw": raw, "g0": g0, "g1": g1, "g2": g2,
        }
        maps.update({"projection_"+k:v for k,v in selected.items()})
        maps.update({"knn_"+k:v for k,v in neighborhood.items()})
        final={}
        for key,val in maps.items():
            final[key]=(val.detach().cpu().numpy() if isinstance(val,torch.Tensor)
                        else np.asarray(val))
        np.savez_compressed(dest/("v42a1_%05d.npz" % row["keyframe_id"]), **final)
