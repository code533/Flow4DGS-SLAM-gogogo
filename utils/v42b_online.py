"""V4.2b: select and audit ONLINE Gaussian propagation from frozen candidates.

Uses the existing G0/G3/G4 + V4.2a.3 Support candidate. No new
loss, tracking weight, dynamic mask or re-projection. The original
update_delta_xyz(..., max_motion=0.3) remains the sole writer.

Mode "off" is the untouched legacy path, "shadow" is G0 with
candidates computed, and "g0" is the instrumented G0 baseline.
"""
import csv
from pathlib import Path

import torch

ONLINE_MODES = ("g0", "g3", "g3_support", "g4")
ALL_MODES = ("off", "shadow", "g0", "g1", "g2", "g3", "g3_support", "g4")


def validate_online_config(mode, cfg, *, use_knn, flow_offset, dynamic_model, flow_ds):
    if mode not in ALL_MODES:
        raise ValueError("Unknown V4.2b propagation mode: %s" % mode)
    if (cfg.get("g4_ablation", False) or cfg.get("g34_shadow", False)) and mode != "shadow":
        raise ValueError("g4_ablation/g34_shadow require mode='shadow'")
    if mode in ("g1", "g2", "g3", "g3_support", "g4"):
        if not (use_knn and flow_offset and dynamic_model == "offset"):
            raise ValueError("Online propagation needs offset dynamic_model, flow_offset and use_knn")
    if mode != "off" and flow_ds != 1:
        raise ValueError("Confidence propagation requires Training.flow_ds=1")
    if mode == "off" and cfg.get("online_trace", False):
        raise ValueError("online_trace requires mode=g0/g3/g3_support/g4")
    if mode == "shadow" and cfg.get("online_trace", False):
        raise ValueError("Use mode=g0 for online_trace baseline; shadow has its own CSV")


@torch.no_grad()
def choose_online_motion(mode, candidates):
    if mode not in ("shadow",) + ONLINE_MODES:
        raise ValueError("choose_online_motion only supports shadow/g0/g3/g3_support/g4")
    needed=("g0","g3","g3_support","g4")
    for key in needed:
        if key not in candidates:
            raise ValueError("Missing candidate: " + key)
    g0=candidates["g0"]
    if g0.ndim!=2 or g0.shape[-1]!=3:
        raise ValueError("Expected [N,3] Gaussian displacements")
    for key in needed:
        if candidates[key].shape!=g0.shape:
            raise ValueError("Candidate shape mismatch: "+key)
        if not torch.isfinite(candidates[key]).all():
            raise FloatingPointError("Non-finite candidate: "+key)
    selected=candidates["g0" if mode=="shadow" else mode].detach()
    return selected


@torch.no_grad()
def online_row(frame_id, mode, g0, selected, q, reasons, g3,
               support, g4, g34diag, stage_gates, *,
               gate_m=.3, compute_ms=0., update_ms=0.):
    if mode not in ONLINE_MODES:
        raise ValueError("online_row requires g0/g3/g3_support/g4")
    if g0.ndim!=2 or g0.shape[-1]!=3 or g0.shape[0]<1:
        raise ValueError("Empty or malformed Gaussian displacements")
    n=len(g0)
    for x in (selected,g3,support,g4):
        if x.shape!=g0.shape:
            raise ValueError("Candidate shapes not aligned")
    if q.shape!=(n,) or reasons.shape!=(n,):
        raise ValueError("Reliability/reasons not aligned")
    if gate_m<=0:
        raise ValueError("gate_m must be positive")
    norm=lambda x:torch.linalg.vector_norm(x.detach(),dim=-1)
    diff=norm(selected-g0)
    norm_g0=norm(g0); norm_selected=norm(selected)
    pass_g0=norm_g0<gate_m; pass_selected=norm_selected<gate_m
    effective_g0=torch.where(pass_g0[:,None],g0,torch.zeros_like(g0))
    effective_selected=torch.where(pass_selected[:,None],selected,torch.zeros_like(selected))
    effective_diff=norm(effective_selected-effective_g0)
    def avg(x):return float(x.float().mean().cpu())
    def frac(x):return avg(x)
    def p95(x):return float(torch.quantile(x.float(),.95).cpu())
    row=dict(
        keyframe_id=int(frame_id),mode=mode,n_points=n,
        q_mean=avg(q),q_near_zero_fraction=frac(q<=1e-6),
        projection_invalid_fraction=frac((reasons==1)|(reasons==2)),
        observation_valid_fraction=frac(g34diag["visible"]),
        support_selected_fraction=frac(stage_gates["support_only"]),
        g4_accepted_fraction=frac(g34diag["g4_accepted"]),
        g0_mean_motion_m=avg(norm_g0),
        selected_mean_motion_m=avg(norm_selected),
        selected_vs_g0_mean_m=avg(diff),
        selected_vs_g0_p95_m=p95(diff),
        selected_vs_g0_gt_10cm_fraction=frac(diff>.1),
        g3_vs_g0_mean_m=avg(norm(g3-g0)),
        support_vs_g0_mean_m=avg(norm(support-g0)),
        g4_vs_g0_mean_m=avg(norm(g4-g0)),
        g0_gate_pass_n=int(pass_g0.sum().cpu()),
        selected_gate_pass_n=int(pass_selected.sum().cpu()),
        selected_gate_pass_fraction=frac(pass_selected),
        new_gate_rejections_n=int((pass_g0&~pass_selected).sum().cpu()),
        new_gate_acceptances_n=int((~pass_g0&pass_selected).sum().cpu()),
        effective_vs_g0_mean_m=avg(effective_diff),
        finite_selected_fraction=frac(torch.isfinite(selected).all(-1)),
        candidate_compute_ms=float(compute_ms),
        gaussian_update_ms=float(update_ms),
    )
    return row


def append_online_row(directory,row):
    folder=Path(directory)
    folder.mkdir(parents=True,exist_ok=True)
    filename=folder/"v42b_per_keyframe.csv"
    new=not filename.exists() or filename.stat().st_size==0
    with filename.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if new:writer.writeheader()
        writer.writerow(row)
