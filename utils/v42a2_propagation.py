"""V4.2a.2: visibility-aware (G3) and motion-consistent (G4) SHADOW proposals.

Read-only proposals based on the EXACT raw deformation, KNN graph and Q_F
used for V4.2a G0/G1/G2. Only call in shadow mode: return proposals, do not
write to Gaussian state. Visibility is observability, NOT staticness or GT.

G3: choose G1 only for temporal correspondences with valid camera projection;
    otherwise preserve the exact original G0 increment.
G4: G3 visibility plus weighted motion compatibility. For high-Q target
    anchors, use its own raw displacement; for low-Q targets, use a robust
    medoid of reliable neighboring raw motions. Require local consensus and
    enough trusted neighbors; otherwise preserve G0 rather than assuming
    zero motion or trusting a possibly different moving object.
"""
import csv
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def shadow_g3_g4(raw, q, dists2, indices, g0, g1, reasons,
                 radius=.2, spatial_sigma=.05, motion_sigma=.08,
                 trusted_q=.3, min_trusted_neighbors=3,
                 min_support=.25, max_consensus_disagreement_m=.12,
                 max_delta_from_g0_m=None, gate_m=.3, eps=1e-8):
    """Compute G3/G4 and diagnostics from already fixed KNN observations.

    `reasons` uses V4.2a.1 projection reason codes. 0 and 8 are
    geometrically and temporally valid. Q=near zero (8) does not
    automatically mean the point is invisible.
    """
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw must have shape [N,3]")
    n=raw.shape[0]
    if n<1 or q.shape!=(n,) or g0.shape!=raw.shape or g1.shape!=raw.shape:
        raise ValueError("Malformed pointwise data")
    if dists2.ndim!=2 or indices.shape!=dists2.shape or dists2.shape[0]!=n:
        raise ValueError("Malformed KNN graph")
    if reasons.shape!=(n,):
        raise ValueError("reason shape mismatch")
    if (radius is not None and radius<=0) or spatial_sigma<=0 or motion_sigma<=0 or gate_m<=0:
        raise ValueError("Invalid radius, sigma, or gate")
    if not 0<=trusted_q<=1 or not 0<=min_support<=1:
        raise ValueError("Q and support thresholds must be within [0,1]")
    if min_trusted_neighbors<1 or max_consensus_disagreement_m<=0:
        raise ValueError("Invalid consensus thresholds")
    if max_delta_from_g0_m is not None and max_delta_from_g0_m<=0:
        raise ValueError("Max delta bound must be positive or None")
    idx=indices.detach().long()
    if idx.numel() and (idx.min()<0 or idx.max()>=n):
        raise ValueError("KNN indices out of range")
    v=((reasons==0)|(reasons==8))
    # Distinguish valid FB correspondence with near-zero numerical confidence
    # (reason 8) from unavailable correspondence (reasons 1..7).
    g3=torch.where(v[:,None],g1,g0)

    raw=raw.detach()
    q=torch.nan_to_num(q.detach().to(dtype=raw.dtype),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    d=dists2.detach()
    finite=torch.isfinite(d)&(d>=0)
    if radius is not None:
        finite=finite&(d<=radius**2)
    w=torch.exp(-d.clamp_min(0)/(2*spatial_sigma**2+eps))
    w=torch.where(finite,w,torch.zeros_like(w))
    # Mimic the original G0/G1 self-weight safety for full comparability.
    w[:,0]=w[:,0].clamp_min(eps)
    neighbor_motion=raw[idx]
    neighbor_q=q[idx]
    # The neighbor must itself have a valid camera projection and temporal
    # correspondence. G1 already encodes invalidity with Q=0, but explicitly
    # enforcing this guards against future changes in Q implementation.
    neighbor_visible=v[idx]
    trusted=finite & neighbor_visible & (neighbor_q>=trusted_q)
    tw=w*neighbor_q*trusted.to(raw.dtype)
    trusted_count=trusted.sum(-1)
    support=tw.sum(-1)/(w.sum(-1)+eps)

    # Weighted L1 medoid among trusted neighboring motions. Unlike a mean,
    # the medoid must be one observed neighbor, reducing mode interpolation.
    # Complexity O(N*K^2), K <= configured KNN count (normally 16).
    pair_d=torch.linalg.vector_norm(
        neighbor_motion[:,:,None,:]-neighbor_motion[:,None,:,:],dim=-1)
    medoid_cost=(pair_d*tw[:,None,:]).sum(-1)/(tw.sum(-1)[:,None]+eps)
    medoid_cost=torch.where(trusted,medoid_cost,torch.full_like(medoid_cost,float("inf")))
    slot=medoid_cost.argmin(-1)
    rows=torch.arange(n,device=raw.device)
    medoid=neighbor_motion[rows,slot]
    # Reliable target self-motion can act as the anchor. For low-Q self
    # motion, its raw displacement is not used to judge neighbor compatibility.
    self_anchor=v&(q>=trusted_q)
    anchor=torch.where(self_anchor[:,None],raw,medoid)

    motion_disagreement=torch.linalg.vector_norm(neighbor_motion-anchor[:,None,:],dim=-1)
    consistency=torch.exp(-motion_disagreement.square()/(2*motion_sigma**2))
    robust_w=tw*consistency
    robust_mass=robust_w.sum(-1)
    estimate=(robust_w[...,None]*neighbor_motion).sum(1)/(robust_mass[:,None]+eps)
    # Measure dispersion around *consensus anchor* on original trusted subset.
    # If the trusted motions are multi-modal, conservative fallback to G0.
    dispersion=(tw*motion_disagreement).sum(-1)/(tw.sum(-1)+eps)
    valid_consensus=(v & (trusted_count>=min_trusted_neighbors) &
                     (support>=min_support) &
                     (dispersion<=max_consensus_disagreement_m) &
                     (robust_mass>10*eps) &
                     torch.isfinite(estimate).all(-1))
    if max_delta_from_g0_m is not None:
        valid_consensus &= (
            torch.linalg.vector_norm(estimate-g0,dim=-1)
            <=max_delta_from_g0_m)
    # V4.2a.3 read-only decomposition: do not modify valid_consensus or g4.
    support_only_gate = (v & (trusted_count >= min_trusted_neighbors) &
                         (support >= min_support) & (tw.sum(-1) > 10 * eps))
    soft_finite_gate = (robust_mass > 10 * eps) & torch.isfinite(estimate).all(-1)
    delta_limit_gate = (torch.ones_like(v) if max_delta_from_g0_m is None
                        else (torch.linalg.vector_norm(estimate-g0, dim=-1)
                              <= max_delta_from_g0_m))
    hard_dispersion_gate = dispersion <= max_consensus_disagreement_m
    g4=torch.where(valid_consensus[:,None],estimate,g0)

    norms=torch.stack([torch.linalg.vector_norm(g,dim=-1)
                       for g in (g0,g1,g3,g4)])
    passed=norms < gate_m  # exact original update_delta_xyz rule: strict <
    applied=torch.stack([torch.where(mask[:,None],g,torch.zeros_like(g))
                          for mask,g in zip(passed,(g0,g1,g3,g4))])
    def magnitude(a):return torch.linalg.vector_norm(a,dim=-1)
    return g3,g4,{
        "visible":v,
        "trusted_count":trusted_count,
        "trusted_support":support,
        "used_own_motion_anchor":self_anchor,
        "consensus_medoid_index":idx[rows,slot],
        "consensus_dispersion_m":dispersion,
        "g4_accepted":valid_consensus,
        # These tensors expose the EXACT original G4 internal gates for the
        # paired mechanism ablation; no recomputation of KNN or medoids.
        "support_only_gate":support_only_gate,
        "soft_finite_gate":soft_finite_gate,
        "delta_limit_gate":delta_limit_gate,
        "hard_dispersion_gate":hard_dispersion_gate,
        "soft_candidate":estimate,
        "g3_fallback_visibility":~v,
        "g4_fallback_visibility":~v,
        "g4_fallback_consensus":v&~valid_consensus,
        "g0_gate_pass":passed[0],
        "g1_gate_pass":passed[1],
        "g3_gate_pass":passed[2],
        "g4_gate_pass":passed[3],
        "g3_new_rejection":passed[0]&~passed[2],
        "g4_new_rejection":passed[0]&~passed[3],
        "g3_new_acceptance":~passed[0]&passed[2],
        "g4_new_acceptance":~passed[0]&passed[3],
        "g3_vs_g0_m":magnitude(g3-g0),
        "g4_vs_g0_m":magnitude(g4-g0),
        "g4_vs_g3_m":magnitude(g4-g3),
        "g3_effective_vs_g0_m":magnitude(applied[2]-applied[0]),
        "g4_effective_vs_g0_m":magnitude(applied[3]-applied[0]),
    }


@torch.no_grad()
def g34_summary(frame_id, mode, g0, g1, g3, g4, reasons, diag):
    n=int(g0.shape[0])
    if n<1:raise ValueError("Empty Gaussian candidate set")
    def fraction(x):return float(x.float().mean().cpu())
    def mean(x):return float(x.float().mean().cpu())
    def quantile(x,p):return float(torch.quantile(x.float(),p).cpu())
    row=dict(keyframe_id=int(frame_id), mode=str(mode), n_points=n,
             valid_observation_fraction=fraction(diag["visible"]),
             invalid_projection_fraction=fraction((reasons==1)|(reasons==2)),
             g3_visibility_fallback_fraction=fraction(diag["g3_fallback_visibility"]),
             g4_consensus_fallback_fraction=fraction(diag["g4_fallback_consensus"]),
             g4_accepted_fraction=fraction(diag["g4_accepted"]),
             consensus_dispersion_mean_m=mean(diag["consensus_dispersion_m"]),
             support_mean=mean(diag["trusted_support"]),
             trusted_neighbor_count_mean=mean(diag["trusted_count"]),
             g1_vs_g0_mean_m=mean(torch.linalg.vector_norm(g1-g0,dim=-1)),
             g3_vs_g0_mean_m=mean(diag["g3_vs_g0_m"]),
             g4_vs_g0_mean_m=mean(diag["g4_vs_g0_m"]),
             g4_vs_g3_mean_m=mean(diag["g4_vs_g3_m"]),
             g3_vs_g0_p95_m=quantile(diag["g3_vs_g0_m"],.95),
             g4_vs_g0_p95_m=quantile(diag["g4_vs_g0_m"],.95),
             g3_vs_g0_p99_m=quantile(diag["g3_vs_g0_m"],.99),
             g4_vs_g0_p99_m=quantile(diag["g4_vs_g0_m"],.99),
             g3_diff_over_10cm_fraction=fraction(diag["g3_vs_g0_m"]>.1),
             g4_diff_over_10cm_fraction=fraction(diag["g4_vs_g0_m"]>.1),
             g0_gate_pass_fraction=fraction(diag["g0_gate_pass"]),
             g3_gate_pass_fraction=fraction(diag["g3_gate_pass"]),
             g4_gate_pass_fraction=fraction(diag["g4_gate_pass"]),
             g3_new_rejections_n=int(diag["g3_new_rejection"].sum().cpu()),
             g4_new_rejections_n=int(diag["g4_new_rejection"].sum().cpu()),
             g3_new_acceptances_n=int(diag["g3_new_acceptance"].sum().cpu()),
             g4_new_acceptances_n=int(diag["g4_new_acceptance"].sum().cpu()),
             g3_effective_vs_g0_mean_m=mean(diag["g3_effective_vs_g0_m"]),
             g4_effective_vs_g0_mean_m=mean(diag["g4_effective_vs_g0_m"]))
    if not torch.isfinite(torch.as_tensor(list(v for v in row.values()
        if isinstance(v,(int,float)) and not isinstance(v,bool)),dtype=torch.float64)).all():
        raise ValueError("Nonfinite G3/G4 summary")
    return row


def save_g34(directory,row,raw,g0,g1,g2,g3,g4,q,reasons,diagnostics,
             selected_indices,xyz,save_points=False):
    """Keyframe CSV every frame; detailed NPZ only for user-selected frames."""
    dest=Path(directory)
    dest.mkdir(parents=True,exist_ok=True)
    csv_path=dest/"v42a2_per_keyframe.csv"
    new=not csv_path.exists() or csv_path.stat().st_size==0
    with csv_path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,row.keys())
        if new:w.writeheader()
        w.writerow(row)
    if save_points:
        arrays={
            "keyframe_id":np.array(row["keyframe_id"],np.int32),
            "selected_gaussian_index":selected_indices,
            "xyz_at_prev":xyz,
            "raw":raw,"g0":g0,"g1":g1,"g2":g2,"g3":g3,"g4":g4,
            "q":q,"visibility_reason":reasons,
        }
        arrays.update({"diag_"+k:v for k,v in diagnostics.items()})
        out={k:(v.detach().cpu().numpy() if isinstance(v,torch.Tensor)
                else np.asarray(v)) for k,v in arrays.items()}
        np.savez_compressed(dest/("v42a2_%05d.npz"%row["keyframe_id"]),**out)
