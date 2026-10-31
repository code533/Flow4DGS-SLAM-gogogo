"""V4.2a.3 paired, read-only decomposition of existing G4 internal gates.

No new propagation algorithm; all predictions share SAME raw Gaussian
motion, precomputed KNN graph, Q, medoid and G0/G3/G4. The stage labels
describe mechanism changes, NOT flow/motion accuracy.

S0 G3: visibility + unfiltered G1.
S1 Support: visibility + trusted-neighbor-only weighted mean.
S2 Soft: same support eligibility + G4 soft compatibility weighting, no
         hard dispersion cutoff.
S3 Full: original G4 including hard dispersion and optional delta cutoff.
"""
import csv
from pathlib import Path
import numpy as np
import torch

STAGES=("g3","support_only","soft_no_hard_dispersion","g4")

@torch.no_grad()
def decompose_g4(raw,q,dists2,indices,reasons,g0,g3,g4,internal,
                 radius=.2,sigma=.05,trusted_q=.3,eps=1e-8,gate=.3):
    n=raw.shape[0]
    if raw.ndim!=2 or raw.shape[1]!=3 or n<1 or q.shape!=(n,) or g0.shape!=raw.shape or g3.shape!=raw.shape or g4.shape!=raw.shape:
        raise ValueError("Bad point shape")
    if indices.shape!=dists2.shape or indices.shape[0]!=n or reasons.shape!=(n,):
        raise ValueError("Bad KNN/reason shape")
    if sigma<=0 or gate<=0 or (radius is not None and radius<=0):
        raise ValueError("Invalid smoothing/gate")
    for key in ("support_only_gate","soft_finite_gate","delta_limit_gate",
                "hard_dispersion_gate","soft_candidate","g4_accepted"):
        if key not in internal:raise ValueError("Missing G4 internal: "+key)
    idx=indices.long().detach()
    if idx.min()<0 or idx.max()>=n:raise ValueError("KNN index out of bounds")
    v=(reasons==0)|(reasons==8)
    valid=torch.isfinite(dists2)&(dists2>=0)
    if radius is not None:valid &= dists2<=radius**2
    w=torch.exp(-dists2.clamp_min(0)/(2*sigma**2+eps))
    w=torch.where(valid,w,torch.zeros_like(w))
    w[:,0]=w[:,0].clamp_min(eps)
    q=torch.nan_to_num(q.detach(),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    trusted=valid&v[idx]&(q[idx]>=trusted_q)
    tw=w*q[idx]*trusted.to(w.dtype)
    mass=tw.sum(-1)
    support_candidate=(tw[...,None]*raw.detach()[idx]).sum(1)/(mass[:,None]+eps)
    sgate=internal["support_only_gate"]
    s1=torch.where(sgate[:,None],support_candidate,g0)
    soft=internal["soft_candidate"]
    sg=sgate & internal["soft_finite_gate"]
    s2=torch.where(sg[:,None],soft,g0)
    # Full G4 exact equality: reconstructed gate must match parent.
    fg=sg & internal["delta_limit_gate"] & internal["hard_dispersion_gate"]
    if not torch.equal(fg,internal["g4_accepted"]):
        raise AssertionError("Full G4 gate differs from original")
    if not torch.equal(torch.where(fg[:,None],soft,g0),g4):
        raise AssertionError("Reconstructed full G4 differs from original")
    out={"g3":g3,"support_only":s1,"soft_no_hard_dispersion":s2,"g4":g4}
    gates={"g3":v,"support_only":sgate,"soft_no_hard_dispersion":sg,"g4":fg}
    return out, gates, {
        "support_candidate":support_candidate,
        "support_gate":sgate,
        "soft_gate":sg,
        "hard_dispersion_new_reject":sg & ~internal["hard_dispersion_gate"],
        "delta_limit_new_reject":sg & internal["hard_dispersion_gate"] & ~internal["delta_limit_gate"],
        "soft_delta_on_shared_support_m":torch.linalg.vector_norm(soft-support_candidate,dim=-1),
    }

@torch.no_grad()
def summarize_ablation(frame_id,stages,gates,internal,extra,gate=.3):
    n=stages["g3"].shape[0]
    g0=internal["g0"]
    def fraction(x):return float(x.float().mean().cpu())
    def avg(x):return float(x.float().mean().cpu())
    def quant(x,p):return float(torch.quantile(x.float(),p).cpu())
    result=[]
    g0pass=torch.linalg.vector_norm(g0,dim=-1)<gate
    g0eff=torch.where(g0pass[:,None],g0,torch.zeros_like(g0))
    for name in STAGES:
        movement=stages[name]
        accepted=gates[name]
        rawdiff=torch.linalg.vector_norm(movement-g0,dim=-1)
        passed=torch.linalg.vector_norm(movement,dim=-1)<gate
        eff=torch.where(passed[:,None],movement,torch.zeros_like(movement))
        ediff=torch.linalg.vector_norm(eff-g0eff,dim=-1)
        conditional=rawdiff[accepted]
        row=dict(
            keyframe_id=int(frame_id), stage=name, n_points=n,
            accepted_fraction=fraction(accepted), accepted_n=int(accepted.sum().cpu()),
            vs_g0_mean_m=avg(rawdiff), vs_g0_p95_m=quant(rawdiff,.95),
            vs_g0_p99_m=quant(rawdiff,.99), change_over_10cm_fraction=fraction(rawdiff>.1),
            on_accepted_vs_g0_mean_m=avg(conditional) if conditional.numel() else float("nan"),
            effective_vs_g0_mean_m=avg(ediff),
            gate_pass_fraction=fraction(passed),
            new_gate_rejection_n=int((g0pass&~passed).sum().cpu()),
            new_gate_acceptance_n=int((~g0pass&passed).sum().cpu()),
            support_gate_fraction=fraction(extra["support_gate"]),
            soft_gate_fraction=fraction(extra["soft_gate"]),
            hard_dispersion_new_rejection_n=int(extra["hard_dispersion_new_reject"].sum().cpu()),
            delta_limit_new_rejection_n=int(extra["delta_limit_new_reject"].sum().cpu()),
            soft_vs_support_shared_mean_m=(
                avg(extra["soft_delta_on_shared_support_m"][extra["soft_gate"]])
                if extra["soft_gate"].any() else float("nan")),
        )
        result.append(row)
    return result

def save_ablation(directory,rows,stages,gates,extra,reasons,selected_idx,xyz,raw,q,g0,g1,g2,
                  save_points=False):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True)
    csv_path=dest/"v42a3_per_keyframe.csv"
    new=not csv_path.exists() or csv_path.stat().st_size==0
    with csv_path.open("a",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        if new:wr.writeheader()
        wr.writerows(rows)
    if save_points:
        m={"keyframe_id":np.asarray(rows[0]["keyframe_id"],np.int32),
           "selected_gaussian_index":selected_idx,"xyz_at_prev":xyz,
           "reason":reasons,"raw":raw,"q":q,"g0":g0,"g1":g1,"g2":g2}
        m.update(stages)
        m.update({"gate_"+k:v for k,v in gates.items()})
        m.update({"internal_"+k:v for k,v in extra.items()})
        arr={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)
             for k,v in m.items()}
        np.savez_compressed(dest/("v42a3_%05d.npz"%rows[0]["keyframe_id"]),**arr)
