"""V2.2 independent RGB and depth weighting for Flow4DGS tracking."""
import math
import torch

def hw(t, name, ref=None):
    if t.ndim==2: out=t
    elif t.ndim==3 and t.shape[0]==1: out=t[0]
    else: raise ValueError(name+" shape")
    if ref is not None:
        out=out.to(device=ref.device,dtype=ref.dtype)
        if out.shape!=ref.shape: raise ValueError(name+" spatial shape")
    return out

@torch.no_grad()
def compute_decoupled_weights(cues,cfg):
    mode=str(cfg.get("mode","decoupled"))
    if mode not in ("decoupled","rgb_only","depth_only","all_ones"): raise ValueError("mode")
    valid=hw(cues["valid"].detach(),"valid").bool()
    h,w=valid.shape
    zero=torch.zeros((h,w),device=valid.device)
    one=torch.ones_like(zero)
    flow=hw(cues["p_flow"].detach(),"p_flow",zero).clamp(0,1)
    rgb=hw(cues["p_rgb"].detach(),"p_rgb",zero).clamp(0,1)
    front=hw(cues["p_front"].detach(),"p_front",zero).clamp(0,1)
    if not (torch.isfinite(flow).all() and torch.isfinite(rgb).all() and torch.isfinite(front).all()):
        raise ValueError("Non-finite cues")
    wf=float(cfg.get("rgb_min_weight",.95))
    wd=float(cfg.get("depth_min_weight",.8))
    fmin=float(cfg.get("flow_prior_min",.35))
    rmin=float(cfg.get("rgb_prior_min",.55))
    dmin=float(cfg.get("depth_prior_min",.65))
    rb=float(cfg.get("rgb_max_fraction",.015))
    db=float(cfg.get("depth_max_fraction",.03))
    rn=int(cfg.get("rgb_min_candidate_pixels",32))
    dn=int(cfg.get("depth_min_candidate_pixels",32))
    if not (0<wf<=1 and 0<wd<=1 and 0<=fmin<=1 and 0<=rmin<=1 and 0<=dmin<=1
            and 0<=rb<=1 and 0<=db<=1 and rn>=0 and dn>=0):
        raise ValueError("Invalid weight settings")
    sr=torch.where(valid,flow*rgb,zero)
    sd=torch.where(valid,flow*front,zero)
    mr=valid&(flow>=fmin)&(rgb>=rmin)
    md=valid&(flow>=fmin)&(front>=dmin)
    def select(raw,score,budget,minimum):
        n=int(raw.sum().item());limit=int(math.floor(budget*valid.sum().item()))
        sel=torch.zeros_like(valid)
        if n==0 or limit<max(minimum,1): return sel,n
        ids=raw.reshape(-1).nonzero(as_tuple=False).flatten()
        if ids.numel()>limit: ids=ids[torch.topk(score.reshape(-1)[ids],limit,sorted=False).indices]
        if ids.numel()<minimum: return sel,n
        sel.reshape(-1)[ids]=True
        return sel,n
    mr,nr=select(mr,sr,rb,rn)
    md,nd=select(md,sd,db,dn)
    if mode in ("depth_only","all_ones"): mr=torch.zeros_like(valid)
    if mode in ("rgb_only","all_ones"): md=torch.zeros_like(valid)
    wr=torch.where(mr,(1-(1-wf)*sr).clamp_min(wf),one)
    wz=torch.where(md,(1-(1-wd)*sd).clamp_min(wd),one)
    diag=bool(cfg.get("diagnostic_only",False))
    active=(not diag) and (mode=="all_ones" or bool((wr<.999999).any()) or bool((wz<.999999).any()))
    reason=("diagnostic_only" if diag else "all_ones_loss_check" if mode=="all_ones"
            else "applied" if active else "no_branch_candidates")
    return dict(rgb_weights=(one if diag else wr).detach(),
                depth_weights=(one if diag else wz).detach(),
                rgb_score=sr.detach(),depth_score=sd.detach(),
                rgb_candidate=mr.detach(),depth_candidate=md.detach(),
                rgb_raw_count=nr,depth_raw_count=nd,
                rgb_selected_count=int(mr.sum().item()),
                depth_selected_count=int(md.sum().item()),
                rgb_changed_fraction=float((wr<.999999).sum().item())/(h*w),
                depth_changed_fraction=float((wz<.999999).sum().item())/(h*w),
                valid=valid.detach(),coverage=float(cues.get("coverage",0)),
                mode=mode,reason=reason,applied=bool(active))

def tracking_loss_decoupled(config,image,depth,opacity,viewpoint,
                            rgb_weights,depth_weights,normalization="mass_preserving"):
    if config["Training"].get("monocular",False):raise ValueError("RGB-D only")
    if normalization not in ("mass_preserving","original_mean"):raise ValueError("normalization")
    if image.ndim!=3 or image.shape[0]!=3 or depth.ndim!=3 or depth.shape[0]!=1:
        raise ValueError("RGB or depth shape")
    h,w=depth.shape[-2:]
    if image.shape[-2:]!=(h,w):raise ValueError("RGB / depth dimensions")
    ref=depth[0].detach()
    rw=hw(rgb_weights.detach(),"rgb_weights",ref)[None]
    dw=hw(depth_weights.detach(),"depth_weights",ref)[None]
    if not (torch.isfinite(rw).all() and torch.isfinite(dw).all()) or (rw<0).any() or (dw<0).any():
        raise ValueError("invalid confidence map")
    pred=torch.exp(viewpoint.exposure_a)*image+viewpoint.exposure_b
    obs=viewpoint.original_image.to(device=image.device,dtype=image.dtype)
    zg=torch.as_tensor(viewpoint.depth,device=depth.device,dtype=depth.dtype).reshape_as(depth)
    op=opacity.reshape_as(depth)
    rm=(obs.sum(0,keepdim=True)>config["Training"]["rgb_boundary_threshold"]).to(image.dtype)
    rm=rm*viewpoint.grad_mask.to(device=image.device,dtype=image.dtype).reshape(1,h,w)
    dm=((zg>.01)&(zg<1000)&(op>.95)).to(depth.dtype)
    if viewpoint.motion_mask is not None and viewpoint.uid>0:
        static=viewpoint.motion_mask.to(image.device,dtype=image.dtype).reshape(1,h,w)
        rm=rm*static;dm=dm*static
    rwm=rm*rw;dwm=dm*dw
    lr=(op*(pred-obs).abs()*rwm).mean()
    ld=((depth-zg).abs()*dwm).mean()
    if normalization=="mass_preserving":
        lr=lr*rm.sum().detach()/rwm.sum().detach().clamp_min(1e-8)
        ld=ld*dm.sum().detach()/dwm.sum().detach().clamp_min(1e-8)
    alpha=config["Training"].get("alpha",.95)
    return alpha*lr+(1-alpha)*ld
