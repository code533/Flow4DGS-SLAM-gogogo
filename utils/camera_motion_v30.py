"""V3.0 camera motion diagnostics and controlled translation cap."""
import torch

@torch.no_grad()
def translation_step(delta_t, max_translation, mode="legacy"):
    if delta_t.shape != (3,) or mode not in ("legacy","capped"):
        raise ValueError("invalid translation shape or mode")
    mx=torch.as_tensor(max_translation,device=delta_t.device,dtype=delta_t.dtype)
    if not bool(torch.isfinite(delta_t).all()) or not bool(torch.isfinite(mx)) or bool(mx<0):
        raise ValueError("non-finite motion or negative limit")
    ratio=mx/(torch.linalg.vector_norm(delta_t)+1e-10)
    scale=ratio if mode=="legacy" else torch.clamp(ratio,max=1.0)
    return scale*delta_t,ratio,scale

@torch.no_grad()
def rigid_transform(R,t):
    m=torch.eye(4,device=R.device,dtype=R.dtype)
    m[:3,:3]=R.detach();m[:3,3]=t.detach()
    return m

@torch.no_grad()
def condition_number_normalized_jacobian(depth,intrinsic,valid_mask,stride=8):
    if depth.ndim!=2 or valid_mask.shape!=depth.shape or intrinsic.shape!=(3,3):
        raise ValueError("invalid geometry shapes")
    stride=int(stride)
    if stride<1: raise ValueError("stride must be positive")
    h,w=depth.shape
    z=depth[::stride,::stride].double()
    valid=valid_mask[::stride,::stride].bool()&torch.isfinite(z)&(z>0)
    n=int(valid.sum().item())
    if n<12:return float("inf"),n
    y,x=torch.meshgrid(torch.arange(0,h,stride,device=depth.device,dtype=z.dtype),
                        torch.arange(0,w,stride,device=depth.device,dtype=z.dtype),indexing="ij")
    k=intrinsic.to(device=depth.device,dtype=z.dtype)
    fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
    X,Y,Z=x[valid]-cx,y[valid]-cy,z[valid];zero=torch.zeros_like(Z)
    jx=torch.stack((-fx/Z,zero,X/Z,X*Y/fy,-(fx+X*X/fx),Y),dim=1)
    jy=torch.stack((zero,-fy/Z,Y/Z,fy+Y*Y/fy,-X*Y/fx,-X),dim=1)
    j=torch.stack((jx,jy),dim=1).reshape(-1,6)
    j=j/torch.linalg.vector_norm(j,dim=0).clamp_min(1e-12)
    eig=torch.linalg.eigvalsh(j.T@j).clamp_min(0)
    if not bool(torch.isfinite(eig).all()) or float(eig[0])<=1e-12:return float("inf"),n
    return float((eig[-1]/eig[0]).item()),n
