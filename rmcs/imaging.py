import numpy as np
import torch

def weights(p,centres,width):
    u=np.abs((p[:,None]-centres[None,:])/np.median(np.diff(centres))/width)
    b=np.where(u<1,(4-6*u*u+3*u**3)/6,np.where(u<2,(2-u)**3/6,0))
    assert np.all(b.sum(1)>0)
    return b/b.sum(1,keepdims=True)

def reconstruct(old, rgb, response, regional_rgb_gains, xb, yb, width=.5):
    """RGB-guided inference from an existing registered cube; not new measurements.
    old: H x W x L; rgb: registered linear RGB H x W x 3;
    response: 3 x L; gains: 100 x 3 in column-major channel order.
    xb/yb: eleven region boundary coordinates each. No target data are bundled.
    This is the original B-spline/RGB-tilt calculation with file I/O removed.
    """
    h,w,n=old.shape
    if len(xb)!=11 or len(yb)!=11:raise ValueError('Expected a 10 x 10 grid')
    lum=np.maximum(rgb@np.array([.2126,.7152,.0722]),.001)
    sums=np.array([old[yb[r]:yb[r+1],xb[c]:xb[c+1]].sum((0,1)) for c in range(10) for r in range(10)])
    totalL=np.array([lum[yb[r]:yb[r+1],xb[c]:xb[c+1]].sum() for c in range(10) for r in range(10)])
    densities=(sums/totalL[:,None]).reshape(10,10,n).transpose(1,0,2)
    gain_grid=np.log(np.maximum(regional_rgb_gains,1e-15)).reshape(10,10,3).transpose(1,0,2)
    wx=weights(np.arange(w),(xb[:-1]+xb[1:]-1)/2,width)
    wy=weights(np.arange(h),(yb[:-1]+yb[1:]-1)/2,width)
    prior=np.einsum('yr,xc,rcl->yxl',wy,wx,densities,optimize=True)*lum[...,None]
    gains=np.exp(np.einsum('yr,xc,rck->yxk',wy,wx,gain_grid,optimize=True)).astype('float32')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def t(a):return torch.as_tensor(a,dtype=torch.float32,device=device)
    scale=max(np.percentile(prior,99),1e-15)
    base=t(prior/scale);target=t(rgb);resp=t(response.T*scale);gg=t(gains)
    # Smooth spectral tilts: wavelength RGB response modes, no new sharp peaks.
    basis=response/np.maximum(response.max(1,keepdims=True),1e-20)
    basis=t(basis)
    coeff=torch.nn.Parameter(torch.zeros((h,w,3),device=device))
    optimizer=torch.optim.Adam([coeff],lr=.04)
    gx=torch.exp(-((target[:,1:]-target[:,:-1])**2).sum(-1)/.02)[...,None]
    gy=torch.exp(-((target[1:]-target[:-1])**2).sum(-1)/.02)[...,None]
    with torch.no_grad():initial=float(((base@resp*gg-target)**2).mean())
    # Smooth overlapping measurements, not rectangular region constraints.
    tx,ty=t(wx),t(wy)
    def observations(z):return torch.einsum('yr,yxl,xc->rcl',ty,z,tx)
    with torch.no_grad():refobs=observations(t(old/scale));obsnorm=refobs.sum(-1,keepdim=True).clamp(min=1e-8)
    best=float('inf');hist=[]
    for it in range(180):
        optimizer.zero_grad()
        correction=1.5*torch.tanh(coeff)
        z=base*torch.exp(correction@basis)
        rgb_loss=((z@resp*gg-target)**2).mean()
        smooth=(gx*(correction[:,1:]-correction[:,:-1]).square()).mean()+(gy*(correction[1:]-correction[:-1]).square()).mean()
        soft=((observations(z)-refobs).abs().sum(-1,keepdim=True)/obsnorm).square().mean()
        loss=rgb_loss/max(initial,1e-5)+.015*correction.square().mean()+.08*smooth+.03*soft
        if float(loss.detach())<best:
            best=float(loss.detach());bestcoeff=correction.detach().clone()
        loss.backward();optimizer.step()
        
    with torch.no_grad():cube=(base*torch.exp(bestcoeff@basis)).cpu().numpy()*scale
    assert np.isfinite(cube).all() and cube.min()>=0
    return cube
