## pytorch 1.2.0

## pool functions && normalization

import torch
import torch.nn.functional as F
import math
from copy import deepcopy

def mac(x):
    return F.adaptive_max_pool2d(x, (1, 1))

def spoc(x):
    return F.adaptive_avg_pool2d(x, (1, 1))

def gem(x, p=3, eps=1e-6):
    N, C, H, W = x.size()
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (H, W)).pow(1./p)

def crow(x, a=2, b=3):
    N, C, H, W = x.size()
    x = x.view(N*C, H, W)
    ## spatial weight
    s = x.sum(dim=0)
    z = ((s.pow(a)).sum()).pow(1./a)
    w_sp = (s/z).pow(1./b)
    ## channel weight
    area = H * W
    nonzeros = x[:,0,0]
    for i in range(N*C):
        nonzeros[i] = x[i].nonzero().size(0) / area
    nzsum = nonzeros.sum()
    w_ch = torch.log(nzsum / nonzeros)
    w_ch[w_ch==float('inf')] = 0.0
    ## weighting
    x = x * w_sp
    x = x.sum(dim=(1,2))
    x = x * w_ch
    return x.view(N, C)

def rmac(x, L=3, eps=1e-6, keep_region=False):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    if keep_region:
        region_feats = deepcopy(v.view(v.size(0), -1))
    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt
                if keep_region:
                    region_feats = torch.cat((region_feats, vt.view(vt.size(0), -1)), dim=0)
    
    return region_feats if keep_region else v

def l2n(x):
    ## x: n * d
    return F.normalize(x, p=2, dim=1)

def root_norm(x, eps=1e-6):
    return torch.sqrt(x / (x.sum()+eps))

def powlaw(x, eps=1e-6):
    x += eps
    return x.abs().sqrt().mul(x.sign())