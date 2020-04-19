import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import net.functional as Fn

class MAC(nn.Module):
    def __init__(self, norm_fn=Fn.l2n):
        super(MAC, self).__init__()
        self.norm_fn = norm_fn

    def forward(self, x):
        x = Fn.mac(x)
        # x.squeeze_()
        x = x.view(x.size(0), -1)
        return self.norm_fn(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SPoC(nn.Module):
    def __init__(self, norm_fn=Fn.l2n):
        super(SPoC, self).__init__()
        self.norm_fn = norm_fn

    def forward(self, x):
        x = Fn.spoc(x)
        x = x.view(x.size(0), -1)
        return self.norm_fn(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CroW(nn.Module):
    def __init__(self, a=2, b=3, norm_fn=Fn.l2n):
        super(CroW, self).__init__()
        self.a = 2
        self.b = 3
        self.norm_fn = norm_fn

    def forward(self, x):
        x = Fn.crow(x, self.a, self.b)
        x = x.view(x.size(0), -1)
        return self.norm_fn(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'a=' + str(self.a) + ',' + 'b=' + str(self.b) + ')'

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, norm_fn=Fn.l2n):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.norm_fn = norm_fn

    def forward(self, x):
        x = Fn.gem(x, p=self.p, eps=self.eps)
        x = x.view(x.size(0), -1)
        return self.norm_fn(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6, keep_region=False, norm_fn=Fn.l2n):
        super(RMAC,self).__init__()
        self.L = L
        self.eps = eps
        self.keep_region = keep_region
        self.norm_fn = norm_fn

    def forward(self, x):
        x = Fn.rmac(x, L=self.L, eps=self.eps, keep_region=self.keep_region)
        if self.keep_region:
            return x
        else:
            x = x.view(x.size(0), -1)
            return self.norm_fn(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'