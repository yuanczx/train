from mmseg.models.builder import MODELS
import torch
import torch.nn as nn

    
@MODELS.register_module()
class SSF(nn.Module):
    def __init__(self,layer=24,dim=1024):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.scales = nn.Parameter(torch.ones([self.layer,self.dim]))
        self.shifts = nn.Parameter(torch.zeros([self.layer,self.dim]))
        nn.init.normal_(self.scales, mean=1, std=.02)
        nn.init.normal_(self.shifts, std=.02)


    def forward(self,x,layer):
        assert self.scales.shape == self.shifts.shape
        if x.shape[-1] == self.scales[layer].shape[0]:
            return x * self.scales[layer] + self.shifts[layer]
        elif x.shape[1] == self.scales[layer].shape[0]:
            return x * self.scales[layer].view(1, -1, 1, 1) + self.shifts[layer].view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')