from mmseg.models.builder import MODELS
import torch
import torch.nn as nn

    
@MODELS.register_module()
class SSF2(nn.Module):
    def __init__(self,layer=24,dim=1024):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.scales = nn.Parameter(torch.ones([self.layer,1025,self.dim]))
        self.shifts = nn.Parameter(torch.zeros([self.layer,1025,self.dim]))
        nn.init.normal_(self.scales, mean=1, std=.02)
        nn.init.normal_(self.shifts, std=.02)


    def forward(self,x,layer):
        return x * self.scales[layer] + self.shifts[layer]