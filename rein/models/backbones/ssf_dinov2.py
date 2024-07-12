from mmseg.models.builder import BACKBONES, MODELS
from torch import nn
import torch

from .ssf import SSF
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
    
    
@BACKBONES.register_module()
class SSFDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self,num_layers=24,embed_dim=1024,**kwargs):
        super().__init__(**kwargs)
        self.ssf = SSF(num_layers,embed_dim)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.ssf(x,idx)
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(B, -1, h // self.patch_size, w // self.patch_size)
                    .contiguous()
                )
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["ssf"])
        set_train(self, ["ssf"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "ssf" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state