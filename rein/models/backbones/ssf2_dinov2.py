from mmseg.models.builder import BACKBONES, MODELS
from torch import nn
import torch

from .ssf2 import SSF2
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class SSF2DinoVisionTransformer(DinoVisionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ssf = SSF2(layer=1,dim=kwargs['embed_dim'])

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.ssf(x,0)
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