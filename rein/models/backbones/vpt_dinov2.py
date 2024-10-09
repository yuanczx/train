from mmseg.models.builder import MODELS
import torch
from torch import nn
import math
from functools import reduce, partial
from operator import mul
from torch import Tensor
from .dino_layers import (
    PatchEmbed,
    MemEffAttention,
    NestedTensorBlock as Block,
)
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train



@MODELS.register_module()
class SimpleVPT(nn.Module):
    def __init__(self, embed_dim, depth, num_tokens) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        # Initialize prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.zeros(depth, num_tokens, embed_dim))
        # Uniform initialization of embeddings
        val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + embed_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def forward(self, idx, layer, x: Tensor, batch_first=True, cls_token=True):
        if not batch_first:
            x = x.permute(1, 0, 2)  # Adjust for batch first convention
        B, _, _ = x.shape
        # Concatenate prompt embeddings
        x = torch.cat([self.prompt_embeddings[idx].expand(B, -1, -1), x], dim=1)
        x = layer(x)
        # Split and permute if necessary
        _, x = torch.tensor_split(x, [self.num_tokens], dim=1)
        if not batch_first:
            x = x.permute(1, 0, 2)
        return x
    
@MODELS.register_module()
class VPTDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ffn_layer="mlp",
        block_chunks=1,
        out_indices=[7, 11, 15, 23],
        upscale_feats=False,
        shallow=False,
        init_cfg=None,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            ffn_bias,
            proj_bias,
            drop_path_rate,
            drop_path_uniform,
            init_values,
            embed_layer,
            act_layer,
            block_fn,
            ffn_layer,
            block_chunks,
            out_indices,
            init_cfg,
        )
        # Initialize VPT
        self.shallow = shallow
        if shallow:
            self.vpt = SimpleVPT(embed_dim=embed_dim, depth=1, num_tokens=150)
        else:
            self.vpt = SimpleVPT(embed_dim=embed_dim, depth=depth, num_tokens=150)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            if self.shallow:
                if idx==0:
                    x = self.vpt.forward(idx, blk, x, batch_first=True, cls_token=True)
                else:
                    x = blk(x)
            else:
                x = self.vpt.forward(idx, blk, x, batch_first=True, cls_token=True)
            if idx in self.out_indices:
                outs.append(x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous())
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["vpt"])
        set_train(self, ["vpt"])    