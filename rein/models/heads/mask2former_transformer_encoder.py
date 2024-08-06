from torch import Tensor
import torch
from mmdet.models.layers.transformer import DeformableDetrTransformerEncoder
from .enhance_transformer import EnhanceLayer


class Mask2FormerTransformerEncoder(DeformableDetrTransformerEncoder):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.enhance_layers = [
            EnhanceLayer(d_model=256,nhead=4,dim_feedforward=1024,dropout=0.1).to(torch.device('cuda'))
            for _ in range(self.num_layers)
        ]

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                reference_points: Tensor, **kwargs) -> Tensor:
        
        for idx, layer in enumerate(self.layers):
            query = self.enhance_layers[idx](
                src = query,
                pos = query_pos,
                src_key_padding_mask = key_padding_mask
            )
            
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query