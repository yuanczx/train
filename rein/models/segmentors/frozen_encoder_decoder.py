from typing import List
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class FrozenBackboneEncoderDecoder(EncoderDecoder):
    def __init__(self,decode_head_grad = True,**kwargs):
        super().__init__(**kwargs)
        if not decode_head_grad:
            for param in self.decode_head.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x
