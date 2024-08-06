from typing import List, Tuple
from mmdet.models.layers.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from .mask2former_transformer_encoder import Mask2FormerTransformerEncoder
from mmseg.registry import MODELS

@MODELS.register_module()
class EnhancePixelDecoder(MSDeformAttnPixelDecoder):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        encoder_config = kwargs['encoder']
        self.encoder = Mask2FormerTransformerEncoder(**encoder_config)
        