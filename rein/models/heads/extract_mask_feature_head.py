from mmseg.models.decode_heads import Mask2FormerHead
from typing import List, Tuple
import torch
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import SampleList,ConfigType
from mmseg.structures.seg_data_sample import SegDataSample
import torch.nn.functional as F


@MODELS.register_module()
class ExtractMaskFeatureHead(Mask2FormerHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_feature_list = []
        self.mask_label_list = []
        self.label_pt_num = 1
        self.feature_pt_num = 1

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward function.

            Args:
                x (list[Tensor]): Multi scale Features from the
                    upstream network, each is a 4D-tensor.
                batch_data_samples (List[:obj:`DetDataSample`]): The Data
                    Samples. It usually includes information such as
                    `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

            Returns:
                tuple[list[Tensor]]: A tuple contains two elements.

                    - cls_pred_list (list[Tensor)]: Classification logits \
                        for each decoder layer. Each is a 3D-tensor with shape \
                        (batch_size, num_queries, cls_out_channels). \
                        Note `cls_out_channels` should includes background.
                    - mask_pred_list (list[Tensor]): Mask logits for each \
                        decoder layer. Each with shape (batch_size, num_queries, \
                        h, w).
            """
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        
        if len(self.mask_feature_list)>=500:
            torch.save(torch.stack(self.mask_feature_list),f'./mask_feature_{self.feature_pt_num}.pt')
            self.mask_feature_list.clear()
            self.feature_pt_num+=1

        self.mask_feature_list.append(mask_features.reshape(256,128,128).to('cpu'))

        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        extract_class_score = F.softmax(mask_cls_results,dim=-1)[...,:-1]
        extract_mask_pred = mask_pred_results.sigmoid()
        extract_seg_logits = torch.einsum('bqc, bqhw->bchw', extract_class_score, extract_mask_pred).to('cpu')
        extract_seg_logits = extract_seg_logits.reshape(19,128,128).argmax(dim=0)

        if len(self.mask_label_list)>=500:
            torch.save(torch.stack(self.mask_label_list),f'./mask_label_{self.label_pt_num}.pt')
            self.mask_label_list.clear()
            self.label_pt_num+=1

        self.mask_label_list.append(extract_seg_logits)

        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


    def __del__(self):
        torch.save(torch.stack(self.mask_feature_list),f'./mask_feature_{self.feature_pt_num}.pt')
        torch.save(torch.stack(self.mask_label_list),f'./mask_label_{self.label_pt_num}.pt')
        super().__del__()
