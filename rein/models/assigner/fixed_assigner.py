from mmdet.models.task_modules.assigners import BaseAssigner,AssignResult
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmseg.registry import TASK_UTILS

@TASK_UTILS.register_module()
class FixedAssigner(BaseAssigner):
    
    def __init__(self) -> None:
        self.offset = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        
        num_gts = len(gt_instances)
        num_preds = len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds,), -1, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_preds,), -1, dtype=torch.long, device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. fixed matching
        min_length = min(num_preds, num_gts)
        assigned_gt_inds[:]=0
        for i in range(min_length):
            assigned_gt_inds[i*5] = i + 1  # assign 1-based index of gt
            assigned_labels[i*5] = gt_labels[i]
        # Assign remaining predictions to background
        # if num_preds > num_gts:
        #     assigned_gt_inds[num_gts:] = 0

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
