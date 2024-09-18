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

    def __init__(self,
                 num_cls):
        self.num_cls = num_cls
        
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
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

        # 2. identity matching
        expand_n = num_preds // self.num_cls
        matched_row_inds = gt_labels
        matched_col_inds = torch.tensor(list(range(len(gt_labels))))
        for n in range(1, expand_n):
            matched_row_inds = torch.concat((matched_row_inds, gt_labels+n*self.num_cls))
            matched_col_inds = torch.concat((matched_col_inds, torch.tensor(list(range(len(gt_labels))))))

        matched_row_inds = matched_row_inds.to(device)
        matched_col_inds = matched_col_inds.to(device)

        # 3. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts=num_gts, gt_inds=assigned_gt_inds,max_overlaps=None, labels=assigned_labels)
