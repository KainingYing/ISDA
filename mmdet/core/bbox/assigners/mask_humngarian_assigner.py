import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 mask_cost=dict(type='MaskIoUCost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)

    def assign(self,
               cls_pred,
               gt_labels,
               mask_pred,
               gt_masks,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_masks = gt_masks.size(0), mask_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = mask_pred.new_full((num_masks, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_masks, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_masks == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']


        # 2. compute the weighted costs
        # classification and maskIoUcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        mask_cost = self.mask_cost(mask_pred, gt_masks)
        # weighted sum of above three costs
        cost = cls_cost + mask_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        # print(cls_pred, cls_cost, mask_cost, cost)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            mask_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            mask_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
