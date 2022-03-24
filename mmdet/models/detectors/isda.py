import numpy as np
import mmcv
import torch

from .single_stage_ins import SingleStageInsDetector
from mmdet.core import ismask2result
from ..builder import DETECTORS
from mmdet.core.visualization import imshow_det_masks


@DETECTORS.register_module
class ISDA(SingleStageInsDetector):
    """use the transformer and dynamic conv to get the instance level mask
    """
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ISDA, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                    test_cfg, pretrained)

    def simple_test(self, img, img_metas, rescale=False):

        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
                                f'mode is supported. Found batch_size {batch_size}.'
        x = self.extract_feat(img)
        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])  # B x 256 x 1/4 x 1/4
        outs = self.bbox_head(x[-4:], img_metas) # 6 x 1 x 300 x 81 and 6 x 1 x 300 x 256
        bbox_list = self.bbox_head.get_masks(
            *outs, mask_feat_pred, img_metas, rescale=rescale)

        bbox_results = [
            ismask2result(det_labels, det_scores, det_masks, self.bbox_head.num_classes)
            for det_labels, det_scores, det_masks in bbox_list
        ]
        return bbox_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        # if isinstance(result, tuple):
        #     bbox_result, segm_result = result
        #     if isinstance(segm_result, tuple):
        #         segm_result = segm_result[0]  # ms rcnn
        # else:
        #     bbox_result, segm_result = result, None

        segm_result, score_result = result
        print(segm_result)

        labels = [
            np.full(segm.shape[0], i, dtype=np.int32)
            for i, segm in enumerate(segm_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            scores = mmcv.concat_list(score_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                scores = torch.stack(scores, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
                scores = np.stack(scores, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_masks(
            img,
            labels,
            segms,
            scores,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
