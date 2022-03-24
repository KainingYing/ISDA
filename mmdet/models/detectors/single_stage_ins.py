import torch.nn as nn
from mmdet.core import bbox2result
from .. import builder
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,):
        super(SingleStageInsDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if mask_feat_head is not None:
            self.mask_feat_head = builder.build_head(mask_feat_head)

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_mask_feat_head(self):
        return hasattr(self, 'mask_feat_head') and \
            self.mask_feat_head is not None

    def init_weights(self, pretrained=None):
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.Sequential):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape


        x = self.extract_feat(img)
        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.
              start_level:self.mask_feat_head.end_level+1])
        losses = self.bbox_head.forward_train(x[-4:], mask_feat_pred, gt_labels, gt_masks, img_metas) # cls loss & dice loss
        return losses # cls dice mass


    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError




