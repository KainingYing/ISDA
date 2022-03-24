import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class ISDAHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=300,
                 num_kernel_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_mask=dict(type='DiceLoss', loss_weight=1.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='MaskHungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.0),
                         mask_cost=dict(type='MaskIoUCost', weight=1.0))),
                 test_cfg=dict(max_per_img=100, mask_thr=0.5),
                 init_cfg=None,
                 **kwargs):

        super(AnchorFreeHead, self).__init__(init_cfg)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is ISDAHead):  # Focal Loss
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            # assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoMaskSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        self.mask_thr = self.test_cfg.get('mask_thr', 0.5)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_kernel_fcs = num_kernel_fcs
        self.train_cfg = train_cfg

        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=False))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        kernel_branch = []
        for _ in range(self.num_kernel_fcs):
            kernel_branch.append(Linear(self.embed_dims, self.embed_dims))
            kernel_branch.append(nn.ReLU(inplace=True))
        kernel_branch.append(Linear(self.embed_dims, 254))
        kernel_branch = nn.Sequential(*kernel_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(num_pred)])
        self.kernel_branches = nn.ModuleList(
            [kernel_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)  # 256 x 2

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        # if self.as_two_stage:
        #     for m in self.reg_branches:
        #         nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        # if not self.as_two_stage:
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=None,  # noqa:E501
            cls_branches=None  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)  # torch.Size([6, 1, 300, 256])
        outputs_classes = []
        outputs_coords = []
        outputs_kernels = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference_temp = reference
            reference_temp = torch.zeros_like(reference)
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])  # torch.Size([1, 300, 80])
            kernel = self.kernel_branches[lvl](hs[lvl])  # 1, 300, 254
            # tmp = self.reg_branches[lvl](hs[lvl]) # torch.Size([1, 300, 4])
            # if reference.shape[-1] == 4:
            #     tmp += reference
            # else:
            assert reference.shape[-1] == 2

            outputs_kernel = torch.cat([kernel, reference], -1)

                # tmp[..., :2] += reference
            # outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            # outputs_coords.append(outputs_coord)
            outputs_kernels.append(outputs_kernel)

        outputs_classes = torch.stack(outputs_classes)
        outputs_kernels = torch.stack(outputs_kernels)
        return outputs_classes, outputs_kernels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_kernel_preds_list'))
    def loss(self,
             mask_feature,
             all_cls_scores_list,
             all_kernel_preds_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_cls_scores = all_cls_scores_list  # 6 x 2 x 300 x 80
        all_kernel_preds = all_kernel_preds_list  # 6 x 2 x 300 x 256
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        mask_feature_list = [mask_feature for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_mask = multi_apply(
            self.loss_single, mask_feature_list, all_cls_scores, all_kernel_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_mask_i in zip(losses_cls[:-1], losses_mask[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            # loss_dict[f'd{num_dec_layer}.loss_mass'] = loss_mass_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    mask_feature,
                    cls_scores,
                    kernel_preds,  # kernel_preds
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # [300 x 80]
        kernel_preds_list = [kernel_preds[i] for i in range(num_imgs)]  # [300 x 256]
        mask_feature_list = [mask_feature[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(mask_feature_list, cls_scores_list, kernel_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, mask_preds_list, mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        mask_preds = torch.cat(mask_preds_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.cat(mask_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)  # torch.Size([300, 80])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # mask loss
        loss_mask = self.loss_mask(mask_preds, mask_targets, mask_weights, avg_factor=num_total_pos)

        return loss_cls, loss_mask

    def get_targets(self,
                    mask_feature_list,
                    cls_scores_list,
                    kernel_preds_list,
                    gt_labels_list,
                    gt_masks_lsit,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, mask_preds_list, mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(
            self._get_target_single, mask_feature_list, cls_scores_list, kernel_preds_list,
            gt_labels_list, gt_masks_lsit, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                mask_preds_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           mask_feature,
                           cls_score,
                           kernel_preds,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):

        device = gt_labels.device

        num_masks = kernel_preds.size(0)
        # using the kernels to get the mask prediction
        resize_kernel = kernel_preds.view(num_masks, 256, 1, 1)  # torch.Size([300, 256, 1, 1])
        mask_preds = F.conv2d(mask_feature.unsqueeze(0), resize_kernel, stride=1).sigmoid().squeeze(
            0)  # torch.Size([300, 128, 100])
        gt_masks = mask_preds.new_tensor(gt_masks.masks)


        if len(gt_masks) == 0:
            gt_masks = mask_preds.new_zeros((0, mask_preds.size(-2), mask_preds.size(-1)))
            mask_targets = gt_masks
        else:
            gt_masks = F.interpolate(gt_masks.unsqueeze(0), scale_factor=1 / 4, mode='bilinear').squeeze(0)
            # padding gt_masks
            mask_targets = mask_preds.new_full((gt_masks.size(0), mask_preds.size(-2), mask_preds.size(-1)), 0)
            mask_targets[:, :gt_masks.shape[-2], :gt_masks.shape[-1]] = gt_masks
            gt_masks = mask_targets

        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, gt_labels,
                                             mask_preds, gt_masks,
                                             img_meta, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, mask_preds, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # mask targets

        mask_preds = mask_preds

        mask_targets = torch.zeros_like(mask_preds)
        mask_weights = mask_targets.new_full((mask_preds.shape[0], 1), 0.0)
        mask_weights[pos_inds] = 1.0

        mask_targets[pos_inds] = sampling_result.pos_gt_masks

        # label targets
        labels = mask_preds.new_full((num_masks,),
                                     self.num_classes,
                                     dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_masks.new_ones(num_masks)

        return (labels, label_weights, mask_preds, mask_targets, mask_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      mask_feat_pred,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      cfg=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training mode.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        outs = self(x, img_metas)  # all_cls_scores, all_kernel_preds->
        mask_feature = mask_feat_pred
        if gt_labels is None:
            loss_inputs = (mask_feature,) + outs + (img_metas)
        else:
            loss_inputs = (mask_feature,) + outs + (gt_labels, gt_masks, img_metas, cfg)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_kernel_preds_list'))
    def get_masks(self,
                  all_cls_scores_list,
                  all_kernel_preds_list,
                  mask_feature,
                  img_metas,
                  rescale=False):
        """
        Transform network outputs for a batch into bbox predictions and masks prediction
        """

        cls_scores = all_cls_scores_list[-1]
        kernel_preds = all_kernel_preds_list[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]  # 300 x 80
            kernel_pred = kernel_preds[img_id]  # 300 x 256
            img_shape = img_metas[img_id]['img_shape']
            origin_img_shape = img_metas[img_id]['ori_shape']
            proposals = self._get_masks_single(cls_score, kernel_pred, mask_feature, img_shape,
                                               origin_img_shape, rescale)
            result_list.append(proposals)
        return result_list

    def _get_masks_single(self,
                          cls_score,
                          kernel_pred,
                          mask_feature,
                          img_shape,
                          origin_img_shape,
                          rescale=False):

        assert len(cls_score) == len(kernel_pred)

        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(self.max_per_img)
            det_labels = indexs % self.num_classes
            kernel_index = indexs // self.num_classes
            kernel_pred = kernel_pred[kernel_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, kernel_index = scores.topk(self.max_per_img)
            kernel_pred = kernel_pred[kernel_index]
            det_labels = det_labels[kernel_index]

        det_masks = F.conv2d(mask_feature, kernel_pred.view(-1, 256, 1, 1), stride=1).sigmoid()
        seg_masks = det_masks.squeeze(0) > self.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float() + 0.00001
        seg_scores = (det_masks.squeeze() * seg_masks.float()).sum((1, 2)) / sum_masks
        scores *= seg_scores.squeeze(0)
        det_masks = F.interpolate(det_masks, scale_factor=4, mode='bilinear')[:, :, :img_shape[0],
                    :img_shape[1]]

        det_masks = F.interpolate(det_masks, size=origin_img_shape[:-1], mode='bilinear').squeeze() > self.mask_thr

        return det_labels, scores, det_masks

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        pass
