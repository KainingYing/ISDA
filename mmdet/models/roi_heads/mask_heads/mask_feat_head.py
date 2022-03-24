import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, normal_init, ConvModule

from mmdet.models.builder import HEADS


import torch
import numpy as np


@HEADS.register_module
class MaskFeatHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 feat_channels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MaskFeatHead, self).__init__()

        self.in_channels = in_channels # 256
        self.out_channels = out_channels # 128
        self.start_level = start_level # 0
        self.end_level = end_level # 2
        assert start_level >= 0 and end_level >= start_level
        self.feat_channels = feat_channels # 256
        self.conv_cfg = conv_cfg # None
        self.norm_cfg = norm_cfg # {'type': 'GN', 'num_groups': 32, 'requires_grad': True}

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential() # 每一个level都有一个卷积集
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels, # 256
                    self.out_channels, # 128
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==end_level else self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.feat_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1) # 1/8 1/16 1/32

        feature_add_all_level = self.convs_all_levels[0](inputs[0]) # 直接进行了
        for i in range(1, len(inputs)):
            input_p = inputs[i] # 这里面就是从1开始
            if i == (len(inputs) - 1):  # 这一处与原来训练的模型有区别的
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device) # x, y 代表了xy的坐标，也就是这个是feature部分以经拥有了位置敏感的特征
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
                
            feature_add_all_level = self.convs_all_levels[i](input_p) + feature_add_all_level

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
