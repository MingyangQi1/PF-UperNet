# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict
from .dcn_v2 import DCN

from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        aa=F.avg_pool2d(x, x.size()[2:])
        atten = self.sigmoid(self.conv_atten(aa))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=256, out_nc=256, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        if in_nc== 320:
            self.offset = Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=None)
        if in_nc== 128:
            self.offset = Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=None)
        if in_nc == 64:
            self.offset = Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=None)
        self.dcpack_L2 = DCN(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        aaa = torch.cat([feat_arm, feat_up * 2], dim=1)
        offset = self.offset(aaa)  # concat for offset by compute the dif

        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm

@HEADS.register_module()
class UPerHead(BaseDecodeHead):
   
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
       
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.qi_lateral_convs3 = FeatureAlign_V2(64, 512, norm='')
        self.qi_lateral_convs2 = FeatureAlign_V2(128, 512, norm='')
        self.qi_lateral_convs1 = FeatureAlign_V2(320, 512, norm='')
        
        self.eca = ECAAttention(kernel_size=3)

    def psp_forward(self, inputs):
       
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def resize_channels_with_1x1(inputs, channels):
       
        resized_inputs = []
        for input in inputs:
            resized_input = nn.Conv2d(input.size(1), channels, kernel_size=1, stride=1, padding=0, bias=False)
            resized_input.weight.data.copy_(input.view(-1, input.size(1)).mean(dim=0))
            resized_inputs.append(resized_input(input))
        return resized_inputs

    def _forward_feature(self, inputs):
        
        inputs = self._transform_inputs(inputs)
        laterals = []
        laterals.append(self.psp_forward(inputs))
        for i in range(0,3):
            if i==0:
                a1 = inputs[2]
                b1 = laterals[i]
                laterals1= self.qi_lateral_convs1(a1,b1)
                laterals.append(laterals1)
            if i==1:
                a2 = inputs[1]
                b2 = laterals[i]
                laterals2= self.qi_lateral_convs2(a2,b2)
                laterals.append(laterals2)
            if i==2:
                a3 = inputs[0]
                b3 = laterals[i]
                laterals3 = self.qi_lateral_convs3(a3,b3)
                laterals.append(laterals3)
        lateralss = laterals[::-1]
        used_backbone_levels = len(lateralss)
        
        fpn_outs = [
            self.fpn_convs[i](lateralss[i])
            for i in range(used_backbone_levels - 1)
        ]
        
        fpn_outs.append(lateralss[-1])
        
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_outs = self.eca(fpn_outs)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
