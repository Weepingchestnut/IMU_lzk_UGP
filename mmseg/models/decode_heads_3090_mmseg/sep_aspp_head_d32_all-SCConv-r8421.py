import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule

import torch.nn.functional as F


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    # expansion = 4
    # pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, pooling_r=4, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.pooling_r = pooling_r
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        # self.conv3 = nn.Conv2d(
        #     group_width * 2, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        # residual = x

        out_a = self.conv1_a(x)
        # print("out_a: {}".format(out_a.shape))
        out_a = self.bn1_a(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)

        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        # out = self.conv3(torch.cat([out_a, out_b], dim=1))
        # out = self.bn3(out)
        out = torch.cat([out_a, out_b], dim=1)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)

        return out


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.c2_channels = 512
        self.c3_channels = 1024
        self.c4_channels = 2048
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_SCConv = SCBottleneck(c1_in_channels, c1_channels, 8)
        else:
            self.c1_SCConv = None
        self.c2_SCConv = SCBottleneck(self.c2_channels, self.c2_channels, 4)
        self.c3_SCConv = SCBottleneck(self.c3_channels, self.c3_channels, 2)
        self.c4_SCConv = SCBottleneck(self.c4_channels, self.c4_channels, 1)

        self.bottleneck2 = ConvModule(
            self.channels + self.c2_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck3 = ConvModule(
            self.channels + self.c3_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck4 = ConvModule(
            self.channels + self.c4_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)     # 3x3conv channels = 512
        # print("aspp_outs bottleneck: {}".format(output.shape))
        # c4跨层
        c4_output = self.c4_SCConv(inputs[3])
        # print("c4_output: ".format(c4_output.shape))
        output = torch.cat([output, c4_output], dim=1)      # channels = 2048+512
        output = self.bottleneck4(output)       # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样
        c3_output = self.c3_SCConv(inputs[2])
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c3_output], dim=1)
        output = self.bottleneck3(output)       # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        c2_output = self.c2_SCConv(inputs[1])
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c2_output], dim=1)
        output = self.bottleneck2(output)       # 3x3conv channels = 512
        # print("bottleneck2: {}".format(output.shape))

        # c1跨层 2倍上采样
        # output = resize(
        #     input=output,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # output = torch.cat([output, inputs[0]], dim=1)

        if self.c1_SCConv is not None:
            c1_output = self.c1_SCConv(inputs[0])
            output = resize(
                input=output,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # print("sep_bottleneck: {}".format(output.shape))
        output = self.cls_seg(output)
        # print("cls_seg: {}".format(output.shape))
        return output
