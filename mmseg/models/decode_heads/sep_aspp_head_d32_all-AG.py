import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule

from torch.nn import functional as F


class AttentionGate2D(nn.Module):
    """attention gate """
    def __init__(self, x_in_channels, g_in_channels, int_channels, out_channel, conv_cfg, norm_cfg, act_cfg, with_out):
        super(AttentionGate2D, self).__init__()
        self.x_c = x_in_channels
        self.g_c = g_in_channels
        self.int_c = int_channels
        self.W_x = ConvModule(
            in_channels=x_in_channels,      # c4 2048
            out_channels=int_channels,     # 2048
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.W_g = ConvModule(
            in_channels=g_in_channels,      # 512
            out_channels=int_channels,     # 2048
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.W_phi = ConvModule(
            in_channels=int_channels,
            out_channels=1,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        # if with_out:
        #     self.out_project = ConvModule(
        #         in_channels=int_channels,
        #         out_channels=out_channel,
        #         kernel_size=1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg)
        # else:
        #     self.out_project = None

    def forward(self, x, g):
        """
        @param self:
        @param x: feature map from encoders
        @param g: feature map from decoders
        @return output: attentioned x
        """
        # assert x.size(0) == g.size(0)
        identity = x
        x = self.W_x(x)
        g = self.W_g(g)
        theta_1 = F.relu(x + g, inplace=True)
        theta_2 = torch.sigmoid(self.W_phi(theta_1))
        # output = theta_2.expand_as(identity) * identity
        output = theta_2 * identity

        # if self.out_project is not None:
        #     output = self.out_project(output)

        return output


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
        # if c1_in_channels > 0:
        #     self.c1_bottleneck = ConvModule(
        #         c1_in_channels,
        #         c1_channels,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        # else:
        #     self.c1_bottleneck = None
        if c1_in_channels > 0:
            self.c1_attention_block = AttentionGate2D(
                x_in_channels=c1_in_channels,     # 256
                g_in_channels=self.channels,
                int_channels=c1_channels//2,       #256
                out_channel=c1_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_out=False)
        else:
            self.c1_attention_block = None
        self.c2_attention_block = AttentionGate2D(
            x_in_channels=self.c2_channels,
            g_in_channels=self.channels,
            int_channels=self.channels//2,
            out_channel=self.c2_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            with_out=False)
        self.c3_attention_block = AttentionGate2D(
            x_in_channels=self.c3_channels,
            g_in_channels=self.channels,
            int_channels=self.channels//2,
            out_channel=self.c3_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            with_out=False)
        self.c4_attention_block = AttentionGate2D(
            x_in_channels=self.c4_channels,
            g_in_channels=self.channels,
            int_channels=self.channels//2,
            out_channel=self.c4_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            with_out=False)

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
        c4_output = self.c4_attention_block(inputs[3], output)
        # print("c4_output: {}".format(c4_output.shape))
        output = torch.cat([output, c4_output], dim=1)      # channels = 2048+512
        output = self.bottleneck4(output)       # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c3_output = self.c3_attention_block(inputs[2], output)
        output = torch.cat([output, c3_output], dim=1)
        output = self.bottleneck3(output)       # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c2_output = self.c2_attention_block(inputs[1], output)
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

        if self.c1_attention_block is not None:
            output = resize(
                input=output,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            c1_output = self.c1_attention_block(inputs[0], output)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # print("sep_bottleneck: {}".format(output.shape))
        output = self.cls_seg(output)
        # print("cls_seg: {}".format(output.shape))
        return output
