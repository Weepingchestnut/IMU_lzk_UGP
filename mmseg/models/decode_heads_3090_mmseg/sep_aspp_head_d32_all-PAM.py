import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Scale

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule

import torch.nn.functional as F
from ..utils import SelfAttentionBlock as _SelfAttentionBlock


class PAM(_SelfAttentionBlock):
    """Position Attention Module (PAM)

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels, channels):
        super(PAM, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        out = super(PAM, self).forward(x, x)

        out = self.gamma(out) + x
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
        self.c1_pam_in_conv = ConvModule(
            c1_in_channels,
            c1_channels,  # 128
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.c2_pam_in_conv = ConvModule(
        #     self.c2_channels,
        #     128,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        # self.c3_pam_in_conv = ConvModule(
        #     self.c3_channels,
        #     128,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        # self.c4_pam_in_conv = ConvModule(
        #     self.c4_channels,
        #     128,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_PAM = PAM(48, 48)
        else:
            self.c1_PAM = None
        # self.c2_PAM = PAM(128, 128)
        # self.c3_PAM = PAM(128, 128)
        # self.c4_PAM = PAM(128, 128)

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
        # c4_output = self.c4_pam_in_conv(inputs[3])
        # c4_output = self.c4_PAM(c4_output)
        # print("c4_output: ".format(c4_output.shape))
        output = torch.cat([output, inputs[3]], dim=1)      # channels = 512+2048
        output = self.bottleneck4(output)       # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样
        # c3_output = self.c3_pam_in_conv(inputs[2])
        # c3_output = self.c3_PAM(c3_output)
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, inputs[2]], dim=1)
        output = self.bottleneck3(output)       # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        # c2_output = self.c2_pam_in_conv(inputs[1])
        # c2_output = self.c2_PAM(c2_output)
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, inputs[1]], dim=1)
        output = self.bottleneck2(output)       # 3x3conv channels = 512
        # print("bottleneck2: {}".format(output.shape))

        # c1跨层 2倍上采样
        # output = resize(
        #     input=output,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # output = torch.cat([output, inputs[0]], dim=1)

        if self.c1_PAM is not None:
            c1_output = self.c1_pam_in_conv(inputs[0])
            c1_output = self.c1_PAM(c1_output)
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
