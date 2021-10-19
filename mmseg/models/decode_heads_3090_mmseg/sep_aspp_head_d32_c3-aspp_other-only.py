import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Scale

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule


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

        self.c3_image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.c3_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.c3_aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.c3_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # self.c2_aspp_modules = DepthwiseSeparableASPPModule(
        #     dilations=self.dilations,
        #     in_channels=self.c2_channels,
        #     channels=self.c2_channels,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

        # if c1_in_channels > 0:
        #     self.c1_CAM = CAM()
        # else:
        #     self.c1_CAM = None
        # self.c2_CAM = CAM()
        # self.c3_CAM = CAM()
        # self.c4_CAM = CAM()

        self.bottleneck2 = ConvModule(
            self.channels + self.c2_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck3 = ConvModule(
            # self.channels + self.c3_channels,
            # self.channels + self.channels,
            self.channels + self.c4_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.bottleneck4 = ConvModule(
        #     self.channels + self.c4_channels,
        #     # self.channels + self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

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
        # x = self._transform_inputs(inputs)      # 2048 * 16 * 32
        # aspp_outs = [
        #     resize(
        #         self.image_pool(x),
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        # aspp_outs.extend(self.aspp_modules(x))
        # aspp_outs = torch.cat(aspp_outs, dim=1)
        # output = self.bottleneck(aspp_outs)  # 3x3conv channels = 512
        # print("aspp_outs bottleneck: {}".format(output.shape))
        # c4跨层
        # print("c4_output: ".format(c4_output.shape))
        # output = torch.cat([output, inputs[3]], dim=1)  # channels = 512+2048
        output = inputs[3]
        # output = self.bottleneck4(output)  # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样 d16 + aspp
        c3_aspp_outs = [
            resize(
                self.c3_image_pool(inputs[2]),
                size=inputs[2].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        c3_aspp_outs.extend(self.c3_aspp_modules(inputs[2]))
        c3_aspp_outs = torch.cat(c3_aspp_outs, dim=1)
        c3_output = self.bottleneck(c3_aspp_outs)
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c3_output], dim=1)
        output = self.bottleneck3(output)  # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        # c2_aspp_outs = self.c2_aspp_modules(inputs[1])
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, inputs[1]], dim=1)
        output = self.bottleneck2(output)  # 3x3conv channels = 512
        # print("bottleneck2: {}".format(output.shape))

        # c1跨层 2倍上采样
        output = resize(
            input=output,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, inputs[0]], dim=1)

        # if self.c1_aspp_modules is not None:
        #     c1_aspp_outs = self.c1_aspp_modules(inputs[0])
        #     output = resize(
        #         input=output,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #     output = torch.cat([output, c1_aspp_outs], dim=1)
        output = self.sep_bottleneck(output)
        # print("sep_bottleneck: {}".format(output.shape))
        output = self.cls_seg(output)
        # print("cls_seg: {}".format(output.shape))
        return output
