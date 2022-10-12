"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_module import *

class LinearBottleNeck(MetaModule):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = MetaSequential(
            MetaConv2d(in_channels, in_channels * t, 1),
            MetaBatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            MetaConv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            MetaBatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            MetaConv2d(in_channels * t, out_channels, 1),
            MetaBatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(MetaModule):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = MetaSequential(
            MetaConv2d(3, 32, 1, padding=1),
            MetaBatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = MetaSequential(
            MetaConv2d(320, 1280, 1),
            MetaBatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = MetaConv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2(num_class):
    return MobileNetV2(class_num=num_class)