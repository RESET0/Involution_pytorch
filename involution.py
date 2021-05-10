from typing import Union, Tuple, Optional

import torch
import torch.nn as nn


class involution(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 group_channels: int = 16,
                 reduction_ratio: int = 4,
                 dilation: int=1,
                 bias: bool = False) -> None:
        super(involution, self).__init__()
        """
        Constructor method
        :param channels: (int) Number of input channels
        :param kernel_size: (int) Kernel size to be used
        :param stride: (int) Stride factor to be utilized
        :param groups: (int) Number of channels of each group
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (int) Dilation in unfold to be employed
        :param bias: (bool) If true bias is utilized in each convolution layer
        """
        self.in_channel = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.reduction_ratio = reduction_ratio
        self.group_channels = group_channels
        self.dilation = dilation
        self.padding = (self.kernel_size-1)//2
        self.groups = self.in_channel // self.group_channels
        self.out_channel_1 = self.in_channel // self.reduction_ratio
        self.out_channel_2 = self.kernel_size**2 * self.groups

        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(kernel_size=self.stride,
                                        stride=self.stride)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.in_channel,
                      self.out_channel_1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=self.bias),
            nn.BatchNorm2d(self.out_channel_1, momentum=0.3), 
            nn.ReLU())

        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.out_channel_1,
                      self.out_channel_2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=self.bias))

        self.unfold = nn.Unfold(kernel_size=self.kernel_size,
                                dilation=self.dilation,
                                padding=self.padding,
                                stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        weight = self.Conv2(
            self.Conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        out = self.unfold(x).view(b, self.groups, self.kernel_size**2, h,
                                  w).unsqueeze(2)
        out = (weight * out).sum(dim=3).view(b, -1, h, w)
        return out