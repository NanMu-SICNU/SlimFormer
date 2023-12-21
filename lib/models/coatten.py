"""
@Author: Hu yiyue
@Detail: SCI830_Pytorch coatten.py
@Time: 2022-09-23 22:59
@E-mail: 1927306867@qq.com
@Description: TODO
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ******************** CooAttention ********************
# 对应论文中的non-linear
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x


class CoorAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoorAttention, self).__init__()
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1, None))
        middle = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, middle, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(middle, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [batch_size, c, h, w]
        identity = x
        batch_size, c, h, w = x.size()  # [batch_size, c, h, w]
        # X Avg Pool
        x_h = self.poolh(x)  # [batch_size, c, h, 1]

        # Y Avg Pool
        x_w = self.poolw(x)  # [batch_size, c, 1, w]
        x_w = x_w.permute(0, 1, 3, 2)  # [batch_size, c, w, 1]

        # following the paper, cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        y = torch.cat((x_h, x_w), dim=2)  # [batch_size, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
        # split
        x_h, x_w = torch.split(y, [h, w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        x_w = x_w.permute(0, 1, 3, 2)  # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))
        # re-weight
        return identity * attention_h * attention_w


# ******************** SE ********************
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ******************** CBAM ********************
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
