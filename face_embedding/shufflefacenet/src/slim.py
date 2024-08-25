"""
https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe
http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf
"""

import torch.nn as nn


def g_name(g_name, m):
    m.g_name = g_name
    return m


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.view(x.data.shape[0], self.groups, x.data.shape[1] // self.groups, x.data.shape[2], x.data.shape[3])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        x = x.view(x.data.shape[0], -1, x.data.shape[3], x.data.shape[4])
        # x = x.permute(0, 1, 2,3)
        return x


def channel_shuffle(name, groups):
    return g_name(name, ChannelShuffle(groups))


class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x):
        x = x.permute(*self.order).contiguous()
        return x


def permute(name, order):
    return g_name(name, Permute(order))


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        assert self.axis == 1
        x = x.view(x.size(0), -1)
        return x


def flatten(name, axis):
    return g_name(name, Flatten(axis))


def conv_bn_prelu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        g_name(name + '/prelu', nn.PReLU(out_channels)),
    )


def conv_bn_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        g_name(name + '/relu', nn.ReLU(inplace=True)),
    )


def conv_bn(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
    )


def conv_dw(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if groups == 512:
        return nn.Sequential(
            g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
            g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        )
    else:
        return nn.Sequential(
            g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, out_channels, False)),
            g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        )
