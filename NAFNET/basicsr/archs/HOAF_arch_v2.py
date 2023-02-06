import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
import time
import random


def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class HOAF_baseline(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels, num_pow):
        super(HOAF_baseline, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_pow = num_pow
        self.ch_per_gp = self.num_channels // self.num_groups
        self.iter = 0

        self.out_channels = 0
        self.out_channels1 = self.ch_per_gp
        self.out_channels2 = self.ch_per_gp * (self.ch_per_gp + 1) // 2

        self.conv1 = nn.Conv2d(self.num_channels, self.out_channels2 * self.num_groups, 1, 1, 0, groups=self.num_groups)
        self.conv2 = nn.Conv2d(self.out_channels2 * self.num_groups, self.num_channels, 1, 1, 0, groups=self.num_groups)

        self.beta2 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)

        # self.channelShuffle = nn.ChannelShuffle(self.ch_per_gp)
        # self.channelUnShuffle = nn.ChannelShuffle(self.num_groups)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        # return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)
        self.iter += 1
        # B, C, H, W = inp.shape
        # output_c1 = torch.zeros([B, self.out_channels1 * self.num_groups])
        x = self.conv1(inp)
        x = self.conv2(x)

        return inp + self.beta2 * x


class HOAF_v3(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels, num_pow):
        super(HOAF_v3, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_pow = num_pow
        self.ch_per_gp = self.num_channels // self.num_groups
        self.iter = 0

        self.out_channels = 0
        self.out_channels1 = self.ch_per_gp
        self.out_channels2 = self.ch_per_gp * (self.ch_per_gp + 1) // 2

        if 1 in self.num_pow:
            self.out_channels += self.out_channels1
            self.conv1 = nn.Conv2d(self.out_channels1 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 2 in self.num_pow:
            self.out_channels += self.out_channels2
            self.conv2 = nn.Conv2d(self.out_channels2 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)

        self.beta2 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)

        # self.channelShuffle = nn.ChannelShuffle(self.ch_per_gp)
        # self.channelUnShuffle = nn.ChannelShuffle(self.num_groups)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        # return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)
        self.iter += 1
        # B, C, H, W = inp.shape
        # output_c1 = torch.zeros([B, self.out_channels1 * self.num_groups])

        output = torch.zeros_like(inp)

        inp_c = torch.chunk(channel_shuffle(inp, self.ch_per_gp), self.ch_per_gp, dim=1)
        # output_c1 = []
        output_c2 = []

        for i in range(self.ch_per_gp):
            for j in range(i, self.ch_per_gp):
                output_c2.append(inp_c[i] * inp_c[j])

        if 1 in self.num_pow:
            output = output + inp
        if 2 in self.num_pow:
            output = output + self.conv2(channel_shuffle(torch.cat(output_c2, dim=1), self.num_groups)) * self.beta2

        return output


if __name__ == "__main__":
    batchsize = 1
    channels = 32
    width = 1024
    _inp = torch.randn([batchsize, channels, width, width]).cuda()
    a = 0.1
    b = 0.2
    _target = a * _inp + b
    net_HOAF = HOAF_v3(channels // 4, channels, [1, 2]).cuda()
    # net_HOAF = HOAF_prune(channels // 4, channels, [1, 2]).cuda()
    # net_HOAF = HOAF_baseline(channels // 4, channels, [1, 2]).cuda()
    net_GELU = nn.GELU().cuda()
    cri = nn.MSELoss()
    optimizer = torch.optim.Adam(net_HOAF.parameters(), lr=1e-2)
    print("start")
    bg = time.time()
    for i in range(10000):
        _output_HOAF = net_HOAF(_inp)
        net_HOAF.zero_grad()
        _loss_HOAF = cri(_output_HOAF, _target)
        # _loss_HOAF.backward()
        optimizer.step()
        if i % 100 == 0:
            print("time cost = {:.2f}s".format(time.time() - bg))
            print("i = {:3d}, loss = {:.4f}".format(i, _loss_HOAF.item()))
            bg = time.time()



