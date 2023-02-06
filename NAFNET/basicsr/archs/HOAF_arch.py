import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
import time


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class HOAF_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, num_groups, ch_per_gp, weight):
        ctx.save_for_backward(weight)
        ctx.num_groups, ctx.ch_per_gp = num_groups, ch_per_gp
        inp_c = torch.chunk(inp, num_groups * ch_per_gp, dim=1)
        ctx.inp_c = inp_c
        output_c = []
        for i in range(num_groups * ch_per_gp):
            output_c.append(torch.zeros_like(inp_c[0]))
        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    res = inp_c[cc1] * inp_c[cc2]
                    output_c[cc1] += res * weight[g][c1][c2]
                    if cc1 != cc2:
                        output_c[cc2] += res * weight[g][c2][c1]
        output = torch.cat(output_c, dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        # print(type(weight))
        inp_c = ctx.inp_c
        num_groups, ch_per_gp = ctx.num_groups, ctx.ch_per_gp
        grad_output_c = torch.chunk(grad_output, num_groups * ch_per_gp, dim=1)
        grad_input_c = []
        for i in range(num_groups * ch_per_gp):
            grad_input_c.append(torch.zeros_like(grad_output_c[0]))
        grad_weight = torch.zeros_like(weight)
        grad_group = grad_ch = None

        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    grad_input_c[cc1] += grad_output_c[cc1] * inp_c[cc2] * weight[g][c1][c2]
                    inp_temp = inp_c[cc1] * inp_c[cc2]
                    grad_weight[g][c1][c2] += (grad_output_c[cc1] * inp_temp).sum()
                    if cc1 != cc2:
                        grad_input_c[cc2] += grad_output_c[cc2] * inp_c[cc1] * weight[g][c2][c1]
                        grad_weight[g][c2][c1] += (grad_output_c[cc2] * inp_temp).sum()
        grad_input = torch.cat(grad_input_c, dim=1)
        # print("backward: {:.2f}".format(time.time() - bg))

        return grad_input, grad_group, grad_ch, grad_weight

class HOAF_Function_without_weight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, num_groups, ch_per_gp):

        # ctx.save_for_backward(inp)
        ctx.num_groups, ctx.ch_per_gp = num_groups, ch_per_gp
        inp_c = torch.chunk(inp, num_groups * ch_per_gp, dim=1)
        ctx.inp_c = inp_c
        output_c = []
        for i in range(num_groups * ch_per_gp):
            output_c.append(torch.zeros_like(inp_c[0]))
        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    res = inp_c[cc1] * inp_c[cc2]
                    output_c[cc1] += res
                    if cc1 != cc2:
                        output_c[cc2] += res
        output = torch.cat(output_c, dim=1)
        return output / num_groups

    @staticmethod
    def backward(ctx, grad_output):
        # inp_c, = ctx.saved_tensors
        inp_c = ctx.inp_c
        num_groups, ch_per_gp = ctx.num_groups, ctx.ch_per_gp
        grad_output_c = torch.chunk(grad_output, num_groups * ch_per_gp, dim=1)
        grad_input_c = []
        for i in range(num_groups * ch_per_gp):
            grad_input_c.append(torch.zeros_like(grad_output_c[0]))
        grad_group = grad_ch = None

        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    grad_input_c[cc1] += grad_output_c[cc1] * inp_c[cc2] / num_groups
                    if cc1 != cc2:
                        grad_input_c[cc2] += grad_output_c[cc2] * inp_c[cc1] / num_groups
        grad_input = torch.cat(grad_input_c, dim=1)

        return grad_input, grad_group, grad_ch


class HOAF(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels, num_pow):
        super(HOAF, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_pow = num_pow
        self.ch_per_gp = self.num_channels // self.num_groups
        # self.weight = nn.Parameter(torch.ones(num_groups, self.ch_per_gp, self.ch_per_gp) / self.ch_per_gp,
        #                            requires_grad=True)
        self.out_channels = 0
        self.out_channels1 = self.ch_per_gp
        self.out_channels2 = self.ch_per_gp * (self.ch_per_gp + 1) // 2
        self.out_channels3 = 0
        for i in range(self.ch_per_gp):
            self.out_channels3 += (i + 1) * (i + 2) // 2
        if 1 in self.num_pow:
            self.out_channels += self.out_channels1
            self.conv1 = nn.Conv2d(self.out_channels1 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 2 in self.num_pow:
            self.out_channels += self.out_channels2
            self.conv2 = nn.Conv2d(self.out_channels2 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 3 in self.num_pow:
            self.out_channels += self.out_channels3
            self.conv3 = nn.Conv2d(self.out_channels3 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        # print("channels: ", self.out_channels, self.out_channels1, self.out_channels2, self.out_channels3)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        # return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)
        inp_c = torch.chunk(inp, self.num_channels, dim=1)
        output_c1 = []
        output_c2 = []
        output_c3 = []
        # for i in range(self.num_channels):
        #     output_c.append(torch.zeros_like(inp_c[0]))
        output = torch.zeros_like(inp)
        for g in range(self.num_groups):
            if 1 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    output_c1.append(inp_c[cc1])

            if 2 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    for c2 in range(c1, self.ch_per_gp):
                        cc2 = c2 + g * self.ch_per_gp
                        output_c2.append(inp_c[cc1] * inp_c[cc2])
            if 3 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    for c2 in range(c1, self.ch_per_gp):
                        cc2 = c2 + g * self.ch_per_gp
                        for c3 in range(c2, self.ch_per_gp):
                            cc3 = c3 + g * self.ch_per_gp
                            output_c3.append(inp_c[cc1] * inp_c[cc2] * inp_c[cc3])
        if 1 in self.num_pow:
            output += self.conv1(torch.cat(output_c1, dim=1))
        if 2 in self.num_pow:
            output += self.conv2(torch.cat(output_c2, dim=1))
        if 3 in self.num_pow:
            output += self.conv3(torch.cat(output_c3, dim=1))
        # output = torch.cat(output_c, dim=1)
        # output = self.conv(output)
        return output


class HOAF_prune(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels, num_pow):
        super(HOAF_prune, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_pow = num_pow
        self.ch_per_gp = self.num_channels // self.num_groups
        self.iter = 0
        # self.weight = nn.Parameter(torch.ones(num_groups, self.ch_per_gp, self.ch_per_gp) / self.ch_per_gp,
        #                            requires_grad=True)
        self.out_channels = 0
        self.out_channels1 = self.ch_per_gp
        self.out_channels2 = self.ch_per_gp * (self.ch_per_gp + 1) // 2
        self.out_channels3 = 0
        for i in range(self.ch_per_gp):
            self.out_channels3 += (i + 1) * (i + 2) // 2
        if 1 in self.num_pow:
            self.out_channels += self.out_channels1
            self.conv1 = nn.Conv2d(self.out_channels1 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 2 in self.num_pow:
            self.out_channels += self.out_channels2
            self.conv2 = nn.Conv2d(self.out_channels2 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
            # self.conv2 = prune.ln_structured(self.conv2, 'weight', 0, n=1, dim=1)
        if 3 in self.num_pow:
            self.out_channels += self.out_channels3
            self.conv3 = nn.Conv2d(self.out_channels3 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
            self.conv3 = prune.ln_structured(self.conv3, 'weight', 0, n=1, dim=1)
        # print("channels: ", self.out_channels, self.out_channels1, self.out_channels2, self.out_channels3)
        self.beta2 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)
        self.beta3 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        # return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)
        self.iter += 1
        # B, C, H, W = inp.shape
        # output_c1 = torch.zeros([B, self.out_channels1 * self.num_groups])

        inp_c = torch.chunk(inp, self.num_channels, dim=1)
        output_c1 = []
        output_c2 = []
        output_c3 = []
        # for i in range(self.num_channels):
        #     output_c.append(torch.zeros_like(inp_c[0]))
        output = torch.zeros_like(inp)
        for g in range(self.num_groups):
            if 1 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    output_c1.append(inp_c[cc1])

            if 2 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    for c2 in range(c1, self.ch_per_gp):
                        cc2 = c2 + g * self.ch_per_gp
                        output_c2.append(inp_c[cc1] * inp_c[cc2])
            if 3 in self.num_pow:
                for c1 in range(self.ch_per_gp):
                    cc1 = c1 + g * self.ch_per_gp
                    for c2 in range(c1, self.ch_per_gp):
                        cc2 = c2 + g * self.ch_per_gp
                        for c3 in range(c2, self.ch_per_gp):
                            cc3 = c3 + g * self.ch_per_gp
                            output_c3.append(inp_c[cc1] * inp_c[cc2] * inp_c[cc3])
        if 1 in self.num_pow:
            output = output + inp
        if 2 in self.num_pow:
            # if 30000 < self.iter < 100000 and self.iter % 20000 == 0:
            #     self.conv2 = prune.ln_structured(self.conv2, 'weight', 0.1, n=1, dim=1)
            output = output + self.conv2(torch.cat(output_c2, dim=1)) * self.beta2
        if 3 in self.num_pow:
            if 30000 < self.iter < 100000 and self.iter % 10000 == 0:
                self.conv3 = prune.ln_structured(self.conv3, 'weight', 0.3, n=1, dim=1)
            output += self.conv3(torch.cat(output_c3, dim=1)) * self.beta3 * min(0.1, 0.1 * max(0., (self.iter / 60000) - 0.1))
        # output = torch.cat(output_c, dim=1)
        # output = self.conv(output)
        return output


class HOAF_v2(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels, num_pow):
        super(HOAF_v2, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_pow = num_pow
        self.ch_per_gp = self.num_channels // self.num_groups
        self.iter = 0
        # self.weight = nn.Parameter(torch.ones(num_groups, self.ch_per_gp, self.ch_per_gp) / self.ch_per_gp,
        #                            requires_grad=True)
        self.out_channels = 0
        self.out_channels1 = self.ch_per_gp
        self.out_channels2 = self.ch_per_gp * (self.ch_per_gp + 1) // 2
        self.out_channels3 = 0
        for i in range(self.ch_per_gp):
            self.out_channels3 += (i + 1) * (i + 2) // 2
        if 1 in self.num_pow:
            self.out_channels += self.out_channels1
            self.conv1 = nn.Conv2d(self.out_channels1 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 2 in self.num_pow:
            self.out_channels += self.out_channels2
            self.conv2 = nn.Conv2d(self.out_channels2 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        if 3 in self.num_pow:
            self.out_channels += self.out_channels3
            self.conv3 = nn.Conv2d(self.out_channels3 * self.num_groups, self.num_channels, kernel_size=1,
                                   groups=self.num_groups)
        # print("channels: ", self.out_channels, self.out_channels1, self.out_channels2, self.out_channels3)
        self.beta2 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)
        self.beta3 = nn.Parameter(torch.ones((1, self.num_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        # return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)
        self.iter += 1
        B, C, H, W = inp.shape
        # output_c1 = torch.zeros([B, self.out_channels1 * self.num_groups, H, W])
        inp_c = []
        for i in range(self.ch_per_gp):
            inp_c.append(inp[:, i::self.ch_per_gp, :, :].clone())
        output = torch.zeros_like(inp)
        if 1 in self.num_pow:
            output = inp
        if 2 in self.num_pow:
            output_c2 = torch.zeros([B, self.out_channels2 * self.num_groups, H, W]).to(inp.device)
            idx = 0
            for i in range(self.ch_per_gp):
                for j in range(i, self.ch_per_gp):
                    output_c2[:, idx::self.out_channels2, :, :] += inp_c[i] * inp_c[j]
            output += self.conv2(output_c2) * self.beta2

        # if 2 in self.num_pow:
        #     # if 30000 < self.iter < 100000 and self.iter % 20000 == 0:
        #     #     self.conv2 = prune.ln_structured(self.conv2, 'weight', 0.1, n=1, dim=1)
        #     output += self.conv2(torch.cat(output_c2, dim=1)) * self.beta2
        if 3 in self.num_pow:
            output_c3 = torch.zeros([B, self.out_channels3 * self.num_groups, H, W])
            if 30000 < self.iter < 100000 and self.iter % 10000 == 0:
                self.conv3 = prune.ln_structured(self.conv3, 'weight', 0.3, n=1, dim=1)
            output += self.conv3(torch.cat(output_c3, dim=1)) * self.beta3 * min(0.1, 0.1 * max(0., (self.iter / 60000) - 0.1))
        # output = torch.cat(output_c, dim=1)
        # output = self.conv(output)
        return output


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

        inp_c = torch.chunk(channel_shuffle(inp, self.num_groups), self.ch_per_gp, dim=1)
        # inp_c = torch.chunk(inp, self.ch_per_gp, dim=1)
        # output_c1 = []
        output_c2 = []

        for i in range(self.ch_per_gp):
            for j in range(i, self.ch_per_gp):
                output_c2.append(inp_c[i] * inp_c[j])

        if 1 in self.num_pow:
            output = output + inp
        if 2 in self.num_pow:
            output = output + self.conv2(channel_shuffle(torch.cat(output_c2, dim=1), self.out_channels2)) * \
                     self.beta2
            # output = output + self.conv2(torch.cat(output_c2, dim=1)) * self.beta2

        return output


class NAFBlock_HOAF(nn.Module):
    def __init__(self, c, num_pow):
        super(NAFBlock_HOAF, self).__init__()

        self.num_pow = num_pow
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1,
                               groups=c, bias=True)
        self.gate1 = HOAF_v2(c // 4, c, self.num_pow)
        # self.gate1 = nn.GELU()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True)
        )
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.gate2 = HOAF_v2(c // 4, c, self.num_pow)
        # self.gate2 = nn.GELU()
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.gate1(x1)
        x1 = x1 * self.sca(x1)
        x1 = self.conv3(x1)
        x1 = x1 * self.beta + x

        x2 = self.norm2(x1)
        x2 = self.conv4(x2)
        x2 = self.gate2(x2)
        x2 = self.conv5(x2)
        return x2 * self.gamma + x1


class NAFBlock_HOAF_prune(nn.Module):
    def __init__(self, c, num_pow):
        super(NAFBlock_HOAF_prune, self).__init__()

        self.num_pow = num_pow
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1,
                               groups=c, bias=True)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.gate1 = HOAF_v3(c // 4, c, self.num_pow)
        # self.gate1 = nn.GELU()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True)
        )
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.gate2 = HOAF_v3(c // 4, c, self.num_pow)
        # self.gate2 = nn.GELU()
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.gate1(x1)
        x1 = self.gelu(x1)
        x1 = x1 * self.sca(x1)
        x1 = self.conv3(x1)
        x1 = x1 * self.beta + x

        x2 = self.norm2(x1)
        x2 = self.conv4(x2)
        x2 = self.gate2(x2)
        x2 = self.gelu(x2)
        x2 = self.conv5(x2)
        return x2 * self.gamma + x1


class NAFBlock_GELU(nn.Module):
    def __init__(self, c):
        super(NAFBlock_GELU, self).__init__()

        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1,
                               groups=c, bias=True)
        # self.gate1 = HOAF(c // 4, c)
        self.gate1 = nn.GELU()
        self.gate_conv1 = nn.Conv2d(c, c, kernel_size=1, groups=c // 4)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True)
        )
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        # self.gate2 = HOAF(c // 4, c)
        self.gate2 = nn.GELU()
        self.gate_conv2 = nn.Conv2d(c, c, kernel_size=1, groups=c // 4)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.GELU()(x1)
        x1 = self.gate1(x1)
        x1 = self.gate_conv1(x1)
        x1 = x1 * self.sca(x1)
        x1 = self.conv3(x1)
        x1 = x1 * self.beta + x

        x2 = self.norm2(x1)
        x2 = self.conv4(x2)
        x2 = self.GELU()(x2)
        x2 = self.gate2(x2)
        x2 = self.gate_conv2(x2)
        x2 = self.conv5(x2)
        return x2 * self.gamma + x1


@ARCH_REGISTRY.register()
class NAFNET_HOAF(nn.Module):

    def __init__(self,
                 img_channels=3,
                 width=32,
                 enc_block_nums=[2, 2, 4, 8],
                 mid_block_nums=12,
                 dec_block_nums=[2, 2, 2, 2],
                 num_pow=[1, 2]):
        super(NAFNET_HOAF, self).__init__()

        self.begin = nn.Conv2d(img_channels, width, 3, 1, 1)
        print(img_channels, width, self.begin)

        self.num_pow = num_pow
        self.enc_block = nn.ModuleList()
        self.down_block = nn.ModuleList()
        ch = width
        for num in enc_block_nums:
            self.enc_block.append(nn.Sequential(*[NAFBlock_HOAF(ch, num_pow=self.num_pow) for _ in range(num)]))
            self.down_block.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.mid_block = nn.Sequential(*[NAFBlock_HOAF(ch, num_pow=self.num_pow) for _ in range(mid_block_nums)])

        self.dec_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for num in dec_block_nums:
            self.up_block.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.dec_block.append(nn.Sequential(*[NAFBlock_HOAF(ch, num_pow=self.num_pow) for _ in range(num)]))

        self.end = nn.Conv2d(width, img_channels, 3, 1, 1)

        self.mod = 2 ** len(enc_block_nums)

    def forward(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        inp = self.pad_to_pow(inp)
        x = inp
        x = self.begin(x)

        skips = []

        for enc, down in zip(self.enc_block, self.down_block):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block(x)

        for dec, up, skip in zip(self.dec_block, self.up_block, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.end(x)
        x = x + inp
        return x[:, :, :h, :w]

    def pad_to_pow(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        h_pad = (self.mod - h % self.mod) % self.mod
        w_pad = (self.mod - w % self.mod) % self.mod
        return F.pad(inp, (0, w_pad, 0, h_pad))


@ARCH_REGISTRY.register()
class NAFNET_HOAF_prune(nn.Module):

    def __init__(self,
                 img_channels=3,
                 width=32,
                 enc_block_nums=[2, 2, 4, 8],
                 mid_block_nums=12,
                 dec_block_nums=[2, 2, 2, 2],
                 num_pow=[1, 2]):
        super(NAFNET_HOAF_prune, self).__init__()

        self.begin = nn.Conv2d(img_channels, width, 3, 1, 1)
        print(img_channels, width, self.begin)

        self.num_pow = num_pow
        self.enc_block = nn.ModuleList()
        self.down_block = nn.ModuleList()
        ch = width
        for num in enc_block_nums:
            self.enc_block.append(nn.Sequential(*[NAFBlock_HOAF_prune(ch, num_pow=self.num_pow) for _ in range(num)]))
            self.down_block.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.mid_block = nn.Sequential(*[NAFBlock_HOAF_prune(ch, num_pow=self.num_pow) for _ in range(mid_block_nums)])

        self.dec_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for num in dec_block_nums:
            self.up_block.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.dec_block.append(nn.Sequential(*[NAFBlock_HOAF_prune(ch, num_pow=self.num_pow) for _ in range(num)]))

        self.end = nn.Conv2d(width, img_channels, 3, 1, 1)

        self.mod = 2 ** len(enc_block_nums)

    def forward(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        inp = self.pad_to_pow(inp)
        x = inp
        x = self.begin(x)

        skips = []

        for enc, down in zip(self.enc_block, self.down_block):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block(x)

        for dec, up, skip in zip(self.dec_block, self.up_block, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.end(x)
        x = x + inp
        return x[:, :, :h, :w]

    def pad_to_pow(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        h_pad = (self.mod - h % self.mod) % self.mod
        w_pad = (self.mod - w % self.mod) % self.mod
        return F.pad(inp, (0, w_pad, 0, h_pad))


@ARCH_REGISTRY.register()
class NAFNET_GELU(nn.Module):

    def __init__(self,
                 img_channels=3,
                 width=32,
                 enc_block_nums=[2, 2, 4, 8],
                 mid_block_nums=12,
                 dec_block_nums=[2, 2, 2, 2]):
        super(NAFNET_GELU, self).__init__()

        self.begin = nn.Conv2d(img_channels, width, 3, 1, 1)
        print(img_channels, width, self.begin)

        self.enc_block = nn.ModuleList()
        self.down_block = nn.ModuleList()
        ch = width
        for num in enc_block_nums:
            self.enc_block.append(nn.Sequential(*[NAFBlock_GELU(ch) for _ in range(num)]))
            self.down_block.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.mid_block = nn.Sequential(*[NAFBlock_GELU(ch) for _ in range(mid_block_nums)])

        self.dec_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for num in dec_block_nums:
            self.up_block.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.dec_block.append(nn.Sequential(*[NAFBlock_GELU(ch) for _ in range(num)]))

        self.end = nn.Conv2d(width, img_channels, 3, 1, 1)

        self.mod = 2 ** len(enc_block_nums)

    def forward(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        inp = self.pad_to_pow(inp)
        x = inp
        x = self.begin(x)

        skips = []

        for enc, down in zip(self.enc_block, self.down_block):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block(x)

        for dec, up, skip in zip(self.dec_block, self.up_block, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.end(x)
        x = x + inp
        return x[:, :, :h, :w]

    def pad_to_pow(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        h_pad = (self.mod - h % self.mod) % self.mod
        w_pad = (self.mod - w % self.mod) % self.mod
        return F.pad(inp, (0, w_pad, 0, h_pad))
