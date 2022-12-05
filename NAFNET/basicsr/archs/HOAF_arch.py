import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class HOAF_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, num_groups, ch_per_gp, weight):
        ctx.save_for_backward(inp, weight)
        ctx.num_groups, ctx.ch_per_gp = num_groups, ch_per_gp
        output = torch.zeros_like(inp)
        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    res = inp[:, cc1, :, :] * inp[:, cc2, :, :] * weight[g][c1][c2]
                    output[:, cc1, :, :] += res
                    if cc1 != cc2:
                        output[:, cc2, :, :] += res
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        num_groups, ch_per_gp = ctx.num_groups, ctx.ch_per_gp
        grad_input = torch.zeros_like(grad_output)
        grad_group = grad_ch = None
        grad_weight = torch.zeros_like(weight)

        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    grad_input[:, cc1, :, :] += grad_output[:, cc1, :, :] * inp[:, cc2, :, :] * weight[g][c1][c2]
                    if cc1 != cc2:
                        grad_input[:, cc2, :, :] += grad_output[:, cc2, :, :] * inp[:, cc1, :, :] * weight[g][c1][c2]
                    grad_weight[g][c1][c2] += ((grad_output[:, cc1, :, :] + grad_output[:, cc2, :, :]) *
                                               inp[:, cc1, :, :] * inp[:, cc2, :, :]).sum()

                    # res = inp[:, cc1, :, :] * inp[:, cc2, :, :] * weight[g][c1][c2]
                    # output[:, cc1, :, :] += res
                    # if cc1 != cc2:
                    #     output[:, cc2, :, :] += res

        return grad_input, grad_group, grad_ch, grad_weight


class HOAF_Function_without_weight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, num_groups, ch_per_gp):
        ctx.save_for_backward(inp)
        ctx.num_groups, ctx.ch_per_gp = num_groups, ch_per_gp
        output = torch.zeros_like(inp)
        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    res = inp[:, cc1, :, :] * inp[:, cc2, :, :]
                    output[:, cc1, :, :] += res
                    if cc1 != cc2:
                        output[:, cc2, :, :] += res
        return output / num_groups

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        num_groups, ch_per_gp = ctx.num_groups, ctx.ch_per_gp
        grad_input = torch.zeros_like(grad_output)
        grad_group = grad_ch = None

        for g in range(num_groups):
            for c1 in range(ch_per_gp):
                for c2 in range(c1, ch_per_gp):
                    cc1 = c1 + g * ch_per_gp
                    cc2 = c2 + g * ch_per_gp
                    grad_input[:, cc1, :, :] += grad_output[:, cc1, :, :] * inp[:, cc2, :, :] / num_groups
                    if cc1 != cc2:
                        grad_input[:, cc2, :, :] += grad_output[:, cc2, :, :] * inp[:, cc1, :, :] / num_groups

                    # res = inp[:, cc1, :, :] * inp[:, cc2, :, :] * weight[g][c1][c2]
                    # output[:, cc1, :, :] += res
                    # if cc1 != cc2:
                    #     output[:, cc2, :, :] += res

        return grad_input, grad_group, grad_ch


"""
0
1.4170293807983398 15.229024648666382
1
1.2995247840881348 14.242572784423828
2
1.1736533641815186 14.207785367965698
3
1.0052779912948608 14.190433263778687
4
0.7590433359146118 14.276637315750122
5
0.7440890073776245 14.2281813621521
6
0.5762566328048706 14.404308319091797
7
0.5205650329589844 14.311109066009521
8
0.5044991374015808 14.367441892623901
9
0.46382617950439453 14.304344654083252
10
0.4057248532772064 14.416651010513306
11
0.36068853735923767 15.096582412719727
"""

class HOAF(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_groups', 'num_channels']
    num_groups: int
    num_channels: int

    def __init__(self, num_groups, num_channels):
        super(HOAF, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels need to be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.ch_per_gp = self.num_channels // self.num_groups
        self.weight = nn.Parameter(torch.ones(num_groups, self.ch_per_gp, self.ch_per_gp) / self.ch_per_gp,
                                   requires_grad=True)

    def forward(self, inp):
        # return HOAF_Function.apply(inp, self.num_groups, self.ch_per_gp, self.weight)
        return HOAF_Function_without_weight.apply(inp, self.num_groups, self.ch_per_gp)


class NAFBlock_HOAF(nn.Module):
    def __init__(self, c):
        super(NAFBlock_HOAF, self).__init__()

        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1,
                               groups=c, bias=True)
        # self.gate1 = HOAF(c // 4, c)
        self.gate1 = nn.GELU()
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


@ARCH_REGISTRY.register()
class NAFNET_HOAF(nn.Module):

    def __init__(self,
                 img_channels=3,
                 width=32,
                 enc_block_nums=[2, 2, 4, 8],
                 mid_block_nums=12,
                 dec_block_nums=[2, 2, 2, 2]):
        super(NAFNET_HOAF, self).__init__()

        self.begin = nn.Conv2d(img_channels, width, 3, 1, 1)
        print(img_channels, width, self.begin)

        self.enc_block = nn.ModuleList()
        self.down_block = nn.ModuleList()
        ch = width
        for num in enc_block_nums:
            self.enc_block.append(nn.Sequential(*[NAFBlock_HOAF(ch) for _ in range(num)]))
            self.down_block.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.mid_block = nn.Sequential(*[NAFBlock_HOAF(ch) for _ in range(mid_block_nums)])

        self.dec_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for num in dec_block_nums:
            self.up_block.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.dec_block.append(nn.Sequential(*[NAFBlock_HOAF(ch) for _ in range(num)]))

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
