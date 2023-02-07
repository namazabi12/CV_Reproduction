import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY

from PIL import Image


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

    def forward(self, inp):
        self.iter += 1

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
        # print("inp: ", inp)
        # print("oup: ", output)
        return output


class ResidualBlockNoBN_HOAF(nn.Module):

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_HOAF, self).__init__()
        self.res_scale = res_scale
        self.norm1 = nn.GroupNorm(1, num_feat)

        # v1.2 conv
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # v1.3 conv -> depth-wise conv
        # self.conv0 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        # self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # self.conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)

        self.gelu = nn.GELU()
        self.hoaf = HOAF_v3(num_feat // 8, num_feat // 2, [1, 2])

        if not pytorch_init:
            default_init_weights(self.conv1, 0.1)

    def forward(self, x):
        identity = x
        # out = self.conv2(self.gelu(self.hoaf(self.conv1(self.norm1(x)))))
        mid = torch.chunk(self.conv1(self.norm1(x)), 2, dim=1)
        out = self.conv2(self.gelu(torch.cat([self.hoaf(mid[0]), mid[1]], dim=1)))
        return identity + out * self.res_scale


class SCA(nn.Module):
    def __init__(self, num_feat=64,):
        super(SCA, self).__init__()

        self.num_feat = num_feat

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

    def forward(self, x):
        return x * self.conv(self.gap(x))


class ResidualBlockNoBN_NAF(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_NAF, self).__init__()
        self.res_scale = res_scale
        self.norm1 = nn.GroupNorm(1, num_feat)
        self.norm2 = nn.GroupNorm(1, num_feat)

        self.conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True, groups=num_feat)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.sca = SCA(num_feat)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.convca = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)

        self.gelu = nn.GELU()

        self.beta1 = nn.Parameter(torch.ones((1, num_feat, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.ones((1, num_feat, 1, 1)), requires_grad=True)

        if not pytorch_init:
            default_init_weights(self.conv1, 0.1)

    def forward(self, x):
        identity = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gelu(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = identity + y * self.beta1

        identity = y
        y = self.norm2(y)
        y = self.conv4(y)
        y = self.gelu(y)
        y = self.conv5(y)
        return identity + y * self.beta2


class ResidualBlockNoBN_NAF_HOAF(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_NAF_HOAF, self).__init__()
        self.res_scale = res_scale
        self.norm1 = nn.GroupNorm(1, num_feat)
        self.norm2 = nn.GroupNorm(1, num_feat)

        self.conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True, groups=num_feat)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)

        self.gelu = nn.GELU()
        # self.hoaf1 = HOAF_v3(num_feat // 8, num_feat // 2, [1, 2])
        # self.hoaf1 = HOAF_v3(num_feat // 4, num_feat, [1, 2])
        self.hoaf2 = HOAF_v3(num_feat // 8, num_feat // 2, [1, 2])
        # self.hoaf2 = HOAF_v3(num_feat // 4, num_feat, [1, 2])
        self.sca = SCA(num_feat)

        self.beta1 = nn.Parameter(torch.ones((1, num_feat, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.ones((1, num_feat, 1, 1)), requires_grad=True)

        if not pytorch_init:
            default_init_weights(self.conv1, 0.1)

    def forward(self, x):
        identity = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        # mid = torch.chunk(y, 2, dim=1)
        # y = torch.cat([mid[0], self.hoaf(mid[1])], dim=1)
        # y = self.hoaf1(y)
        y = self.gelu(y)
        y = self.sca(y)
        # y = y * self.convca(self.gap(y))
        y = self.conv3(y)
        y = identity + y * self.beta1

        identity = y
        y = self.norm2(y)
        y = self.conv4(y)
        mid = torch.chunk(y, 2, dim=1)
        y = torch.cat([mid[0], self.hoaf2(mid[1])], dim=1)
        # y = self.hoaf2(y)
        y = self.gelu(y)
        y = self.conv5(y)
        return identity + y * self.beta2


@ARCH_REGISTRY.register()
class EDSR_HOAF(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=1.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR_HOAF, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN_HOAF, num_block, num_feat=num_feat, res_scale=res_scale,
                               pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x


@ARCH_REGISTRY.register()
class EDSR_NAF(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=1.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR_NAF, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN_NAF, num_block, num_feat=num_feat, res_scale=res_scale,
                               pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x


@ARCH_REGISTRY.register()
class EDSR_NAF_HOAF(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=1.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR_NAF_HOAF, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN_NAF_HOAF, num_block, num_feat=num_feat, res_scale=res_scale,
                               pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x


