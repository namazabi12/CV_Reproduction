import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY
from math import sqrt

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
            self.norm2 = nn.GroupNorm(1, self.out_channels2 * self.num_groups)

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
            output = output + self.conv2(self.norm2(channel_shuffle(torch.cat(output_c2, dim=1), self.num_groups)))\
                     * self.beta2
        # print("inp: ", inp)
        # print("oup: ", output)
        return output


class HOAF_v4(nn.Module):
    """HOAF(High Order Activation Function) structure.

        Args:
            num_channels (int): number of channels expected in input
    """
    __constants__ = ['num_channels']
    num_channels: int

    def __init__(self, num_channels, scale=2):
        super(HOAF_v4, self).__init__()

        self.num_channels = num_channels
        self.downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.norm = nn.GroupNorm(1, num_channels)
        # self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
        # self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic')


    def forward(self, inp):
        x = self.norm(inp)
        return inp + x * (x - self.upsample(self.downsample(x)))


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


class SCA_v2(nn.Module):
    def __init__(self, num_feat=64, scale=8):
        super(SCA_v2, self).__init__()

        self.num_feat = num_feat
        self.scale = scale
        self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

    def forward(self, x):
        return x * self.conv2(self.gap(self.conv1(self.pool(x))))


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()

        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_k, dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)

        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        batch, c, n = x.shape
        assert n == self.dim_q

        q = self.linear_q(x)  # batch, c, dim_k
        k = self.linear_k(x)  # batch, c, dim_k
        v = self.linear_v(x)  # batch, c, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, c, c

        dist = torch.softmax(dist, dim=-1)  # batch, c, c

        att = torch.bmm(dist, v)

        return att


class SelfAttention2(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention2, self).__init__()

        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_k, dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        # self.linear_v = nn.Linear(dim_q, dim_v, bias=False)

        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        batch, c, n = x.shape
        assert n == self.dim_q

        q = self.linear_q(x)  # batch, c, dim_k
        k = self.linear_k(x)  # batch, c, dim_k
        # v = self.linear_v(x)  # batch, c, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, c, c

        dist = torch.softmax(dist, dim=-1)  # batch, c, c

        # att = torch.bmm(dist, v)

        return dist


class SCA_v3(nn.Module):
    def __init__(self, num_feat=64, gap_size=8):
        super(SCA_v3, self).__init__()

        self.num_feat = num_feat

        self.gap1 = nn.AdaptiveAvgPool2d(gap_size)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.GroupNorm(1, num_feat)
        self.gap_size = gap_size
        self.win_size = gap_size ** 2

        self.SA = SelfAttention(self.win_size, self.win_size, self.win_size)

    def forward(self, x):
        batch, c, h, w = x.shape
        # y = self.norm(x)
        y = self.gap1(x)
        y = y.reshape([batch, c, self.win_size])
        y = self.SA(y)
        y = y.reshape([batch, c, self.gap_size, self.gap_size])
        y = self.gap2(y)
        return x + y


class SCA_v4(nn.Module):
    def __init__(self, num_feat=64, gap_size=8):
        super(SCA_v4, self).__init__()

        self.num_feat = num_feat

        self.gap1 = nn.AdaptiveAvgPool2d(gap_size)
        self.norm = nn.GroupNorm(1, num_feat)
        self.gap_size = gap_size
        self.win_size = gap_size ** 2

        self.SA = SelfAttention2(self.win_size, self.win_size, self.win_size)

    def forward(self, x):
        batch, c, h, w = x.shape
        y = self.norm(x)
        y = self.gap1(y)
        y = y.reshape([batch, c, self.win_size])
        dist = self.SA(y)
        # y = y.reshape([batch, c, self.gap_size, self.gap_size])
        # y = self.gap2(y)
        x = x.reshape([batch, c, h * w])
        x = torch.bmm(dist, x).reshape([batch, c, h, w])
        return x


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
        # self.hoaf2 = HOAF_v3(num_feat // 8, num_feat // 2, [1, 2])
        # self.hoaf2 = HOAF_v3(num_feat // 4, num_feat, [1, 2])
        # self.hoaf1 = HOAF_v4(num_channels=num_feat, scale=2)
        # self.hoaf2 = HOAF_v4(num_channels=num_feat, scale=2)
        # self.sca = SCA(num_feat)
        # self.sca = SCA_v2(n  um_feat, scale=8)
        self.sca = SCA_v3(num_feat, gap_size=4)
        # self.sca = SCA_v4(num_feat, gap_size=4)

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
        # y = self.norm(y)
        y = self.gelu(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = identity + y * self.beta1

        identity = y
        y = self.norm2(y)
        y = self.conv4(y)
        # mid = torch.chunk(y, 2, dim=1)
        # y = self.gelu(y)
        # y = torch.cat([y, self.hoaf2(y)], dim=1)
        # y = self.hoaf2(y)
        y = self.gelu(y)
        y = self.conv5(y)
        return identity + y * self.beta2


# @ARCH_REGISTRY.register()
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


# @ARCH_REGISTRY.register()
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
        self.mod = 4
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN_NAF_HOAF, num_block, num_feat=num_feat, res_scale=res_scale,
                               pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        # h, w = x.shape[-2], x.shape[-1]
        # x = self.pad_to_pow(x)
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
        # return x[:, :, :h * self.upscale, :w * self.upscale]

    def pad_to_pow(self, inp):
        h, w = inp.shape[-2], inp.shape[-1]
        h_pad = (self.mod - h % self.mod) % self.mod
        w_pad = (self.mod - w % self.mod) % self.mod
        return F.pad(inp, (0, w_pad, 0, h_pad))


if __name__ == "__main__":
    a = torch.randn([4, 64, 32, 32])
    sca = SCA_v4(64, 8)
    print(a)
    print(sca(a))