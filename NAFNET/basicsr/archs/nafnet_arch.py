import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expend=2):
        super(NAFBlock, self).__init__()
        dw_channels = c * DW_Expand
        ffn_channels = c * FFN_Expend

        self.norm1 = nn.LayerNorm([1, c, 1, 1])
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channels, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channels, out_channels=dw_channels, kernel_size=3, stride=1, padding=1,
                               groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channels // 2, out_channels=dw_channels // 2, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True)
        )
        self.conv3 = nn.Conv2d(in_channels=dw_channels // 2, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)

        self.norm2 = nn.LayerNorm([1, c, 1, 1])
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channels, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channels // 2, out_channels=c, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)


    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = x1 * self.sca(x1)
        x1 = self.conv3(x1)
        x1 = x1 + x

        x2 = self.norm2(x1)
        x2 = self.conv4(x2)
        x2 = self.sg(x2)
        x2 = self.conv5(x2)
        return x2 + x1


@ARCH_REGISTRY.register()
class NAFNET(nn.Module):
    """NAFNET network structure.

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
                 img_channels=3,
                 width=32,
                 enc_block_nums=[2, 2, 4, 8],
                 mid_block_nums=12,
                 dec_block_nums=[2, 2, 2, 2]):
        super(NAFNET, self).__init__()

        self.begin = nn.Conv2d(img_channels, width, 3, 1, 1)

        self.enc_block = []
        self.down_block = []
        ch = width
        for num in enc_block_nums:
            self.enc_block.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.down_block.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.mid_block = nn.Sequential(*[NAFBlock(ch) for _ in range(mid_block_nums)])

        self.dec_block = []
        self.up_block = []
        for num in dec_block_nums:
            self.up_block.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch /= 2
            self.dec_block.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

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
