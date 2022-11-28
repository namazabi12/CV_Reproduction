import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand, FFN_Expend):
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
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(NAFNET, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.ln = nn.LayerNorm()

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
