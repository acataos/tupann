""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(mid_channels),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )
        self.one_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
        )

    def forward(self, x):
        return self.double_conv(x) + self.one_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)

        return self.conv(x)


class S(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.bn_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            )
            self.first_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)),
            )
            self.second_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x3 = x1 + x2
        x4 = self.first_conv(x3)

        return (self.second_conv(x4 + x2) + self.bn_conv(x3)) / 2


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=0):
        super(OutConv, self).__init__()
        if norm == 2:
            self.conv = nn.Sequential(
                nn.LeakyReLU(),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)),
            )
        elif norm == 0 or norm == 1:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)), nn.Tanh()
            )

    def forward(self, x):
        return self.conv(x)
