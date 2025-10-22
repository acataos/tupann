""" Parts of the GAN model """

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# class DoubleConv2(nn.Module):
#     def __init__(self, in_channel, out_channel,bias = False):
#         super(DoubleConv2,self).__init__()
#         slope = 0.2
#         self.double_conv = nn.Sequential(
#             nn.BatchNorm2d(in_channel),
#             nn.LeakyReLU(slope, inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
#             spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=bias)),
#             nn.BatchNorm2d(in_channel),
#             nn.LeakyReLU(slope, inplace=True),
#             spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias))
#         )
#         self.one_conv = nn.Sequential(
#             nn.BatchNorm2d(in_channel),
#             nn.LeakyReLU(slope, inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
#             spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias))
#         )

#     def forward(self, input):
#         return self.double_conv(input)+ self.one_conv(input)


class D3_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias):
        super(D3_Block, self).__init__()
        slope = 0.2
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=bias)),
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.one_conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.BatchNorm3d(out_channel),
        )

    def forward(self, input):
        return self.double_conv(input) + self.one_conv(input)


class D2_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super(D2_Block, self).__init__()
        slope = 0.2
        self.double_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.one_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, input):
        return self.double_conv(input) + self.one_conv(input)


# class Down(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             D2_Block(in_channels, out_channels, kernel_size, stride, padding)
#         )

#     def forward(self, x):
#         x = self.maxpool_conv(x)
#         return x


class L3_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias):
        super(L3_Block, self).__init__()
        slope = 0.2
        self.double_conv = nn.Sequential(
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
        )
        self.one_conv = nn.Sequential(
            nn.BatchNorm3d(in_channel),
            spectral_norm(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
        )

    def forward(self, input):
        return self.double_conv(input) + self.one_conv(input)


class L2_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias):
        super(L2_Block, self).__init__()
        slope = 0.2
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.one_conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            ),
        )

    def forward(self, input):
        return self.double_conv(input) + self.one_conv(input)


# input_length = 16
class Noise_Projector(nn.Module):
    def __init__(self, input_length, output_length=None):
        super(Noise_Projector, self).__init__()
        self.input_length = input_length
        if output_length is None:
            output_length = self.input_length * 32
        self.output_length = output_length
        self.conv_first = spectral_norm(nn.Conv2d(self.input_length, self.input_length * 2, kernel_size=3, padding=1))
        self.L1 = ProjBlock(self.input_length * 2, self.input_length * 4)
        self.L2 = ProjBlock(self.input_length * 4, self.input_length * 8)
        self.L3 = ProjBlock(self.input_length * 8, self.input_length * 16)
        self.L4 = ProjBlock(self.input_length * 16, self.output_length)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        return x


class ProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ProjBlock, self).__init__()
        self.one_conv = spectral_norm(nn.Conv2d(in_channel, out_channel - in_channel, kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        x1 = torch.cat([x, self.one_conv(x)], dim=1)
        x2 = self.double_conv(x)
        output = x1 + x2
        return output


class AvgPool(nn.Module):
    def __init__(self, in_channel, out_channel, output_size, *args, **kwargs):
        super(AvgPool, self).__init__(*args, **kwargs)

        self.output_size = output_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=output_size),
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.avgpool(x)
