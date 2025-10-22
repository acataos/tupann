import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.models.nowcastnet.gan.gan_parts import L2_Block, L3_Block, Noise_Projector
from src.models.nowcastnet.gan.unet_parts import DoubleConv, Down, OutConv, Up
from src.models.nowcastnet.layers.generation.generative_network import Generative_Decoder, Generative_Encoder

ni = 192  # size of image
ndf = 32  # size of discriminator feature map


class Discriminator2D(nn.Module):
    def __init__(self, channel_in):
        super(Discriminator2D, self).__init__()
        self.channel_in = channel_in
        kernel_size = 4
        stride = 2
        padding = 1
        self.main2d = nn.Sequential(
            # input is ``1 x 16 (n_after+n_before) x 192 x 192`
            L2_Block(self.channel_in, ndf, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf)x 8 x 96 x 96``
            L2_Block(ndf, ndf * 2, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*2) x 4 x 48 x 48``
            L2_Block(ndf * 2, ndf * 4, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*4) x 2 x 24 x 24``
            L2_Block(ndf * 4, ndf * 8, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*8)x 1 x 12 x 12``
            # After this the input should be transform to 2d
            L2_Block(ndf * 8, ndf * 16, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*16) x 6 x 6``
            L2_Block(ndf * 16, ndf * 32, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*32) x 3 x 3``
            L2_Block(ndf * 32, 1, kernel_size + 1, stride + 1, padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main2d(input)


class Discriminator3D(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator3D, self).__init__()
        kernel_size = 4
        stride = 2
        padding = 1
        self.in_channel = in_channel
        self.main3d = nn.Sequential(
            # input is ``1 x 16 (n_after+n_before) x 192 x 192`
            L3_Block(1, ndf, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf)x 8 x 96 x 96``
            L3_Block(ndf, ndf * 2, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*2) x 4 x 48 x 48``
            L3_Block(ndf * 2, ndf * 4, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*4) x 2 x 24 x 24``
        )
        if in_channel >= 16 and in_channel < 32:
            self.tor2dim = L3_Block(ndf * 4, ndf * 8, kernel_size, stride, padding, bias=False)
        elif in_channel >= 8 and in_channel < 16:
            self.tor2dim = L2_Block(ndf * 4, ndf * 8, kernel_size, stride, padding, bias=False)
        else:
            ValueError("Incorrect number of contex and predictions.")
            # After this the input should be transform to 2d
        self.main2d = nn.Sequential(
            # state size. ``(ndf*8)x 1 x 12 x 12`` or  ``(ndf*8)x 1 x 14 x 14``
            L2_Block(ndf * 8, ndf * 16, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*16) x 6 x 6``
            L2_Block(ndf * 16, ndf * 32, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*32) x 3 x 3``
            L2_Block(ndf * 32, 1, kernel_size + 1, stride + 1, padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.main3d(input)
        if self.in_channel >= 16 and self.in_channel < 32:
            x = self.tor2dim(x)
            x = x[:, :, 0, :, :]
        elif self.in_channel >= 8 and self.in_channel < 16:
            x = self.tor2dim(x[:, :, 0, :, :])
        else:
            ValueError("Incorrect number of contex and predictions.")
        return self.main2d(x)


class TemporalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super(TemporalDiscriminator, self).__init__()
        kernel_size = 4
        stride = 2
        padding = 1
        size = 16
        after_flatten = 64 + (in_channel - 3) * 4 + (in_channel // 4 + 1) * 8
        self.conv2d = spectral_norm(nn.Conv2d(in_channel, 64, kernel_size=9, stride=2, padding=4))
        self.conv3d_1 = spectral_norm(nn.Conv3d(1, 4, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4)))
        self.conv3d_2 = spectral_norm(nn.Conv3d(1, 8, kernel_size=(4, 9, 9), stride=(4, 2, 2), padding=(2, 4, 4)))
        self.main2d = nn.Sequential(
            L2_Block(after_flatten, ndf * 4, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*4) x96 x 96``
            L2_Block(ndf * 4, ndf * 8, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*8) x 48 x 48``
            L2_Block(ndf * 8, ndf * 16, kernel_size, stride, padding, bias=False),
            # state size. ``(ndf*8) x 24 x 24``
            L2_Block(ndf * 16, ndf * 16, kernel_size - 1, stride - 1, padding, bias=False),
            # state size. ``(ndf*8) x 12 x 12``
            nn.BatchNorm2d(ndf * 16),
            spectral_norm(nn.Conv2d(ndf * 16, 1, kernel_size=3, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            # returns 1x12x12 or 1x14x14 (in the new dataset is 1x16x16)
            nn.Flatten(),
            spectral_norm(torch.nn.Linear(size * size, 1)),
            nn.Sigmoid(),
        )

    def forward(self, input):
        input3d = input[:, None, :, :, :]
        x2 = torch.flatten(self.conv3d_1(input3d), start_dim=1, end_dim=2)
        x3 = torch.flatten(self.conv3d_2(input3d), start_dim=1, end_dim=2)
        x = torch.cat([self.conv2d(input), x2, x3], dim=1)

        return self.main2d(x)


class Join_Discriminator(nn.Module):
    def __init__(self, channel_in):
        super(Join_Discriminator, self).__init__()
        self.channel_in = channel_in
        self.dis2d = Discriminator2D(self.channel_in)
        self.dis3d = Discriminator3D(self.channel_in)
        self.alpha = 0.2
        self.beta = 1 - self.alpha

    def forward(self, input):
        return self.alpha * self.dis2d(input) + self.beta * self.dis3d(input[:, None, :, :, :])


class Generator(nn.Module):
    def __init__(self, channel_in, latent_dim, n_after, normilize):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.normalize = normilize
        self.deeptospace = nn.PixelShuffle(2)
        self.noise = Noise_Projector(self.latent_dim, 512 * 4)
        self.bilinear = True

        # Generator Unet
        self.inc = DoubleConv(channel_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, n_after, norm=normilize)

        # self.up_noise1 = (DoubleConv2(512,256))
        # self.up_noise2 = (DoubleConv2(256,128))
        # self.up_noise3 = (DoubleConv2(128,64))
        # self.up_noise4 = (DoubleConv2(64,64))
        self.out_noise = OutConv(64, n_after, norm=normilize)

        self.up_noise1 = Up(1024, 512 // factor, self.bilinear)
        self.up_noise2 = Up(512, 256 // factor, self.bilinear)
        self.up_noise3 = Up(256, 128 // factor, self.bilinear)
        self.up_noise4 = Up(128, 64, self.bilinear)

        # self.relu = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm2d(n_after)
        # self.bn2 = nn.BatchNorm2d(n_after)

        # self.outres = (OutConv(n_after,n_after,norm=normilize))

    def forward(self, x, z):
        with torch.no_grad():
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x_encoded = self.down4(x4)

            x5 = self.up1(x_encoded, x4)
            x6 = self.up2(x5, x3)
            x7 = self.up3(x6, x2)
            x8 = self.up4(x7, x1)

            logits = self.outc(x8)

        # x_encoded = self.encoder(input)

        z_encoded = self.deeptospace(self.noise(z))

        # x_concat = torch.concat([x_encoded, self.deeptospace(z_encoded)], dim=1)
        # Both need to have the same number of channel
        # z_concat = x_encoded + z_encoded

        z1 = self.up_noise1(z_encoded, x4)
        # z2 = x5 + z1

        z2 = self.up_noise2(z1, x3)
        # z4 = x6 + z3

        z3 = self.up_noise3(z2, x2)
        # z5 = x7 + z4

        z4 = self.up_noise4(z3, x1)
        # z7 = x8 + z6

        z_out = self.out_noise(z4)

        # xout = torch.concat([self.relu(self.bn1(logits)), self.relu(self.bn2(z_out))],dim=1)
        xout = (logits + z_out) / 2

        # x_decoded = self.decoder(x_concat)
        return xout


class NowcasnetGenerator(nn.Module):
    def __init__(self, channel_in, latent_dim, n_after):
        super(NowcasnetGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channel_in = channel_in
        self.n_after = n_after
        self.ngf = 32

        # if self.sat:
        #     self.upnoise = nn.Upsample(size=30, mode="bilinear", align_corners=True)

        self.gen_encoder = Generative_Encoder(n_channels=channel_in, base_c=self.ngf)
        self.gen_decoder = Generative_Decoder(nf=self.ngf, ic=self.ngf * 10, gen_oc=n_after, evo_ic=n_after)
        self.noise_projector = Noise_Projector(input_length=self.ngf)
        self.deeptospace = nn.PixelShuffle(4)

    def forward(self, x, z):
        pred = x[:, -self.n_after :, :, :]
        batch, time, height, width = x.size()
        x_encoded = self.gen_encoder(x)
        pred = pred / 40
        noise_feature = self.noise_projector(z)
        noise_feature = self.deeptospace(noise_feature)
        # if self.sat:
        #     noise_feature = self.upnoise(noise_feature)
        feature = torch.cat([x_encoded, noise_feature], dim=1)

        return self.gen_decoder(feature, pred)
