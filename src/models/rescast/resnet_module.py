"""
ResNet-based nowcasting model for sequence prediction.
"""
import torch
import torch.nn as nn

from src.models.unet.unet_parts import DoubleConv, Down, OutConv, Up


class ResNetNowcasting(nn.Module):
    """
    ResNet-based model for nowcasting that predicts all future timesteps at once.

    Takes autoencoder latent representations and predicts future latents using
    convolutional layers with residual connections.

    Args:
        input_channels: Number of input channels (autoencoder latent dimension)
        output_timesteps: Number of future timesteps to predict
        latent_height: Height of latent feature maps
        latent_width: Width of latent feature maps
        base_channels: Base number of channels for the network
        num_blocks: Number of residual blocks per stage
    """

    def __init__(
        self,
        input_channels: int,
        output_timesteps: int,
        latent_height: int,
        latent_width: int,
        base_channels: int = 64,
        num_blocks: int = 2,
        bilinear: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, input_channels * output_timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet.

        Args:
            x: Input latent tensor of shape (batch, input_channels, height, width)

        Returns:
            Predicted future latents of shape (batch, output_timesteps, input_channels, height, width)
        """
        batch_size = x.shape[0]

        rep_x = x[:, None, :, :, :].repeat(1, self.output_timesteps, 1, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        x = x.view(batch_size, self.output_timesteps, self.input_channels, self.latent_height, self.latent_width)
        x = x + rep_x

        return x
