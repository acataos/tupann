from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helpers


def exists(val):
    """Check if val is not none

    Args:
        val (any): val

    Returns:
        bool: if val is not none
    """
    return val is not None


def default(val, d):
    """Return val if exists, else return the default d value.

    Args:
        val (any ): value to check if it exists
    d (any): default value

    Returns:
        any: val or d
    """
    return val if exists(val) else d


# down and upsample

# they use maxpool for downsample, and convtranspose2d for upsample
# todo: figure out the 4x upsample from 4km to 1km

Downsample2x = partial(nn.MaxPool2d, kernel_size=2, stride=2)


def Upsample2x(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.ConvTranspose2d(dim, dim_out, kernel_size=2, stride=2)


# conditionable resnet block


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = ChanLayerNorm(dim_out)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, *, cond_dim=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.mlp = None

        if exists(cond_dim):
            self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(cond_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond=None):
        scale_shift = None

        assert not (exists(self.mlp) ^ exists(cond))
        if exists(self.mlp) and exists(cond):
            cond = self.mlp(cond)
            cond = rearrange(cond, "b c -> b c 1 1")
            scale_shift = cond.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlocks(nn.Module):
    def __init__(self, dim, *, dim_in=None, depth=1, cond_dim=None):
        super().__init__()
        curr_dim = default(dim_in, dim)
        blocks = []
        for _ in range(depth):
            blocks.append(ResnetBlock(dim=curr_dim, dim_out=dim, cond_dim=cond_dim))
            curr_dim = dim

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond=None):
        for block in self.blocks:
            x = block(x, cond=cond)

        return x


# they use layernorms after the conv in the resnet blocks for some reason


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=self.eps).rsqrt() * self.g + self.b


# center crop


class CenterPad(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        target_dim = self.target_dim
        *_, height, width = x.shape
        assert target_dim >= height and target_dim >= width

        height_pad = target_dim - height
        width_pad = target_dim - width
        left_height_pad = height_pad // 2
        left_width_pad = width_pad // 2

        return F.pad(
            x, (left_height_pad, height_pad - left_height_pad, left_width_pad, width_pad - left_width_pad), value=0.0
        )


class CenterCrop(nn.Module):
    def __init__(self, crop_dim):
        super().__init__()
        self.crop_dim = crop_dim

    def forward(self, x):
        crop_dim = self.crop_dim
        *_, height, width = x.shape
        assert (height >= crop_dim) and (width >= crop_dim)

        cropped_height_start_idx = (height - crop_dim) // 2
        cropped_width_start_idx = (width - crop_dim) // 2

        height_slice = slice(cropped_height_start_idx, cropped_height_start_idx + crop_dim)
        width_slice = slice(cropped_width_start_idx, cropped_width_start_idx + crop_dim)
        return x[..., height_slice, width_slice]
