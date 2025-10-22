from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from beartype.typing import Optional
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, einsum, nn
from torch.nn import Module, ModuleList, Sequential

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


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def safe_div(num, den, eps=1e-10):
    return num / den.clamp(min=eps)


# tensor helpers


def l2norm(t):
    return F.normalize(t, dim=-1)


# prepare batch norm in maxvit for distributed training


def MaybeSyncBatchnorm2d(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d


@contextmanager
def freeze_batchnorm(bn):
    assert not exists(next(bn.parameters(), None))

    was_training = bn.training
    was_tracking_stats = (
        bn.track_running_stats
    )  # in some versions of pytorch, running mean and variance still gets updated even in eval mode it seems..

    bn.eval()
    bn.track_running_stats = False

    yield bn

    bn.train(was_training)
    bn.track_running_stats = was_tracking_stats


# multi-headed rms normalization, for query / key normalized attention


class RMSNorm(Module):
    def __init__(self, dim, *, heads):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


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


# MBConv


class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    batchnorm_klass = MaybeSyncBatchnorm2d()

    net = Sequential(
        # Conv 1x1
        nn.Conv2d(dim_in, hidden_dim, 1),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        # Deepwise Conv 3x3
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        # SE
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        # Conv 1x1
        nn.Conv2d(hidden_dim, dim_out, 1),
        batchnorm_klass(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class Attention(Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        heads=32,
        dim_head=32,
        dropout=0.0,
        window_size=8,
        num_registers=1,
        block=False,  # 4 a int the
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, "dimension should be divisible by dimension per head"

        dim_inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.block = block

        self.has_cond = exists(cond_dim)

        self.film = None

        if self.has_cond:
            self.film = Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange("b (r d) -> r b d", r=2),
            )

        self.norm = nn.LayerNorm(dim, elementwise_affine=not self.has_cond)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.q_norm = RMSNorm(dim_head, heads=heads)
        self.k_norm = RMSNorm(dim_head, heads=heads)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value=num_rel_pos_bias)
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x: Tensor, r: Tensor, cond: Optional[Tensor] = None, x_size=None, window_size=8):
        _, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        # conditioning

        if exists(self.film):
            assert exists(cond)

            lead_cond = self.film(cond)
            lead_cond = repeat(lead_cond, "r b d -> r b d x y", x=x_size, y=x_size)

            if self.block:
                lead_cond = rearrange(
                    lead_cond, "r b d (x w1) (y w2) -> r b x y w1 w2 d", w1=window_size, w2=window_size
                )
                lead_cond, _ = pack_one(lead_cond, "r b x y * d")
                lead_cond, _ = pack_one(lead_cond, "r * n d")
            else:
                lead_cond = rearrange(
                    lead_cond, "r b d (w1 x) (w2 y) -> r b x y w1 w2 d", w1=window_size, w2=window_size
                )
                lead_cond, _ = pack_one(lead_cond, "r b x y * d")
                lead_cond, _ = pack_one(lead_cond, "r * n d")

            gamma, beta = lead_cond

            x = x * gamma + beta

        # Pack the tensor

        x_i, register_ps = pack([r, x], "b * d")

        x = self.norm(x_i)

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # scale

        q, k = self.q_norm(q), self.k_norm(k)

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # combine heads out

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), register_ps, x_i


class MaxViT(Module):
    def __init__(
        self,
        *,
        dim,
        depth,  # Tuple that indicates the number of steps and how m
        cond_dim=32,  # for conditioniong on lead time embedding
        heads=32,
        dim_head=32,
        window_size=8,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        num_register_tokens=4
    ):
        super().__init__()
        depth = (depth,) if isinstance(depth, int) else depth
        assert num_register_tokens > 0

        self.cond_dim = cond_dim

        # variables

        num_stages = len(depth)

        # maps duplicating the number of dimentions each block.
        # dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages+1)))
        # but in our case we will keep it fixed.
        dims = tuple(map(lambda i: (1**i) * dim, range(num_stages + 1)))

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                # is_first = stage_ind == 0
                is_first = False
                # First layer downsample, we do not downsample and not change the value of the channels
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample=is_first,
                    expansion_rate=mbconv_expansion_rate,
                    shrinkage_rate=mbconv_shrinkage_rate,
                )

                block_attn = Attention(
                    dim=layer_dim,
                    cond_dim=cond_dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    window_size=window_size,
                    num_registers=num_register_tokens,
                    block=True,
                )

                grid_attn = Attention(
                    dim=layer_dim,
                    cond_dim=cond_dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    window_size=window_size,
                    num_registers=num_register_tokens,
                    block=False,
                )

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([conv, block_attn, grid_attn]))

                self.register_tokens.append(register_tokens)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        if exists(self.cond_dim):
            assert cond.shape == (x.shape[0], self.cond_dim)

        b, w = x.shape[0], self.window_size

        for (conv, block_attn, grid_attn), register_tokens in zip(self.layers, self.register_tokens):
            xskip = x
            x = conv(x)

            # block-like attention

            x = rearrange(x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w)

            # prepare register tokens

            r = repeat(register_tokens, "n d -> b x y n d", b=b, x=x.shape[1], y=x.shape[2])
            r, register_batch_ps = pack_one(r, "* n d")

            x, window_ps = pack_one(x, "b x y * d")
            x, batch_ps = pack_one(x, "* n d")
            # x, register_ps = pack([r, x], "b * d")

            x, register_ps, x_i = block_attn(x, r, cond=cond, x_size=xskip.shape[2], window_size=w)

            x += x_i

            r, x = unpack(x, register_ps, "b * d")

            x = unpack_one(x, batch_ps, "* n d")
            x = unpack_one(x, window_ps, "b x y * d")
            x = rearrange(x, "b x y w1 w2 d -> b d (x w1) (y w2)")

            r = unpack_one(r, register_batch_ps, "* n d")

            # grid-like attention

            x = rearrange(x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w)

            # prepare register tokens

            r = reduce(r, "b x y n d -> b n d", "mean")
            r = repeat(r, "b n d -> b x y n d", x=x.shape[1], y=x.shape[2])
            r, register_batch_ps = pack_one(r, "* n d")

            x, window_ps = pack_one(x, "b x y * d")
            x, batch_ps = pack_one(x, "* n d")
            # x, register_ps = pack([r, x], "b * d")

            x, register_ps, x_i = grid_attn(x, r, cond=cond, x_size=xskip.shape[2], window_size=w)

            x += x_i

            r, x = unpack(x, register_ps, "b * d")

            x = unpack_one(x, batch_ps, "* n d")
            x = unpack_one(x, window_ps, "b x y * d")
            x = rearrange(x, "b x y w1 w2 d -> b d (w1 x) (w2 y)")

            x = x + xskip

        return x
