import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from src.models.metnet.Max_Vit import MaxViT
from src.models.metnet.modules import Downsample2x, ResnetBlocks, Upsample2x


class Metnet(nn.Module):
    def __init__(
        self,
        channels_in: int = 32,
        channels_out: int = 32,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        depth: int = 4,
        dim: int = 128,
        resnet_block_depth: int = 2,
        data_modification: list = None,
        lead_time_dim_embedding: int = 32,
        cond: bool = False,
        target_length: int = 6,
        downsampling: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.depth = depth
        self.dim = dim
        self.resnet_block_depth = resnet_block_depth
        self.data_modification = data_modification
        self.lead_time_dim_embedding = lead_time_dim_embedding
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.cond = cond
        self.cond_dim = self.lead_time_dim_embedding if self.cond else None
        self.downsampling = downsampling
        self.resnet_blocks_down_1 = ResnetBlocks(
            dim=self.dim, dim_in=self.channels_in, cond_dim=self.cond_dim, depth=self.resnet_block_depth
        )

        self.target_length = target_length

        if self.cond:
            self.lead_time_embedding = nn.Embedding(self.target_length, self.lead_time_dim_embedding)

        if self.downsampling == 1:
            self.down_and_pad_1 = Downsample2x()

        # self.resnet_2_dim_in = self.dim // 2

        # self.resnet_blocks_down_2 = ResnetBlocks(
        #     dim=dim, dim_in=self.resnet_2_dim_in, cond_dim=self.cond_dim, depth=resnet_block_depth
        # )
        # # If we want to crop the output of the model
        # # self.to_skip_connect_2 = CenterCrop(crop_size_post_16km * 2)

        self.dim_head = self.dim // 4

        # self.down_2 = Downsample2x()

        self.maxvitparams = {
            "dim": self.dim,
            "depth": self.depth,
            "dim_head": self.dim_head,
            "heads": 32,
            "dropout": 0.1,
            "cond_dim": self.cond_dim,
            "window_size": 8,
            "mbconv_expansion_rate": 4,
            "mbconv_shrinkage_rate": 0.25,
        }

        self.maxvit = MaxViT(**self.maxvitparams)

        # # If we want to predict in a smaller size we need to crop the output of the maxvit
        # # self.crop_post_16km = CenterCrop(crop_size_post_16km)

        if self.downsampling == 1:
            self.upsample_16km_to_8km = Upsample2x(self.dim)

        # self.resnet_blocks_up_1 = ResnetBlocks(
        #     dim=self.dim // 2, dim_in=dim + self.dim // 2, cond_dim=self.cond_dim, depth=resnet_block_depth
        # )
        # self.upsample_8km_to_4km = Upsample2x(self.dim // 2)

        # self.crop_to_half = CenterCrop(240)

        self.resnet_blocks_up_4km = ResnetBlocks(
            dim=self.channels_out, dim_in=2 * dim, cond_dim=self.cond_dim, depth=resnet_block_depth
        )

    def forward(self, x, cond=None):
        if self.cond:
            assert cond is not None
            cond = self.lead_time_embedding(cond).squeeze(1)
        else:
            cond = None
        x_in = x

        x1 = self.resnet_blocks_down_1(x_in, cond)

        if self.downsampling == 1:
            x2 = self.down_and_pad_1(x1)
        else:
            x2 = x1

        x3 = self.maxvit(x2, cond=cond)

        if self.downsampling == 1:
            x3 = self.upsample_16km_to_8km(x3)

        x3 = torch.cat([x3, x1], dim=1)
        logits = self.resnet_blocks_up_4km(x3, cond)
        return logits

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            if self.weight_init == "trunc_normal":
                trunc_normal_(m.weight, std=self.std_init)
            if self.weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            if self.weight_init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            if self.weight_init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if self.weight_init == "kaming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
