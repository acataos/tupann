import torch
import torch.nn as nn
from torch.nn import Sequential

from src.models.lightning_module import LModule
from src.models.metnet.Max_Vit import MaxViT
from src.models.metnet.modules import CenterCrop, CenterPad, Downsample2x, ResnetBlocks, Upsample2x


class model(LModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        context: torch.Tensor = None,
        truth: torch.Tensor = None,
        elev: torch.Tensor = None,
        context_val: torch.Tensor = None,
        truth_val: torch.Tensor = None,
        elev_val: torch.Tensor = None,
        loss: int = 1,
        n_before: int = 10,
        n_after: int = 20,
        normalized: int = 0,
        xmax: float = 0,
        x_mean: float = 0,
        x_std: float = 1,
        depth: int = 4,
        dim: int = 128,
        resnet_block_depth: int = 2,
        data_modification: list = None,
        lead_time_dim_embedding: int = None,
        merge: bool = False,
        correct_context: bool = False,
        satellite: bool = False,
        **kwargs,
    ):
        super().__init__(
            truth=truth,
            context=context,
            truth_val=truth_val,
            context_val=context_val,
            n_before=n_before,
            n_after=n_after,
            normalized=normalized,
            loss=loss,
            xmax=xmax,
            x_mean=x_mean,
            x_std=x_std,
            data_modification=data_modification,
            merge=merge,
            satellite=satellite,
        )

        self.save_hyperparameters()

        self.lr = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay

        self.weighted = self.hparams.weights
        self.depth = self.hparams.depth
        self.cond = self.dm_option["Lead_time_cond"] and not self.dm_option["Add_lead_to_input"]

        # The dimention throught the model.
        self.dim = self.hparams.dim
        self.resnet_block_depth = self.hparams.resnet_block_depth
        self.lead_time_dim_embedding = self.hparams.lead_time_dim_embedding

        self.correct_context = self.hparams.correct_context

        # Since we are using only n_before channels we need to subtract the rest
        if not self.old or self.merge:
            if not self.dm_option["No_satellite"]:
                self.channels_in = self.channels_in - self.n_before
        else:
            self.channels_in = self.channels_in

        if self.dm_option["Lead_time_cond"] and self.dm_option["Add_lead_to_input"]:
            self.channels_in += 1

        # If we want to crop the output of the model
        # self.to_skip_connect_1 = CenterCrop(crop_size_post_16km * 4)

        self.cond_dim = self.lead_time_dim_embedding if self.cond else None

        if self.cond:
            self.lead_time_embedding = nn.Embedding(self.n_after, self.lead_time_dim_embedding)

        self.resnet_blocks_down_1 = ResnetBlocks(
            dim=self.dim // 2, dim_in=self.channels_in, cond_dim=self.cond_dim, depth=self.resnet_block_depth
        )

        self.down_and_pad_1 = Sequential(Downsample2x(), CenterPad(256))

        self.resnet_2_dim_in = self.dim // 2 if self.old else self.dim // 2 + self.n_before
        if self.dm_option["No_satellite"]:
            self.resnet_2_dim_in -= self.n_before

        self.resnet_blocks_down_2 = ResnetBlocks(
            dim=dim, dim_in=self.resnet_2_dim_in, cond_dim=self.cond_dim, depth=resnet_block_depth
        )
        # If we want to crop the output of the model
        # self.to_skip_connect_2 = CenterCrop(crop_size_post_16km * 2)

        self.dim_head = self.dim // 4

        self.down_2 = Downsample2x()

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

        # If we want to predict in a smaller size we need to crop the output of the maxvit
        # self.crop_post_16km = CenterCrop(crop_size_post_16km)

        self.upsample_16km_to_8km = Upsample2x(self.dim)

        self.resnet_blocks_up_1 = ResnetBlocks(
            dim=self.dim // 2, dim_in=dim + self.dim // 2, cond_dim=self.cond_dim, depth=resnet_block_depth
        )
        self.upsample_8km_to_4km = Upsample2x(self.dim // 2)

        if not self.old or not self.sat:
            self.crop_to_half = CenterCrop(256)
        else:
            self.crop_to_half = CenterCrop(240)

        self.resnet_blocks_up_4km = ResnetBlocks(
            dim=self.channels_out, dim_in=dim // 2 + self.channels_in, cond_dim=self.cond_dim, depth=resnet_block_depth
        )

    def forward(self, x, lead_times=None):
        try:
            self.correct_context
        except AttributeError:
            self.correct_context = False

        if self.correct_context:
            if self.merge:
                if self.dm_option["No_satellite"]:
                    x_in = x
                else:
                    x_in = x[:, 0 : 4 * self.n_before : 2]
            else:
                x_in = x[:, 0 : 2 * self.n_before : 2]

            if not self.old or self.merge:
                if not self.dm_option["No_satellite"]:
                    x_in_2 = x[:, 1 : 2 * self.n_before + 1 : 2]
        else:
            if self.merge:
                if self.dm_option["No_satellite"]:
                    x_in = x
                else:
                    x_in = x[:, : 2 * self.n_before]
            else:
                x_in = x[:, : self.n_before]

            if not self.old or self.merge:
                if not self.dm_option["No_satellite"]:
                    x_in_2 = x[:, -self.n_before :]

        if self.cond:
            assert lead_times is not None
            cond = self.lead_time_embedding(lead_times)
        else:
            cond = None

        x1 = self.resnet_blocks_down_1(x_in, cond)

        x2 = self.down_and_pad_1(x1)

        if not self.old or self.merge:
            if not self.dm_option["No_satellite"]:
                x2_m = torch.cat((x2, x_in_2), dim=1)
            else:
                x2_m = x2
        else:
            x2_m = x2

        x3 = self.resnet_blocks_down_2(x2_m, cond)

        x3 = self.down_2(x3)

        x3 = self.maxvit(x3, cond=cond)

        x = self.upsample_16km_to_8km(x3)

        x_m = torch.cat((x, x2), dim=1)

        x = self.resnet_blocks_up_1(x_m, cond)

        x = self.upsample_8km_to_4km(x)

        x = self.crop_to_half(x)

        x_m = torch.cat((x, x_in), dim=1)

        logits = self.resnet_blocks_up_4km(x_m, cond)

        return logits

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [opt], []
