import torch

from src.models.earthformer.earthformer_module import CuboidTransformerModel
from src.models.lightning import LModule
from src.utils.lightning_utils import calc_concat_shape_dict


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        batch_train: tuple = (),
        batch_val: tuple = (),
        loss: str = "l1",
        weights: str = "uniform",
        xmax: float = 3 * 20 / 5,
        base_units: int = 128,
        num_heads: int = 4,
        num_global_vectors: int = 8,
        target_is_imerg: bool = False,
        crop_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_shape_dict,
            target_shape_dict,
            target_shape_dict_val,
            batch_train,
            batch_val,
            loss=loss,
            xmax=xmax,
            weights=weights,
            target_is_imerg=target_is_imerg,
            crop_predict=crop_predict,
            **kwargs,
        )
        self.save_hyperparameters()

        # Earthformer concatenates input and target
        input_shape = calc_concat_shape_dict(input_shape_dict)
        target_shape = calc_concat_shape_dict(target_shape_dict)
        input_shape = input_shape + (1,)
        target_shape = target_shape + (1,)

        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.weighted = weights
        self.crop_predict = crop_predict

        # just like cascast earthformer default params
        num_blocks = 2
        enc_attn_patterns = ["axial"] * num_blocks
        dec_self_attn_patterns = ["axial"] * num_blocks
        dec_cross_attn_patterns = ["cross_1x1"] * num_blocks

        self.earthformer_module = CuboidTransformerModel(
            input_shape=input_shape,
            target_shape=target_shape,
            base_units=base_units,
            block_units=None,
            scale_alpha=1.0,
            enc_depth=[1, 1],
            dec_depth=[1, 1],
            enc_use_inter_ffn=True,
            dec_use_inter_ffn=True,
            dec_hierarchical_pos_embed=False,
            downsample=2,
            downsample_type="patch_merge",
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=None,
            dec_use_first_self_attn=False,
            num_heads=num_heads,
            attn_drop=0.1,
            proj_drop=0.1,
            ffn_drop=0.1,
            upsample_type="upsample",
            ffn_activation="gelu",
            gated_ffn=False,
            norm_layer="layer_norm",
            # global vectors
            num_global_vectors=num_global_vectors,
            use_dec_self_global=False,
            dec_self_update_global=True,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,
            use_global_self_attn=True,
            separate_global_qkv=True,
            global_dim_ratio=1,
            # initial_downsample
            initial_downsample_type="stack_conv",
            initial_downsample_activation="leaky",
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=3,
            initial_downsample_stack_conv_dim_list=[
                base_units // 4,
                base_units // 2,
                base_units,
            ],
            initial_downsample_stack_conv_downscale_list=[3, 2, 2],
            initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
            # misc
            padding_type="zeros",
            z_init_method="zeros",
            checkpoint_level=0,
            pos_embed_type="t+h+w",
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            # initialization
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            conv_init_mode="0",
            down_up_linear_init_mode="0",
            norm_init_mode="0",
        )

    def forward(self, x, cond=None):
        x = x.unsqueeze(-1)
        if cond is not None:
            cond = cond.reshape(x.shape[0], 1, 1, 1, 1).repeat(1, 1, x.shape[-2], x.shape[-2], 1)
            x = torch.cat([x, cond], dim=1)

        y = self.earthformer_module.forward(x)
        return y.squeeze(-1)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [opt], []
