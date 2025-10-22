import pathlib
from collections.abc import Iterable
from functools import partial

import torch
import yaml
from einops import repeat
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.lightning import LModule
from src.models.metnet.metnet import Metnet
from src.models.nowcastrio.autoenc_lightning import model as AutoencoderKL
from src.models.nowcastrio.utils import make_grid, warp

# from src.models.nowcastrio.mamba_parts.vmamba import VSSM
from src.utils.lightning_utils import transform_multiple_loc

scheduler_dict = {
    "exp": partial(LambdaLR, lr_lambda=lambda epoch: 0.99**epoch),
    "cosine": CosineAnnealingLR,
}


def fetch_scheduler(parameter_dict):
    return lambda optimizer: scheduler_dict[parameter_dict["name"]](optimizer, **parameter_dict["params"])


class model(LModule):
    # flake8: noqa: C901
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        autoenc_hash=None,
        autoenc_ckpt=None,
        dataset="sevir",
        learning_rate: float = 0.0002,
        optim: str = "adam",
        lead_time_cond: int = 0,
        cos_weight=0.33,
        image_weight=0.5,
        vector_weight: float = 0.1,
        maxvit_dim: int = 16,
        maxvit_depth: int = 4,
        loss: str = "l1",
        weights: str = "uniform",
        xmax: float = 3 * 20 / 5,
        maxvit_downsample: int = 0,
        target_is_imerge: bool = False,
        transforms={},
        inv_transforms={},
        scheduler_params={"name": "exp", "params": {}},
        train_autoenc: bool = False,
        latent_distribution: bool = False,
        **kwargs,
    ):
        true_target_dict = {list(target_shape_dict.keys())[0]: target_shape_dict[list(target_shape_dict.keys())[0]]}
        super().__init__(
            input_shape_dict,
            true_target_dict,
            target_shape_dict_val,
            target_is_imerge=target_is_imerge,
            transforms=transforms,
            inv_transforms=inv_transforms,
            loss=loss,
            xmax=xmax,
            weights=weights,
            lead_time_cond=lead_time_cond,
            **kwargs,
        )
        self.save_hyperparameters()

        self.optim = optim
        self.cond = True
        self.train_autoenc = train_autoenc

        self.maxvit_dim = maxvit_dim
        self.maxvit_depth = maxvit_depth
        self.maxvit_downsample = maxvit_downsample
        self.scheduler_params = scheduler_params

        self.img_size = target_shape_dict[list(target_shape_dict.keys())[0]][-1]
        # self.target_length = target_shape[0]
        self.input_length = input_shape_dict[list(input_shape_dict.keys())[0]][0]

        self.automatic_optimization = False
        self.val_loss_steps = []
        self.name = "nowcastrio"

        input_shape_dict_autoenc = input_shape_dict

        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        # Used just in the no lead_time case
        # self.register_buffer("alphas", torch.linspace(1, 0, self.target_length))

        # load autoencoder
        if autoenc_ckpt is not None:
            autoencoder_path = "models/nowcastrio_autoenc/" + autoenc_hash + "/train/" + dataset + "/" + autoenc_ckpt
        else:
            autoencoder_path = "models/nowcastrio_autoenc/" + autoenc_hash + "/train/" + dataset + "/model_train.pt"
        with open(pathlib.Path(autoencoder_path).parent / "h_params.yaml") as f:
            params = yaml.safe_load(f)
        # To match the train target length used in the autoencoder
        dummy_val = torch.zeros(1, self.img_size, self.img_size)
        # if we use the latent
        val_autoencoder_dict = {list(target_shape_dict.keys())[-1]: dummy_val.shape}

        if not self.train_autoenc:
            autoencoder = AutoencoderKL(
                **params,
                input_shape_dict=input_shape_dict_autoenc,
                target_shape_dict=true_target_dict,
                target_shape_dict_val=val_autoencoder_dict,
            ).requires_grad_(False)
        else:
            autoencoder = AutoencoderKL(
                **params,
                input_shape_dict=input_shape_dict_autoenc,
                target_shape_dict=true_target_dict,
                target_shape_dict_val=val_autoencoder_dict,
            )
        try:
            autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=self.device, weights_only=False), strict=True
            )
        except Exception as e:
            print(e, "... loaded state_dict")
            autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=self.device, weights_only=False)["state_dict"], strict=True
            )

        self.add_module("autoencoder", autoencoder)
        # make grid
        sample_tensor = torch.zeros(1, 1, self.img_size, self.img_size)
        self.register_buffer("grid", make_grid(sample_tensor))

        self.reduc_factor = self.autoencoder.reduc_factor

        channels_in = self.autoencoder.embed_dim * self.dim_factor
        channels_out = self.autoencoder.embed_dim * self.dim_factor

        # Define latent model
        self.maxvitparams = {
            "dim": self.maxvit_dim,
            "depth": self.maxvit_depth,
            "dim_head": 32,
            "heads": 32,
            "dropout": 0.1,
            "cond_dim": False,
            "window_size": 8,
            "mbconv_expansion_rate": 4,
            "mbconv_shrinkage_rate": 0.25,
            "downsampling": self.maxvit_downsample,
        }
        # target_length needs always to be the whole sequence length
        self.latent_model = Metnet(
            channels_in=channels_in,
            channels_out=channels_out,
            cond=True,
            target_length=self.target_length_val,
            **self.maxvitparams,
        )

        self.register_buffer("weight_time_decay", torch.ones((self.target_length_val), device=self.device))

    def forward(self, x, x_ini, lead_time=None, transform=None, inv_transform=None):
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        current_image = x_ini

        future_image = None
        out = torch.empty((x.shape[0], self.target_length, x.shape[-1], x.shape[-1]), device=self.device)

        first_latent_field = self.autoencoder.encode(x).sample()

        latent = first_latent_field

        future_latent_fields = self.latent_model(latent, lead_time)

        latent_decoded = self.autoencoder.decode(future_latent_fields)
        field_pred, intensity_pred = torch.tensor_split(latent_decoded, (2), dim=1)

        future_image = (
            warp(inv_transform(current_image), field_pred, grid, padding_mode="zeros", fill_value=0) + intensity_pred
        )
        future_image = transform(future_image)

        out[:, 0] = future_image.squeeze()

        return out

    def forward_lead_time_all(self, x, x_ini, transform=None, inv_transform=None):
        # latent_first = None
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        current_image = x_ini

        future_image = None
        out = torch.empty((x.shape[0], self.target_length_val, x.shape[-1], x.shape[-1]), device=self.device)

        first_latent_field = self.autoencoder.encode(x).sample()

        latent_decoded = self.autoencoder.decode(first_latent_field)
        field_pred, intensity_pred = torch.tensor_split(latent_decoded, (2), dim=1)
        future_image = (
            warp(inv_transform(current_image), field_pred, grid, padding_mode="zeros", fill_value=0) + intensity_pred
        )
        future_image = transform(future_image)
        current_image = future_image
        out[:, 0] = future_image.squeeze()

        for i in range(1, self.target_length_val):
            lead_time = torch.Tensor([i - 1]).int()
            lead_time = repeat(lead_time, "c -> b c", b=x.shape[0]).to(device=self.device)
            # lead_time = torch.ones((x.shape[0]), device=self.device, dtype=torch.int) * int(i)
            future_latent_fields = self.latent_model(first_latent_field, lead_time)  # Reshape the i

            latent_decoded = self.autoencoder.decode(future_latent_fields)
            field_pred, intensity_pred = torch.tensor_split(latent_decoded, (2), dim=1)
            future_image = (
                warp(inv_transform(current_image), field_pred, grid, padding_mode="zeros", fill_value=0)
                + intensity_pred
            )
            future_image = transform(future_image)

            current_image = future_image

            out[:, i] = future_image.squeeze()

            return out

    def compute_loss(self, image_pred, x_after, split="train"):
        train_loss_func = self.train_loss().to(self.device)

        loss = train_loss_func(image_pred, x_after)
        self.log(split + "_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        schs = self.lr_schedulers()
        if not isinstance(schs, Iterable):
            schs = [schs]
        opts = self.optimizers()
        if not isinstance(opts, Iterable):
            opts = [opts]

        for opt in opts:
            opt.zero_grad()

        X_dict, Y_dict = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(Y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(Y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        X = X_dict[list(X_dict.keys())[0]]
        Y = Y_dict[list(Y_dict.keys())[0]]

        lead_time = batch[2]
        x_ini = Y[:, 0][:, None, :, :]
        pred = self(X, x_ini, lead_time, transform=loc_transform, inv_transform=loc_inv_transform)
        Y = Y[:, 1][:, None, :, :]

        train_loss = self.compute_loss(
            pred,
            Y,
            split="train",
        )

        self.manual_backward(train_loss)
        for opt in opts:
            opt.step()

        if self.trainer.is_last_batch:
            for sch in schs:
                sch.step()
        return train_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        parameters = self.latent_model.parameters()
        optimizers = []
        schedulers = []

        if self.optim == "adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
            if self.train_autoenc:
                optimizer_autoenc = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001 * lr)
                optimizers.append(optimizer_autoenc)
                scheduler_autoenc = fetch_scheduler(self.scheduler_params)(optimizer_autoenc)
                schedulers.append(scheduler_autoenc)

        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=lr)
            if self.train_autoenc:
                optimizer_autoenc = torch.optim.AdamW(self.autoencoder.parameters(), lr=0.001 * lr)
                optimizers.append(optimizer_autoenc)
                scheduler_autoenc = fetch_scheduler(self.scheduler_params)(optimizer_autoenc)
                schedulers.append(scheduler_autoenc)

        scheduler = fetch_scheduler(self.scheduler_params)(optimizer)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        return optimizers, scheduler

    def validation_step(self, batch, batch_idx):
        X_dict, Y_dict = batch[:2]
        X = X_dict[list(X_dict.keys())[0]]
        Y = Y_dict[list(Y_dict.keys())[0]]

        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(Y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(Y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        x_ini = X[:, -1][:, None, :, :]

        pred = self.forward_lead_time_all(X, x_ini, transform=loc_transform, inv_transform=loc_inv_transform)
        _ = self.compute_loss(pred, Y, "val")

        pred_dict = {list(Y_dict.keys())[0]: pred[0]}

        return pred_dict

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        X_dict = batch[0]
        y_dict = batch[1]
        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(y_dict.keys())[0]][locations[i]](t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        # transformation = self.inv_transforms[list(y_dict.keys())[0]]
        Y = y_dict[list(y_dict.keys())[0]]
        x_before = X_dict[list(X_dict.keys())[0]].to(self.device)

        x_ini = x_before[:, -1][:, None, :, :]

        pred = self.forward_lead_time_all(x_before, x_ini, transform=loc_transform, inv_transform=loc_inv_transform)

        y_hat = {
            list(y_dict.keys())[0]: pred,
        }

        y_trans = {}
        y_hat_trans = {}
        for key in y_dict.keys():
            transformation = self.inv_transforms[key]
            if n_locations == 1:
                location = locations[0]
                y_hat_trans[key] = transformation[location](y_hat[key])
                y_trans[key] = transformation[location](y_dict[key])
            else:
                y_hat_trans[key] = transform_multiple_loc(transformation, y_hat[key], locations)
                y_trans[key] = transform_multiple_loc(transformation, y_dict[key], locations)
        if update_metrics:
            self.eval_metrics_agg.update(
                target=y_trans[list(y_dict.keys())[0]][:, :, None],
                pred=y_hat_trans[list(y_dict.keys())[0]][:, :, None],
                metadata=None,
            )
            self.eval_metrics_lag.update(
                target=y_trans[list(y_dict.keys())[0]][:, :, None],
                pred=y_hat_trans[list(y_dict.keys())[0]][:, :, None],
                metadata=metadata,
            )
        if return_full:
            return y_hat_trans
        return y_hat_trans[list(y_dict.keys())[0]]
