import torch
import torch.nn.functional as F
from timm.scheduler import create_scheduler

from src.models.lightning import LModule
from src.utils.lightning_utils import transform_multiple_loc

from .autoencoder_kl import autoencoder_kl
from .lpipsWithDisc import lpipsWithDisc
from .misc import dictToObj


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        predict_latent: bool = False,
        transforms={},
        inv_transforms={},
        **kwargs,
    ):
        super().__init__(
            input_shape_dict,
            target_shape_dict,
            target_shape_dict_val,
            transforms=transforms,
            inv_transforms=inv_transforms,
            **kwargs,
        )

        self.automatic_optimization = False
        self.config_autoencoder = kwargs.get("autoencoder_kl", None)
        self.config_lpipsWithDisc = kwargs.get("lpipsWithDisc", None)
        self.lr_scheduler_config = kwargs.get("lr_scheduler", {})
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.scale_factor = 1.0
        self.predict_latent = predict_latent

        self.autoencoder_kl = autoencoder_kl(config=self.config_autoencoder)
        self.lpipsWithDisc = lpipsWithDisc(config=self.config_lpipsWithDisc)

        self.automatic_optimization = False

    def forward(self, inputs):
        """
        Forward pass through the model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Forward pass through the autoencoder

        reconstruction, _ = self.autoencoder_kl(
            sample=inputs, sample_posterior=True, return_posterior=True, generator=None
        )
        return reconstruction

    def get_last_layer(self):
        last_layer = self.autoencoder_kl.net.decoder.conv_out.weight

        return last_layer

    def training_step(self, batch, batch_idx):
        # data_dict = self.data_preprocess(batch_data)
        inp, tar = batch[:2]
        inp = inp[list(inp.keys())[0]]
        tar = tar[list(tar.keys())[0]]

        optimizer_auto, optimizer_disc = self.optimizers()
        schd_auto, schd_disc = self.lr_schedulers()
        if schd_auto is not None:
            schd_auto.step(self.global_step)
        if schd_disc is not None:
            schd_disc.step(self.global_step)
        ## first: encoder+decoder+logvar ##
        reconstruction, posterior = self.autoencoder_kl(
            sample=inp, sample_posterior=True, return_posterior=True, generator=None
        )
        aeloss, _ = self.lpipsWithDisc(
            inputs=tar,
            reconstructions=reconstruction,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=batch_idx,
            mask=None,
            last_layer=self.get_last_layer(),
            split="train",
        )

        with torch.no_grad():
            self.log(
                "l1_loss",
                F.l1_loss(reconstruction, inp),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "l2_loss",
                F.mse_loss(reconstruction, inp),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("aeloss", aeloss, prog_bar=True, on_epoch=True, sync_dist=True)

        optimizer_auto.zero_grad()
        # Use manual backward to avoid automatic optimization
        self.manual_backward(aeloss)
        # optimizer_auto.step()  # Step the optimizer manually
        optimizer_auto.step()
        # print learning rate
        with torch.no_grad():
            self.log(
                "learning_rate",
                optimizer_auto.param_groups[0]["lr"],
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )

        ## second: the discriminator ##
        disloss, _ = self.lpipsWithDisc(
            tar,
            reconstruction,
            posterior,
            optimizer_idx=1,
            global_step=batch_idx,
            mask=None,
            last_layer=self.get_last_layer(),
            split="train",
        )
        optimizer_disc.zero_grad()
        # Use manual backward to avoid automatic optimization
        self.manual_backward(disloss)
        optimizer_disc.step()
        with torch.no_grad():
            self.log("disloss", disloss, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch_data, batch_idx):
        inp_dict, tar_dir = batch_data[:2]
        inp = inp_dict[list(inp_dict.keys())[0]]
        tar = tar_dir[list(tar_dir.keys())[0]]
        reconstruction, posterior = self.autoencoder_kl(
            sample=inp, sample_posterior=True, return_posterior=True, generator=None
        )
        aeloss, _ = self.lpipsWithDisc(
            inputs=tar,
            reconstructions=reconstruction,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=0,
            mask=None,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val_aeloss", aeloss, prog_bar=True, on_epoch=True, sync_dist=True)

        ## second: the discriminator ##
        disloss, _ = self.lpipsWithDisc(
            tar,
            reconstruction,
            posterior,
            optimizer_idx=1,
            global_step=0,
            mask=None,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val_disloss", disloss, prog_bar=True, on_epoch=True, sync_dist=True)

        return {list(tar_dir.keys())[0]: reconstruction}

    def configure_optimizers(self):
        opt_auto = torch.optim.AdamW(
            self.autoencoder_kl.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        opt_disc = torch.optim.AdamW(
            self.lpipsWithDisc.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        if self.lr_scheduler_config:
            lr_scheduler_auto = create_scheduler(
                dictToObj(self.lr_scheduler_config),
                opt_auto,
            )[0]
            lr_scheduler_disc = create_scheduler(
                dictToObj(self.lr_scheduler_config),
                opt_disc,
            )[0]

            return [opt_auto, opt_disc], [lr_scheduler_auto, lr_scheduler_disc]
        else:
            return [opt_auto, opt_disc], []

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    @torch.no_grad()
    def encode_stage(self, x):
        z = self.autoencoder_kl.net.encode(x)
        return z.sample() * self.scale_factor

    @torch.no_grad()
    def decode_stage(self, z):
        z = z / self.scale_factor
        z = self.autoencoder_kl.net.decode(z)
        return z

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        inp_dict, tar_dict = batch[:2]
        inp = inp_dict[list(inp_dict.keys())[0]].to(self.device)
        tar = tar_dict[list(tar_dict.keys())[0]].to(self.device)
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        transformation = self.inv_transforms[list(tar_dict.keys())[0]]

        z_input = self.encode_stage(inp)
        reconstruction = self.decode_stage(z_input)

        if n_locations == 1:
            location = locations[0]
            reconstruction = transformation[location](reconstruction)
            tar = transformation[location](tar)
        else:
            reconstruction = transform_multiple_loc(transformation, reconstruction, locations)
            tar = transform_multiple_loc(transformation, tar, locations)

        if update_metrics:
            # Calculate the metrics
            self.eval_metrics_agg.update(target=tar[:, :, None], pred=reconstruction[:, :, None])
            self.eval_metrics_lag.update(target=tar[:, :, None], pred=reconstruction[:, :, None])

        if self.predict_latent:
            return z_input[:, None]
        else:
            return {list(inp_dict.keys())[0]: reconstruction}
