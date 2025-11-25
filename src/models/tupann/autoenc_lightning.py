import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.models.lightning import LModule
from src.models.tupann.autoenc_parts.autoenc_utils import DiagonalGaussianDistribution
from src.models.tupann.autoenc_parts.modules import Decoder, Encoder
from src.models.tupann.utils import make_grid, warp
from src.utils.lightning_utils import transform_multiple_loc

# def wait():
#     estructure_time = time.localtime()
#     if estructure_time.tm_wday < 5 and estructure_time.tm_hour < 19 and estructure_time.tm_hour >= 9:
#         time.sleep(2.5)
#     elif estructure_time.tm_hour < 19 and estructure_time.tm_hour >= 9:
#         time.sleep(1)


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        learning_rate: float = 0.0001,
        embed_dim: int = 6,
        reduc_factor: int = 4,
        cos_weight=0.33,
        kl_weight=0.00001,
        vector_weight=0.5,
        future_image_loss=0.0,
        lanbda_: float = 0.0,
        channels=64,
        dropout=0.0,
        n_fields=1,
        lead_time_cond: int = 0,
        predict_latent: bool = False,
        **kwargs,
    ):
        true_target_dict = {list(target_shape_dict.keys())[
            0]: target_shape_dict[list(target_shape_dict.keys())[0]]}
        super().__init__(
            input_shape_dict,
            true_target_dict,
            **kwargs,
        )
        self.automatic_optimization = False
        self.lr = learning_rate
        self.loss_weight = cos_weight
        self.cos_loss = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
        self.reduc_factor = reduc_factor
        self.kl_weight = kl_weight
        self.vector_weight = vector_weight
        self.embed_dim = embed_dim
        self.future_image_loss = future_image_loss
        self.n_fields = n_fields
        self.lambda_ = lanbda_
        self.predict_latent = predict_latent
        self.lead_time_cond = lead_time_cond
        self.fields_intensities_key = "fields_intensities"

        self.g1 = torch.broadcast_to(
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                         dtype=torch.float32), (1, 1, 3, 3)
        )

        self.g2 = torch.broadcast_to(
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                         dtype=torch.float32), (1, 1, 3, 3)
        )

        # assert len(list(true_target_dict.values())) == 1, "Target should a dictionary with a single value"
        # assert len(list(input_shape_dict.values())) == 1, "Input should a dictionary with a single value"

        self.in_shape = list(input_shape_dict.values())[0]
        self.input_length = self.in_shape[0]
        self.target_length = list(target_shape_dict.values())[0][0]
        self.img_size = self.in_shape[-1]
        self.register_buffer("sample_tensor", torch.zeros(
            1, 1, self.img_size, self.img_size))
        self.register_buffer("grid", make_grid(self.sample_tensor))
        self.name = "autoenc"

        z_channels = 4
        double_z = True
        ch_mult = [2**i for i in range(int(np.log2(reduc_factor) // 2) + 1)]

        self.encoder = Encoder(
            ch=channels,
            in_channels=self.input_length,
            resolution=self.img_size,
            z_channels=z_channels,
            num_res_blocks=2,
            attn_resolutions=[],
            ch_mult=ch_mult,
            dropout=dropout,
            double_z=double_z,
        )

        self.decoder = Decoder(
            ch=channels,
            out_ch=(2 * self.n_fields + 1),
            in_channels=self.input_length,
            resolution=self.img_size,
            z_channels=z_channels,
            num_res_blocks=2,
            attn_resolutions=[],
            ch_mult=ch_mult,
            dropout=dropout,
        )

        self.classifier = torch.nn.Conv2d(self.input_length, 1, 1)

        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def classify(self, z):
        classes = self.classifier(z)
        return classes

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def loss(
        self,
        motions_field_real,
        intensities_real,
        reconstructions,
        posteriors,
        split="train",
    ):
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        field_pred = reconstructions[:, :2].squeeze()
        intensity_pred = reconstructions[:, 2].squeeze()
        l1_vector = torch.abs(motions_field_real - field_pred).mean()
        cossim = -self.cos_loss(motions_field_real, field_pred).mean()
        l1_intensities = torch.abs(intensities_real - intensity_pred).mean()

        rec_loss_vector = self.loss_weight * cossim + \
            (1.0 - self.loss_weight) * l1_vector
        rec_loss_intensities = l1_intensities

        loss = (
            self.vector_weight * rec_loss_vector
            + (1 - self.vector_weight) * rec_loss_intensities
            + self.kl_weight * kl_loss
        )

        log = {
            "{}/total_loss".format(split): loss.detach().mean(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/rec_loss_vector".format(split): rec_loss_vector.detach().mean(),
            "{}/l1_vector".format(split): l1_vector.detach().mean(),
            "{}/cossim".format(split): cossim.detach().mean(),
            "{}/rec_loss_intensities".format(split): rec_loss_intensities.detach().mean(),
        }
        return loss, log

    def loss_val(
        self,
        motion_truth,
        motion_predict,
        intensities_truth,
        intensities_pred,
        posteriors,
        split="val",
    ):
        kl_loss = []
        for i in range(self.target_length):
            kl_loss_i = posteriors[i].kl()
            kl_loss.append(torch.sum(kl_loss_i) / kl_loss_i.shape[0])

        kl_loss = torch.stack(kl_loss).mean()
        l1_vector = torch.abs(motion_truth - motion_predict).mean()
        cossim = -self.cos_loss(motion_truth, motion_predict).mean()
        l1_intensities = torch.abs(intensities_truth - intensities_pred).mean()

        rec_loss_vector = self.loss_weight * cossim + \
            (1.0 - self.loss_weight) * l1_vector
        rec_loss_intensities = l1_intensities
        loss = (
            self.vector_weight * rec_loss_vector
            + (1 - self.vector_weight) * rec_loss_intensities
            + self.kl_weight * kl_loss
        )

        log = {
            "{}/total_loss".format(split): loss.detach().mean(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/rec_loss_vector".format(split): rec_loss_vector.detach().mean(),
            "{}/l1_vector".format(split): l1_vector.detach().mean(),
            "{}/cossim".format(split): cossim.detach().mean(),
            "{}/rec_loss_intensities".format(split): rec_loss_intensities.detach().mean(),
        }

        return loss, log

    def training_step(self, batch, batch_idx):
        X, Y = batch[:2]

        motion_fields = Y[self.fields_intensities_key][:, :, :2].squeeze()
        intensities = Y[self.fields_intensities_key][:, :, 2].squeeze()

        assert len(X.keys()) == 1, "Expected a single key in the input batch."
        input = X[list(X.keys())[0]]
        reconstructions, posterior = self(input)
        opt_ae = self.optimizers()
        sch = self.lr_schedulers()


        self.toggle_optimizer(opt_ae)

        aeloss, log_dict_ae = self.loss(
            motion_fields, intensities, reconstructions, posterior, split="train"
        )

        self.log_dict(
            log_dict_ae,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.manual_backward(aeloss)
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        if self.trainer.is_last_batch:
            sch.step()

    def forward_lead_time_all(self, x, x_ini, latent_ini=None, transform=None, inv_transform=None):
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        current_image = x_ini
        future_image = None
        out = torch.empty(
            (x.shape[0], self.target_length, x.shape[-1], x.shape[-1]),
            device=self.device,
        )
        out_intensities = torch.empty(
            (x.shape[0], self.target_length, 1, x.shape[-1], x.shape[-1]),
            device=self.device,
        )
        out_motion_field = torch.empty(
            (x.shape[0], self.target_length, 2, x.shape[-1], x.shape[-1]),
            device=self.device,
        )

        posteriors = []

        for i in range(self.target_length):
            reconstructions, posterior = self(x)
            field_pred = reconstructions[:, :2]
            intensity_pred = reconstructions[:, 2]
            future_image = warp(
                inv_transform(current_image),
                field_pred,
                grid,
                padding_mode="zeros",
                fill_value=0,
            ) + intensity_pred.unsqueeze(1)
            future_image = transform(future_image)

            out[:, i] = future_image.squeeze()
            out_motion_field[:, i] = field_pred
            out_intensities[:, i] = intensity_pred[:, None]

            posteriors.append(posterior)

            x = torch.cat((x[:, 1:], future_image), axis=1)
            current_image = future_image
        return out, out_intensities, out_motion_field, posteriors

    def validation_step(self, batch, batch_idx):
        X, Y = batch[:2]

        motion_fields = Y[self.fields_intensities_key][:, :, :2].squeeze()
        intensities = Y[self.fields_intensities_key][:, :, 2].squeeze()

        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(Y.keys())[1]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])], dim=0
            )

        def loc_inv_transform(t):
            return torch.cat(
                [self.inv_transforms[list(Y.keys())[1]][locations[i]](
                    t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        assert len(X.keys()) == 1, "Expected a single key in the input batch."
        input = X[list(X.keys())[0]]
        x_ini = input[:, -1][:, None, :, :]

        pred = self.forward_lead_time_all(
            input, x_ini, transform=loc_transform, inv_transform=loc_inv_transform)
        image_pred, intensities_pred, motion_fields_pred, posterior = pred

        val_loss, log_dict_ae = self.loss_val(
            motion_fields,
            motion_fields_pred,
            intensities,
            intensities_pred[:, :, 0],
            posterior,
            split="val",
        )

        self.log("val/total_loss", val_loss, sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        concat_pred = torch.cat((motion_fields_pred, intensities_pred), dim=2)
        return {list(Y.keys())[0]: concat_pred, list(Y.keys())[1]: image_pred}

    def predict_step(self, batch, batch_idx, update_metrics=False, return_full=False):
        X = batch[0]
        y_dict = batch[1]
        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(y_dict.keys())[1]][locations[i]](
                    t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(y_dict.keys())[1]][locations[i]](
                        t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        assert len(X.keys()) == 1, "Expected a single key in the input batch."
        input = X[list(X.keys())[0]].to(self.device)
        x_ini = input[:, -1][:, None, :, :]
        pred, intensities_hat, fields_hat, _ = self.forward_lead_time_all(
            input, x_ini, transform=loc_transform, inv_transform=loc_inv_transform
        )

        y_hat = {
            list(y_dict.keys())[1]: pred,
            self.fields_intensities_key: torch.cat([fields_hat, intensities_hat], dim=2),
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
                y_hat_trans[key] = transform_multiple_loc(
                    transformation, y_hat[key], locations)
                y_trans[key] = transform_multiple_loc(
                    transformation, y_dict[key], locations)

        if update_metrics:
            self.eval_metrics.update(
                target=y_trans[list(y_dict.keys())[1]][:, :, None],
                pred=y_hat_trans[list(y_dict.keys())[1]][:, :, None],
            )
        if self.predict_latent == "sample" or self.predict_latent == "mean_logvar":
            z = self.encode(input)
            if self.predict_latent == "mean_logvar":
                mean = z.mean
                logvar = z.logvar
                first_latent_field = torch.cat((mean, logvar), dim=1)
                return first_latent_field[:, None]
            elif self.predict_latent == "sample":
                return z.sample()[:, None]
        if return_full:
            return y_hat_trans
        return y_hat_trans[list(y_dict.keys())[0]]

    def configure_optimizers(self):
        lr = self.lr
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        scheduler = LambdaLR(opt_ae, lr_lambda=lambda epoch: 0.99**epoch)
        return {"optimizer": opt_ae, "lr_scheduler": {"scheduler": scheduler}}
