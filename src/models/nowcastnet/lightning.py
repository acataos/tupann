import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lightning import LModule
from src.models.nowcastnet.gan.model import NowcasnetGenerator, TemporalDiscriminator
from src.utils.lightning_utils import calc_concat_shape_dict, calc_weights, transform_multiple_loc


# define the LightningModule
class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        latent_dim: int = 32,
        discriminator_learning_rate: float = 0.0002,
        generator_learning_rate: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        context: torch.Tensor = torch.zeros(1),
        alpha: float = 6,
        beta: float = 20,
        generation_steps: int = 1,
        amplification: float = 1,
        target_is_imerg=False,
        **kwargs,
    ):
        super(model, self).__init__(
            input_shape_dict,
            target_shape_dict,
            context=context,
            target_is_imerg=target_is_imerg,
            **kwargs,
        )

        self.save_hyperparameters()

        self.automatic_optimization = False
        self.h_dim_noise = 8
        self.context = context
        self.latent_dim = latent_dim
        self.dlr = discriminator_learning_rate
        self.glr = generator_learning_rate
        self.lr = (self.dlr + self.glr) / 2
        self.alpha = alpha
        self.beta = beta
        self.generation_steps = generation_steps
        self.amplification = amplification
        self.b1 = b1
        self.b2 = b2

        assert (
            len(list(input_shape_dict.values())) == 2
        ), "Input should a dictionary with two values: real input and Evolution Network prediction"
        self.input_shape = calc_concat_shape_dict(input_shape_dict)
        assert len(list(target_shape_dict.values())) == 1, "Target should a dictionary with a single value"
        self.target_shape = list(target_shape_dict.values())[0]
        self.n_before = self.input_shape[0] - self.target_shape[0]
        self.register_buffer(
            "fixed_noise",
            torch.randn((1, self.latent_dim, self.h_dim_noise, self.h_dim_noise)),
        )

        self.discriminator = TemporalDiscriminator(in_channel=self.input_shape[0])

        self.generator = NowcasnetGenerator(
            channel_in=self.input_shape[0],
            latent_dim=self.latent_dim,
            n_after=self.target_shape[0],
        )

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def forward(self, x):
        z = torch.randn(x.shape[0], self.latent_dim, self.h_dim_noise, self.h_dim_noise)
        z = z.type_as(x)
        return self.generator(x, z)

    def grid_cell_regularizer(self, generated_samples, batch_targets):
        """Grid cell regularizer with max pool as Nowcastnet.

        Args:
        generated_samples: Tensor of size [n_samples, batch_size, time, 256, 256].
        batch_targets: Tensor of size [batch_size, time, 256, 256].

        Returns:
        loss: A tensor of shape [batch_size].
        """
        max_pool = nn.MaxPool2d(kernel_size=5, stride=2)
        max_pred = [max_pool(generated_samples[i]) for i in range(self.generation_steps)]
        x_pred = torch.mean(torch.stack(max_pred, dim=0), dim=0)
        batch_targets = max_pool(batch_targets)

        weights = calc_weights(batch_targets, self.xmax, self.type_loss, self.weighted)
        loss = torch.mean(torch.abs(x_pred - batch_targets) * weights)
        return loss

    def training_step(self, batch):
        inputs, targets = batch[:2]
        x_before = torch.cat([inputs[key] for key in inputs.keys()], dim=1)
        assert len(list(targets.keys())) == 1, "Target should a dictionary with a single value"
        x_after = targets[list(targets.keys())[0]]

        optimizer_g, optimizer_d = self.optimizers()
        real = torch.concat([x_before[:, : self.n_before, :, :], x_after], dim=1)

        for _ in range(1):
            valid = torch.ones(real.size(0), 1)
            valid = valid.type_as(real)

            predictions = [self.forward(x_before) for _ in range(self.generation_steps)]
            generated_samples = torch.stack(predictions, dim=0)

            x_pred = torch.mean(generated_samples, dim=0)

            # y = [x_after, x_pred]

            # self.log("l2_loss_train", F.mse_loss(y[1], y[0]), prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
            # self.log("l1_loss_train", F.l1_loss(y[1], y[0]), prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

            pred = torch.concat([x_before[:, : self.n_before, :, :], x_pred], dim=1)

            for i in range(self.generation_steps):
                disc_val = torch.zeros(self.generation_steps)
                pred = torch.concat([x_before[:, : self.n_before, :, :], generated_samples[i]], dim=1)

                score_generated = self.discriminator(pred)
                disc_val[i] = self.adversarial_loss(score_generated, valid)

            g_loss = torch.mean(disc_val)

            cell_loss = self.grid_cell_regularizer(generated_samples, x_after)

            generator_loss = self.alpha * g_loss + self.beta * cell_loss

            self.log(
                "g_loss_train",
                g_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )
            self.log(
                "g_cell_train",
                generator_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )

            optimizer_d.zero_grad()
            optimizer_g.zero_grad()

            self.manual_backward(generator_loss)
            optimizer_g.step()

        fake = torch.zeros(real.size(0), 1)

        fake = fake.type_as(real)

        for _ in range(1):
            pred = torch.concat([x_before[:, : self.n_before, :, :], self.forward(x_before)], dim=1).detach()

            real_loss = self.adversarial_loss(self.discriminator(real), valid)
            fake_loss = self.adversarial_loss(self.discriminator(pred), fake)

            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss_train", d_loss, prog_bar=True)

            optimizer_d.zero_grad()
            self.manual_backward(d_loss)

            optimizer_d.step()

    def configure_optimizers(self):
        b1 = self.b1
        b2 = self.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.glr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.dlr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[:2]
        x_before = torch.cat([inputs[key] for key in inputs.keys()], dim=1)
        assert len(list(targets.keys())) == 1, "Target should a dictionary with a single value"
        x_after = targets[list(targets.keys())[0]]
        real = torch.concat([x_before[:, : self.n_before, :, :], x_after], dim=1)
        valid = torch.ones(real.size(0), 1)
        fake = torch.zeros(real.size(0), 1)

        valid = valid.type_as(real)
        fake = fake.type_as(real)

        predictions = [self.forward(x_before) for _ in range(self.generation_steps)]
        generated_samples = torch.stack(predictions, dim=0)
        x_pred = torch.mean(generated_samples, dim=0)

        # y = self.normalization([x_after, x_pred])

        # self.log("l2_loss_val", F.mse_loss(y[1], y[0]), prog_bar=True, on_epoch=True, sync_dist=True)
        # self.log("l1_loss_val", F.l1_loss(y[1], y[0]), prog_bar=True, on_epoch=True, sync_dist=True)
        pred = torch.concat([x_before[:, : self.n_before, :, :], x_after], dim=1)

        score_generated = self.discriminator(pred)

        g_loss = self.adversarial_loss(score_generated, valid)

        cell_loss = self.grid_cell_regularizer(generated_samples, x_after)

        generator_loss = self.alpha * g_loss + self.beta * cell_loss

        self.log("g_loss_val", g_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("g_cell_val", generator_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        real_loss = self.adversarial_loss(self.discriminator(real), valid)
        fake_loss = self.adversarial_loss(self.discriminator(pred), fake)

        d_loss = (real_loss + fake_loss) / 2

        self.log("d_loss_val", d_loss, prog_bar=True)

        return {list(targets.keys())[0]: x_pred}

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        inputs, targets = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        x_before = torch.cat([inputs[key] for key in inputs.keys()], dim=1).to(self.device)
        assert len(list(targets.keys())) == 1, "Assume target to be a dictionary with a single value"
        predictions = [self.forward(x_before) for _ in range(self.generation_steps)]
        generated_samples = torch.stack(predictions, dim=0)
        y_hat_tensor = torch.mean(generated_samples, dim=0)
        y_hat = {list(targets.keys())[0]: y_hat_tensor}

        y_trans = {}
        y_hat_trans = {}
        for key in targets.keys():
            transformation = self.inv_transforms[key]
            if n_locations == 1:
                location = locations[0]
                y_hat_trans[key] = transformation[location](y_hat[key])
                y_trans[key] = transformation[location](targets[key])
            else:
                y_hat_trans[key] = transform_multiple_loc(transformation, y_hat[key], locations)
                y_trans[key] = transform_multiple_loc(transformation, targets[key], locations)
        if update_metrics:
            self.eval_metrics_agg.update(
                target=y_trans[list(targets.keys())[0]][:, :, None],
                pred=y_hat_trans[list(targets.keys())[0]][:, :, None],
                metadata=None,
            )
            self.eval_metrics_lag.update(
                target=y_trans[list(targets.keys())[0]][:, :, None],
                pred=y_hat_trans[list(targets.keys())[0]][:, :, None],
                metadata=metadata,
            )
        if return_full:
            return y_hat_trans
        return y_hat_trans[list(targets.keys())[0]]
