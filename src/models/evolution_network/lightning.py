import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from src.models.evolution_network.evolution_network import Evolution_Encoder_Decoder
from src.models.evolution_network.utils import make_grid, warp
from src.models.lightning import LModule
from src.utils.lightning_utils import (
    FIELDS_INTENSITIES_KEY_OPTIONS,
    calc_concat_shape_dict,
    calc_weights,
    transform_multiple_loc,
)


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        learning_rate: float = 0.0001,
        lambda_: float = 1e-2,
        n_epochs: int = 100,
        cos_weight: float = 0.5,
        vector_weight: float = 0.05,
        fields_intensities_weight: float = 0,
        **kwargs,
    ):
        super(model, self).__init__(
            input_shape_dict,
            target_shape_dict,
            **kwargs,
        )

        self.save_hyperparameters()

        self.lr = learning_rate
        self.lambda_ = lambda_
        self.n_epochs = n_epochs

        assert len(list(target_shape_dict.values())) <= 2, "Target should a dictionary with at most two values"
        self.out_shape = list(target_shape_dict.values())[0]
        self.channels_in = calc_concat_shape_dict(input_shape_dict)[0]
        self.channels_out = self.out_shape[0]
        self.image_size = self.out_shape[-2]

        self.g1 = torch.broadcast_to(
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32), (self.channels_out, 1, 3, 3)
        )

        self.g2 = torch.broadcast_to(
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32), (self.channels_out, 1, 3, 3)
        )
        sample_tensor = torch.zeros(1, 1, self.image_size, self.image_size)
        self.grid = make_grid(sample_tensor)

        self.evo_net = Evolution_Encoder_Decoder(self.channels_in, self.channels_out, base_c=32)

        self.cos_loss = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
        self.cos_weight = cos_weight
        self.vector_weight = vector_weight
        self.fields_intensities_weight = fields_intensities_weight
        self.fields_intensities_key = None

    def forward(self, x, transform=None, inv_transform=None):
        intensity, motion = self.evo_net(x)
        motion_ = motion.reshape(x.shape[0], self.channels_out, 2, x.shape[2], x.shape[3])
        intensity_ = intensity.reshape(x.shape[0], self.channels_out, 1, x.shape[2], x.shape[3])
        series = []
        x_bili = []
        last_frames = x[:, -1, :, :].detach().unsqueeze(1)
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        for i in range(self.channels_out):
            warped = transform(
                warp(inv_transform(last_frames), motion_[:, i], grid.to(self.device), padding_mode="border")
            )
            x_bili.append(warped)
            with torch.no_grad():
                last_frames = warp(
                    inv_transform(last_frames),
                    motion_[:, i],
                    grid.to(self.device),
                    mode="nearest",
                    padding_mode="border",
                )
            last_frames = transform(last_frames + intensity_[:, i])
            series.append(last_frames)
            last_frames = last_frames.detach()
        evo_result = torch.cat(series, dim=1)
        bili_results = torch.cat(x_bili, dim=1)

        return evo_result, bili_results, motion_, intensity

    def intensity_vector_loss(self, motion_pred, intensity_pred, motion_field_truth, intensities_truth, split):
        size = torch.numel(intensity_pred)
        l1_vector = torch.abs(motion_field_truth.contiguous() - motion_pred.contiguous()).sum() / (2 * size)
        cossim = -self.cos_loss(motion_field_truth, motion_pred).sum() / (2 * size)
        l1_intensities = torch.abs(intensities_truth.contiguous() - intensity_pred.squeeze().contiguous()).sum() / size

        rec_loss_vector = self.cos_weight * cossim + (1.0 - self.cos_weight) * l1_vector
        rec_loss_intensities = l1_intensities
        loss = self.vector_weight * rec_loss_vector + (1.0 - self.vector_weight) * rec_loss_intensities

        log = {
            "{}/total_loss".format(split): loss.detach().mean(),
            "{}/rec_loss_vector".format(split): rec_loss_vector.detach().mean(),
            "{}/l1_vector".format(split): l1_vector.detach().mean(),
            "{}/cossim".format(split): cossim.detach().mean(),
            "{}/rec_loss_intensities".format(split): rec_loss_intensities.detach().mean(),
        }

        return loss, log

    def general_step(self, batch, split):
        inputs, targets = batch[:2]
        x = torch.cat([inputs[key] for key in inputs.keys()], dim=1)
        y = targets[list(targets.keys())[0]]

        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(targets.keys())[0]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(targets.keys())[0]][locations[i]](t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        y_hat, y_hat1, v, intensity = self.forward(x, transform=loc_transform, inv_transform=loc_inv_transform)

        weights = calc_weights(y, self.xmax, self.type_loss, self.weighted)
        train_loss_f = self.train_loss()

        loss = train_loss_f(y_hat * weights, y * weights) + train_loss_f(y_hat1 * weights, y * weights)

        self.log(f"l1_loss_{split}", F.l1_loss(y_hat, y), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"l2_loss_{split}", F.mse_loss(y_hat, y), prog_bar=True, on_epoch=True, sync_dist=True)

        v1 = v[:, :, 0, :, :]
        v2 = v[:, :, 1, :, :]
        dv1x = F.conv2d(v1, self.g1.to(self.device), groups=self.channels_out, padding=1)
        dv1y = F.conv2d(v1, self.g2.to(self.device), groups=self.channels_out, padding=1)
        dv2x = F.conv2d(v2, self.g1.to(self.device), groups=self.channels_out, padding=1)
        dv2y = F.conv2d(v2, self.g2.to(self.device), groups=self.channels_out, padding=1)

        loss += self.lambda_ * torch.mean((torch.pow(dv1x, 2) + torch.pow(dv1y, 2)) * weights)
        loss += self.lambda_ * torch.mean((torch.pow(dv2x, 2) + torch.pow(dv2y, 2)) * weights)

        y_hat = {list(targets.keys())[0]: y_hat}
        try:
            try:
                fields_intensities = targets[self.fields_intensities_key]
            except KeyError:
                for option in FIELDS_INTENSITIES_KEY_OPTIONS:
                    if option in targets.keys():
                        self.fields_intensities_key = option
                if self.fields_intensities_key is None:
                    raise ValueError("No valid key for motion fields and intensities found in target dictionary.")
                fields_intensities = targets[self.fields_intensities_key]
            fields_intensities_hat = torch.cat([v, intensity.unsqueeze(2)], dim=2)
            fields_target = fields_intensities[:, :, :2]
            intensities_target = fields_intensities[:, :, 2]
            fields_intensities_loss, log = self.intensity_vector_loss(
                v, intensity, fields_target, intensities_target, "train"
            )

            loss = (
                self.fields_intensities_weight * fields_intensities_loss + (1 - self.fields_intensities_weight) * loss
            )
            y_hat[self.fields_intensities_key] = fields_intensities_hat
            self.log_dict(log, prog_bar=False, on_epoch=True, sync_dist=True)
        except ValueError:
            pass

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, y_hat = self.general_step(batch, "val")
        return y_hat

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        inputs, targets = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        x = torch.cat([inputs[key] for key in inputs.keys()], dim=1).to(self.device)

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(targets.keys())[0]][locations[i]](t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(targets.keys())[0]][locations[i]](t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        y_hat, _, fields_hat, intensities_hat = self.forward(
            x, transform=loc_transform, inv_transform=loc_inv_transform
        )

        for option in FIELDS_INTENSITIES_KEY_OPTIONS:
            if option in targets.keys():
                self.fields_intensities_key = option
        if self.fields_intensities_key is not None:
            y_hat = {
                list(targets.keys())[0]: y_hat,
                self.fields_intensities_key: torch.cat([fields_hat, intensities_hat.unsqueeze(2)], dim=2),
            }
        else:
            y_hat = {list(targets.keys())[0]: y_hat}
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
            )
        if return_full:
            return y_hat_trans
        return y_hat_trans[list(targets.keys())[0]]

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(opt, step_size=self.n_epochs // 2, gamma=0.5)
        return [opt], [scheduler]
