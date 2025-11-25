from functools import partial

import matplotlib as mlp
import numpy as np
import skimage.measure
import torch
from einops import rearrange
from matplotlib.colors import ListedColormap
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from src.eval.metrics.csi_metrics import csi_denominator_torch, true_positive_torch
from src.eval.metrics.metrics import SEVIRSkillScore, is_dist_avail_and_initialized
from src.utils.data_utils import center_crop
from src.utils.lightning_utils import calc_weights, make_gif_tensor, transform_multiple_loc

CSI_THR = torch.tensor([1, 2, 4, 8, 16, 32, 64])
TP_dict = dict(
    [
        (
            f"TP{thr}",
            {
                "log": False,
                "function": partial(true_positive_torch, threshold=thr, dim=[0, 2, 3]),
            },
        )
        for thr in CSI_THR
    ]
)
DEN_dict = dict(
    [
        (
            f"DEN{thr}",
            {
                "log": False,
                "function": partial(csi_denominator_torch, threshold=thr, dim=[0, 2, 3]),
            },
        )
        for thr in CSI_THR
    ]
)

metrics_by_data = {
    "imerg": {
        "l1_loss": {"log": True, "function": torch.nn.L1Loss()},
        "l2_loss": {"log": True, "function": torch.nn.MSELoss()},
        **TP_dict,
        **DEN_dict,
    },
    "goes16_rrqpe": {
        "l1_loss": {"log": True, "function": torch.nn.L1Loss()},
        "l2_loss": {"log": True, "function": torch.nn.MSELoss()},
        **TP_dict,
        **DEN_dict,
    },
    "fields_intensities": {
        "l1_loss": {"log": True, "function": torch.nn.L1Loss()},
        "l2_loss": {"log": True, "function": torch.nn.MSELoss()},
    },
    "nowcastrio_autoenc": {
        "l1_loss": {"log": True, "function": torch.nn.L1Loss()},
        "l2_loss": {"log": True, "function": torch.nn.MSELoss()},
    },
}

aggregated_metrics = {
    "csi": "tp2/csi_denominator",
}


class LModule(LightningModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        batch_train: tuple = (),
        batch_val: tuple = (),
        loss: str = "l1",
        xmax: float = 3 * 20 / 5,
        needs_prediction: bool = False,
        weights: str = "uniform",
        transforms={},
        inv_transforms={},
        crop_predict=False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.transforms = transforms
        self.inv_transforms = inv_transforms
        self.crop_predict = crop_predict
        # TODO: in latent space this assert is broken
        # assert len(input_shape) == 3 and len(target_shape) == 3
        # Assume input/target shape = (T,H,W)
        lead_time_cond = kwargs.get("lead_time_cond", False)
        if lead_time_cond:
            self.target_length = 1
        else:
            self.target_length = list(target_shape_dict.values())[0][0]

        self.target_length_val = list(target_shape_dict_val.values())[0][0]
        self.type_loss = loss
        self.weighted = weights

        self.needs_prediction = needs_prediction

        self.batch_train = batch_train
        self.batch_val = batch_val

        # self.csi_option = torch.tensor([16, 74, 133, 160, 181, 219])

        self.compute_csi = False
        self.register_buffer("CSI", torch.zeros(
            (2 * len(CSI_THR), self.target_length_val)))
        self.show_im = 1
        self.xmax = xmax

        rainbow = mlp.colormaps["rainbow"]
        white = np.array([1, 1, 1, 1])

        # Create colormap for radar
        list_color = rainbow(np.linspace(0, 1, 256))
        list_color[: (256 // 6), :] = white
        self.cmpradar = ListedColormap(list_color)

        self.cmp = self.cmpradar

        # Use the Sevir skill score
        self.eval_metrics_lag = SEVIRSkillScore(
            layout="NTCHW",
            seq_len=self.target_length_val,
            preprocess_type="identity",
            mode="1",
            dist_eval=True if is_dist_avail_and_initialized() else False,
        )

        self.eval_metrics_agg = SEVIRSkillScore(
            layout="NTCHW",
            seq_len=self.target_length_val,
            preprocess_type="identity",
            mode="0",
            dist_eval=True if is_dist_avail_and_initialized() else False,
        )

    def train_loss(self):
        if self.type_loss == "l1":
            loss = torch.nn.L1Loss()
        elif self.type_loss == "l2":
            loss = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss.")
        return loss.to(self.device)

    # default behavior is to concatenate inputs and outputs
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        for v in x.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        for v in y.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        x = torch.cat([v for v in x.values()], dim=1)
        y = torch.cat([v for v in y.values()], dim=1)
        y_hat = self.forward(x)
        if self.crop_predict:
            y_hat = center_crop(y_hat)
            y = center_crop(y)

        weights = calc_weights(y, self.xmax, self.type_loss, self.weighted)

        train_loss_f = self.train_loss()
        train_loss = train_loss_f(y_hat * weights, y * weights)

        with torch.no_grad():
            self.log(
                "l1_loss",
                F.l1_loss(y_hat, y),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "l2_loss",
                F.mse_loss(y_hat, y),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("train_loss", train_loss, prog_bar=True,
                     on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y_dict = batch[:2]
        for v in x.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        for v in y_dict.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        x = torch.cat([v for v in x.values()], dim=1)
        y = torch.cat([v for v in y_dict.values()], dim=1)
        y_hat = self.forward(x)
        if type(y_hat) is tuple:
            y_hat = y_hat[0]
        if self.crop_predict:
            y_hat = center_crop(y_hat)
            y = center_crop(y)

        weights = calc_weights(y, self.xmax, self.type_loss, self.weighted)
        train_loss_f = self.train_loss()
        val_loss = train_loss_f(y_hat * weights, y * weights)

        self.log("val_loss", val_loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        return {list(y_dict.keys())[0]: y_hat}

    def on_train_epoch_start(self) -> None:
        with torch.no_grad():
            # if self.return_full:
            #     self.plot_fields(self.batch_train, "train")
            #     self.plot_fields(self.batch_val, "validation")

            # Train images

            self.plot_preds(self.batch_train, "train")

            # Validation images
            self.plot_preds(self.batch_val, "validation")

    def on_validation_batch_end(self, outputs, batch, batch_idx) -> None:
        y = batch[1]
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        y_hat = outputs
        for target in y.keys():
            if target == "nowcastrio_autoenc":
                y[target] = y[target][:, :-1]
            for key in self.inv_transforms.keys():
                if target in key:
                    inv_transform_key = key  # Assuming the key is unique for each target
            if n_locations == 1:
                location = locations[0]
                y_hat_trans = self.inv_transforms[inv_transform_key][location](
                    y_hat[target])
                y_trans = self.inv_transforms[inv_transform_key][location](
                    y[target])
            else:
                y_hat_trans = transform_multiple_loc(
                    self.inv_transforms[inv_transform_key], y_hat[target], locations)
                y_trans = transform_multiple_loc(
                    self.inv_transforms[inv_transform_key], y[target], locations)
            for metric_name, value in metrics_by_data[target].items():
                metric_function = value["function"]
                metric_log = value["log"]

                if self.crop_predict:
                    y_trans = center_crop(y_trans)
                    y_hat_trans = center_crop(y_hat_trans)

                computed_metric = metric_function(y_hat_trans, y_trans)

                if metric_log:
                    self.log(
                        metric_name + "_val",
                        computed_metric,
                        prog_bar=True,
                        on_epoch=True,
                        sync_dist=True,
                    )
                if metric_name[:2] == "TP":
                    self.compute_csi = True
                    csi_value = int(metric_name[2:])
                    index_csi = np.where(CSI_THR == csi_value)[0][0]
                    self.CSI[index_csi] += computed_metric

                elif metric_name[:3] == "DEN":
                    csi_value = int(metric_name[3:])
                    index_csi = np.where(CSI_THR == csi_value)[0][0]
                    self.CSI[len(CSI_THR) + index_csi] += computed_metric

    def on_validation_epoch_end(self) -> None:
        if self.compute_csi:
            Calculated_CSI = self.CSI[: len(CSI_THR)] / self.CSI[len(CSI_THR):]
            Sumed_CSI = torch.sum(self.CSI, dim=1)
            CSI_res = Sumed_CSI[: len(CSI_THR)] / Sumed_CSI[len(CSI_THR):]
            Average_CSI = torch.mean(Calculated_CSI, dim=0)

            for i, value in enumerate(CSI_THR):
                for j in range(self.target_length_val):
                    self.log(
                        f"CSI{value}/Lead_time: lag {(j + 1)}",
                        (Calculated_CSI[i][j]).cpu(),
                        on_epoch=True,
                        sync_dist=True,
                    )

                self.log(
                    f"CSI{value}",
                    (CSI_res[i]).cpu(),
                    on_epoch=True,
                    sync_dist=True,
                )

            for j in range(self.target_length_val):
                self.log(
                    f"CSI_average/Lead_time: lag {(j + 1)}",
                    (Average_CSI[j]).cpu(),
                    on_epoch=True,
                    sync_dist=True,
                )

            self.log(
                "CSI_average",
                (CSI_res.mean()).cpu(),
                on_epoch=True,
                sync_dist=True,
            )
            self.CSI[:] = 0

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        X_dict, y_dict = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1

        for v in X_dict.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        for v in y_dict.values():
            assert v.ndim == 4, "Expected input shape to be (B, T, H, W)"
        x = torch.cat([v for v in X_dict.values()], dim=1).to(self.device)
        pred = self.forward(x)

        if type(pred) is tuple:
            pred = pred[0]

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
                y_hat_trans[key] = transform_multiple_loc(
                    transformation, y_hat[key], locations)
                y_trans[key] = transform_multiple_loc(
                    transformation, y_dict[key], locations)

        if update_metrics:
            self.eval_metrics_agg.update(
                target=y_trans[list(y_dict.keys())[0]][:,
                                                       :, None].to(self.device),
                pred=y_hat_trans[list(y_dict.keys())[0]][:,
                                                         :, None].to(self.device),
                metadata=None,
            )
            self.eval_metrics_lag.update(
                target=y_trans[list(y_dict.keys())[0]][:,
                                                       :, None].to(self.device),
                pred=y_hat_trans[list(y_dict.keys())[0]][:,
                                                         :, None].to(self.device),
                metadata=metadata,
            )

        if return_full:
            return y_hat_trans
        return y_hat_trans[list(y_dict.keys())[0]]

    @torch.no_grad()
    def on_after_backward(self):
        with torch.no_grad():
            if self.trainer.global_step % 1000 == 0:
                params = self.named_parameters()
                for p in params:
                    name = p[0]
                    w = p[1]
                    grads = w.grad
                    if grads is None:
                        continue
                    self.logger.experiment.add_histogram(
                        tag=f"weights/grads-{name}",
                        values=grads,
                        global_step=self.trainer.global_step,
                    )
                    change_rate = ((self.lr * grads).std() /
                                   w.data.std()).log10().item()
                    self.logger.experiment.add_scalar(
                        tag=f"weights/grad:data-{name}",
                        scalar_value=change_rate,
                        global_step=self.trainer.global_step,
                    )

    def plot_preds(self, batch, split):
        _, target = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        y_hat = self.predict_step(
            batch, 0, update_metrics=False, return_full=True)
        for key in y_hat.keys():
            if key == "nowcastrio_autoenc":
                continue
            if n_locations == 1:
                location = locations[0]
                y_transf = self.inv_transforms[key][location](target[key])
            else:
                y_transf = transform_multiple_loc(
                    self.inv_transforms[key], target[key], locations)

            y_hat_key = y_hat[key]
            if self.crop_predict:
                y_hat_key = center_crop(y_hat[key])
                y_transf = center_crop(y_transf)

            for i in range(y_transf.shape[0]):
                gif_train = make_gif_tensor(
                    y=y_transf[i].cpu().detach(),
                    y_hat=y_hat_key[i].cpu().detach(),
                    dataset_name=key,
                )

                gif_train = skimage.measure.block_reduce(
                    gif_train,
                    (1, 2, 2, 1),
                    np.median,
                )
                gif_train = rearrange(
                    torch.Tensor(gif_train).unsqueeze(0),
                    "b t h w c -> b t c h w",
                ).to(torch.uint8)
                self.logger.experiment.add_video(
                    f"video/pred/dataset={key}/index={i}/" + f"{split}",
                    gif_train,
                    self.current_epoch,
                    fps=50,
                    walltime=100.0,
                )
