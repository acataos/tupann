"""
Lightning module for ResNet-based nowcasting model.
"""
import pathlib

import torch
import yaml
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR

from src.models.lightning import LModule
from src.models.tupann.autoenc_lightning import model as AutoencoderKL
from src.models.tupann.utils import make_grid, warp
from src.models.rescast.resnet_module import ResNetNowcasting
from src.utils.lightning_utils import FIELDS_INTENSITIES_KEY_OPTIONS, transform_multiple_loc


class model(LModule):
    """
    Lightning module for ResNet-based nowcasting.

    This model follows the same structure as the original nowcasting model but uses
    a simple ResNet to predict all future timesteps at once instead of iteratively.

    Key differences:
    - Uses ResNet backbone instead of transformer
    - Predicts all timesteps simultaneously
    - Simpler architecture with convolutional layers
    """

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
        base_channels: int = 64,
        num_blocks: int = 2,
        loss: str = "l1",
        weights: str = "uniform",
        xmax: float = 3 * 20 / 5,
        target_is_imerge: bool = False,
        transforms={},
        inv_transforms={},
        **kwargs,
    ):
        """
        Initialize the ResNet nowcasting model.

        Args:
            input_shape_dict: Dictionary with input shapes
            target_shape_dict: Dictionary with target shapes
            target_shape_dict_val: Validation target shapes
            autoenc_hash: Hash for autoencoder model directory
            autoenc_ckpt: Specific autoencoder checkpoint to load
            dataset: Dataset name for autoencoder path
            learning_rate: Learning rate for optimizer
            optim: Optimizer type ('adam' or 'adamw')
            base_channels: Base number of channels for ResNet
            num_blocks: Number of residual blocks per stage
            loss: Loss function type
            weights: Loss weighting strategy
            xmax: Maximum value for normalization
            target_is_imerge: Whether target is in imerge format
            transforms: Data transforms
            inv_transforms: Inverse transforms
        """
        # Extract first target for parent initialization
        true_target_dict = {list(target_shape_dict.keys())[
            0]: target_shape_dict[list(target_shape_dict.keys())[0]]}

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
            **kwargs,
        )

        self.save_hyperparameters()

        # Store configuration
        self.automatic_optimization = False
        self.optim = optim
        self.learning_rate = learning_rate
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.name = "resnet_nowcasting"
        self.fields_intensities_key = None

        # Extract dimensions
        self.img_size = target_shape_dict[list(
            target_shape_dict.keys())[0]][-1]
        self.input_length = input_shape_dict[list(
            input_shape_dict.keys())[0]][0]
        self.target_length = target_shape_dict[list(
            target_shape_dict.keys())[0]][0]

        # Check if we have latent input (2 keys) or just image input (1 key)
        if len(list(input_shape_dict.keys())) == 2:
            self.add_latent = True
        elif len(list(input_shape_dict.keys())) == 1:
            self.add_latent = False
        else:
            raise ValueError(
                "Input shape dictionary should have 1 or 2 keys, " f"got {len(input_shape_dict.keys())}")

        # Set up autoencoder input shape
        if self.add_latent:
            input_autoencoder = torch.zeros(9, self.img_size, self.img_size)
            input_shape_dict_autoenc = {list(input_shape_dict.keys())[
                0]: input_autoencoder.shape}
        else:
            input_shape_dict_autoenc = input_shape_dict

        # Load autoencoder
        self._load_autoencoder(autoenc_hash, autoenc_ckpt,
                               dataset, input_shape_dict_autoenc, true_target_dict)

        # Create grid for warping operations
        sample_tensor = torch.zeros(1, 1, self.img_size, self.img_size)
        self.register_buffer("grid", make_grid(sample_tensor))

        # Initialize ResNet model
        self.reduc_factor = self.autoencoder.reduc_factor
        latent_size = self.img_size // self.reduc_factor

        self.resnet = ResNetNowcasting(
            input_channels=self.autoencoder.embed_dim,
            output_timesteps=self.target_length,
            latent_height=latent_size,
            latent_width=latent_size,
            base_channels=base_channels,
            num_blocks=num_blocks,
        )

    def _load_autoencoder(self, autoenc_hash, autoenc_ckpt, dataset, input_shape_dict_autoenc, true_target_dict):
        """Load and setup the pretrained autoencoder."""
        # Construct autoencoder path
        if autoenc_ckpt is not None:
            autoencoder_path = f"models/tupann_autoenc/{autoenc_hash}/train/" f"{dataset}/{autoenc_ckpt}"
        else:
            autoencoder_path = f"models/tupann_autoenc/{autoenc_hash}/train/" f"{dataset}/model_train.pt"

        # Load autoencoder hyperparameters
        with open(pathlib.Path(autoencoder_path).parent / "h_params.yaml") as f:
            params = yaml.safe_load(f)

        # Create validation target dict for autoencoder
        dummy_val = torch.zeros(1, self.img_size, self.img_size)
        val_autoencoder_dict = {
            list(true_target_dict.keys())[-1]: dummy_val.shape}

        # Initialize autoencoder
        autoencoder = AutoencoderKL(
            **params,
            input_shape_dict=input_shape_dict_autoenc,
            target_shape_dict=true_target_dict,
            target_shape_dict_val=val_autoencoder_dict,
        )

        # Load weights
        try:
            autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=self.device, weights_only=False), strict=True
            )
        except Exception as e:
            print(f"Direct ckpt load failed: {e}, trying to load state dict")
            autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=self.device,
                           weights_only=False)["state_dict"], strict=True
            )
        autoencoder = autoencoder.requires_grad_(False)
        autoencoder.eval()
        self.add_module("autoencoder", autoencoder)

    def forward(self, x, x_ini, latent_first=None, transform=None, inv_transform=None):
        """
        Forward pass through the model.

        Args:
            x: Input sequence tensor
            x_ini: Initial frame for warping
            latent_first: Optional pre-computed initial latent
            transform: Transform function for locations
            inv_transform: Inverse transform function

        Returns:
            Tuple of (predictions, intensities, motion_fields, latent_fields)
        """
        batch_size = x.shape[0]
        grid = self.grid.repeat(batch_size, 1, 1, 1)

        # Initialize output tensors
        out = torch.empty((batch_size, self.target_length,
                          x.shape[-1], x.shape[-1]), device=self.device)
        out_intensities = torch.empty(
            (batch_size, self.target_length, 1, x.shape[-1], x.shape[-1]), device=self.device)
        out_motion_field = torch.empty(
            (batch_size, self.target_length, 2, x.shape[-1], x.shape[-1]), device=self.device
        )

        # Encode initial frame to latent space
        if latent_first is None:
            initial_latent = self.autoencoder.encode(x).sample()
        else:
            initial_latent = latent_first

        # Predict all future latents using ResNet
        future_latents = self.resnet(initial_latent)
        # Shape: (batch, target_length, embed_dim, latent_h, latent_w)

        current_image = x_ini.detach()

        # Process each predicted timestep
        for i in range(self.target_length):
            # Get latent for this timestep
            latent_t = future_latents[:, i]

            # Decode latent to motion field and intensity
            decoded = self.autoencoder.decode(latent_t)
            field_pred, intensity_pred = torch.tensor_split(
                decoded, (2,), dim=1)

            # Apply motion field to current image
            with torch.no_grad():
                future_image = transform(
                    warp(inv_transform(current_image), field_pred,
                         grid, padding_mode="zeros", fill_value=0)
                    + intensity_pred
                )

            # Store outputs
            out[:, i] = future_image.squeeze()
            out_motion_field[:, i] = field_pred
            out_intensities[:, i] = intensity_pred

            # Update current image for next iteration
            current_image = future_image.detach()

        # Return latent fields for loss computation if needed
        latent_fields = future_latents

        return out, out_intensities, out_motion_field, latent_fields

    def compute_loss(self, pred, x_after, motion_field_truth, intensities_truth, latent_truth=None, split="train"):
        """Compute loss based on predicted and target values."""
        train_loss_func = self.train_loss().to(self.device)

        image_pred = pred[0]
        loss = train_loss_func(image_pred, x_after)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        X_dict, Y_dict = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]

        # Location-specific transforms
        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(Y_dict.keys())[0]][locations[i]](
                    t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(Y_dict.keys())[0]][locations[i]](
                        t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        # Extract data
        X = X_dict[list(X_dict.keys())[0]]
        Y = Y_dict[list(Y_dict.keys())[0]]

        # Get motion fields and intensities
        try:
            motion_field_truth = Y_dict[self.fields_intensities_key][:, :, :2]
            intensities_truth = Y_dict[self.fields_intensities_key][:, :, 2]
        except (KeyError, TypeError):
            for option in FIELDS_INTENSITIES_KEY_OPTIONS:
                if option in Y_dict.keys():
                    self.fields_intensities_key = option
                    break
            if self.fields_intensities_key is None:
                raise ValueError(
                    "No valid key for motion fields and intensities found in target dictionary.")
            motion_field_truth = Y_dict[self.fields_intensities_key][:, :, :2]
            intensities_truth = Y_dict[self.fields_intensities_key][:, :, 2]

        # Handle latent input if present
        # if self.add_latent:
        #     latent_ini = X_dict[list(X_dict.keys())[1]].squeeze()
        #     latent_truth = Y_dict[list(Y_dict.keys())[2]][:, 1:]
        # else:
        #     latent_ini = None
        #     latent_truth = torch.zeros(
        #         (Y.shape[0], 2, Y.shape[-1], Y.shape[-1], Y.shape[1])
        #     ).cuda()
        #     latent_truth = rearrange(
        #         latent_truth, "b c x y na -> b na c x y")[:, 1:]
        #
        # Set initial frame
        x_ini = X[:, -1][:, None, :, :]
        optim = self.optimizers()
        sch = self.lr_schedulers()
        # Forward pass
        batch_size = X.shape[0]
        grid = self.grid.repeat(batch_size, 1, 1, 1)

        # Initialize output tensors
        # out = torch.empty(
        #     (batch_size, self.target_length, X.shape[-1], X.shape[-1]),
        #     device=self.device
        # )
        # out_intensities = torch.empty(
        #     (batch_size, self.target_length, 1, X.shape[-1], X.shape[-1]),
        #     device=self.device
        # )
        # out_motion_field = torch.empty(
        #     (batch_size, self.target_length, 2, X.shape[-1], X.shape[-1]),
        #     device=self.device
        # )

        # Encode initial frame to latent space
        # if latent_ini is None:
        #     initial_latent = self.autoencoder.encode(X).sample()
        # else:
        #     initial_latent = latent_ini
        initial_latent = self.autoencoder.encode(X).sample()

        # Predict all future latents using ResNet
        future_latents = self.resnet(initial_latent)
        # Shape: (batch, target_length, embed_dim, latent_h, latent_w)

        current_image = x_ini.detach()

        optim.zero_grad()
        total_loss = torch.tensor([0.0], device=self.device)
        # Process each predicted timestep
        retain_graph = True
        for i in range(self.target_length):
            if i == self.target_length - 1:
                retain_graph = False
            # Get latent for this timestep
            latent_t = future_latents[:, i]

            # Decode latent to motion field and intensity
            decoded = self.autoencoder.decode(latent_t)
            field_pred, intensity_pred = torch.tensor_split(
                decoded, (2,), dim=1)

            # Apply motion field to current image
            future_image = loc_transform(
                warp(loc_inv_transform(current_image), field_pred,
                     grid, padding_mode="zeros", fill_value=0)
                + intensity_pred
            )

            # Store outputs
            # out[:, i] = future_image.squeeze()
            # out_motion_field[:, i] = field_pred
            # out_intensities[:, i] = intensity_pred
            # TODO: check future_image dimension
            partial_loss = self.compute_loss(
                (future_image, intensity_pred, field_pred, latent_t),
                Y[:, i],
                motion_field_truth[:, i],
                intensities_truth[:, i],
            )
            self.manual_backward(partial_loss, retain_graph=retain_graph)
            total_loss += partial_loss.detach()

            # Update current image for next iteration
            current_image = future_image.detach()
        optim.step()
        sch.step()

        # Compute loss
        self.log("train_loss", total_loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X_dict, Y_dict = batch[:2]
        metadata = batch[3]
        locations = metadata["location"]

        # Location-specific transforms
        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(Y_dict.keys())[0]][locations[i]](
                    t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(Y_dict.keys())[0]][locations[i]](
                        t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        # Extract data
        X = X_dict[list(X_dict.keys())[0]]
        Y = Y_dict[list(Y_dict.keys())[0]]

        # Get motion fields and intensities
        try:
            motion_fields_truth = Y_dict[self.fields_intensities_key][:, :, :2]
            intensities_truth = Y_dict[self.fields_intensities_key][:, :, 2]
        except (KeyError, TypeError):
            for option in FIELDS_INTENSITIES_KEY_OPTIONS:
                if option in Y_dict.keys():
                    self.fields_intensities_key = option
                    break
            if self.fields_intensities_key is None:
                raise ValueError(
                    "No valid key for motion fields and intensities found in target dictionary.")
            motion_fields_truth = Y_dict[self.fields_intensities_key][:, :, :2]
            intensities_truth = Y_dict[self.fields_intensities_key][:, :, 2]

        # Handle latent input if present
        if self.add_latent:
            latent_ini = X_dict[list(X_dict.keys())[1]].squeeze()
            latent_truth = Y_dict[list(Y_dict.keys())[2]][:, :-1]
        else:
            latent_ini = None
            latent_truth = torch.zeros(
                (Y.shape[0], 2, Y.shape[-1], Y.shape[-1], Y.shape[1])).cuda()
            latent_truth = rearrange(latent_truth, "b c x y na -> b na c x y")

        # Set initial frame
        x_ini = X[:, -1][:, None, :, :]

        # Forward pass
        pred = self(X, x_ini, latent_ini, transform=loc_transform,
                    inv_transform=loc_inv_transform)

        # Compute validation loss
        _ = self.compute_loss(pred, Y, motion_fields_truth,
                              intensities_truth, latent_truth, "val")

        # Return predictions in expected format
        pred_dict = {list(Y_dict.keys())[0]: pred[0], list(Y_dict.keys())[
            1]: torch.cat((pred[2], pred[1]), dim=2)}
        if self.add_latent:
            pred_dict[list(Y_dict.keys())[2]] = pred[3]

        return pred_dict

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        self.lr = self.hparams.learning_rate

        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.resnet.parameters(), lr=self.lr)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.resnet.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")

        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99**epoch)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        X_dict = batch[0]
        y_dict = batch[1]
        metadata = batch[3]
        locations = metadata["location"]

        def loc_transform(t):
            return torch.cat(
                [self.transforms[list(y_dict.keys())[0]][locations[i]](
                    t[i]).unsqueeze(0) for i in range(t.shape[0])],
                dim=0,
            )

        def loc_inv_transform(t):
            return torch.cat(
                [
                    self.inv_transforms[list(y_dict.keys())[0]][locations[i]](
                        t[i]).unsqueeze(0)
                    for i in range(t.shape[0])
                ],
                dim=0,
            )

        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        # breakpoint()
        # transformation = self.inv_transforms[list(y_dict.keys())[0]]
        Y = y_dict[list(y_dict.keys())[0]]
        x_before = X_dict[list(X_dict.keys())[0]].to(self.device)
        if len(list(X_dict.keys())) == 1:
            latent_ini = None
            latent_truth = torch.zeros(
                (Y.shape[0], 2, Y.shape[-1], Y.shape[-1], Y.shape[1])).cuda()
            latent_truth = rearrange(latent_truth, "b c x y na -> b na c x y")
        else:
            latent_ini = X_dict[list(X_dict.keys())[1]
                                ].squeeze().to(self.device)
            latent_truth = y_dict[list(y_dict.keys())[2]][:, :-1]

        if latent_ini is None:
            x_ini = x_before[:, -1][:, None, :, :]
        else:
            x_ini = x_before

        pred, intensities_hat, fields_hat, latent_hat = self(
            x_before, x_ini, latent_ini, transform=loc_transform, inv_transform=loc_inv_transform
        )

        for option in FIELDS_INTENSITIES_KEY_OPTIONS:
            if option in y_dict.keys():
                self.fields_intensities_key = option
        if self.fields_intensities_key is None:
            self.fields_intensities_key = "fields_intensities"
        y_hat = {
            list(y_dict.keys())[0]: pred,
            self.fields_intensities_key: torch.cat([fields_hat, intensities_hat], dim=2),
        }
        if self.add_latent:
            y_hat[list(y_dict.keys())[2]] = latent_hat
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
                target=y_trans[list(y_dict.keys())[0]][:, :, None],
                pred=y_hat_trans[list(y_dict.keys())[0]][:, :, None],
            )
        if return_full:
            return y_hat_trans
        return y_hat_trans[list(y_dict.keys())[0]]
