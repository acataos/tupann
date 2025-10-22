import pathlib

import torch
import torch.distributed as dist
from einops import rearrange
from timm.scheduler import create_scheduler
from tqdm import tqdm

from src.models.cascast.casformer import CasFormer
from src.models.lightning import LModule
from src.utils.lightning_utils import transform_multiple_loc

from ..autoencoderklgan import misc as utils
from ..autoencoderklgan.lightning import model as autoencoder_kl
from ..autoencoderklgan.misc import dictToObj


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.01,
        weights: str = "uniform",
        name: str = "cascast",
        beta1: float = 0.9,
        beta2: float = 0.95,
        transforms={},
        inv_transforms={},
        scale_factor: float = 1.0,
        **kwargs,
    ):
        dum = torch.zeros([1, 256, 256])
        true_target_dict = {"autoencoderklgan": target_shape_dict["autoencoderklgan"]}
        auto_target = {"autoencoderklgan": dum.shape}
        super().__init__(
            input_shape_dict,
            true_target_dict,
            target_shape_dict_val,
            transforms=transforms,
            inv_transforms=inv_transforms,
            **kwargs,
        )
        self.save_hyperparameters()

        self.name = name
        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.weighted = weights
        self.betas = (beta1, beta2)
        self.input_shape = input_shape_dict["earthformer"][0]

        # just like cascast earthformer default params
        config_autoencoder_loc = list(self.inv_transforms.keys())[0].split("#")[1]

        self.config_castformet = kwargs.get("casformer", {})
        self.castformer = CasFormer(**self.config_castformet)
        self.lr_scheduler_config = kwargs.get("lr_scheduler", {})

        self.config_autoencoder = kwargs.get("autoencoder_kl", {})
        autoenc_hash = list(self.inv_transforms.keys())[0].split("#")[2]

        autoenc_ckpt = self.config_autoencoder.get("ckpt", None)
        self.autoncoder_arch = self.config_autoencoder.get("config", None)

        # transfrom_path = "data/transform_params/goes16_rrqpe.20200101-20221231_10min.yaml"
        # transfrom_params_dict = yaml.safe_load(pathlib.Path(transfrom_path).read_text())
        # # self.inv_transform_autoenc = partial(
        # #     inv_standardize,
        # #     **transfrom_params,
        # # )

        # load autoencoder
        if autoenc_ckpt is not None:
            autoencoder_path = (
                "models/autoencoderklgan/" + autoenc_hash + "/train/" + config_autoencoder_loc + "/" + autoenc_ckpt
            )
        else:
            autoencoder_path = (
                "models/autoencoderklgan/" + autoenc_hash + "/train/" + config_autoencoder_loc + "/model_train.pt"
            )
        print(f"Loading autoencoder from {autoencoder_path}")
        if pathlib.Path(autoencoder_path).suffix == ".ckpt":
            autoencoder = autoencoder_kl.load_from_checkpoint(
                autoencoder_path,
                map_location=self.device,
                input_shape_dict=auto_target,
                target_shape_dict=auto_target,
                target_shape_dict_val=auto_target,
                **self.autoncoder_arch,
            ).requires_grad_(False)
        else:
            autoencoder = autoencoder_kl(
                autoencoder_kl=self.autoncoder_arch,
                lpipsWithDisc=kwargs.get("lpipsWithDisc", {}),
                input_shape_dict=auto_target,
                target_shape_dict=auto_target,
                target_shape_dict_val=auto_target,
            ).requires_grad_(False)
            autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device), strict=False)
        autoencoder.eval()
        self.add_module("autoencoder", autoencoder)

        self.diffusion_kwargs = kwargs.get("diffusion_kwargs", {})

        ## init noise scheduler ##
        self.noise_scheduler_kwargs = self.diffusion_kwargs.get("noise_scheduler", {})

        self.noise_scheduler_type = list(self.noise_scheduler_kwargs.keys())[0]
        if self.noise_scheduler_type == "DDPMScheduler":
            from diffusers import DDPMScheduler

            self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]["num_train_timesteps"]
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        elif self.noise_scheduler_type == "DPMSolverMultistepScheduler":
            from diffusers import DPMSolverMultistepScheduler

            self.noise_scheduler = DPMSolverMultistepScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]["num_train_timesteps"]
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        else:
            raise NotImplementedError

        ## init noise scheduler for sampling ##
        self.sample_noise_scheduler_type = "DDIMScheduler"
        if self.sample_noise_scheduler_type == "DDIMScheduler":
            print("############# USING SAMPLER: DDIMScheduler #############")
            from diffusers import DDIMScheduler

            self.sample_noise_scheduler = DDIMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            ## set num of inference
            self.sample_noise_scheduler.set_timesteps(20)
        elif self.sample_noise_scheduler_type == "DDPMScheduler":
            print("############# USING SAMPLER: DDPMScheduler #############")
            from diffusers import DDPMScheduler

            self.sample_noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            self.sample_noise_scheduler.set_timesteps(1000)
        else:
            raise NotImplementedError

        ## important: scale the noise to get a reasonable noise process ##
        self.noise_scale = self.noise_scheduler_kwargs.get("noise_scale", 1.0)
        # self.logger.info(f'####### noise scale: {self.noise_scale} ##########')

        ## scale factor ##
        self.register_buffer("scale_factor", torch.tensor(scale_factor, dtype=torch.float32))
        # self.logger.info(f'####### USE SCALE_FACTOR: {self.scale_factor} ##########')

        ## classifier free guidance ##
        self.classifier_free_guidance_kwargs = self.diffusion_kwargs.get("classifier_free_guidance", {})
        self.p_uncond = self.classifier_free_guidance_kwargs.get("p_uncond", 0.0)
        self.guidance_weight = self.classifier_free_guidance_kwargs.get("guidance_weight", 0.0)

        # turn off automatic optimization
        self.automatic_optimization = False

    @torch.no_grad()
    def init_scale_factor(self, z_tar):
        del self.scale_factor
        # self.logger.info("### USING STD-RESCALING ###")
        _std = z_tar.std()
        if utils.get_world_size() == 1:
            pass
        else:
            dist.barrier()
            dist.all_reduce(_std)
            _std = _std / dist.get_world_size()
        scale_factor = 1 / _std
        # self.logger.info(f'####### scale factor: {scale_factor.item()} ##########')
        self.register_buffer("scale_factor", scale_factor)

    def forward(self, x, timesteps, cond, context=None, **kwargs):
        """
        x: (b, t, c, h, w)
        cond: (b, t, c, h, w)
        """
        b, t, _, h, w = x.shape
        inp = torch.cat([x, cond], dim=2)
        out = self.model(x=inp, t=timesteps)
        return out

    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device)  # torch.manual_seed(0)
        generator.manual_seed(0)
        latents = torch.randn(
            (bs * ensemble_member, t, c, h, w),
            generator=generator,
            device=template_data.device,
        )
        latents = latents * self.sample_noise_scheduler.init_noise_sigma

        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps):
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.castformer(x=latents, timesteps=timestep, cond=cond_data)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            return latents
        else:
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member * bs : (member + 1) * bs, ...]
                for t in self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs * 2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents] * 2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.castformer(x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            # print("end sampling")
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents

    @torch.no_grad()
    def encode_stage(self, x):
        z = self.autoencoder.autoencoder_kl.net.encode(x)

        return z.sample() * self.scale_factor

    @torch.no_grad()
    def decode_stage(self, z):
        z = z / self.scale_factor

        z = self.autoencoder.autoencoder_kl.net.decode(z)

        return z

    def data_preprocess(self, batch_data):
        """Leaves the data ready to train

        Args:
            batch_data (_type_): _description_
        """
        input_dict = batch_data[0]
        # input_tensors = [arr for arr in input_dict.values()]
        # inp = torch.cat(input_tensors, dim=1)
        inp = input_dict[list(input_dict.keys())[0]].float()

        tar = batch_data[1][list(batch_data[1].keys())[0]].float()

        original_tar = batch_data[1][list(batch_data[1].keys())[1]].float()
        return inp, tar, original_tar

    def training_step(self, batch_data, batch_idx):
        batch = self.data_preprocess(batch_data)
        # need input target and original target
        inp, tar, _ = batch

        # Init the optimizer
        optimazer_diff = self.optimizers()
        b, t, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar
        z_coarse_prediction = inp
        ## init scale_factor ##
        if self.scale_factor == 1.0:
            self.init_scale_factor(tar)
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## classifier free guidance ##
        p = torch.rand(1)
        if p < self.p_uncond:  # discard condition
            z_coarse_prediction_cond = torch.zeros_like(z_coarse_prediction)
        else:
            z_coarse_prediction_cond = z_coarse_prediction
        ## sample noise to add ##
        noise = torch.randn_like(z_tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(z_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.castformer(x=noisy_tar, timesteps=timesteps, cond=z_coarse_prediction_cond)
        train_loss_f = self.train_loss()
        loss = train_loss_f(noise_pred, noise)  # important: rescale the loss
        loss.backward()
        self.log("train_l2_noise", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        ## update params of diffusion model ##
        optimazer_diff.step()
        optimazer_diff.zero_grad()

    def validation_step(self, batch, batch_idx):
        data_dict = self.data_preprocess(batch)
        inp, tar, _ = data_dict
        b, t, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar
        z_coarse_prediction = inp
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample noise to add ##
        noise = torch.randn_like(z_tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(z_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.castformer(x=noisy_tar, timesteps=timesteps, cond=z_coarse_prediction)
        train_loss_f = self.train_loss()
        loss = train_loss_f(noise_pred, noise)  # important: rescale the loss
        self.log("val_l2_noise", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return noise_pred

    def on_validation_batch_end(self, outputs, batch, batch_idx) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        opt_diff = torch.optim.AdamW(
            self.castformer.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        if self.lr_scheduler_config:
            lr_scheduler_auto = create_scheduler(
                dictToObj(self.lr_scheduler_config),
                opt_diff,
            )[0]

            return [opt_diff], [lr_scheduler_auto]
        else:
            return [opt_diff], []

    def forward_denoise_out_latent(self, inp, cfg_weight=1, ens_member=1):
        b, t, c, h, w = inp.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_coarse_prediction = inp
        ## scale ##
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample image ##
        z_sample_prediction = self.denoise(
            template_data=z_coarse_prediction,
            cond_data=z_coarse_prediction,
            bs=b,
            cfg=cfg_weight,
            ensemble_member=ens_member,
        )

        len_shape_prediction = len(z_sample_prediction.shape)
        assert len_shape_prediction == 6
        n = z_sample_prediction.shape[1]
        sample_predictions = []
        for i in range(n):
            member_z_sample_prediction = z_sample_prediction[:, i, ...]
            member_z_sample_prediction = rearrange(member_z_sample_prediction, "b t c h w -> (b t) c h w").contiguous()
            member_sample_prediction = self.decode_stage(member_z_sample_prediction)
            member_sample_prediction = rearrange(member_sample_prediction, "(b t) c h w -> b t c h w", t=t)
            sample_predictions.append(member_sample_prediction)
        sample_predictions = torch.stack(sample_predictions, dim=1)

        sample_prediction = sample_predictions.mean(dim=1)

        return sample_prediction

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def predict_step(self, batch, batch_idx, update_metrics=True, return_full=False):
        "The prediction is done in the image space"
        input, target = batch[:2]
        latent_target = target[list(target.keys())[0]].float().to(self.device)

        metadata = batch[3]
        locations = metadata["location"]
        n_locations = len(set(locations)) if isinstance(locations, list) else 1
        for v in input.values():
            assert v.ndim == 5, "Expected input shape to be (B, T, C, H, W)"

        assert latent_target.ndim == 5, "Expected input shape to be (B, T,C, H, W)"
        # The input is the latent space prediction of the earthformer, in this case the key 1
        x = input[list(input.keys())[0]].float().to(self.device)

        y_hat = self.forward_denoise_out_latent(x, cfg_weight=2, ens_member=1)
        y_trans = {}
        y_hat_trans = {}

        # We only care about the true rain data. In this case is the second key
        key = list(target.keys())[1]
        y_hat_dict = {key: y_hat}

        for key in y_hat_dict.keys():
            transformation = self.inv_transforms[key]
            if n_locations == 1:
                location = locations[0]
                y_hat_trans[key] = transformation[location](y_hat_dict[key])
                y_trans[key] = transformation[location](target[key].to(self.device))
            else:
                y_hat_trans[key] = transform_multiple_loc(transformation, y_hat_dict[key], locations)
                y_trans[key] = transform_multiple_loc(transformation, target[key].to(self.device), locations)

        if type(y_hat) is tuple:
            y_hat = y_hat[0]

        if update_metrics:
            self.eval_metrics_agg.update(
                target=y_trans[list(target.keys())[1]][:, :, None],
                pred=y_hat_trans[list(target.keys())[1]],
                metadata=None,
            )
            self.eval_metrics_lag.update(
                target=y_trans[list(target.keys())[1]][:, :, None],
                pred=y_hat_trans[list(target.keys())[1]],
                metadata=metadata,
            )

        if return_full:
            return {key: y_hat[:, :, 0]}
        return y_hat[:, :, 0]
