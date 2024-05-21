import importlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from threestudio.utils.ops import cylin2cart


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt) #, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("zero123n-guidance")
class Zero123nGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "load/zero123/zero123-xl.ckpt"
        pretrained_config: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = True

        guidance_scale: float = 5.0

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        cond_cam_pos: str="nearest" # or "random"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero123 ...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        # self.prepare_embeddings(self.cfg.cond_image_path)

        threestudio.info(f"Loaded Zero123!")

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_conds(
        self, 
        all_rgb: Float[Tensor, "N H W 3"], 
        all_mask: Float[Tensor, "N H W 1"],
        all_elevation: Float[Tensor, "N"],
        all_azimuth: Float[Tensor, "N"],
        all_camera_distances: Float[Tensor, "N"],
    ):
        
        # compute cond image for zero123
        all_rgb = all_rgb * all_mask + (1- all_mask)
        
        # make sure input image have same height and width, and rescale it to 256x256, 
        # assert all_rgb.shape[1] == all_rgb.shape[2]
        self.all_rgb: Float[Tensor, "N 3 H W"] = \
            torch.nn.functional.interpolate(all_rgb.permute(0,3,1,2), (256,256),mode='bilinear')
        self.all_c_crossattn, self.all_c_concat = self.get_img_embeds(self.all_rgb)
        
        # load conditional camera position informations
        self.all_c_elevation, self.all_c_azimuth, self.all_c_camera_distance = all_elevation, all_azimuth, all_camera_distances
        self.all_c_camera_xyz = cylin2cart(all_elevation, all_azimuth, all_camera_distances, rad=False)
        
        # # initialize the condition as the first image and its position
        # self.update_cond(0)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def update_cond(
        self,
        cond_idx:int = 0
    ):
        self.c_concat = self.all_c_concat[cond_idx]
        self.c_crossattn = self.all_c_crossattn[cond_idx]
        self.c_elevation = self.all_c_elevation[cond_idx]
        self.c_azimuth = self.all_c_azimuth[cond_idx]
        self.c_camera_distance = self.all_c_camera_distance[cond_idx]

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        **kwargs,
    ) -> dict:
        
        # update condition using closest camera in space
        if self.cfg.cond_cam_pos == "nearest":
            mean_camera_xyz = cylin2cart(elevation, azimuth, camera_distances).mean(dim=0)
            delta_xyz_norm = torch.norm(mean_camera_xyz - self.all_c_camera_xyz,dim=-1)
            cond_idx = torch.argmin(delta_xyz_norm)
        elif self.cfg.cond_cam_pos == "random":
            cond_idx = np.random.choice(len(self.all_c_concat))
        self.update_cond(cond_idx)

        T = torch.stack(
            [
                torch.deg2rad(
                    (90 - elevation) - (90 - self.c_elevation)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth - self.c_azimuth)),
                torch.cos(torch.deg2rad(azimuth - self.c_azimuth)),
                camera_distances - self.c_camera_distance,
            ],
            dim=-1,
        )[:, None, :].to(self.device)
        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (self.c_crossattn).repeat(
                        len(T), 1, 1
                    ),
                    T,
                ],
                dim=-1,
            )
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat)
                    .repeat(len(T), 1, 1, 1)
                    .to(self.device),
                    (self.c_concat).repeat(
                        len(T), 1, 1, 1
                    ),
                ],
                dim=0,
            )
        ]
        return cond

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs, # returned batch
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(rgb_BCHW, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        cond = self.get_cond(elevation, azimuth, camera_distances)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(cond, t, latents_noisy, noise_pred)
        else:
            guidance_eval_out = {}
        return guidance_out, guidance_eval_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, cond, t_orig, latents, noise_pred):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents.shape[0]  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(
            [bs, -1]
        ) > t_orig.unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(len(t)):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1) # B x H x W x C
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1) # B x H x W x C

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            c = {
                "c_crossattn": [cond["c_crossattn"][0][b * 2 : b * 2 + 2]],
                "c_concat": [cond["c_concat"][0][b * 2 : b * 2 + 2]],
            }
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                x_in = torch.cat([latents] * 2)
                t_in = torch.cat([t.reshape(1)] * 2).to(self.device)
                noise_pred = self.model.apply_model(x_in, t_in, c)
                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def generate(
        self,
        image,  # image tensor [1, 3, H, W] in [0, 1]
        elevation=0,
        azimuth=0,
        camera_distances=0,  # new view params
        c_crossattn=None,
        c_concat=None,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        if c_crossattn is None:
            c_crossattn, c_concat = self.get_img_embeds(image)

        cond = self.get_cond(
            elevation, azimuth, camera_distances, c_crossattn, c_concat
        )

        imgs = self.gen_from_cond(cond, scale, ddim_steps, post_process, ddim_eta)

        return imgs

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def gen_from_cond(
        self,
        cond,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        # produce latents loop
        B = cond["c_crossattn"][0].shape[0] // 2
        latents = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)

        for t in self.scheduler.timesteps:
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)[
                "prev_sample"
            ]

        imgs = self.decode_latents(latents)
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs
