import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.systems.criterions import PSNR, SSIM
from threestudio.models.materials.neuspir_light import cubemap_to_latlong,_load_env_hdr
import cv2
import time

@threestudio.register("zero123n-system")
class Zero123n(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'refine']
        freq: dict = field(default_factory=dict)
        test_verbose: bool = True
        update_gen_envmap: str = "False"

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any],verbose=False) -> Dict[str, Any]:
        start_time = time.time()
        if self.cfg.renderer_type == "nvdiff-render":
            render_out = self.renderer(**batch, render_normal=True, render_rgb=True)
        else:
            render_out = self.renderer(**batch)
        end_time = time.time()
        if verbose:
            print(f"rendering time: {end_time - start_time}")
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        train_dataset = self.trainer.datamodule.train_dataloader().dataset

        # visualize all training images
        all_images = train_dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        # visualize all training depths if available:
        if hasattr(train_dataset, "all_depths"):
            all_depths = train_dataset.all_depths
            self.save_image_grid(
                "all_training_depths.png",
                [
                    {"type": "grayscale", "img": image,"kwargs": {}}
                    for image in all_depths
                ],
                name="on_fit_start",
                step=self.true_global_step,
            )
        
        # visualize all training features
        if hasattr(train_dataset, "all_features"):
            all_features = train_dataset.all_features
            if all_features is not None:
                _,H,W,_ = all_images.shape
                N,h,w,C = all_features.shape
                features = torch.nn.functional.interpolate(all_features.permute(0,3,1,2), (H,W),mode='bilinear').permute(0,2,3,1) # in format (N,H,W,C)
                features = torch.nn.functional.normalize(features,dim=-1)
                features = torch.from_numpy(train_dataset.pca_fit.transform(features.reshape(-1,C).detach().cpu())).reshape(N,H,W,3)
                self.save_image_grid(
                    "all_training_features.png",
                    [
                        {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC","data_range": (-0.5, 0.5)}}
                        for image in features
                    ],
                    name="on_fit_start",
                    step=self.true_global_step,
                )
                # "data_range":[features.min().item(),features.max().item()]

        # no prompt processor
        all_mask = train_dataset.get_all_masks()
        all_elevation, all_azimuth, all_distance, all_camera_position = train_dataset.get_all_positions()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.prepare_conds(all_images,all_mask,all_elevation, all_azimuth, all_distance)


    def training_substep(self, batch, batch_idx, guidance: str, verbose=False):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        start_time = time.time()
        if guidance == "ref":
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
            bg_color = None
            batch['env_light']=None
            
        elif guidance == "zero123":
            env_light = _load_env_hdr(batch["env_light_file"]) if batch["env_light_file"]!="" else None
            batch = batch["random_camera"]
            if self.cfg.update_gen_envmap=="True":
                batch["env_light"]=env_light
            if random.random() > 0.5:
                bg_color = None
            else:
                bg_color = torch.rand(3).to(self.device)
            ambient_ratio = 0.1 + 0.9 * random.random()
            

        batch["bg_color"] = bg_color
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]
            
            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))
            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                gt_depth = batch["depth"]
                valid_gt_depth = gt_depth[gt_mask].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))   

        elif guidance == "zero123":
            guidance_inp = out['comp_rgb']
            # zero123
            guidance_out, guidance_eval_out = self.guidance(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            set_loss("sds", guidance_out["loss_sds"])

        # apply regularizations on materials and envronment light
        if self.C(self.cfg.loss.lambda_occ) > 0:
            if "comp_ks" not in out:
                raise ValueError(
                    "comp_ks is required for orientation loss, no normal is found in the output."
                )
            set_loss("occ", out["comp_ks"][...,2].mean())
        if self.C(self.cfg.loss.lambda_envmap) > 0:
            if not hasattr(self.renderer, "env_light"):
                raise ValueError(
                    "env_light is required for orientation loss, no normal is found in the output."
                )
            light_base = self.renderer.env_light.base
            white = (light_base[..., 0:1] + light_base[..., 1:2] + light_base[..., 2:3]) / 3.0
            set_loss("envmap", torch.mean(torch.abs(light_base - white)))

        # apply renderer related regularizations on geometry properties
        if self.cfg.renderer_type == 'nvdiff-render':
            set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())
        else:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                set_loss(
                    "normal_smooth",
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                    + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
                )

            if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for normal smooth loss, no normal is found in the output."
                    )
                if "normal_perturb" not in out:
                    raise ValueError(
                        "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                    )
                normals = out["normal"]
                normals_perturb = out["normal_perturb"]
                set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

            if guidance != "ref":
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3) # proximately range (0,1)
            set_loss("opaque", binary_cross_entropy(opacity_clamped, opacity_clamped))

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(out["comp_rgb"].detach(), guidance_eval_out)
        end_time = time.time()
        if verbose:
            print(f"sub_training with {guidance} guidance time: {end_time - start_time}")
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.freq.ref_or_zero123 == "accumulate":
            do_ref = True
            do_zero123 = True
        elif self.cfg.freq.ref_or_zero123 == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_zero123 = not do_ref

        total_loss = 0.0
        if do_zero123:
            out = self.training_substep(batch, batch_idx, guidance="zero123",verbose=False)
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref",verbose=False)
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def merge12(self, x):
        # merge N x C x H x W to C x H x W
        return x.reshape(-1, *x.shape[2:])

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"
        self.save_image_grid(
            filename,
            [
                {
                    "name":"comp_rgb",
                    "type": "rgb",
                    "img": self.merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "name":"img_noisy",
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "name":"img_1step",
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "name":"img_origin",
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "name":"img_final",
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
        )

        img = Image.open(self.get_save_path(filename))
        draw = ImageDraw.Draw(img)
        for i, n in enumerate(guidance_eval_out["noise_levels"]):
            draw.text((1, (img.size[1] // B) * i + 1), f"{n:.02f}", (255, 255, 255))
            draw.text((0, (img.size[1] // B) * i), f"{n:.02f}", (0, 0, 0))
        img.save(self.get_save_path(filename))

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        val_dataset = self.trainer.datamodule.val_dataloader().dataset
        N,H,W,_ = out["comp_normal"].shape # N==1
        if hasattr(self.renderer, "env_light"):
            env_map = cubemap_to_latlong(self.renderer.env_light.base, [H, H*2]).detach().cpu()
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [{"name":"rgb","type": "rgb","img": batch["rgb"][0],"kwargs": {"data_format": "HWC"},}]
                if "rgb" in batch else []
            )
            + [{"name":"comp_rgb","type": "rgb","img": out["comp_rgb"][0],"kwargs": {"data_format": "HWC"},},]
            + (
                [{"name":"comp_normal","type": "rgb","img": out["comp_normal"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                if "comp_normal" in out else []
            )
            + (
                [{"name":"depth","type": "grayscale","img": out["depth"][0],"kwargs": {}}] 
                if "depth" in out else []
            )
            + (
                [{"name":"comp_kd","type": "rgb","img": out["comp_kd"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                if "comp_kd" in out else []
            )
            + (
                [
                    {"name":"roughness","type": "grayscale", "img": out["comp_ks"][0][...,0], "kwargs": {}},
                    {"name":"metallic","type": "grayscale", "img": out["comp_ks"][0][...,1], "kwargs": {}},
                    {"name":"occlusion","type": "grayscale", "img": out["comp_ks"][0][...,2], "kwargs": {}}
                ] 
                if "comp_ks" in out else []
            )
            + [{"name":"opacity","type": "grayscale","img": out["opacity"][0, :, :, 0],"kwargs": {"cmap": None, "data_range": (0, 1)},},]
            + (
                [{"name":"env_light","type": "rgb","img": env_map,"kwargs": {"data_format": "HWC"},},] 
                if hasattr(self.renderer, "env_light") else []
            ),
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
            align = "ori"
        )


    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def test_step(self, batch, batch_idx):
        if "env_light_file" in batch.keys():
            # batch["env_light"] = _load_env_hdr(batch["env_light_file"][0])
            batch["env_light"] = _load_env_hdr(batch["env_light_file"][0],shift=0.5) if batch["env_light_file"][0]!="" else None
        out = self(batch)
        # self.test_dir = f"it{self.true_global_step}-test"
        if batch['generated'] == True:
            self.test_dir = f"it{self.true_global_step}-test-gen"
        else:
            self.test_dir = f"it{self.true_global_step}-test-{batch['test_method'][0]}"

        if self.cfg.test_verbose:
            test_dataset = self.trainer.datamodule.test_dataloader().dataset
            N,H,W,_ = out["comp_normal"].shape # N==1
            if hasattr(self.renderer, "env_light"):
                env_map = cubemap_to_latlong(self.renderer.env_light.base, [H, H*2]).detach().cpu()
            self.save_image_grid(
                f"{self.test_dir}/{batch['index'][0]}.png",
                (
                    [{"name":"gt_rgb","type": "rgb","img": batch["rgb"][0],"kwargs": {"data_format": "HWC"},}]
                    if "rgb" in batch else []
                )
                + [{"name":"comp_rgb","type": "rgb","img": out["comp_rgb"][0],"kwargs": {"data_format": "HWC"},},]
                + (
                    [{"name":"comp_normal","type": "rgb","img": out["comp_normal"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                    if "comp_normal" in out else []
                )
                + (
                    [{"name":"depth","type": "grayscale","img": out["depth"][0],"kwargs": {}}] 
                    if "depth" in out else []
                )
                +(
                    [{"name":"gt_albedo","type": "rgb","img": batch["albedo"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                    if "albedo" in batch else []
                )
                +(
                    [{"name":"gt_glossy","type": "rgb","img": batch["glossy"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                    if "glossy" in batch else []
                )
                +(
                    [{"name":"comp_kd","type": "rgb","img": out["comp_kd"][0],"kwargs": {"data_format": "HWC", "data_range": (0, 1)},}]
                    if "comp_kd" in out else []
                )
                + (
                    [
                        {"name":"roughness","type": "grayscale", "img": out["comp_ks"][0][...,0], "kwargs": {}},
                        {"name":"metallic","type": "grayscale", "img": out["comp_ks"][0][...,1], "kwargs": {}},
                        {"name":"occlusion","type": "grayscale", "img": out["comp_ks"][0][...,2], "kwargs": {}}
                    ] 
                    if "comp_ks" in out else []
                )
                + [{"name":"opacity","type": "grayscale","img": out["opacity"][0, :, :, 0],"kwargs": {"cmap": None, "data_range": (0, 1)},},]
                + (
                    [{"name":"mask","type": "grayscale","img": batch['mask'][0, :, :, 0].float(),"kwargs": {"cmap": None, "data_range": (0, 1)},},]
                    if "mask" in batch.keys() else []
                )
                + (
                    [{"name":"env_light","type": "rgb","img": env_map,"kwargs": {"data_format": "HWC"},},] 
                    if hasattr(self.renderer, "env_light") else []
                ),
                name="test_step",
                step=self.true_global_step,
                align = "ori"
            )
        else:
            self.save_image_grid(
                f"{self.test_dir}/{batch['index'][0]}.png",
                (
                    [
                        {
                            "name":"gt_rgb",
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch else []
                )
                + [
                    {
                        "name":"comp_rgb",
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "name":"comp_normal",
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out else []
                )
                + (
                    [
                        {
                            "name":"depth",
                            "type": "grayscale", 
                            "img": out["depth"][0], 
                            "kwargs": {}
                        }
                    ]
                    if "depth" in out else []
                )
                + [
                    {
                        "opacity"
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        # convert test images to mp4 videos
        self.save_img_sequence(
            self.test_dir, # f"it{self.true_global_step}-test",
            self.test_dir, # f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )

        # log evaluation result
        criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            # 'lpips': LPIPS(net_type='alex',version='0.1')
        }
        self.calc_metric_imgs(
            f"{self.test_dir}.csv", #f"it{self.true_global_step}-test.csv",
            f"{self.test_dir}", #f"it{self.true_global_step}-test",
            "(\d+)\.json",
            criterions,
            name="test",
            step=self.true_global_step,
        )
