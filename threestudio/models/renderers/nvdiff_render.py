from dataclasses import dataclass, field

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Renderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.models.materials import neuspir_light as light
import time 

def invRT(RT):
    new_RT = torch.eye(4,dtype=RT.dtype,device=RT.device)
    if RT.shape == torch.Size([3, 4]):
        new_RT[:3,:4] = RT
    elif RT.shape == torch.Size([4, 4]):
        new_RT = RT
    else:
        raise TypeError(f"unrecognized RT shape: {RT.shape}")
    return torch.inverse(new_RT)

def blend_fg_bg(
    selector: Float[Tensor,"B H W"], 
    fg: Float[Tensor,"N C"], 
    gb_bg: Float[Tensor,"B H W C"], 
    mask: Float[Tensor,"B H W 1"]
    ) -> Float[Tensor,"B H W C"]:
    """
    Perform a blend of foreground and background image using mask.
    """
    B,H,W,C = gb_bg.shape
    gb_fg = torch.zeros(B, H, W, C).to(fg)
    gb_fg[selector] = fg
    gb = torch.lerp(gb_bg, gb_fg, mask.float())
    return gb

    

@threestudio.register("nvdiff-render")
class NVDiffRender(Renderer):
    @dataclass
    class Config(Renderer.Config):
        context_type: str = "cuda"
        light: dict = field(
            default_factory=lambda: {
                "base_res": "512",
                "brdf_lut_path": "./threestudio/models/materials/renderutils/bsdf_256_256.bin",
            }
        )

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        self.env_light = light.create_trainable_env_rnd(self.cfg.light).to(self.device)

    def load_env_fn(self,fn,scale=1.0,contrast=1.0):
        self.env_light = light.load_env(fn,scale,contrast)
    
    def load_env_base(self,env_base):
        self.env_light = light.EnvironmentLight(env_base)

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        height:int,
        width: int,
        c2w: Float[Tensor, "B 3 4"],
        render_rgb: bool = True,
        verbose:bool = False,
        env_light: light.EnvironmentLight = None,
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        w2c: Float[Tensor, "B 4 4"] = torch.stack([invRT(RT) for RT in c2w])
        if env_light==None:
            self.env_light.build_mips(verbose=verbose) # important to update mips for each batch
            env_light = self.env_light
        assert type(env_light)== light.EnvironmentLight # assure type for light


        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        start_time = time.time()
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        end_time = time.time()
        if verbose:
            print(f"rasterization time:{end_time - start_time}")
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb: 
            # this is the majority of time spent
            start_time = time.time()
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            if verbose:
                end_time = time.time()
                print(f"acc_interpolate time:{end_time - start_time}")

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)
            if verbose:
                end_time = time.time()
                print(f"acc_geo time:{end_time - start_time}")

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]
            
            if verbose:
                end_time = time.time()
                print(f"acc_geo_extra time:{end_time - start_time}")

            
            comp_values_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                env_light=env_light,
                **extra_geo_info,
                **geo_out
            )
            if verbose:
                end_time = time.time()
                print(f"acc_material time:{end_time - start_time}")

            # dealing with depth
            gb_pos_ext:Float[Tensor, "B H W 4"] = torch.concat([gb_pos,torch.ones_like(gb_pos[...,:1])],dim=-1)
            loc_pos_ext:Float[Tensor, "B H W 4"] = torch.matmul(gb_pos_ext,w2c.permute(0,2,1).unsqueeze(1))
            comp_depth = (-loc_pos_ext[...,2:3]/loc_pos_ext[...,3:4])[selector]

            # unravel values
            comp_rgb_fg = comp_values_fg[:,:3]
            comp_kd_fg = comp_values_fg[:,3:6]
            comp_ks_fg = comp_values_fg[:,6:9] # ks values: ordered (roughness, metallic, occlusion)
            comp_sem_feat_fg = comp_values_fg[:,9:] # for all semantic features, should have shape: (Nr, 64)
            

            # background models
            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            # add background for additional properties
            gb_values_bg = torch.ones([batch_size,height,width,64]).to(comp_kd_fg) # create random background with shape (B,H,W,3) 

            # add background for RGB
            gb_rgb = blend_fg_bg(selector, comp_rgb_fg, gb_rgb_bg, mask)
            gb_kd =  blend_fg_bg(selector, comp_kd_fg, gb_values_bg[...,:3], mask)
            gb_ks = blend_fg_bg(selector, comp_ks_fg, gb_values_bg[...,:3], mask)
            gb_sem_feat = blend_fg_bg(selector, comp_sem_feat_fg, gb_values_bg[...,:64], mask)
            gb_depth = blend_fg_bg(selector, comp_depth.to(torch.float32), gb_values_bg[...,:1], mask)

            if verbose:
                end_time = time.time()
                print(f"acc_bkg time:{end_time - start_time}")

            # antialiasing
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
            gb_kd_aa = self.ctx.antialias(gb_kd, rast, v_pos_clip, mesh.t_pos_idx)
            gb_ks_aa = self.ctx.antialias(gb_ks, rast, v_pos_clip, mesh.t_pos_idx)
            gb_sem_feat_aa = self.ctx.antialias(gb_sem_feat, rast, v_pos_clip, mesh.t_pos_idx)
            gb_sem_feat_aa = torch.nn.functional.normalize(gb_sem_feat_aa,dim=-1)
            # gb_depth = self.ctx.antialias(gb_depth, rast, v_pos_clip, mesh.t_pos_idx)

            if verbose:
                end_time = time.time()
                print(f"acc_antialiasing time:{end_time - start_time}")

            out.update(
                {
                    "comp_rgb": gb_rgb_aa, 
                    "comp_rgb_bg": gb_rgb_bg,
                    "comp_kd": gb_kd_aa,
                    "comp_ks": gb_ks_aa,
                    "comp_sem_feat": gb_sem_feat_aa,
                    "depth":gb_depth
                }
            )
            

        return out