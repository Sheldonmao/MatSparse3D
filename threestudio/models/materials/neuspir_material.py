import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *
from threestudio.models.materials import neuspir_light as light

@threestudio.register("neuspir-material")
class NeuSPIRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 72 # first 64 corresponds to semantic feature, the rest is non-semantic ones
        use_bump: bool = False
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
    cfg: Config
    requires_normal: bool = True
    
    def configure(self) -> None:
        self.requires_tangent = self.cfg.use_bump
        if self.cfg.use_bump:
            self.bump_net = get_mlp(self.cfg.input_feature_dims, 3, self.cfg.mlp_network_config)      # normal bump depend on all features
        self.mat_net = get_mlp(self.cfg.input_feature_dims, 5, self.cfg.mlp_network_config)      # material depend on all features
        self.occ_net = get_mlp(self.cfg.input_feature_dims-64, 1, self.cfg.mlp_network_config)   # occ only depend on non-semantic features
        
    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        shading_normal: Float[Tensor, "*B 3"],
        env_light: light.EnvironmentLight,
        tangent: Optional[Float[Tensor, "B ... 3"]] = None,
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:

        # print('features.shape',features.shape)
        network_inp = features.view(-1, features.shape[-1]) # N_samples x feature_channel
        semantic_feats = network_inp[:,:64]
        # print('network_inp.shape',network_inp.shape)
        material = self.mat_net(network_inp).view(*features.shape[:-1], 5).float()
        material = torch.sigmoid(material) # limit the material to range (0,1)

        occ = self.occ_net(network_inp[:,-(self.cfg.input_feature_dims-64):]).view(*features.shape[:-1], 1).float()
        occ = torch.sigmoid(occ) # limit the occ to range (0,1)


        if self.cfg.use_bump:
            assert tangent is not None
            bump = self.bump_net(network_inp).view(*features.shape[:-1], 3).float()
            bump = torch.sigmoid(bump) # limit the occ to range (0,1)
            # perturb_normal is a delta to the initialization [0, 0, 1]
            perturb_normal = (bump * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            ) # from range[0,1] to range [-1,1]
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)

            # apply normal perturbation in tangent space
            bitangent = F.normalize(torch.cross(tangent, shading_normal), dim=-1)
            shading_normal = (
                tangent * perturb_normal[..., 0:1]
                - bitangent * perturb_normal[..., 1:2]
                + shading_normal * perturb_normal[..., 2:3]
            )
            shading_normal = F.normalize(shading_normal, dim=-1)

        rgb = env_light.shade(
            viewdirs,
            shading_normal,
            kd=material[...,:3],
            roughness=material[...,3:4],
            metallic=material[...,4:5],
            occlusion=occ,
        )# (N_samples,3)
        
        # represent in order: diffuse(red,gree,blue), specular(roughness,metalness) and occlusion
        output = torch.cat([rgb, material,occ,semantic_feats],dim=-1) 
        return output
    
    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        network_inp = features.view(-1, features.shape[-1]) # N_samples x feature_channel
        if self.mat_net.network.params.device == 'cpu':
            self.mat_net = self.mat_net.to(network_inp.device)
        material = self.mat_net(network_inp).view(*features.shape[:-1], 5).float()
        material = torch.sigmoid(material)
        out = {
                "albedo": material[...,:3],
                "roughness":material[...,3:4],
                "metallic":material[...,4:5]
            }

        if self.cfg.use_bump:
            perturb_normal = (material[..., 5:8] * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)
            perturb_normal = (perturb_normal + 1) / 2
            out.update(
                {
                    "bump": perturb_normal,
                }
            )
        return out