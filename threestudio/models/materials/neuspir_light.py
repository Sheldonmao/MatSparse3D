# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import torchvision.transforms.functional as tff
from . import renderutils as ru
import imageio
import time

######################################################################################
# Utility functions
######################################################################################
#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# sRGB color transforms
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

#----------------------------------------------------------------------------
# cube map transforms
#----------------------------------------------------------------------------
def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(-v[..., 1:2], v[..., 0:1]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 2:3], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*cosphi,  
        -sintheta*sinphi,
        costheta
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)


def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255


######################################################################################
# Load, modify and store hdr files
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0,contrast=1.0,shift=0):
    latlong_img = torch.tensor(load_image(fn), dtype=torch.float32, device='cuda') # H x W x 3
    if shift>0:
        split_idx = int(shift*latlong_img.shape[1])
        H,W = latlong_img.shape[:2]
        # print(f"split_idx:{split_idx}, H:{H}, W:{W}")
        new_latlong_img = torch.zeros_like(latlong_img)
        new_latlong_img[:,:split_idx] = latlong_img[:,-split_idx:]
        new_latlong_img[:,split_idx:] = latlong_img[:,:-split_idx]
        latlong_img = new_latlong_img
    latlong_img = tff.adjust_contrast(latlong_img.permute(2,0,1),contrast).permute(1,2,0)*scale 
    cubemap = latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by adding block hight light
def mod_env_hdr(light, x=0, y=0,bs=100,black=False,intensity=1):
    '''
        x in range(-1,1) indicates the vertical location of bright block
        y in range(-1,1) indicates the horizontal location of bright block
        bs indicates the bright block size 
    '''
    latlong_img = cubemap_to_latlong(light.base, [512, 1024])
    max_val = intensity #latlong_img.max()
    if black:
        latlong_img[:,:,:]=0
    if type(bs)==int:
        bs_x = bs_y = bs
    else:
        bs_x,bs_y = bs
    H,W,_ = latlong_img.shape
    x_start = int((W-bs_x)/2*(x+1))
    y_start = int((H-bs_y)/2*(y+1))

    latlong_img[y_start:y_start+bs_y,x_start:x_start+bs_x] = max_val

    cubemap = latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by horizontally moving the light
def mod_env_hdr_mov(light,idx, period):
    latlong_img = cubemap_to_latlong(light.base, [512, 1024])
    div = int(idx/period*1024)
    left_part = latlong_img[:,:div].clone()
    latlong_img[:,:1024-div] = latlong_img[:,div:] 
    latlong_img[:,1024-div:] = left_part
    cubemap = latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by cycling the darkness
def mod_env_hdr_cycle(light, idx, period = 20):
    '''
    '''
    latlong_img = cubemap_to_latlong(light.base, [512, 1024])
    idx = np.abs(idx % (period*2) - period)
    latlong_img = latlong_img / period * idx
    cubemap = latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l


def load_env(fn, scale=1.0,contrast=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr" or os.path.splitext(fn)[1].lower() == ".png" :
        return _load_env_hdr(fn, scale, contrast)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = cubemap_to_latlong(light.base, [512, 1024])
    save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(config, scale=0.5, bias=0.25):
    base = torch.rand(6, config.base_res, config.base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base,config.brdf_lut_path)
      
class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base,brdf_lut_path='threestudio/models/materials/renderutils/bsdf_256_256.bin'):
        '''
        input:
            base: torch.tensor with shape 6xHxWx3, highest-resolution env_map
        '''
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.brdf_lut_path = brdf_lut_path
        self.register_parameter('env_base', self.base)
        

    def xfm(self, mtx):
        ''' store transform matrix for vector transformation **unused**
        '''
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99,verbose=False):
        ''' build mips for env_map
        '''
        start_time = time.time()
        with torch.no_grad():
            self.clamp_(min=0.0) # make sure the env_map is in valid range
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
        end_time = time.time()
        if verbose:
            print(f"buil_mips time: {end_time - start_time}")

    def regularizations(self,out):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return {'light_white': torch.mean(torch.abs(self.base - white))}
        
    def shade(self, view_dir, normal, kd, roughness, metallic, occlusion, specular=True):
        ''' perform shading of a point.
        
        input: 
            view_dir: viewing direction pointing from surface to camera (N_samples x 3)
            normal: normal direction (N_samples x 3)
            kd: diffuse albedo, channels are RGB (N_samples x 3)
            roughness: roughness value (N_samples x 1)
            metallic: metallic value (N_samples x 1)
            occlusion: occlusion value (N_samples x 1)
            specular: weather render using specular lobe or not
        '''
        assert kd.shape[1]==3 and roughness.shape[1]==1 and metallic.shape[1]==1 and occlusion.shape[1]==1
        if specular:
            # roughness = ks[..., 1:2] # y component
            # metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = safe_normalize(reflect(view_dir, normal))
        nrmvec = normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec[None,None,:,:].contiguous(), filter_mode='linear', boundary_mode='cube')[0,0]
        diffuse = torch.clamp(diffuse,min=0)
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(dot(view_dir, normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile(self.brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv[None,None,:,:], filter_mode='linear', boundary_mode='clamp')[0,0]

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness[None,None,:,:])
            spec = dr.texture(self.specular[0][None, ...], reflvec[None,None,:,:].contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')[0,0]
            spec = torch.clamp(spec,min=0)
            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        shaded_col = shaded_col  * (1.0 - occlusion) # Modulate by ambient occlusion *hemisphere visibility* 
        shaded_col = torch.clamp(shaded_col,0,1)
        return rgb_to_srgb(shaded_col)
    
    def render_bg(self, rays):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        env_bg = dr.texture(self.base[None, ...], rays_d[None,None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0,0]
        return env_bg

