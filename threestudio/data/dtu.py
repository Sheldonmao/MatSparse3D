import gzip
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_projection_matrix_from_intrinsic,
    get_ray_directions,
    get_rays,
)
from threestudio.models.materials import neuspir_light
from threestudio.utils.typing import *
import re

from typing import Tuple

def downsample(img, factor):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  if not (sh[0] % factor == 0 and sh[1] % factor == 0):
    raise ValueError(f'Downsampling factor {factor} does not '
                     f'evenly divide image shape {sh[:2]}')
  img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
  img = img.mean((1, 3))
  return img

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]

def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform

def listdir(pth):
  return os.listdir(pth)

def open_file(pth, mode='r'):
  return open(pth, mode=mode)

def load_img(pth: str, downsample_factor) -> np.ndarray:
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
    image = image / 255.
    if downsample_factor > 1:
        image = downsample(image, downsample_factor)
  return image

def load_renderings(config):
    """Load images from disk."""

    images = [] # (N, H,W,3) images
    masks = [] # (N, H,W,1) images
    depths = [] # (N, H,W,1) images
    camtoworlds = [] # (N,3,4) camera extrinsics 3x4 matrix
    camera_mats = [] # (N,3,3) camera intrinsics 3x3 matrix

    # Find out whether the particular scan has 49 or 65 images.
    n_images = len(listdir(config.root_dir)) //3

    # Loop over all images.
    for i in range(1, n_images + 1):
        # Set light condition string accordingly.
        if config.dtu_light_cond < 7:
            light_str = f'{config.dtu_light_cond}_r' + ('5000'
                                                        if i < 50 else '7000')
        else:
            light_str = 'max'

        # Load RGBA image.
        fname = os.path.join(config.root_dir, f'rect_{i:03d}_{light_str}_rgba.png')
        image = load_img(fname,config.factor)
        masks.append(image[...,-1:])
        images.append(image[...,:3])

        # Load Depth image.
        fname = os.path.join(config.root_dir, f'rect_{i:03d}_{light_str}_depth.png')
        depth = load_img(fname,config.factor)
        depths.append(depth[...,None])


        # Load projection matrix from file.
        fname = os.path.join(config.root_dir, f'../../Calibration/cal18/pos_{i:03d}.txt')
        with open_file(fname, 'rb') as f:
            projection = np.loadtxt(f, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)

        if config.factor > 0:
            # Scale camera matrix according to downsampling factor.
            camera_mat = np.diag([1. / config.factor, 1. / config.factor, 1.
                                    ]).astype(np.float32) @ camera_mat
        # pixtocams.append(np.linalg.inv(camera_mat))
        camera_mats.append(camera_mat)

    camera_mats = np.stack(camera_mats)
    camtoworlds = np.stack(camtoworlds)
    images = np.stack(images)
    masks = np.stack(masks)
    depths = np.stack(depths)

    def rescale_poses(poses):
        """Rescales camera poses according to maximum x/y/z value."""
        s = np.max(np.abs(poses[:, :3, -1]))
        out = np.copy(poses)
        out[:, :3, -1] /= s
        return out

    # # Center and scale poses.
    # camtoworlds, _ = recenter_poses(camtoworlds)
    camtoworlds = rescale_poses(camtoworlds)
    # Flip y and z axes to get poses in OpenGL coordinate system.
    camtoworlds = camtoworlds @ np.diag([1., -1., -1., 1.]).astype(np.float32)
    # manually move the camera positions
    Trans = np.array([[0., 0., -0.75]])
    camtoworlds[:,:3,-1] = camtoworlds[:,:3,-1] + Trans

    camtoworlds = np.diag([1., -1., -1.]).astype(np.float32) @ camtoworlds
    camtoworlds[:,:3,-1] *= 3


    height, width = images.shape[1:3]

    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % config.dtuhold == 0],
        "train": all_indices[all_indices % config.dtuhold != 0],
    }

    return images, depths, masks, camera_mats, camtoworlds, height, width, split_indices, all_indices


@dataclass
class DTUDataModuleConfig:
    root_dir: str = ""
    factor: int = 2
    dtu_light_cond: int = 4
    dtuhold: int = 3 # test: 1/dtuhold, train: 1 - 1/dtuhold
    batch_size: int = 1
    train_num_rays: int = -1
    train_views: int = 3
    
    scale_radius: float = 1.0
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 0.0
    render_path: str = "circle"
    near_plane: float = 0.1  # near plane distance for rasterization
    far_plane: float = 100 # far plane distance for rasterization

    train_split: str = "train"
    val_split: str = "test"
    test_split: str = "test"

class DTUDatasetBase:
    
    def setup(self, cfg, split):
        self.all_images, self.all_depths, self.all_fg_masks, self.all_intrinsics, self.all_c2w, self.height, self.width, self.i_split, self.all_frame_id = load_renderings(cfg)
        self.cfg = cfg
        self.split = split
        self.rank = get_rank()

        ## save loaded data for visualization
        all_directions = []
        for intrinsic in self.all_intrinsics:
            fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
            directions: Float[Tensor, "B H W C"] = get_ray_directions(self.height,self.width,(fx,fy),(cx,cy))[None]
            all_directions.append(directions[0])
        all_directions = np.stack(all_directions)
        save_dict = {
                "all_images":self.all_images, 
                "all_fg_masks":self.all_fg_masks,
                "all_c2w":self.all_c2w,
                "all_intrinsics":self.all_intrinsics,
                "all_directions":all_directions,
                "i_split":self.i_split,
                "all_frame_id":self.all_frame_id,
                "all_depths":self.all_depths
            }
        processed_npy = os.path.join(self.cfg.root_dir, "processed.npy")
        np.save(processed_npy,save_dict)

        # modify random camera distance to align with the mean distance of all input cameras
        cam_pose: Float[np_array, "N 3"] = self.all_c2w[..., :3, -1]
        all_distance = np.sqrt(cam_pose[:,0]**2 + cam_pose[:,1]**2 + cam_pose[:,2]**2) # camera distance
        mean_distance = float(all_distance.mean())
        self.cfg.random_camera['eval_camera_distance'] = mean_distance
        self.cfg.random_camera['camera_distance_range'] = [float(all_distance.min())*0.9, float(all_distance.max())*1.1]
        print("[INFO] random camera distance range: ", self.cfg.random_camera['camera_distance_range'])
        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        if self.split == "train":
            num_frames = len(self.i_split['train'])
            if self.cfg.train_views>1:
                sparsity = (num_frames-1)//(self.cfg.train_views-1)
                print(f"sparsity: {sparsity}, original num_frames= {num_frames}")
                self.i_split['train'] = self.i_split['train'][::sparsity]
            else:
                self.i_split['train'] = self.i_split['train'][num_frames//2:num_frames//2+1]
            print("[INFO] num of train views: ", len(self.i_split['train']))
            print("[INFO] train view ids = ", np.array(self.all_frame_id)[self.i_split['train']].tolist())        

        # select, convert to torch on device
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_intrinsics, self.all_depths = (
            self.to_cuda_tensor(self.all_c2w[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_images[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_fg_masks[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_intrinsics),
            self.to_cuda_tensor(self.all_depths[self.i_split[self.split]])
        )
        

    def to_cuda_tensor(self, np_array):
        '''convert to torch on device'''
        return torch.from_numpy(np_array).float().to(self.rank)


    def get_all_images(self):
        return self.all_images
    
    def get_all_features(self):
        return self.all_features

    def get_all_positions(self):
        cam_pose: Float[Tensor, "N 3"] = self.all_c2w[..., :3, -1]

        x,y,z = cam_pose[:,0],cam_pose[:,1],cam_pose[:,2]
        r = torch.sqrt(x**2 + y**2) # camera to origin in xoy plane
        azimuth = torch.atan2(y,x)/np.pi*180 # in degree
        elevation = torch.atan2(z,r)/np.pi*180  # in degree

        distance = torch.sqrt(x**2 + y**2 + z**2) # camera distance

        return elevation, azimuth, distance, cam_pose

    def get_all_masks(self):
        return self.all_fg_masks


class DTUDataset(Dataset, DTUDatasetBase):
    # for test and evaluation only
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return len(self.random_pose_generator)
            else:
                return len(self.all_images)
        else:
            return len(self.random_pose_generator)
            # return len(self.all_images)
    
    def prepare_data(self, index):
        # prepare batch data for validation and test, only when corresponding data are provided
        c2w = self.all_c2w[index][None] # (B=1,3,3)
        camera_positions = c2w[:,:,3] # (B,3)
        
        intrinsic: Float[Tensor, "4 4"] = self.all_intrinsics[index]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        directions: Float[Tensor, "B H W C"] = get_ray_directions(self.height,self.width,(fx,fy),(cx,cy))[None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb: Float[Tensor, "B H W 3"] = self.all_images[index][None]
        depth: Float[Tensor, "B H W 1"] = self.all_depths[index][None]
        mask: Float[Tensor, "B H W 1"] = self.all_fg_masks[index][None]

        # get projection matrix for OpengGL format for nvdiffrast rendering
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_intrinsic(
            intrinsic[None], self.height, self.width, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # calc elevation, azimuth and camera_distance for zero-123
        x,y,z = camera_positions[:,0],camera_positions[:,1],camera_positions[:,2]
        r = torch.sqrt(x**2 + y**2) # camera distance in xoy plane
        azimuth = torch.atan2(y,x)/np.pi*180 # in degree
        elevation = torch.atan2(z,r)/np.pi*180  # in degree
        
        lgt_index = self.i_split[self.split][index]
        # env_light_file = self.all_light_prob[lgt_index] #neuspir_light._load_env_hdr(self.all_light_prob[lgt_index])

        batch = {
            "index":index, #idx of training images
            "rays_o": rays_o[0],
            "rays_d": rays_d[0],
            "mvp_mtx": mvp_mtx[0],
            "c2w":c2w[0],
            "camera_positions": camera_positions[0],
            "elevation": elevation[0],
            "azimuth": azimuth[0],
            "camera_distances": torch.sqrt(x**2 + y**2 + z**2)[0],
            "rgb": rgb[0],
            "mask": mask.bool()[0],
            "height":self.height,
            "width":self.width,
            "depth":depth,
            "env_light_file":"",
            "generated":False,
            "test_method":self.cfg.render_path
        }

        return batch

    def __getitem__(self, index):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return self.random_pose_generator[index]
            else:
                return self.prepare_data(index)
        else:
            return self.random_pose_generator[index]


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        index = self.image_perm[self.idx]
        # prepare batch data here
        c2w = self.all_c2w[index][None] # (B=1,3,3)
        camera_positions = c2w[:,:,3] # (B,3)
        intrinsic: Float[Tensor, "4 4"] = self.all_intrinsics[index]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        directions: Float[Tensor, "B H W C"] = get_ray_directions(self.height,self.width,(fx,fy),(cx,cy))[None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )

        rgb: Float[Tensor, "B H W 3"] = self.all_images[index][None]
        depth: Float[Tensor, "B H W 1"] = self.all_depths[index][None]
        mask: Float[Tensor, "B H W 1"] = self.all_fg_masks[index][None]

        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.height * self.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, width, size=(self.cfg.train_num_rays,), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(self.cfg.train_num_rays,), device=rays_o.device
            )

            rays_o = rays_o[:, y, x].unsqueeze(-2)
            rays_d = rays_d[:, y, x].unsqueeze(-2)
            directions = directions[:, y, x].unsqueeze(-2)
            rgb = rgb[:, y, x].unsqueeze(-2)
            mask = mask[:, y, x].unsqueeze(-2)
            depth = depth[:,y,x].unsqueeze(-2)

        # get projection matrix for OpengGL format for nvdiffrast rendering
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_intrinsic(
            intrinsic[None], self.height, self.width, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # calc elevation, azimuth and camera_distance for zero-123
        x,y,z = camera_positions[:,0],camera_positions[:,1],camera_positions[:,2]
        r = torch.sqrt(x**2 + y**2) # camera distance
        azimuth = torch.atan2(y,x)/np.pi*180 # in degree
        elevation = torch.atan2(z,r)/np.pi*180  # in degree
        
        # rand_idx = np.random.choice(len(self.all_light_prob),1)
        # env_light_file = self.all_light_prob[rand_idx[0]] #neuspir_light._load_env_hdr(self.all_light_prob[rand_idx[0]])
        
        batch = {
            "index":index, #index of training images
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "c2w":c2w,
            "camera_positions": camera_positions,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": torch.sqrt(x**2 + y**2 + z**2),
            "rgb": rgb,
            "mask": mask.bool(),
            "height":self.height,
            "width":self.width,
            "depth":depth,
            "env_light_file":""
        }

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))

        return batch


@register("dtu-datamodule")
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DTUDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit","test","validate","predict"]:
            self.train_dataset = DTUIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DTUDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = DTUDataset(self.cfg, self.cfg.test_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        sampler = None
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            # pin_memory=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)
