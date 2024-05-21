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
from sklearn.decomposition import PCA
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

@dataclass
class SparseRelightDataModuleConfig:
    root_dir: str = ""
    batch_size: int = 1
    height: int = 256
    width: int = 256
    load_preprocessed: bool = False
    train_num_rays: int = -1
    train_views: int = 3
    train_split: str = "train"
    val_split: str = "test"
    test_split: str = "test"
    scale_radius: float = 1.0
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 0.0
    render_path: str = "circle"
    near_plane: float = 0.1  # near plane distance for rasterization
    far_plane: float = 100 # far plane distance for rasterization

class SparseRelightDatasetBase:

    def load_from_annotate(self):
        """ load all images for train/val/test from annotation """
        base_dir = os.path.dirname(os.path.dirname(self.cfg.root_dir))
        self.light_dir = os.path.join(base_dir,'light-probes')

        # loaded datas
        self.all_c2w, self.all_images, self.all_depths, self.all_fg_masks, self.all_frame_id = [], [], [], [], []

        # specific datas for val/test
        self.all_light_prob, self.all_albedo_images = [],[]
        
        # all idx
        self.i_split = {"train": [], "test": []}
        g_idx = 0

        for split in ['train','test']: 
            # load meta file
            meta_file = os.path.join(self.cfg.root_dir, f"transforms_{split}.json")
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            self.near, self.far = self.cfg.near_plane, self.cfg.far_plane
            self.focal = 0.5 * self.cfg.width / np.tan(0.5 * meta['camera_angle_x']) # scaled focal length

            # ray directions for all pixels, same for all images (same H, W, focal)
            self.directions = get_ray_directions(
                self.cfg.width, self.cfg.height, 
                self.focal, 
                [self.cfg.width//2, self.cfg.height//2],
                ) # (h, w, 3)           
            self.intrinsic = np.array(
                [
                    [self.focal, 0.0, self.cfg.width//2, 0.0],
                    [0.0, self.focal, self.cfg.height//2, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            
            img_wh = [self.cfg.width, self.cfg.height]
            if split=='train' or (split=='test' and self.cfg.render_path == "novel_view"):
                num_frames = len(meta['frames'])
                if split=="train":      # downsample the views if necessary
                    if self.cfg.train_views>1:
                        sparsity = (num_frames-1)//(self.cfg.train_views-1)
                        print(f"sparsity: {sparsity},num_frames= {num_frames}")
                        meta['frames'] = meta['frames'][::sparsity]
                    else:
                        meta['frames'] = meta['frames'][num_frames//2:num_frames//2+1]
                for i, frame in enumerate(meta['frames']):
                    c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                    self.all_c2w.append(c2w)

                    # load rgb and mask
                    img_path = os.path.join(self.cfg.root_dir, f"{frame['file_path']}.png")
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    self.all_fg_masks.append(img[..., -1:]) # (h, w,1)
                    self.all_images.append(img[...,:3])

                    # load depth
                    depth_path = os.path.join(os.path.dirname(img_path),"pred_depth.png")
                    depth = Image.open(depth_path)
                    depth = depth.resize(img_wh, Image.BICUBIC)
                    depth = TF.to_tensor(depth).permute(1, 2, 0) # (1, h, w) => (h, w, 1)
                    self.all_depths.append(depth)
                    

                    frame_id = int(re.search(r"\d+", f"{frame['file_path']}.png").group())
                    self.all_frame_id.append(frame_id) # save like 0,1,2, .....

                    # Null value for light probes and albedos
                    self.all_light_prob.append("")
                    if split=="train":
                        self.all_albedo_images.append(torch.zeros_like(img[...,:3]))
                    else:
                        ## load albedo
                        albdeo_path = os.path.join(self.cfg.root_dir, f"{frame['file_path'][:-4]}albedo.png")
                        albedo_img = Image.open(albdeo_path)
                        albedo_img = albedo_img.resize(img_wh, Image.BICUBIC)
                        albedo_img = TF.to_tensor(albedo_img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                        self.all_albedo_images.append(albedo_img[...,:3])
                    
                    self.i_split[split].append(g_idx)
                    g_idx +=1
            else: # load for relight dataset
                light_probs = [ fname.split(".")[0] for fname in os.listdir(os.path.join(self.light_dir,split))]
                frame_list = meta['frames']
               
                for i, frame in enumerate(frame_list): # align with NeRFactor test set
                    c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                    for light_prob in light_probs:
                        ## load test specific light probe and albedo information
                        self.all_light_prob.append(os.path.join(self.light_dir,'test',f"{light_prob}.hdr"))
                        
                        self.all_c2w.append(c2w)
                        # load rgb and mask
                        img_path = os.path.join(self.cfg.root_dir, f"{frame['file_path']}_{light_prob}.png")
                        img = Image.open(img_path)
                        img = img.resize(img_wh, Image.BICUBIC)
                        img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                        self.all_fg_masks.append(img[..., -1:]) # (h, w,1)
                        self.all_images.append(img[...,:3])

                        # load depth
                        depth_path = os.path.join(os.path.dirname(img_path),"pred_depth.png")
                        depth = Image.open(depth_path)
                        depth = depth.resize(img_wh, Image.BICUBIC)
                        depth = TF.to_tensor(depth).permute(1, 2, 0) # (1, h, w) => (h, w, 1)
                        self.all_depths.append(depth)

                        frame_id = int(re.search(r"\d+", f"{frame['file_path']}.png").group())
                        self.all_frame_id.append(frame_id) # save like 0

                        ## load albedo
                        albdeo_path = os.path.join(self.cfg.root_dir, f"{frame['file_path'][:-4]}albedo.png")
                        albedo_img = Image.open(albdeo_path)
                        albedo_img = albedo_img.resize(img_wh, Image.BICUBIC)
                        albedo_img = TF.to_tensor(albedo_img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                        self.all_albedo_images.append(albedo_img[...,:3])


                        

                        self.i_split[split].append(g_idx)
                        g_idx +=1

        # convert all features to numpy as unified interface
        self.all_images, self.all_fg_masks, self.all_c2w, self.all_albedo_images, self.directions, self.all_depths= (
            torch.stack(self.all_images).numpy(),
            torch.stack(self.all_fg_masks).numpy(), 
            torch.stack(self.all_c2w).numpy(),
            torch.stack(self.all_albedo_images).numpy(),
            self.directions.numpy(),
            torch.stack(self.all_depths).numpy(),
        )
    
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SparseRelightDataModuleConfig = cfg

        processed_npy = os.path.join(self.cfg.root_dir, "processed.npy")
        # load data from preprocessed
        if self.cfg.load_preprocessed and os.path.exists(processed_npy):
            np_data = np.load(processed_npy,allow_pickle=True).item()
            # select, convert to torch on device
            self.all_c2w, self.all_images, self.all_fg_masks, self.intrinsic, \
                self.directions, self.all_albedo_images, self.all_light_prob, self.i_split, self.all_frame_id,self.all_depths = (
                np_data['all_c2w'], 
                np_data['all_images'], 
                np_data['all_fg_masks'], 
                np_data['intrinsic'],
                np_data['directions'], 
                np_data['all_albedo_images'],
                np_data['all_light_prob'],
                np_data['i_split'],
                np_data['all_frame_id'],
                np_data['all_depths']
            )
        
        # load data from SparseRelight data annotations 
        else:
            self.load_from_annotate()
            save_dict = {
                "all_images":self.all_images, 
                "all_fg_masks":self.all_fg_masks,
                "all_c2w":self.all_c2w,
                "all_albedo_images":self.all_albedo_images,
                "intrinsic":self.intrinsic,
                "directions":self.directions,
                "i_split":self.i_split,
                "all_light_prob":self.all_light_prob,
                "all_frame_id":self.all_frame_id,
                "all_depths":self.all_depths
            }
            # save data for fast loading next time
            np.save(processed_npy,save_dict)


        # modify light probes for training images
        if split =="train":
            self.all_light_prob = []
            rand_light_dir = os.path.join(os.path.dirname(os.path.dirname(self.cfg.root_dir)),'light-probes',"random")
            for hdr_fname in os.listdir(rand_light_dir):
                self.all_light_prob.append(os.path.join(rand_light_dir,hdr_fname))
            

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
            print("[INFO] num of train views: ", len(self.i_split['train']))
            print("[INFO] train view ids = ", np.array(self.all_frame_id)[self.i_split['train']].tolist())        

        # select, convert to torch on device
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_albedo_images, self.directions, self.intrinsic, self.all_depths = (
            self.to_cuda_tensor(self.all_c2w[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_images[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_fg_masks[self.i_split[self.split]]), 
            self.to_cuda_tensor(self.all_albedo_images[self.i_split[self.split]]),
            self.to_cuda_tensor(self.directions),
            self.to_cuda_tensor(self.intrinsic),
            self.to_cuda_tensor(self.all_depths[self.i_split[self.split]])
        )

    def to_cuda_tensor(self, np_array):
        '''convert to torch on device'''
        return torch.from_numpy(np_array).float().to(self.rank)


    def get_all_images(self):
        return self.all_images

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


class SparseRelightDataset(Dataset, SparseRelightDatasetBase):
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
        c2w = self.all_c2w[index][None]
        light_positions = camera_positions = c2w[..., :3, -1]
        directions: Float[Tensor, "B H W C"] = self.directions[None]
        intrinsic: Float[Tensor, "B 4 4"] = self.intrinsic[None]        
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb: Float[Tensor, "B H W 3"] = self.all_images[index][None]
        albedo: Float[Tensor, "B H W 3"] = self.all_albedo_images[index][None]
        depth: Float[Tensor, "B H W 1"] = self.all_depths[index][None]
        mask: Float[Tensor, "B H W 1"] = self.all_fg_masks[index][None]

        # get projection matrix for OpengGL format for nvdiffrast rendering
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_intrinsic(
            intrinsic, self.cfg.height, self.cfg.width, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # calc elevation, azimuth and camera_distance for zero-123
        x,y,z = camera_positions[:,0],camera_positions[:,1],camera_positions[:,2]
        r = torch.sqrt(x**2 + y**2) # camera distance
        azimuth = torch.atan2(y,x)/np.pi*180 # in degree
        elevation = torch.atan2(z,r)/np.pi*180  # in degree
        
        lgt_index = self.i_split[self.split][index]
        env_light_file = self.all_light_prob[lgt_index] #neuspir_light._load_env_hdr(self.all_light_prob[lgt_index])

        batch = {
            "index":index, #idx of training images
            "rays_o": rays_o[0],
            "rays_d": rays_d[0],
            "mvp_mtx": mvp_mtx[0],
            "c2w":c2w[0],
            "camera_positions": camera_positions[0],
            "light_positions": light_positions[0],
            "elevation": elevation[0],
            "azimuth": azimuth[0],
            "camera_distances": torch.sqrt(x**2 + y**2 + z**2)[0],
            "rgb": rgb[0],
            "albedo":albedo[0],
            "mask": mask.bool()[0],
            "height":self.cfg.height,
            "width":self.cfg.width,
            "depth":depth,
            "env_light_file":env_light_file,
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


class SparseRelightIterableDataset(IterableDataset, SparseRelightDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]
        # prepare batch data here
        c2w = self.all_c2w[idx][None]
        light_positions = camera_positions = c2w[..., :3, -1]
        directions: Float[Tensor, "B H W C"] = self.directions[None]
        intrinsic: Float[Tensor, "B 4 4"] = self.intrinsic[None]        
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb: Float[Tensor, "B H W 3"] = self.all_images[idx][None]
        depth: Float[Tensor, "B H W 1"] = self.all_depths[idx][None]
        mask: Float[Tensor, "B H W 1"] = self.all_fg_masks[idx][None]
        # sample training rays on full image
        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.cfg.height * self.cfg.width
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
            intrinsic, self.cfg.height, self.cfg.width, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # calc elevation, azimuth and camera_distance for zero-123
        x,y,z = camera_positions[:,0],camera_positions[:,1],camera_positions[:,2]
        r = torch.sqrt(x**2 + y**2) # camera distance
        azimuth = torch.atan2(y,x)/np.pi*180 # in degree
        elevation = torch.atan2(z,r)/np.pi*180  # in degree
        
        rand_idx = np.random.choice(len(self.all_light_prob),1)
        env_light_file = self.all_light_prob[rand_idx[0]] #neuspir_light._load_env_hdr(self.all_light_prob[rand_idx[0]])
        
        
        batch = {
            "index":idx, #idx of training images
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "c2w":c2w,
            "camera_positions": camera_positions,
            "light_positions": light_positions,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": torch.sqrt(x**2 + y**2 + z**2),
            "rgb": rgb,
            "mask": mask.bool(),
            "height":self.cfg.height,
            "width":self.cfg.width,
            "depth":depth,
            "env_light_file":env_light_file
        }

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))
        # self.idx = (self.idx + 1) % len(self.all_images)

        return batch


@register("sparserelight-datamodule")
class SparseRelightDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SparseRelightDataModuleConfig, cfg)

    def setup(self, stage=None):
        print(f"[INFO] setup datamodule for stage: {stage}")
        if stage in [None, "fit","test","validate","predict"]:
            self.train_dataset = SparseRelightIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SparseRelightDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = SparseRelightDataset(self.cfg, self.cfg.test_split)

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
