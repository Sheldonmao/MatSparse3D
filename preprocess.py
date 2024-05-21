import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from rembg import remove

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device

        from dpt import DPTDepthModel

        if task == 'depth':
            path = 'load/omnidata/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else: # normal
            path = 'load/omnidata/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)


    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, help="path to scene directions for images (png, jpeg, etc.)")
    parser.add_argument('--use_normal', action="store_true")
    opt = parser.parse_args()

    dpt_depth_model = DPT(task='depth')
    if opt.use_normal:
        dpt_normal_model = DPT(task='normal')
    for frame_dir in tqdm(os.listdir(opt.scene_dir)): # loop over list of all dirs
        # check for valid image directions
        if frame_dir.startswith("test") or frame_dir.startswith("val") or frame_dir.startswith("train"):
            print(f"processing frame:{frame_dir}")
            img_path = os.path.join(opt.scene_dir, frame_dir,"rgba.png")
            out_dir = os.path.dirname(img_path)
            # out_rgba = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_rgba.png')
            out_depth = os.path.join(out_dir, os.path.basename(img_path).replace("rgba.png", 'pred_depth.png'))
            out_normal = os.path.join(out_dir, os.path.basename(img_path).replace("rgba.png", 'pred_normal.png'))
            # out_caption = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_caption.txt')

            # load image
            print(f'[INFO] loading image...')
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] == 4:
                # use current background
                print(f'[INFO] use current mask in rgba image...')
                mask = image[:, :, 3]>0
                if image.dtype == np.uint16:
                    image = (image.astype(np.float32) / 65535.0 *255).astype(np.uint8) # in BGRA uint8
                carved_image = image[:,:,[2,1,0,3]] # in RGBA uint8 
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) # in RGB uint8

            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # carve background
                print(f'[INFO] background removal...')
                carved_image = remove(image) # [H, W, 4]
                mask = carved_image[..., -1] > 0

            # predict depth
            print(f'[INFO] depth estimation...') 
            depth = dpt_depth_model(image)[0]
            depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
            depth[~mask] = 0
            depth = (depth * 255).astype(np.uint8)
            
            # write output
            cv2.imwrite(out_depth, depth)
            

            # predict normal
            if opt.use_normal:
                print(f'[INFO] normal estimation...')
                normal = dpt_normal_model(image)[0]
                normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
                normal[~mask] = 0
                cv2.imwrite(out_normal, normal)
