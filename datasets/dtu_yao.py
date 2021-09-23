from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random
import torch.nn.functional as F
import torch
from torchvision import transforms


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train = False, small_image=True):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        if small_image:
            self.img_wh = (640, 512)
        else:
            self.img_wh = (1280, 1024)
        self.small_image = small_image
        self.robust_train = robust_train
        

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "Cameras_1/pair.txt"
            
            with open(os.path.join(self.datapath, pair_file)) as f:
                self.num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode=='train':
            img = self.color_augment(img)
        # scale 0~255 to -1~1
        np_img = 2*np.array(img, dtype=np.float32) / 255. - 1
        h, w, _ = np_img.shape
        if not self.small_image:
            target_h, target_w = self.img_wh[1], self.img_wh[0]
            start_h, start_w = (h - target_h)//2, (w - target_w)//2
            np_img = np_img[start_h: start_h + target_h, start_w: start_w + target_w]
        h, w, _ = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR), 
            "stage_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms
    

    def prepare_img(self, hr_img):
        #downsample
        h, w = hr_img.shape
        if self.small_image:
            # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
            hr_img = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            #crop
            h, w = hr_img.shape
        target_h, target_w = self.img_wh[1], self.img_wh[0]
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        return np_img.astype(np.bool_)
        
    def read_view_weight(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = cv2.resize(np_img, self.img_wh, interpolation=cv2.INTER_NEAREST)
        h, w = np_img.shape
        np_img = (np_img > 10).astype(np.float32)
        
        np_img_ms = {
            "stage_3": np.expand_dims(cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),2), 
            "stage_2": np.expand_dims(cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),2),
            "stage_1": np.expand_dims(cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),2),
            "stage_0": np.expand_dims(np_img,2)
        }
        
        
        return np_img_ms

    def read_depth_mask(self, filename, mask_filename, depth_min, depth_max, scale):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_hr = np.squeeze(depth_hr,2)
        depth_lr = self.prepare_img(depth_hr)
        
        mask = self.read_mask(mask_filename)
        mask = mask & (depth_lr>=depth_min) & (depth_lr<=depth_max)
        mask = mask.astype(np.float32)
        # if not self.small_image:
        #     depth_hr = cv2.resize(depth_hr, self.img_wh, interpolation=cv2.INTER_NEAREST)
        #     mask = cv2.resize(mask, self.img_wh, interpolation=cv2.INTER_NEAREST)

        h, w = depth_lr.shape
        depth_lr_ms = {}
        # normal_ms = {}
        mask_ms = {}

        for i in range(self.stages):
            depth_cur = cv2.resize(depth_lr, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)

            depth_lr_ms[f"stage_{i}"] = depth_cur
            mask_ms[f"stage_{i}"] = mask_cur
            
        # intrinsics[:2,:] *= 0.5

        return depth_lr_ms, mask_ms


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        
        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        mask = None
        depth = None
        depth_min = None
        depth_max = None
        
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []


        # trans from source to ref
        relative_extrinsic = []

        for i, vid in enumerate(view_ids):
            if self.small_image:
                img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/train/{:0>8}_cam.txt').format(vid)
            else:
                img_filename = os.path.join(self.datapath,
                                            'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/{:0>8}_cam.txt').format(vid)

            mask_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            extrinsics[:3,3] *= scale
            
            if i==0:
                ref_extrinsics = extrinsics
            if i>0:
                relative_extrinsic.append(np.matmul(ref_extrinsics, np.linalg.inv(extrinsics)))
            if self.small_image:
                intrinsics[0] *= 4
                intrinsics[1] *= 4
            else:
                # intrinsics[0] *= self.img_wh[0]/1600
                # intrinsics[1] *= self.img_wh[1]/1200
                original_w = 1600
                original_h = 1200
                target_h, target_w = self.img_wh[1], self.img_wh[0]
                intrinsics[0,2] -= 0.5*(original_w-target_w)
                intrinsics[1,2] -= 0.5*(original_h-target_h)
            
          
            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 0.125
            # intrinsics_inv_3 = np.linalg.inv(intrinsics)
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            intrinsics_inv_2 = np.linalg.inv(intrinsics)
            # intrinsics_2 = intrinsics
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            # intrinsics_inv_1 = np.linalg.inv(intrinsics)
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            # intrinsics_inv_0 = np.linalg.inv(intrinsics)
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:  # reference view
                depth_min = depth_min_ * scale
                depth_max = depth_max_ * scale
                
                depth, mask = self.read_depth_mask(depth_filename, mask_filename, depth_min, depth_max, scale)
                intrinsics_inv = {}
                # intrinsics_inv['stage_0'] = intrinsics_inv_0
                # intrinsics_inv['stage_1'] = intrinsics_inv_1
                intrinsics_inv['stage_2'] = intrinsics_inv_2
                # intrinsics_inv['stage_3'] = intrinsics_inv_3
                # intrinsics = {'stage_2': intrinsics_2}
                
                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'],2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2,0,1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'],2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2,0,1])

                
        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3

        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        
        proj={}
        proj['stage_3']=proj_matrices_3
        proj['stage_2']=proj_matrices_2
        proj['stage_1']=proj_matrices_1
        proj['stage_0']=proj_matrices_0

        relative_extrinsic = np.stack(relative_extrinsic)

        
        # data is numpy array
        return {"imgs": imgs,                   # [N, 3, H, W]
                "proj_matrices": proj,          # [N,4,4]
                "relative_extrinsic": relative_extrinsic, #[N-1,4,4]
                "depth": depth,                 # [1, H, W]
                # "view_weights": view_weights,   # [N-1,1,H,W]
                # "intrinsics": intrinsics,
                "intrinsics_inv": intrinsics_inv,#[3, 3]
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "mask": mask}                   # [1, H, W]

