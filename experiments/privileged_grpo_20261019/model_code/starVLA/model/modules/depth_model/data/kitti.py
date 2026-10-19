from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
import h5py
from PIL import Image
import torchvision.transforms.functional as TF

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'kitti'
        splits = open(self.cfg.split_path, 'r').readlines()
        self.rgb_files = []
        self.depth_files = []
        for split in splits:
            rgb_file, depth_file, _ = split.strip().split(' ')
            if depth_file != 'None':
                self.rgb_files.append(join(self.cfg.data_root, rgb_file))
                self.depth_files.append(join(self.cfg.data_root, depth_file))

    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        start_time = time.time()
        rgb = cv2.imread(img_path)
        end_time = time.time()
        if end_time - start_time > 1:
            Log.warn(f'Long time to read {img_path}: {end_time - start_time}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.asarray(rgb / 255.).astype(np.float32)

        ######## benchmark crop
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = rgb.shape[:2]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)

        rgb = rgb[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
                :
            ]
        return rgb
            
    def read_depth(self, index):
        depth = imageio.imread(self.depth_files[index]) / 256.

        ######## benchmark crop
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = depth.shape
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)

        depth = depth[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]

        valid_mask = np.logical_and(
            depth > 0.1, ~np.isnan(depth)) & (~np.isinf(depth))
        valid_mask = np.logical_and(valid_mask, depth < 80.)
        if valid_mask.sum() == 0:
            Log.warn('No valid mask in the depth map of {}'.format(
                self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[valid_mask].max()


        ####### benchmark crop
        eval_mask = np.zeros_like(valid_mask, dtype=bool)
        gt_height, gt_width = eval_mask.shape
        eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
        
        valid_mask = np.logical_and(valid_mask, eval_mask)
        ####### benchmark crop
        depth[valid_mask == 0] = 0
        return depth, valid_mask.astype(np.uint8)
    
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])