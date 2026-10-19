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
        self.dataset_name = 'scannet'
        splits = open(self.cfg.split_path, 'r').readlines()
        self.rgb_files = []
        self.depth_files = []
        for split in splits:
            rgb_file, depth_file = split.strip().split(' ')
            self.rgb_files.append(join(self.cfg.data_root, rgb_file))
            self.depth_files.append(join(self.cfg.data_root, depth_file))

    def read_depth(self, index, depth=None):
        depth = (np.asarray(imageio.imread(self.depth_files[index])) / 1000.).astype(np.float32)
        valid_mask = np.logical_and(
            depth > 0.01, ~np.isnan(depth)) & (~np.isinf(depth))
        valid_mask = np.logical_and(valid_mask, depth < 10.)
        if valid_mask.sum() == 0:
            Log.warn('No valid mask in the depth map of {}'.format(
                self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[valid_mask].max()

        return depth, valid_mask.astype(np.uint8)
    

    def read_rgb_name(self, index):
        rgb_name = '__'.join(self.rgb_files[index].split('/')[-3:])
        return rgb_name.replace(".jpg", ".png")
