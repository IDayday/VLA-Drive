from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
import h5py
from PIL import Image
import torchvision.transforms.functional as TF
from scipy import ndimage

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'diode'
        splits = open(self.cfg.split_path, 'r').readlines()
        self.rgb_files = []
        self.depth_files = []
        self.mask_files = []
        for split in splits:
            rgb_file, depth_file, mask_file = split.strip().split(' ')
            self.rgb_files.append(join(self.cfg.data_root, rgb_file))
            self.depth_files.append(join(self.cfg.data_root, depth_file))
            self.mask_files.append(join(self.cfg.data_root, mask_file))

    def read_depth(self, index, depth=None):
        depth = np.load(self.depth_files[index])[:, :, 0]
        valid_mask = np.load(self.mask_files[index])
        valid_mask = valid_mask == 1
        valid_mask = (
            valid_mask & (depth >= 0.6) & (depth <= 350) & (~np.isnan(depth)) & (~np.isinf(depth)))

        dx = ndimage.sobel(depth, 0)  # horizontal derivative
        dy = ndimage.sobel(depth, 1)  # vertical derivative
        grad = np.abs(dx) + np.abs(dy)
        valid_mask[grad>0.3] = 0
        depth[valid_mask == 0] = 0

        return depth, valid_mask.astype(np.uint8)

    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])