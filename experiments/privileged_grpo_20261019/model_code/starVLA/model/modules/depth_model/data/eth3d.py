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
        self.dataset_name = 'eth3d'
        splits = open(self.cfg.split_path, 'r').readlines()
        self.rgb_files = []
        self.depth_files = []
        for split in splits:
            rgb_file, depth_file = split.strip().split(' ')
            self.rgb_files.append(join(self.cfg.data_root, rgb_file))
            self.depth_files.append(join(self.cfg.data_root, depth_file))

    def read_depth(self, index, depth=None):
        depth_path = self.depth_files[index]
        with open(depth_path, "rb") as file:
            binary_data = file.read()

        # Convert the binary data to a numpy array of 32-bit floats
        depth = np.frombuffer(binary_data, dtype=np.float32).copy()

        HEIGHT, WIDTH = 4032, 6048
        depth = depth.reshape((HEIGHT, WIDTH))

        valid_mask = np.logical_and(
            depth > 0.01, ~np.isnan(depth)) & (~np.isinf(depth))
        if valid_mask.sum() == 0:
            Log.warn('No valid mask in the depth map of {}'.format(
                self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[valid_mask].max()

        resized_depth = cv2.resize(depth, (2048, 1360), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(valid_mask.astype(np.uint8), (2048, 1360), interpolation=cv2.INTER_NEAREST)
        return resized_depth, resized_mask

    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        start_time = time.time()
        rgb = cv2.imread(img_path)
        end_time = time.time()
        if end_time - start_time > 1:
            Log.warn(f'Long time to read {img_path}: {end_time - start_time}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.asarray(rgb / 255.).astype(np.float32)
        resized_rgb = cv2.resize(rgb, (2048, 1360), interpolation=cv2.INTER_AREA)
        return resized_rgb

    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:]).replace(".JPG", ".png")
