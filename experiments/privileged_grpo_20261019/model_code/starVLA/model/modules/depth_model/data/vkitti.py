from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
import os
import json
from ppd.utils.logger import Log


class Dataset(BaseDataset):
    def build_metas(self):
        self.dataset_name = 'vkitti'
        lines = open(self.cfg.split_path).readlines()
        self.rgb_files = []
        self.depth_files = []
        for line in lines:
            rgb_path, dpt_path = line.strip().split(' ')
            self.rgb_files.append(os.path.join(self.cfg.data_root, rgb_path))
            self.depth_files.append(os.path.join(self.cfg.data_root, dpt_path))
        assert len(self.rgb_files) == len(self.depth_files)

    def read_depth(self, index, depth=None):
        depth = cv2.imread(self.depth_files[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0 
        min_val, max_val = 0.1, 80
        tiankong_mask = depth > 200.
        valid_mask = np.logical_and(
            depth > min_val, ~np.isnan(depth)) & (~np.isinf(depth))
        valid_mask = np.logical_and(valid_mask, depth < max_val)
        if valid_mask.sum() == 0:
            Log.warn('No valid mask in the depth map of {}'.format(
                self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[valid_mask].max()
            
        depth = np.clip(depth, min_val, max_val)
        depth[tiankong_mask] = depth.max() + 1.0
        valid_mask = np.logical_or(valid_mask, tiankong_mask)
        return depth, valid_mask.astype(np.uint8)
