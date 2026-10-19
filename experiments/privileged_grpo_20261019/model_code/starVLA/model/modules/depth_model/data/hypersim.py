from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
import os
import json
from ppd.utils.logger import Log


class Dataset(BaseDataset):

    def build_metas(self):
        self.dataset_name = 'hypersim'
        splits = json.load(open(self.cfg.split_path))
        key_rgb = f"{self.cfg.split}_rgb_paths"
        rgb_paths = splits[key_rgb]
        key_dpt = f"{self.cfg.split}_dpt_paths"
        dpt_paths = splits[key_dpt]
        assert len(rgb_paths) == len(dpt_paths)

        self.rgb_files = [os.path.join(self.cfg.data_root, rgb_path)
                          for rgb_path in rgb_paths]
        self.depth_files = [os.path.join(
            self.cfg.data_root, dpt_path) for dpt_path in dpt_paths]


    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])

    def read_depth(self, index, depth=None):
        if not hasattr(self, 'depth_files'):
            return None, None
        Log.debug(index, self.depth_files[index])
        start_time = time.time()
        depth_path = self.depth_files[index]
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                               cv2.IMREAD_ANYDEPTH) / 1000.
        end_time = time.time()
        if end_time - start_time > 1:
            Log.warn(
                f'Long time to read {self.depth_files[index]}: {end_time - start_time}')
        valid_mask = np.logical_and(
            (depth > 0.1) & (depth < 65.0), ~np.isnan(depth)) & (~np.isinf(depth))
        if valid_mask.sum() == 0:
            Log.warn('No valid mask in the depth map of {}'.format(
                self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[valid_mask].max()
        return depth, valid_mask.astype(np.uint8)

