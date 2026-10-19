from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
import os
import json
from ppd.utils.logger import Log
import OpenEXR

def read_exr(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = exr_file.header()["channels"]
    data = {}
    for channel in channels:
        raw_bytes = exr_file.channel(channel)
        data[channel] = np.frombuffer(raw_bytes, dtype=np.float32).reshape(height, width)
    return data

class Dataset(BaseDataset):

    def build_metas(self):
        self.dataset_name = 'urbansyn'
        self.rgb_files = []
        self.depth_files = []
        folder_path = os.path.join(self.cfg.data_root, 'rgb')
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"RGB data folder does not exist: {folder_path}")
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.png'):
                rgb_path = os.path.join(folder_path, file_name)
                dpt_path = rgb_path.replace('rgb/rgb_', 'depth/depth_').replace('.png', '.exr')

                if os.path.isfile(rgb_path) and os.path.isfile(dpt_path):
                    self.rgb_files.append(rgb_path)
                    self.depth_files.append(dpt_path)

        assert len(self.rgb_files) == len(self.depth_files)


    def read_depth(self, index, depth=None):
        data = read_exr(self.depth_files[index])
        depth = (data["Y"] * 1e5).astype(np.float32)

        min_val, max_val = 0.1, 80
        tiankong_mask = depth > 200.
        valid_mask = np.logical_and(
            depth > 0.1, ~np.isnan(depth)) & (~np.isinf(depth))
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

