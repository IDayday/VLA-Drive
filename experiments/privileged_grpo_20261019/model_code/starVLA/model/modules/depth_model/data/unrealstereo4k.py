from ppd.data.depth_estimation import Dataset as BaseDataset
from ppd.data.depth_estimation import *
import os
import json
from ppd.utils.logger import Log


def load_in_extrinsic(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
        lines = [[float(x) for x in line] for line in lines]

    intrinsic = np.array(lines[0]).reshape(3,3)
    extrinsic_ = np.array(lines[1]).reshape(3,4)
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = extrinsic_[:3,:3]
    extrinsic[:3,3] = extrinsic_[:3,3]
    return intrinsic, extrinsic

def get_baseline_focal(left_path, right_path):
    intrinsic0, extrinsic0 = load_in_extrinsic(left_path)
    intrinsic1, extrinsic1 = load_in_extrinsic(right_path)
    cam_extrinsic = extrinsic1 @ np.linalg.inv(extrinsic0) # ref:https://github.com/fabiotosi92/SMD-Nets/issues/8
    baseline = abs(cam_extrinsic[0, 3])
    return baseline, (intrinsic0[0,0] + intrinsic0[1,1]) / 2

def get_depth(disp, focal, baseline):
    """
    get depth from reference frame disparity and camera intrinsics
    """
    return focal * baseline / disp

class Dataset(BaseDataset):

    def build_metas(self):
        self.focal = 1920
        self.dataset_name = 'unrealstereo4k'
        folder_names = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007']
        self.rgb_files = []
        self.depth_files = []
        for folder_name in folder_names:
            folder_path = os.path.join(self.cfg.data_root, folder_name, 'Image0')
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.png'):
                    rgb_path = os.path.join(folder_path, file_name)
                    dpt_path = rgb_path.replace('Image0', 'Disp0').replace('.png', '.npy')

                    if os.path.isfile(rgb_path) and os.path.isfile(dpt_path):
                        self.rgb_files.append(rgb_path)
                        self.depth_files.append(dpt_path)

        assert len(self.rgb_files) == len(self.depth_files)


    def read_depth(self, index, depth=None):
        disp = np.load(self.depth_files[index])
        # set nan to 0
        invalid_mask = np.isnan(disp) | (disp == 0)
        disp[invalid_mask] = 1e-4
        left_extrin_path = self.depth_files[index].replace('Disp0', 'Extrinsics0').replace('.npy', '.txt')
        right_extrin_path = self.depth_files[index].replace('Disp0', 'Extrinsics1').replace('.npy', '.txt')
        baseline, focal = get_baseline_focal(left_extrin_path, right_extrin_path)
        depth = get_depth(disp, focal, baseline).astype(np.float32)
        
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


