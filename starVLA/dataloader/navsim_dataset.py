import copy
import random
import time
from typing import Callable, Iterable, List, Optional, Sequence

from torch.utils.data import Dataset
import pickle
import json
import os
from PIL import Image
import torch

from starVLA.cache.navsim_feature_cache import (
    NavsimFeatureCacheReader,
    parse_components,
)
import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
import torchvision.transforms as transforms

from nuplan.common.actor_state.state_representation import StateSE2
from typing import List

import numpy as np
import numpy.typing as npt

# we add an idx to the original func.
# from nuplan.common.geometry.convert import absolute_to_relative_poses

from nuplan.common.geometry.convert import pose_from_matrix, matrix_from_pose

def absolute_to_relative_poses(absolute_poses: List[StateSE2], idx=0) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from absolute to relative coordinates with the first pose being the origin
    :param absolute_poses: list of absolute poses to convert
    :return: list of converted relative poses
    """
    absolute_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[idx])
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms]

    return relative_poses

# for rgb data loading
from starVLA.model.modules.video_model.videox_fun.data.utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)

# for gs data (gs_model removed; MEAN/STD only used when load_3d_data=True)
# from starVLA.model.modules.gs_model.storm.dataset.constants import MEAN, STD
MEAN = STD = None

import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

def collate_fn(batch):
    """Simple collate that returns the raw list of samples.

    Training code in this repo expects a batch to be a list of per-sample
    dicts (further collated by higher-level collators). Keep the default
    behaviour minimal.
    """
    return batch


def to_float_tensor(d):
    if isinstance(d, dict):
        return {k: to_float_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_float_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.float()
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d


def gt_2_ego(gt_xy, velo=None, acc=None, k_ahead=1, min_step=1):
    # theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy).to(torch.float32)            # shape [T, 2]

    T = gt.shape[0]
    t0 = 3

    # ---- 平移到原点 --------------------------------------------------------
    origin = gt[t0]             # p_3
    rel_gt = gt - origin        # 平移: 使 t0 -> (0,0)

    # 用多帧位移稳健估计速度朝向
    k = k_ahead
    v = gt[t0 + k] - gt[t0]  # v = p_{3+k} - p_3
    if torch.linalg.norm(v) < min_step:
        # 找到第一个位移够大的帧
        for j in range(t0+1, T):
            v = gt[j] - gt[t0]
            if torch.linalg.norm(v) >= min_step:
                break

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    heading = v
    theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # 逆时针旋转
                    [ torch.sin(theta),  torch.cos(theta)]]).to(torch.float32)  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()

    if velo is not None:
        velo = torch.from_numpy(velo)
        velo_local = torch.matmul(velo, R)
        velo_local[:, [0,1]] = velo_local[:, [1, 0]]
        velo_local[:, 0] = -velo_local[:, 0]
        velo = velo_local.numpy()
    
    if acc is not None:
        acc = torch.from_numpy(acc)
        acc_local = torch.matmul(acc, R)
        acc_local[:, [0,1]] = acc_local[:, [1, 0]]
        acc_local[:, 0] = -acc_local[:, 0]
        acc = acc_local.numpy()

    return gt, velo, acc

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

q01 = np.array([
        -0.01789146974507183,
        -0.19088272509455573,
        -0.1892357842470911
      ], dtype=np.float64)
q99 = np.array([
        6.199554522088146,
        0.24262804072441968,
        0.1804889553518122
      ], dtype=np.float64)

x_mean = 10.172484
x_std = 8.805105

y_mean = 0.360762
y_std = 2.277741

import cv2, numpy as np


def resolve_navsim_data_path(file_name):
    """Map absolute paths embedded during preprocessing to this runtime root."""
    if file_name is None:
        return None

    file_path = os.fspath(file_name)
    runtime_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    marker = f"{os.sep}navsim_dataset_raw{os.sep}"
    if runtime_root and marker in file_path:
        relative_path = file_path.split(marker, 1)[1]
        trainval_prefix = os.path.join("sensor_blobs", "trainval") + os.sep
        trainval_sensor_root = os.environ.get("NAVSIM_TRAINVAL_SENSOR_ROOT", "")
        if trainval_sensor_root and relative_path.startswith(trainval_prefix):
            return os.path.join(
                trainval_sensor_root,
                relative_path[len(trainval_prefix):],
            )
        return os.path.join(runtime_root, relative_path)
    return file_path


def bev_rgb_to_occ(bev_rgb):  # (500,500,3) uint8
    gray = cv2.cvtColor(bev_rgb[:496, :496], cv2.COLOR_RGB2GRAY)
    occ = (gray < 255).astype(np.uint8)          # 白底≈255，框/线更暗
    # occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return occ  # (500,500) {0,1}


class NavSimDataset(Dataset):
    """Wrap a list of navsim records (datalist_path) into a PyTorch Dataset.

    The dataset accepts a `datalist_path` which is an iterable of raw records. Each
    raw record will be passed through an optional `meta_processor` which is a
    callable that converts the raw record into the training sample format used
    across this project. The expected output of `meta_processor` is a dict
    containing at least a `conversations` key (list of message dicts). Typical
    additional keys are `images`, `videos`, `data_path`, etc.

    Args:
        datalist_path: Iterable of raw records.
        meta_processor: Optional[callable(raw_record) -> dict]. If None the
            raw record is assumed to already be in the sample format.
        shuffle: Whether to shuffle samples on dataset construction.
        max_samples: Optional cap on number of samples to keep (for fast
            debugging).
    """

    def __init__(
        self,
        datalist_path: Iterable[dict],
        split: str = "mini",
        video_data_cfg = None,
        gs_data_cfg = None,
        reward_data_cfg = None,
        ver_1225 = False,
        dataset_cfg = None,
        all_cfg = None,
        max_samples: Optional[int] = None,
        s2_pred_dir: Optional[str] = None,
        data_root: Optional[str] = None,
    ) -> None:
        with open(datalist_path, "rb") as f:
            raw_list = json.load(f)

        if max_samples is not None:
            raw_list = raw_list[: max_samples]

        self.raw_list = raw_list
        self.split = split
        cache_root = os.environ.get("NAVSIM_FEATURE_CACHE_ROOT", "").strip()
        cache_components = parse_components(os.environ.get("NAVSIM_CACHE_COMPONENTS"))
        cache_strict = os.environ.get("NAVSIM_CACHE_STRICT", "1") == "1"
        self.feature_cache = (
            NavsimFeatureCacheReader(
                cache_root=cache_root,
                components=cache_components,
                strict=cache_strict,
            )
            if cache_root
            else None
        )
        if self.feature_cache is not None:
            print(
                f"loading NAVSIM feature cache {cache_root} "
                f"components={','.join(cache_components)} strict={int(cache_strict)}"
            )
        agent_dino_cache_root = os.environ.get("NAVSIM_AGENT_DINO_CACHE_ROOT", "").strip()
        agent_dino_cache_strict = os.environ.get("NAVSIM_AGENT_DINO_CACHE_STRICT", "1") == "1"
        self.agent_dino_cache = (
            NavsimFeatureCacheReader(
                cache_root=agent_dino_cache_root,
                components=("agent_dino",),
                strict=agent_dino_cache_strict,
            )
            if agent_dino_cache_root
            else None
        )
        if self.agent_dino_cache is not None:
            print(
                f"loading NAVSIM agent dino cache {agent_dino_cache_root} strict={int(agent_dino_cache_strict)}"
            )
        _cfg_data_root = getattr(dataset_cfg, "data_root", None) if dataset_cfg is not None else None
        _data_root = data_root or _cfg_data_root or os.environ.get("OPENSCENE_DATA_ROOT", "")
        if self.split == "mini":
            self.base_dir = os.path.join(_data_root, "meta/mini")
        elif self.split == "mini_test" or self.split == "test":
            self.base_dir = os.path.join(_data_root, "meta/test")
        elif self.split == "train":
            self.base_dir = os.path.join(_data_root, "meta/train")
        elif self.split == "navhard_two_stage":
            self.base_dir = os.path.join(_data_root, "meta/navhard_two_stage")
        elif self.split == 'waymo_train':
            self.base_dir = os.path.join(_data_root, "waymo_video_20_4hz/waymo_train/meta")
            print("loading waymo train set")
        else:
            raise ValueError(f"Split {self.split} not supported yet in NavSimDataset")
        self.s2_pred_dir = s2_pred_dir
        
        self.video_data_cfg = video_data_cfg
        self.video_source = os.environ.get("NAVSIM_VIDEO_SOURCE", "mp4").lower()
        if self.video_source not in {"mp4", "images"}:
            raise ValueError(
                "NAVSIM_VIDEO_SOURCE must be either 'mp4' or 'images', "
                f"got {self.video_source!r}"
            )
        if self.video_data_cfg.load_2d_data:
            print(f'loading video data {self.video_data_cfg.rgb_meta_dir}')
            print(f'video source: {self.video_source}')
            self.rgb_meta_dir = os.path.join(self.video_data_cfg.rgb_meta_dir, split)
            # sample_size = (self.video_data_cfg.sample_size, self.video_data_cfg.sample_size)
            sample_size = (self.video_data_cfg.sample_size[0], self.video_data_cfg.sample_size[1])
            self.video_transforms = transforms.Compose(
                [
                    transforms.Resize(sample_size),
                    # transforms.CenterCrop(sample_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

        self.gs_data_cfg = gs_data_cfg
        if self.gs_data_cfg.load_3d_data:
            print(f'loading gs data')
            self.gs_meta_dir = os.path.join(self.gs_data_cfg.gs_meta_dir, split)

            self.gs_transformation = transforms.Compose(
                [
                    # transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                    # transforms.ToTensor(),  # to 0-1
                    transforms.Normalize(mean=MEAN, std=STD),
                ]
            )
            if self.gs_data_cfg.debug:
                self.raw_list = self.raw_list[:1]   # one sample
                # pass

        self.reward_data_cfg = reward_data_cfg
        if self.reward_data_cfg.load_reward_data:
            print(f'loading reward data')
            self.reward_meta_dir = os.path.join(self.reward_data_cfg.reward_meta_dir, split, "rnd_0")


        self.dataset_cfg = dataset_cfg
        self.w_neg_traj = dataset_cfg.w_neg_traj
        if self.w_neg_traj is not None:
            self.neg_list = os.path.join(self.w_neg_traj, split)
            self.neg_dirs = os.listdir(self.neg_list)
            # not self for now
            self.neg_dirs.remove("rnd_0")

        self.pixel_values = None
        self.text = None
        self.raw_data = None
        self.raw = None

        self.ver_1225 = ver_1225

        self.act_norm = dataset_cfg.act_norm

        self.all_cfg = all_cfg
        try:
            self.enable_image_aug = all_cfg.enable_image_aug
        except:
            self.enable_image_aug = 0
        if self.enable_image_aug:
            print('using img aug')
        self.image_aug = T.Compose([
            # 颜色增强 (最重要)
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.3,
                    contrast=0.2, 
                    saturation=0.2,
                    hue=0.03
                )
            ], p=0.8),
            
            # 随机灰度 (模拟黑白相机/夜视)
            T.RandomGrayscale(p=0.05),
            
            # 高斯模糊 (模拟失焦/运动模糊)
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.15),
            
            # 随机擦除 (模拟遮挡) - 可选
            # T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])

        print(f'loading {len(self.raw_list)} data.')

        try:
            self.w_depth = all_cfg.w_depth
            if self.w_depth:
                print(f'using depth')
        except:
            self.w_depth = 0

        try:
            self.doing_s2 = all_cfg.doing_s2
        except:
            self.doing_s2 = 0
        
        try:
            self.vit_pre = all_cfg.vit_pre
        except:
            self.vit_pre = 0

        if self.doing_s2:
            rew_path = os.path.join(self.s2_pred_dir, "train_reward.json")
            with open(rew_path, 'rb') as f:
                self.rew_dict = json.load(f)



    def __len__(self) -> int:
        return len(self.raw_list)

    def __getitem__(self, idx):
        # mirror the pattern used in other dataset wrappers: return a single
        # sample dict (no further tensorization here)
        raw = self.raw_list[idx]
        raw_dir = os.path.join(self.base_dir, raw+'.pkl')
        try:
            with open(raw_dir, "rb") as f:
                raw_data = pickle.load(f)
            self.raw_data = raw_data
            self.raw = raw
            if self.split == 'navhard_two_stage':
                print(f"Loading sample {raw}")
            if self.w_neg_traj:
                if random.random() < 0.1:
                # if True:
                    rnd = np.random.choice(self.neg_dirs)
                    jsonl_path = os.path.join(self.neg_list, rnd, raw+'.jsonl')
                    rnd_dicts = []
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            one = json.loads(line)   # one 是 list，比如 [str, float]
                            rnd_dicts.append(one)
                    new_token = rnd_dicts[0][0]
                    try:
                        new_path = os.path.join(self.base_dir, new_token+'.pkl')
                        with open(new_path, "rb") as f:
                            new_raw_data = pickle.load(f)
                        raw_data['glo_status']['global_poses'] = new_raw_data['glo_status']['global_poses']
                        print(f"Loading negative sample {new_token}")
                    except:
                        print('loading neg failed!')


        except:
            print('loading failed!')
            raw_data = self.raw_data
            raw = self.raw

        if self.doing_s2:
            pred_dir = os.path.join(self.s2_pred_dir, f"train/{raw}.npy")
            self_pred = np.load(pred_dir)
        else:
            self_pred = None

        bev = None

        cached_features = {}
        if self.feature_cache is not None:
            for component in self.feature_cache.components:
                if self.feature_cache.has_component(component):
                    payload = self.feature_cache.get(component, idx, raw)
                    if payload is not None:
                        cached_features[component] = payload
        if self.agent_dino_cache is not None and self.agent_dino_cache.has_component("agent_dino"):
            payload = self.agent_dino_cache.get("agent_dino", idx, raw)
            if payload is not None:
                cached_features["agent_dino"] = payload

        if self.vit_pre:
            bev_path = os.path.join(self.base_dir, raw+'-bev.pkl')
            with open(bev_path, "rb") as f:
                bev_data = pickle.load(f)   # 500, 500, 3

        if self.w_depth and "ppd" not in cached_features:
            depth_map_path = os.path.join(self.base_dir, raw+'.pkl-depth.pkl')
            with open(depth_map_path, "rb") as f:
                depth_map_data = pickle.load(f)

        sample = self._get_sample(
            raw_data,
            raw,
            self_pred,
            cached_features=cached_features,
        )
        if self.vit_pre:
            sample['bev'] = bev_rgb_to_occ(bev_data)    # np array unit8  hwc
        if self.w_depth and "ppd" in cached_features:
            ppd_cache = cached_features["ppd"]
            sample['depth_data'] = {
                'image': ppd_cache['image_uint8'].float().div_(255.0),
                'depth': ppd_cache['depth'].float(),
                'mask': ppd_cache['mask'].bool(),
                'semantics': ppd_cache['semantics'],
            }
        elif self.w_depth:
            views = ['cam_l0', 'cam_f0', 'cam_r0']
            depths = []
            masks = []
            for view in views:
                depth_m = depth_map_data[view]
                # resize
                depth_m = cv2.resize(depth_m, (256, 144), interpolation=cv2.INTER_NEAREST)
                depths.append(depth_m)
                masks.append(depth_m>0.1)
            depths = np.array(depths)   # [H, 3W]
            masks  = np.array(masks)    # [H, 3W]

            rgbs = sample['image']

            rgbs = [rgbs[1], rgbs[0], rgbs[2]]
            resized_rgbs = []
            for rgb in rgbs:
                # to 256 * 144
                rgb_ = rgb.resize((256, 144), resample=Image.LANCZOS)
                rgb_ = (np.array(rgb_) / 255.).astype(np.float32)
                resized_rgbs.append(rgb_)

            rgbs = np.array(resized_rgbs)   # [H, 3W]   # 0,1

            sample['depth_data'] = {}
            sample['depth_data']['image'] = torch.from_numpy(rgbs).float().permute(0, 3, 1, 2) # view, c, h, w
            sample['depth_data']['depth'] = torch.from_numpy(depths).float().unsqueeze(1)
            sample['depth_data']['mask'] = torch.from_numpy(masks).unsqueeze(1)  # bool


            # rgb, (dpt, msk) = self.read_rgb(index), self.read_depth(index)
            # if dpt is not None:
            #     self.check_shape(rgb, dpt)
            # sample = {
            #     'image': rgb,
            # }
            # if dpt is not None:
            #     sample['depth'] = dpt
            #     sample['mask'] = msk

            # sample = self.transform(sample)

        return sample

    def _get_sample(self, raw, token, self_pred=None, cached_features=None):
        cached_features = cached_features or {}
        glo_images = raw['glo_images']
        # temporarily only use one view
        views = ['cam_f0', 'cam_l0', 'cam_r0']
        images = []
        # if self.w_depth:
        #     depth_feats = []

        need_policy_images = (
            "qwen" not in cached_features
            or (self.w_depth and "ppd" not in cached_features)
        )
        if need_policy_images:
            for view in views:
                cur_image_path = glo_images[view]['image_paths'][3]
                if self.split == 'waymo_train':
                    cur_image_path = glo_images[view]['image_paths'][0]
                images.append(self._load_image(cur_image_path))
            # if self.w_depth:
            #     # img, not good
            #     # depth_image_path = cur_image_path + '-depth.jpg'
            #     # images.append(self._load_image(depth_image_path))
            #     depth_feat = cur_image_path + '-depth_feat.npy'
            #     depth_feat = np.load(depth_feat)
            #     depth_feats.append(torch.from_numpy(depth_feat))
        
        # if self.w_depth:
        #     depth_feats = torch.stack(depth_feats)
        # else:
        #     depth_feats = None

        if not self.split == 'waymo_train':
            glo_pose = raw['glo_status']['global_poses']
            # glo_velocity = raw['glo_status']['velocities']
            # glo_acceleration = raw['glo_status']['accelerations']
            command = raw['glo_status']['commands']
        else:
            cur_command = 'unknown'
            glo_pose = None


        # ego_pose, ego_velo, ego_acc = gt_2_ego(glo_pose[:12,:2], glo_velocity[:12,:2], glo_acceleration[:12,:2])
        # cur_state = np.array(ego_pose[3], ego_velo[3], ego_acc[3]).reshape(1, -1)

        # use pose only for now: always 0 in fact.
        # ego_pose, ego_velo, ego_acc = gt_2_ego(glo_pose[:12,:2])
        # ego_pose[:, 1] /= 4.5912
        # action = (ego_pose[4:]-ego_pose[3:-1]).astype(np.float32)
        action_copy = None
        if self.ver_1225==1:
            ego_pose_se2 = [StateSE2(float(x), float(y), float(yaw)) for x, y, yaw in glo_pose[:12]]
            rel_ego_pose_se2 = absolute_to_relative_poses(ego_pose_se2, 3)
            rel_ego_pose_se2 = [[pose.x, pose.y, pose.heading] for pose in rel_ego_pose_se2]
            ego_pose = np.array(rel_ego_pose_se2)
            if self.doing_s2:
                ego_pose_copy = ego_pose.copy()
                ego_pose_copy[4:12] = self_pred
            # 1225: update to 0 as x
            if not self.act_norm:
                ego_pose[:, 0] /= 4.5912
                if self.doing_s2:
                    ego_pose_copy[:, 0] /= 4.5912
            else:
                pass
            theta = ego_pose[:, 2].astype(np.float64)
            if self.doing_s2:
                theta_copy = ego_pose_copy[:, 2].astype(np.float64)
            # delta heading with wrap
            if self.split in ['train', 'mini', 'test']:
                dtheta = (theta[4:] - theta[3:4])
                dtheta = wrap_to_pi(dtheta)
                dtheta_sc = np.stack([np.sin(dtheta), np.cos(dtheta)], axis=-1).astype(np.float32)  # (H,2)
                dxy = (ego_pose[4:, :2] - ego_pose[3:4, :2]).astype(np.float32)  # (H,2)
                if self.act_norm:
                    dxy[:, 0] = (dxy[:, 0] - x_mean) / x_std
                    dxy[:, 1] = (dxy[:, 1] - y_mean) / y_std
                action = np.concatenate([dxy, dtheta_sc], axis=-1).astype(np.float32)  # (H,4)

                if self.doing_s2:
                    dtheta_copy = (theta_copy[4:] - theta_copy[3:4])
                    dtheta_copy = wrap_to_pi(dtheta_copy)
                    dtheta_sc_copy = np.stack([np.sin(dtheta_copy), np.cos(dtheta_copy)], axis=-1).astype(np.float32)  # (H,2)
                    dxy_copy = (ego_pose_copy[4:, :2] - ego_pose_copy[3:4, :2]).astype(np.float32)  # (H,2)
                    if self.act_norm:
                        dxy_copy[:, 0] = (dxy_copy[:, 0] - x_mean) / x_std
                        dxy_copy[:, 1] = (dxy_copy[:, 1] - y_mean) / y_std
                    action_copy = np.concatenate([dxy_copy, dtheta_sc_copy], axis=-1).astype(np.float32)  # (H,4)
            else:
                action = None
        
        if self.ver_1225==2:
            ego_pose_se2 = [StateSE2(float(x), float(y), float(yaw)) for x, y, yaw in glo_pose[:12]]
            rel_ego_pose_se2_per_time = []
            for i in range(len(ego_pose_se2)-1):
                if i < 2:
                    rel_ego_pose_se2_per_time.append([0,0,0])
                    continue
                rel_ego_pose_se2 = absolute_to_relative_poses(ego_pose_se2, i)
                next_point = rel_ego_pose_se2[i+1]
                rel_ego_pose_se2_per_time.append([next_point.x, next_point.y, wrap_to_pi(next_point.heading)])
            ego_pose = np.array(rel_ego_pose_se2_per_time)  # 11
            action = ego_pose
            normalized = 2 * (action - q01) / (q99 - q01 + 1e-8) - 1
            normalized = np.clip(normalized, -1, 1)
            action = normalized[3:]
            print(action)

        # cur_state = np.array(ego_pose[3] - ego_pose[2]).reshape(1, -1)
        if self.ver_1225==1:
            dtheta = wrap_to_pi(ego_pose[3, 2] - ego_pose[2, 2])
            dx = ego_pose[3, 0] - ego_pose[2, 0]
            dy = ego_pose[3, 1] - ego_pose[2, 1]
            if self.act_norm:
                dx = (dx - x_mean) / x_std
                dy = (dy - y_mean) / y_std
            cur_state = np.array([dx, dy, np.sin(dtheta), np.cos(dtheta)], dtype=np.float32).reshape(1, -1)  # (1,4)
        if self.ver_1225==2:
            cur_state = normalized[2:3].reshape(1, -1)
        cur_state = cur_state.astype(np.float32)
        # else:
        #     cur_state = None
        #     action = None


        # loader use all action
        # cur_action = ego_pose[1:4] - ego_pose[0:3]

        if not self.split == 'waymo_train':
            cur_command = command[3]

            LABELS = ["turn left", "keep straight", "turn right", "unknown"]

            cur_command = np.asarray(cur_command, dtype=float).ravel()

            cur_command = LABELS[int(cur_command.argmax())]

        cur_instruction = f"The navigation command for the current timestep is {cur_command}. Your task is to" \
            f"plan future actions based on the understanding of the driving scene."

        # 1109
        cur_instruction = f"You are an autonomous driving agent. The navigation command for the current timestep is {cur_command}. Your task is to" \
            f" plan future actions based on the understanding of the driving scene."
        
        # if self.w_depth:
        #     cur_instruction = f"{cur_command}"

        if self.doing_s2:
            self_rew = self.rew_dict[token]
            # 50%          0.9968        0.9559        0.9468           N/A        0.962
            if cur_command == 'turn left':
                thresh = 0.9968
            elif cur_command == 'keep straight':
                thresh = 0.9559
            elif cur_command == 'turn right':
                thresh = 0.9468
            else:
                assert False
            self_rew = True if self_rew >=thresh else False
        else:
            self_rew = None

        # target
        cmd = cur_command.replace(' ', '_')
        # target = self.targets[cmd]

        sample = {
            'image': images,
            'state': cur_state, # vector 3
            'action': action,
            'lang': cur_instruction,
            # for infer
            'token': token,
            # for s2
            'action_copy': action_copy,
            'adv': self_rew,
            # for depth
            # 'depth_feat': depth_feats,
            # 'target': target    # 16, 2
        }
        if "qwen" in cached_features:
            sample['qwen_feature_cache'] = cached_features["qwen"]
        if "agent_dino" in cached_features:
            sample['agent_dino_feature_cache'] = cached_features["agent_dino"]

        # if self.video_data_cfg.load_2d_data:
        #     sample['2d_gen_data'] = {}
        #     for view in views[:1]:  # only front for now
        #         video_dir = os.path.join(self.video_data_cfg.rgb_meta_dir, self.split, view, token+'.mp4')
        #         # -1~1, f,c,h,w
        #         pixel_values, name = self.rgb_gen_get_batch(video_dir)

        #         sample['2d_gen_data']["pixel_values"] = pixel_values
        #         sample['2d_gen_data']["text"] = name
        #         sample['2d_gen_data']["idx"] = f'{token}-{view}'

        #         mask = get_random_mask(pixel_values.size(), image_start_only=True)
        #         mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
        #         sample['2d_gen_data']["mask_pixel_values"] = mask_pixel_values
        #         sample['2d_gen_data']["mask"] = mask

        #         clip_pixel_values = sample['2d_gen_data']["pixel_values"][0].permute(1, 2, 0).contiguous()
        #         clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
        #         # 512x512x3, 0~255: why float?
        #         sample['2d_gen_data']["clip_pixel_values"] = clip_pixel_values

        if self.video_data_cfg.load_2d_data and "wan" in cached_features:
            wan_cache = dict(cached_features["wan"])
            # Keep this scalar as a Python bool so Accelerate does not move it
            # to the device and force a GPU->CPU synchronization every step.
            mask_flag = wan_cache.get("mask_is_all_ones", False)
            if isinstance(mask_flag, torch.Tensor):
                mask_flag = bool(mask_flag.item())
            wan_cache["mask_is_all_ones"] = mask_flag
            sample['2d_gen_data'] = {
                "idx": f"{token}-cam_l0_cam_f0_cam_r0",
                "feature_cache": wan_cache,
            }
        elif self.video_data_cfg.load_2d_data:
            views = ['cam_l0', 'cam_f0', 'cam_r0']   # 左前、前、右前

            sample['2d_gen_data'] = {}

            pixel_values_list = []
            name_list = []

            # 1. Read each view either from the pre-built MP4 or directly from
            # the same nine source frames. The latter avoids a redundant
            # encode/decode pass when a complete camera tree is available.
            for view in views:
                if self.video_source == "images":
                    image_paths = glo_images[view]["image_paths"][3:12]
                    pixel_values, name = self.rgb_gen_get_batch_from_images(
                        image_paths
                    )
                else:
                    video_path = os.path.join(
                        self.video_data_cfg.rgb_meta_dir,
                        self.split,
                        view,
                        token + '.mp4'
                    )

                    # -1~1, shape: (F, C, H, W)
                    pixel_values, name = self.rgb_gen_get_batch(video_path)

                pixel_values_list.append(pixel_values)
                name_list.append(name)

            # 3. 按宽度维拼接：新视频 shape = (F, C, H, W * num_views)
            pixel_values = torch.cat(pixel_values_list, dim=-1)   # dim=-1 即 W 维

            # ====== 填 sample['2d_gen_data'] ======
            sample['2d_gen_data']["pixel_values"] = pixel_values
            # 文本你可以只用前视的 name，也可以简单拼一下
            sample['2d_gen_data']["text"] = " | ".join(name_list)
            sample['2d_gen_data']["idx"] = f"{token}-{'_'.join(views)}"

            # 4. 生成 mask（在拼好之后再做）
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample['2d_gen_data']["mask_pixel_values"] = mask_pixel_values
            sample['2d_gen_data']["mask"] = mask

            # 5. clip_pixel_values 还是用新视频的第 0 帧
            #    pixel_values[0]: (C, H, 3W)
            clip_pixel_values = pixel_values[0].permute(1, 2, 0).contiguous()  # (H, 3W, C)
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255          # 变回 0~255
            sample['2d_gen_data']["clip_pixel_values"] = clip_pixel_values

            if self.video_data_cfg.text_input:
                txt_path = os.path.join(
                    self.video_data_cfg.rgb_meta_dir,
                    self.split,
                    views[1],
                    token + '.txt'
                )
                with open(txt_path, "r") as txt_file:
                    text = txt_file.read()
                sample['2d_gen_data']["text"] = text

        if self.gs_data_cfg.load_3d_data:
            pkl_path = os.path.join(self.gs_meta_dir, token+'.pkl')
            with open(pkl_path, "rb") as f:
                storm_data = pickle.load(f) # data of 5 timesteps and 3 views
            gs_data = self.get_storm_data(storm_data, token)
            sample['3d_gs_data'] = gs_data
        
        if self.reward_data_cfg.load_reward_data:
            jsonl_path = os.path.join(self.reward_meta_dir, token+'.jsonl')
            reward_dicts = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    one = json.loads(line)   # one 是 list，比如 [str, float]
                    reward_dicts.append(one)

            # currently self reward
            reward_dicts.sort(key=lambda x: x[1])
            reward = np.array([r[-1] for r in reward_dicts]).reshape(-1, 1).astype(np.float32)
            sample['reward_data'] = reward


        # 1109: check pose: may use delta for generation?

        return sample

    def _maybe_aug_image(self, img_path_or_pil):
        # 输出 PIL.Image 给 processor
        im = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        if self.enable_image_aug:
            im = self.image_aug(im)
        return im
    
    def _load_image(self, file_name, target_height=576, target_width=1024):
        file_name = resolve_navsim_data_path(file_name)
        if file_name is not None:
            image = Image.open(file_name)
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise ValueError(f"Invalid image file {file_name}")

        ori_w, ori_h = image.size
        ar_src = ori_w / ori_h
        ar_dst = target_width / target_height

        # --- 记录裁剪参数（新增，但不影响原逻辑） ---
        left = top = 0
        crop_w, crop_h = ori_w, ori_h

        if ar_src > ar_dst:
            tmp_w = int(target_width / target_height * ori_h)
            left  = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            crop_w = tmp_w
            image = image.crop((left, 0, right, ori_h))
        elif ar_src < ar_dst:
            tmp_h = int(target_height / target_width * ori_w)
            top    = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            crop_h = tmp_h
            image = image.crop((0, top, ori_w, bottom))

        image = image.resize((target_width, target_height), resample=Image.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return self._maybe_aug_image(image)

    def rgb_gen_get_batch(self, video_dir):

        with VideoReader_contextmanager(video_dir, num_threads=1) as video_reader:
            # min_sample_n_frames = min(
            #     self.video_sample_n_frames, 
            #     int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            # )
            min_sample_n_frames = 9

            if min_sample_n_frames == 0:
                raise ValueError(f"No Frames in video.")

            # video_length = int(self.video_length_drop_end * len(video_reader))
            # clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
            # start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
            # batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            video_length = len(video_reader)
            clip_length = 9
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
            except:
                print(f'loading fail: {video_dir}')
                pixel_values = self.pixel_values
                text = self.text
                return pixel_values, text

            # if not self.enable_bucket:
            if True:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                del video_reader
            else:
                pixel_values = pixel_values

            # if not self.enable_bucket:
            if True:
                pixel_values = self.video_transforms(pixel_values)
            
            # Random use no text generation
            # if random.random() < self.text_drop_ratio:
            #     text = ''

            text = ''

            self.pixel_values = pixel_values
            self.text = text

            return pixel_values, text

    def rgb_gen_get_batch_from_images(self, image_paths):
        if len(image_paths) != 9:
            raise ValueError(
                f"Expected 9 source frames, received {len(image_paths)}"
            )

        frames = []
        for image_path in image_paths:
            image_path = resolve_navsim_data_path(image_path)
            with Image.open(image_path) as image:
                frames.append(np.asarray(image.convert("RGB")).copy())

        pixel_values = torch.from_numpy(np.stack(frames))
        pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        pixel_values = self.video_transforms(pixel_values)
        return pixel_values, ''

    def get_storm_data(self, data_dict, token):
        # value range:
        # rgb -1,1
        # depth >0 valid
        # 
        '''
            # shape 5 x 3 x h x w
            gs_pkl = {
                'view_order': ['cam_f0', 'cam_l0', 'cam_r0'],
                'rgb': gs_rgbs,
                'depth': depths,
                'cam2ego': cam2egos,
                'intrinsic': intrinsics,
                'ego2global': ego2globals,
                'cam2globals': cam2globals,
                'dynamic_mask': dynamic_masks,
                'ground_mask': ground_masks,
                'sky_masks': sky_masks,
            }
        
        '''

        images = self.gs_transformation(torch.from_numpy(data_dict['rgb']).permute(0, 1, 4, 2, 3)[:, [1,0,2]]/255.)
        cam2globals = torch.from_numpy(data_dict['cam2globals'])[:, [1,0,2]]
        intrinsics = torch.from_numpy(data_dict['intrinsic'])[:, [1,0,2]]
        depths = torch.from_numpy(data_dict['depth'][..., 0])[:, [1,0,2]]
        
        sky_masks = torch.from_numpy(data_dict['sky_masks'])[:, [1,0,2]]
        sky_masks = (sky_masks > 0).float()   # (5,1,Ht,Wt), 0/1
        
        flow = torch.from_numpy(data_dict['depth'][..., 1:])[:, [1,0,2]]  # all zero

        cam2ego = torch.from_numpy(data_dict['cam2ego'])[:, [1,0,2]]

        frame_idx = torch.tensor([i for i in range(12)]).long()
        time = torch.tensor([
            [i*0.5]*3
        for i in range(12)]).float()

        first_frame = 3

        # first frame, front view
        world_to_canonical = torch.linalg.inv(cam2globals[first_frame][1])
        # sync to the first frame
        # camtoworld = world_to_canonical @ cam2globals
        camtoworld = (
            torch.from_numpy(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])).float()
            @ world_to_canonical.float()
            @ cam2globals.float()
            @ torch.from_numpy(np.eye(4)).float()
        )

        frames = np.array([first_frame+i for i in range(4)])
        timespan = 2.
        fps = 2.
        num_context_timesteps = 1
        num_target_timesteps = 1
        context_frame_idx = first_frame

        num_max_future_frames = int(timespan * fps)

        context_frame_idx = np.arange(
            context_frame_idx,
            context_frame_idx + num_max_future_frames,
            num_max_future_frames // num_context_timesteps,
        )

        # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
        all_target_frame_idx = np.arange(
            context_frame_idx[0],
            context_frame_idx[0] + num_max_future_frames,
        )

        # randomly sample "num_target_timesteps" frames
        target_frame_idx = np.random.choice(
            np.arange(
                context_frame_idx[0],
                context_frame_idx[0] + num_max_future_frames,
            ),
            num_target_timesteps,
            replace=False,
            p=np.array([0.1, 0.3, 0.3, 0.3])
        )

        target_frame_idx = sorted(target_frame_idx)

        context_dict = {
            'image': images[context_frame_idx],
            'camtoworld': camtoworld[context_frame_idx],
            'intrinsics': intrinsics[context_frame_idx],
            'frame_idx': frame_idx[context_frame_idx] - frame_idx[context_frame_idx[:1]],
            'depth': depths[context_frame_idx],
            'sky_masks': sky_masks[context_frame_idx],
            'flow': flow[context_frame_idx],
            'time': time[context_frame_idx] - time[context_frame_idx[:1]]
        }

        target_dict = {
            'image': images[target_frame_idx],
            'camtoworld': camtoworld[target_frame_idx],
            'intrinsics': intrinsics[target_frame_idx],
            'frame_idx': frame_idx[target_frame_idx] - frame_idx[context_frame_idx[:1]],
            'depth': depths[target_frame_idx],
            'sky_masks': sky_masks[target_frame_idx],
            'flow': flow[target_frame_idx],
            'time': time[target_frame_idx] - time[context_frame_idx[:1]]
        }

        # for vis
        all_target_dict = {
            'image': images[all_target_frame_idx],
            'camtoworld': camtoworld[all_target_frame_idx],
            'intrinsics': intrinsics[all_target_frame_idx],
            'frame_idx': frame_idx[all_target_frame_idx] - frame_idx[context_frame_idx[:1]],
            'depth': depths[all_target_frame_idx],
            'sky_masks': sky_masks[all_target_frame_idx],
            'flow': flow[all_target_frame_idx],
            'time': time[all_target_frame_idx] - time[context_frame_idx[:1]]
        }

        for k, v in context_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                context_dict[k] = torch.cat([d for d in v], dim=0)
        for k, v in target_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                target_dict[k] = torch.cat([d for d in v], dim=0)
        
        for k, v in all_target_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                all_target_dict[k] = torch.cat([d for d in v], dim=0)

        sample = {
            'context': context_dict,
            'target': target_dict,
            "scene_id": token,
            "scene_name": token,
            "width": 256,
            "height": 144,
            "fps": fps,
            "timespan": timespan,  # 1 for fixed input
            "all_target": all_target_dict
        }
        sample = to_float_tensor(sample)
        return sample






        


if __name__ == "__main__":
    # Tiny smoke-test / example usage
    import sys
    datalist = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NAVSIM_DATALIST_PATH", "mini_meta.json")
    ds = NavSimDataset(datalist_path=datalist, split="mini")
    example_data = ds[0]
    print(f"Loaded {len(ds)} samples. Example: {example_data}")
