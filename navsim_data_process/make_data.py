import os
import json
import math
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from pathlib import Path
import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent
sys.path.append(f"{project_root}/")
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim

from PIL import Image
from moviepy.editor import ImageSequenceClip
def images_to_video(image_folder, output_video, fps=20):
    # images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
    # images.sort()  # Ensure the images are in the correct order
    images = image_folder
    # images.sort(key=lambda p: int(p.split('/')[-1].split('.')[0].split('_')[-1]))

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_video, codec="libx264",
        verbose=False,   # 关闭 MoviePy 自己的进度条
        logger=None      # 禁止写自定义 logger（否则仍会输出）
    )

def gt_2_ego(gt_xy, heading=None, k_ahead=1, min_step=1):
    # theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy)            # shape [T, 2]

    # ---- 平移到原点 --------------------------------------------------------
    origin  = gt[0]
    rel_gt  = gt - origin

    # 用多帧位移稳健估计速度朝向
    k = min(k_ahead, len(gt)-1)
    v = gt[k] - gt[0]
    if torch.linalg.norm(v) < min_step:
        # 找到第一个位移够大的帧
        for j in range(1, len(gt)):
            v = gt[j] - gt[0]
            if torch.linalg.norm(v) >= min_step:
                break

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    if heading is None:
        # heading = rel_gt[1]                 # (Δx, Δy)
        heading = v
        theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    else:
        theta = heading
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # 逆时针旋转
                    [ torch.sin(theta),  torch.cos(theta)]])  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()

    return gt


import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="mini")
parser.add_argument("--data_root", default="navsim_dataset")
parser.add_argument("--make_video", action="store_true")
args = parser.parse_args()

root = args.data_root
video_root = f"{root}/navsim_video"

import cv2, numpy as np

def gen_info(root):
    os.makedirs(root, exist_ok=True)

    for mode in [args.split]:

        # tip: omit --make_video to skip video generation and speed up processing
        make_video = args.make_video

        all_x = []
        all_y = []

        meta_dir = os.path.join(root, mode)
        os.makedirs(meta_dir, exist_ok=True)
        dataset = VLMNavsim(mode=mode)
        video_dir = os.path.join(video_root, mode)
        os.makedirs(video_dir, exist_ok=True)

        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):

            container = dataset.get_container_in(sid)

            if container is None:
                continue

            this_glo_status = {}
            this_glo_images = {}

            for idx, ego_status in enumerate(container["ego_status"]):

                this_glo_status[idx] = {
                    "global_pose": ego_status.ego_pose,
                    "velocity": ego_status.ego_velocity,
                    "acceleration": ego_status.ego_acceleration,
                    "command": ego_status.driving_command
                }

            # stack status in time
            glo_status = {
                    "global_poses": np.array([this_glo_status[i]["global_pose"] for i in range(len(this_glo_status))]),
                    "velocities": np.array([this_glo_status[i]["velocity"] for i in range(len(this_glo_status))]),
                    "accelerations": np.array([this_glo_status[i]["acceleration"] for i in range(len(this_glo_status))]),
                    "commands": np.array([this_glo_status[i]["command"] for i in range(len(this_glo_status))]),
                }

            views = ['cam_f0', 'cam_l0', 'cam_r0', 'cam_l1', 'cam_r1', 'cam_l2', 'cam_r2', 'cam_b0']
            for idx, frame_data in enumerate(container["images"]):
                if idx == 13:
                    # no frames for last frame
                    continue

                for view in views:
                    if idx not in this_glo_images:
                        this_glo_images[idx] = {}

                    this_glo_images[idx][view] = {
                        "image_path": frame_data[view]["image_path"],
                        "sensor2lidar_rotation": frame_data[view]["sensor2lidar_rotation"],
                        "sensor2lidar_translation": frame_data[view]["sensor2lidar_translation"],
                        "intrinsics": frame_data[view]["intrinsics"],
                        "distortion": frame_data[view]["distortion"]
                    }

            # stack sensors in time
            glo_images = {}
            for view in views:
                glo_images[view] = {
                            "image_paths": [this_glo_images[i][view]["image_path"] for i in range(len(this_glo_images))],
                            "sensor2lidar_rotations": np.array([this_glo_images[i][view]["sensor2lidar_rotation"] for i in range(len(this_glo_images))]),
                            "sensor2lidar_translations": np.array([this_glo_images[i][view]["sensor2lidar_translation"] for i in range(len(this_glo_images))]),
                            "intrinsics": np.array([this_glo_images[i][view]["intrinsics"] for i in range(len(this_glo_images))]),
                            "distortions": np.array([this_glo_images[i][view]["distortion"] for i in range(len(this_glo_images))]),
                        }
            meta_res = {
                "glo_status": glo_status,
                "glo_images": glo_images,
            }

            with open(f"{meta_dir}/{container['token']}.pkl", "wb") as f:
                pickle.dump(meta_res, f, protocol=pickle.HIGHEST_PROTOCOL)
                # print(f'writing: {container['token']}.pkl')


            if make_video:
                for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                    img_list = glo_images[view]['image_paths'][3:12]
                    video_view_dir = os.path.join(video_dir, view)
                    os.makedirs(video_view_dir, exist_ok=True)
                    video_path = os.path.join(video_view_dir, container['token'] + '.mp4')
                    if not os.path.isfile(video_path):
                        images_to_video(img_list, video_path, fps=2)
                    ext = img_list[0][-4:]
                    still_path = os.path.join(video_view_dir, container["token"] + ext)
                    if not os.path.isfile(still_path):
                        shutil.copy2(img_list[0], still_path)

            all_x.append(gt_2_ego(glo_status['global_poses'][3:12,:2])[1:,0])
            all_y.append(gt_2_ego(glo_status['global_poses'][3:12,:2])[1:,1])
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        std_x = np.std(all_x.reshape(-1, 1))
        print(f'data samples: {all_x.shape[0]}')
        std_y = np.std(all_y.reshape(-1, 1))
        scale = std_y / (std_x+1e-6)
        print(scale)


gen_info(root=f"{root}/meta/")
