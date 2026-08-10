# from data_engine.datasets.navsim.loaders.pipelines import *
from data_engine.datasets.navsim.loaders.navsim.planning.training.agent_lightning_module import AgentLightningModule
from data_engine.datasets.navsim.loaders.navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from data_engine.datasets.navsim.loaders.navsim.common.dataloader import SceneLoader
from data_engine.datasets.navsim.loaders.navsim.common.dataclasses import Scene, SceneFilter, SceneMetadata, SensorConfig, Lidar
from data_engine.datasets.navsim.loaders.navsim.agents.abstract_agent import AbstractAgent
from data_engine.datasets.navsim.loaders.navsim.visualization.plots import plot_bev_with_agent, plot_bev_with_agent_ori
# from data_engine.common_misc.external_helpers.openai_query import construct_external_query
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate
from PIL import Image
import hydra
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Tuple
import os
import ray
import json
import math

import hydra
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
import pickle
from easydict import EasyDict

from pathlib import Path
script_path = Path(__file__).resolve()
_cfg_root = script_path.parent  # data_engine/datasets/navsim/
project_root = script_path.parent.parent.parent.parent.parent
project_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
from data_engine.datasets.navsim.loaders.navsim.visualization.camera import render_dynamic_object_mask

# append current directory to path
import sys

import torch
sys.path.append(os.path.dirname(__file__))


logger = logging.getLogger(__name__)

# PIPELINES = {
#     "metadata": PromptNavsimMetadata,
#     "ego_status": PromptNavsimEgoStatus,
#     "planning": PromptNavsimPlanning,
#     'meta_planning': PromptNavsimMetaPlanning,
#     "road_agent_analysis": PromptNavsimRoadAgentAnalysis,
#     "scene_description": PromptNavsimSceneDescription,
# }

v0_pipelines = [
    {"type": "metadata", "use_image": "3v"},
    {"type": "ego_status", "mode": "x-y"},
    {"type": "planning", "mode": "x-y"},
    # ! the constructed queries of this stage are dependent on the mode of previous planning pipeline
    {"type": "meta_planning"},
    {"type": "road_agent_analysis"},
    {"type": "scene_description"},
]
v0_container_out_key_comb = ["scene_description",
                             'road_agent_analysis', 'meta_planning', 'planning']
# v0_container_out_key_comb = ['meta_planning', 'planning']

va_pipelines = [
    {"type": "metadata", "use_image": "3v"},
    {"type": "ego_status", "mode": "dist-dtheta"},
    # {"type": "scene_description"},
    # {"type": "road_agent_analysis_text"},
    {"type": "planning", "mode": "dist-dtheta"},
    # {"type": "meta_planning"},  #! the constructed queries of this stage are dependent on the mode of previous planning pipeline
]
va_container_out_key_comb = ['planning']


import numpy as np
import cv2

def figax_to_bev500(fig, ax, out_hw=(500, 500), rgb=True, bg=(255,255,255), pad=0):
    """
    裁剪 ax 内容区域 -> 固定尺寸
    - 先 RGBA 合成到白底，避免透明变黑边
    - bbox 用 renderer 像素坐标，避免 dpi 换算误差
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 整张 figure RGBA (H,W,4)
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)

    # ✅ 关键：RGBA -> RGB（铺白底）
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb_img = rgba[..., :3].astype(np.float32) * a + np.array(bg, np.float32) * (1.0 - a)
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    # ✅ 用像素 bbox（直接可切）
    bbox = ax.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = map(int, bbox.extents)  # (left,bottom,right,top) in pixels

    # 可选 padding（防止切掉1px边）
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)

    # 注意：bbox 的 y 是以左下为原点，但数组是左上
    crop = rgb_img[h - y1+4 : h - y0-4, x0+4 : x1-4]

    if not rgb:
        crop = crop.mean(-1).astype(np.uint8)

    out_h, out_w = out_hw
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return crop


class VLMNavsim(torch.utils.data.Dataset):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None, pipelines=[], container_out_key_comb=[]):
        cfg = OmegaConf.load(
            _cfg_root / "loaders/navsim/planning/script/config/training/default_training.yaml")
        navsim = self.build_datasets(cfg, mode)

        self.navsim = navsim

        self.mode = mode

        self.length = len(self.navsim)
        logger.info(f"Dataset mode {mode} has {self.length} samples.")
        if length is not None:
            self.length = min(self.length, length)

        self.pipelines = []
        # self.external_helper = construct_external_query(
            # "Qwen/Qwen2.5-VL-72B-Instruct")
        # for pipeline in pipelines:
        #     if pipeline["type"] in PIPELINES:
        #         self.pipelines.append(PIPELINES[pipeline["type"]](
        #             navsim=self.navsim, **pipeline, external_helper=self.external_helper))
        #     else:
        #         raise ValueError("Pipeline {} not found".format(pipeline))
        self.container_out_key_comb = container_out_key_comb

    def __len__(self):
        return self.length

    def build_datasets(self, cfg: DictConfig, mode="train") -> Dataset:
        """
        Builds training and validation datasets from omega config
        :param cfg: omegaconf dictionary
        :param agent: interface of agents in NAVSIM
        :return: tuple for training and validation dataset
        """
        sensor_config = SensorConfig.build_all_sensors(include=[i for i in range(13)])

        navtrain_filter_cfg = OmegaConf.load(
            _cfg_root / "loaders/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
        navtest_cfg = OmegaConf.load(
            _cfg_root / "loaders/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
        navmini_cfg = OmegaConf.load(
            _cfg_root / "loaders/navsim/planning/script/config/common/train_test_split/scene_filter/navmini.yaml")

        split_logs = OmegaConf.load(
            _cfg_root / "loaders/navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
        cfg.train_logs, cfg.val_logs, cfg.test_logs = split_logs.train_logs, split_logs.val_logs, split_logs.test_logs

        cfg.mini_logs = split_logs.mini_logs

        if mode == "train":

            trainval_scene_filter: SceneFilter = instantiate(
                navtrain_filter_cfg)
            if trainval_scene_filter.log_names is not None:
                trainval_scene_filter.log_names = [
                    log_name for log_name in trainval_scene_filter.log_names if log_name in cfg.train_logs or log_name in cfg.val_logs
                ]
            else:
                trainval_scene_filter.log_names = cfg.train_logs + cfg.val_logs

            data_path_trainval = Path(
                f"{project_root}/navsim_logs/trainval")
            sensor_blobs_path_trainval = Path(
                f"{project_root}/sensor_blobs/trainval")

            trainval_scene_loader = SceneLoader(
                original_sensor_path=sensor_blobs_path_trainval,
                data_path=data_path_trainval,
                scene_filter=trainval_scene_filter,
                sensor_config=sensor_config,
            )

            trainval_data = Dataset(
                scene_loader=trainval_scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return trainval_data

        elif mode == "mini":
            
            mini_scene_filter: SceneFilter = instantiate(navmini_cfg)
            if mini_scene_filter.log_names is not None:
                mini_scene_filter.log_names = [
                    log_name for log_name in mini_scene_filter.log_names if log_name in cfg.mini_logs]
            else:
                mini_scene_filter.log_names = mini_logs

            data_path_mini = Path(
                f"{project_root}/navsim_logs/mini")
            sensor_blobs_path_mini = Path(
                f"{project_root}/sensor_blobs/mini")

            mini_scene_loader = SceneLoader(
                original_sensor_path=sensor_blobs_path_mini,
                data_path=data_path_mini,
                scene_filter=mini_scene_filter,
                sensor_config=sensor_config
            )

            mini_data = Dataset(
                scene_loader=mini_scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return mini_data

        elif mode == "test":

            test_scene_filter: SceneFilter = instantiate(navtest_cfg)
            if test_scene_filter.log_names is not None:
                test_scene_filter.log_names = [
                    log_name for log_name in test_scene_filter.log_names if log_name in cfg.test_logs]
            else:
                test_scene_filter.log_names = cfg.test_logs

            data_path_test = Path(os.environ.get("NAVSIM_TEST_LOG_ROOT", f"{project_root}/navsim_logs/test"))
            sensor_blobs_path_test = Path(os.environ.get("NAVSIM_TEST_SENSOR_ROOT", f"{project_root}/sensor_blobs/test"))

            test_scene_loader = SceneLoader(
                original_sensor_path=sensor_blobs_path_test,
                data_path=data_path_test,
                scene_filter=test_scene_filter,
                sensor_config=sensor_config
            )

            test_data = Dataset(
                scene_loader=test_scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return test_data
        
        elif mode == "warmup_two_stage":

            # try 3 later
            sensor_config = SensorConfig.build_all_sensors(include=[i for i in range(4)])

            CONFIG_PATH = 'data/data_engine/datasets/navsim/loaders/navsim/planning/script/config/pdm_scoring/'
            CONFIG_NAME = 'default_run_pdm_score'

            abs_config_dir = os.path.abspath(CONFIG_PATH)

            with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
                cfg = compose(config_name=CONFIG_NAME, overrides=["train_test_split=warmup_two_stage"])

            synthetic_sensor_path = os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", ""), "warmup_two_stage/sensor_blobs")
            synthetic_scenes_path = os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", ""), "warmup_two_stage/synthetic_scene_pickles")

            scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
            # grouped by two stages
            scene_loader = SceneLoader(
                synthetic_sensor_path=Path(synthetic_sensor_path),
                original_sensor_path=Path(cfg.original_sensor_path),
                data_path=Path(cfg.navsim_log_path),
                synthetic_scenes_path=Path(synthetic_scenes_path),
                scene_filter=scene_filter,
                sensor_config=sensor_config,
            )

            test_data = Dataset(
                scene_loader=scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return test_data

        elif mode == "navhard_two_stage":

            # try 3 later
            sensor_config = SensorConfig.build_all_sensors(include=[i for i in range(4)])

            CONFIG_PATH = 'data/data_engine/datasets/navsim/loaders/navsim/planning/script/config/pdm_scoring/'
            CONFIG_NAME = 'default_run_pdm_score'

            abs_config_dir = os.path.abspath(CONFIG_PATH)

            with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
                cfg = compose(config_name=CONFIG_NAME, overrides=["train_test_split=navhard_two_stage"])

            synthetic_sensor_path = os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", ""), "navhard_two_stage/sensor_blobs")
            synthetic_scenes_path = os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", ""), "navhard_two_stage/synthetic_scene_pickles")

            scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
            # grouped by two stages
            scene_loader = SceneLoader(
                synthetic_sensor_path=Path(synthetic_sensor_path),
                original_sensor_path=Path(cfg.original_sensor_path),
                data_path=Path(cfg.navsim_log_path),
                synthetic_scenes_path=Path(synthetic_scenes_path),
                scene_filter=scene_filter,
                sensor_config=sensor_config,
            )

            test_data = Dataset(
                scene_loader=scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return test_data


        else:
            raise ValueError("Mode {} not found".format(mode))

    def cache_data(self, cache_filename):
        assert cache_filename.endswith(".json")
        # create cache file if not exists
        cache_filename = os.path.join("data", "navsim", cache_filename)

        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

        all_iters = list(range(len(self)))
        # Parallel processing even slower than serial processing. do not know why. by c7w
        # num_cpus = min(math.ceil(os.cpu_count() * 0.8), 16)
        # print("Using {} CPUs for caching".format(num_cpus))
        # all_jsons = p_map(self.__getitem__, all_iters, num_cpus=num_cpus, desc="Caching data")
        all_jsons = []
        for i in tqdm(all_iters, desc="Caching data"):
            all_jsons.append(self.__getitem__(i))

        # DUMP a json file!
        with open(cache_filename, "w") as f:
            f.write("[\n")
            all_len = len(all_jsons)
            for idx, json_obj in enumerate(all_jsons):
                f.write(json.dumps(json_obj))
                if idx != all_len - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("]")

        # cleanup all pipelines
        for pipeline in self.pipelines:
            if hasattr(pipeline, "cleanup"):
                pipeline.cleanup()

    def evaluate(self, jsonl_file):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))

        assert len(predicted_data) == len(
            self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))
        # evaluate every single prediction
        # import pdb;pdb.set_trace()
        for idx, pred in tqdm(enumerate(predicted_data)):
            # import pdb;pdb.set_trace()

            container_in = self.get_container_in(idx)
            container_in["idx"] = idx
            container_out = self.__getitem__(idx)

            # evaluate the prediction
            for pipeline in self.pipelines:
                pipeline.evaluation_update(pred, container_out, container_in)
        # import pdb;pdb.set_trace()

        results = {}
        for pipeline in self.pipelines:
            pipeline.evaluation_compute(results)
        print(results)
        return results

    def viz_all_results(self, jsonl_file, interval):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))

        assert len(predicted_data) == len(
            self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))
        # evaluate every single prediction
        # import pdb;pdb.set_trace()

        viz_path = jsonl_file.replace(".json", "")
        os.makedirs(viz_path, exist_ok=True)
        print(f"Saving visualization to {viz_path}")

        for idx, pred in tqdm(enumerate(predicted_data)):
            # import pdb;pdb.set_trace()
            if idx % interval != 0:
                continue

            container_in = self.get_container_in(idx)
            container_in["idx"] = idx
            container_out = self.__getitem__(idx)
            # import pdb;pdb.set_trace()
            save_name = f"{idx:06d}.html"
            id = container_out["id"]
            images = container_out['images']
            query = container_out['messages'][0]['content']
            gt = container_out['messages'][1]['content']
            prediction = pred['predict']

            def escape_html(text):
                return text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            from PIL import Image
            image_paths = images
            target_size = (576, 384)
            output_dir_imgs = os.path.join(viz_path, f"images_{idx}")
            output_img_names = []

            os.makedirs(output_dir_imgs, exist_ok=True)

            # Process each image
            for image_path in image_paths:
                # Extract the camera status from the path
                camera_status = image_path.split('/')[-2]

                # Open the image
                with Image.open(image_path) as img:
                    # Resize the image
                    img_resized = img.resize(
                        target_size, Image.Resampling.NEAREST)

                    # Create the output file name
                    output_file_name = f"{camera_status}.jpg"
                    output_file_path = os.path.join(
                        output_dir_imgs, output_file_name)
                    output_img_names.append(output_file_name)

                    # Save the resized image
                    img_resized.save(output_file_path, "JPEG")
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>All In One</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }}
                    h2 {{
                        color: #333;
                    }}
                    p {{
                        font-family: monospace;
                    }}
                    .section {{
                        margin-bottom: 20px;
                    }}
                    .images {{
                        display: flex;
                        flex-wrap: wrap;
                    }}
                    .images img {{
                        margin: 5px;
                        max-width: 200px;
                    }}
                </style>
            </head>
            <body>
                <div class="section">
                    <h2>Query</h2>
                    <p>{escape_html(query)}</p>
                </div>
                <div class="section">
                    <h2>Ground Truth</h2>
                    <p>{escape_html(gt)}</p>
                </div>
                <div class="section">
                    <h2>Prediction</h2>
                    <p>{escape_html(prediction)}</p>
                </div>
                <div class="section">
                    <h2>Images</h2>
                    <div class="images">
            """
            # Add the images to the HTML content
            for image_file in output_img_names:
                image_path = os.path.join(output_dir_imgs, image_file)
                html_content += f'            <img src="images_{idx}/{image_file}" alt="{image_file}">\n'

            # Close the HTML tags
            html_content += """</div>
                </div>
            </body>
            </html>"""

        # import pdb;pdb.set_trace()
            with open(os.path.join(viz_path, save_name), "w") as f:
                f.write(html_content)

    def get_container_in(self, idx, load_lidar=0, load_bev=0, meta_test_mini_list=None):
        container_in = {}  # first, aggregate container_in
        this_token = self.navsim._scene_loader.tokens[idx]
        
        synthetic = False

        try:
            scene_dict_list = self.navsim._scene_loader.scene_frames_dicts[this_token]
            sensor_blobs_path = self.navsim._scene_loader._original_sensor_path
        except:
            # print(f'loading synthetic:')
            file_path = self.navsim._scene_loader.synthetic_scenes[this_token][0]
            with open(file_path, "rb") as f:
                scene_dict_list = pickle.load(f)
            sensor_blobs_path = self.navsim._scene_loader._synthetic_sensor_path
            synthetic = True
        num_history_frames, num_future_frames = self.navsim._scene_loader._scene_filter.num_history_frames, self.navsim._scene_loader._scene_filter.num_future_frames
        sensor_config = self.navsim._scene_loader._sensor_config

        if not synthetic:
            this_scene_metadata = SceneMetadata(
                log_name=scene_dict_list[num_history_frames - 1]["log_name"],
                scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
                map_name=scene_dict_list[num_history_frames - 1]["map_location"],
                initial_token=scene_dict_list[num_history_frames - 1]["token"],
                num_history_frames=num_history_frames,
                num_future_frames=num_future_frames,
            )
        else:
            this_scene_metadata = SceneMetadata(**scene_dict_list["scene_metadata"])

            # only use frame
            scene_dict_list = scene_dict_list['frames']

        container_in["token"] = this_token
        container_in["scene_metadata"] = this_scene_metadata

        if load_bev:
            # scene = self.navsim._scene_loader.get_scene_from_token(this_token)
            ann = scene_dict_list[3]['anns']
            # add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            fig, ax = plot_bev_with_agent(ann, None)

            gt_bev = figax_to_bev500(fig, ax, out_hw=(500, 500), rgb=True)

            bev = gt_bev
        else:
            bev = None

        frame_data_dict = []
        for frame_idx in range(len(scene_dict_list)):

            if synthetic:
                global_ego_status = scene_dict_list[frame_idx]['ego_status']
                annotations = scene_dict_list[frame_idx]['annotations']
            else:
                global_ego_status = Scene._build_ego_status(
                    scene_dict_list[frame_idx])
                annotations = Scene._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            if len(sensor_names) > 0:
                sensor_names = sensor_names[:-1]  # DROP lidar_pc

            this_frame_cameras = {}
            if not synthetic:
                camera_dict = scene_dict_list[frame_idx]["cams"]
            else:
                camera_dict = scene_dict_list[frame_idx]['camera_dict']
            data_dict = {}
            for camera_name in camera_dict.keys():
                camera_identifier = camera_name.lower()
                if camera_identifier in sensor_names:
                    image_path = sensor_blobs_path / \
                        camera_dict[camera_name]["data_path"]
                    data_dict[camera_identifier] = {
                        "image_path": str(image_path),
                        "sensor2lidar_rotation": camera_dict[camera_name]["sensor2lidar_rotation"],
                        "sensor2lidar_translation": camera_dict[camera_name]["sensor2lidar_translation"],
                        "intrinsics": camera_dict[camera_name]["cam_intrinsic"],
                        "distortion": camera_dict[camera_name]["distortion"],
                    }

                else:
                    data_dict[camera_identifier] = {}  # empty camera
            this_frame_cameras = data_dict  # rename it

            if load_lidar == 1 and frame_idx <12 and frame_idx == 3:
                if synthetic:
                    pass
                    
                lidar_path = scene_dict_list[frame_idx]["lidar_path"]
                lidar = Lidar.from_paths(sensor_blobs_path, lidar_path, ['lidar_pc']).lidar_pc

                # ego2global
                print("lidar2ego: ", scene_dict_list[frame_idx]["lidar2ego"])
                ego2global = scene_dict_list[frame_idx]["ego2global"]
                dynamic_masks = {}
                for camera_name in camera_dict.keys():
                    camera_identifier = camera_name.lower()
                    if camera_identifier in sensor_names:
                        cam = camera_dict[camera_name]
                        ann = scene_dict_list[frame_idx]['anns']
                        dynamic_mask, _ = render_dynamic_object_mask(ann, cam, lidar)
                        dynamic_masks[camera_identifier] = dynamic_mask
            else:
                lidar = None
                ego2global = None
                dynamic_masks = None


            frame_data_dict.append({
                "token": scene_dict_list[frame_idx]["token"],
                "timestamp": scene_dict_list[frame_idx]["timestamp"],
                "ego_status": global_ego_status if not synthetic else EasyDict(global_ego_status),
                "annotations": annotations if not synthetic else EasyDict(annotations),
                "cameras": this_frame_cameras,
                # :5
                "lidar": lidar,
                "ego2global": ego2global,
                "dynamic_mask": dynamic_masks
            })
        container_in["frame_data"] = frame_data_dict

        container_in["ego_status"] = []
        # test set still has pos?
        for frame in frame_data_dict:
            container_in["ego_status"].append(frame["ego_status"])

        container_in["images"] = []
        for frame in frame_data_dict:
            container_in["images"].append(frame["cameras"])

        container_in["bev"] = bev
        return container_in


    def get_container_in_only_lidar(self, idx, load_lidar=0, load_bev=0, meta_test_mini_list=None):
        container_in = {}  # first, aggregate container_in
        this_token = self.navsim._scene_loader.tokens[idx]

        # if self.mode == 'test' and not this_token in meta_test_mini_list:
        #     return None
        
        synthetic = False

        try:
            scene_dict_list = self.navsim._scene_loader.scene_frames_dicts[this_token]
            sensor_blobs_path = self.navsim._scene_loader._original_sensor_path
        except:
            # print(f'loading synthetic:')
            file_path = self.navsim._scene_loader.synthetic_scenes[this_token][0]
            with open(file_path, "rb") as f:
                scene_dict_list = pickle.load(f)
            sensor_blobs_path = self.navsim._scene_loader._synthetic_sensor_path
            synthetic = True
        num_history_frames, num_future_frames = self.navsim._scene_loader._scene_filter.num_history_frames, self.navsim._scene_loader._scene_filter.num_future_frames
        sensor_config = self.navsim._scene_loader._sensor_config

        if not synthetic:
            this_scene_metadata = SceneMetadata(
                log_name=scene_dict_list[num_history_frames - 1]["log_name"],
                scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
                map_name=scene_dict_list[num_history_frames - 1]["map_location"],
                initial_token=scene_dict_list[num_history_frames - 1]["token"],
                num_history_frames=num_history_frames,
                num_future_frames=num_future_frames,
            )
        else:
            this_scene_metadata = SceneMetadata(**scene_dict_list["scene_metadata"])

            # only use frame
            scene_dict_list = scene_dict_list['frames']

        container_in["token"] = this_token
        container_in["scene_metadata"] = this_scene_metadata

        if load_bev:
            # scene = self.navsim._scene_loader.get_scene_from_token(this_token)
            ann = scene_dict_list[3]['anns']
            # add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            fig, ax = plot_bev_with_agent(ann, None)

            gt_bev = figax_to_bev500(fig, ax, out_hw=(500, 500), rgb=True)

            bev = gt_bev
        else:
            bev = None

        frame_data_dict = [0, 0, 0]
        for frame_idx in range(len(scene_dict_list)):

            if frame_idx !=3:
                continue

            if load_lidar == 1 and frame_idx <12 and frame_idx == 3:
                if synthetic:
                    pass
                    
                lidar_path = scene_dict_list[frame_idx]["lidar_path"]
                lidar = Lidar.from_paths(sensor_blobs_path, lidar_path, ['lidar_pc']).lidar_pc

            # if synthetic:
            #     global_ego_status = scene_dict_list[frame_idx]['ego_status']
            #     annotations = scene_dict_list[frame_idx]['annotations']
            # else:
            #     global_ego_status = Scene._build_ego_status(
            #         scene_dict_list[frame_idx])
            #     annotations = Scene._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            if len(sensor_names) > 0:
                sensor_names = sensor_names[:-1]  # DROP lidar_pc

            this_frame_cameras = {}
            if not synthetic:
                camera_dict = scene_dict_list[frame_idx]["cams"]
            else:
                camera_dict = scene_dict_list[frame_idx]['camera_dict']
            data_dict = {}
            for camera_name in camera_dict.keys():
                camera_identifier = camera_name.lower()
                if camera_identifier in sensor_names:
                    image_path = sensor_blobs_path / \
                        camera_dict[camera_name]["data_path"]
                    data_dict[camera_identifier] = {
                        "image_path": str(image_path),
                        "sensor2lidar_rotation": camera_dict[camera_name]["sensor2lidar_rotation"],
                        "sensor2lidar_translation": camera_dict[camera_name]["sensor2lidar_translation"],
                        "intrinsics": camera_dict[camera_name]["cam_intrinsic"],
                        "distortion": camera_dict[camera_name]["distortion"],
                    }

                else:
                    data_dict[camera_identifier] = {}  # empty camera
            this_frame_cameras = data_dict  # rename it


            frame_data_dict.append({
                "token": scene_dict_list[frame_idx]["token"],
                "timestamp": scene_dict_list[frame_idx]["timestamp"],
                # "ego_status": global_ego_status if not synthetic else EasyDict(global_ego_status),
                # "annotations": annotations if not synthetic else EasyDict(annotations),
                "cameras": this_frame_cameras,
                # :5
                "lidar": lidar,
                # "ego2global": ego2global,
                # "dynamic_mask": dynamic_masks
            })
        container_in["frame_data"] = frame_data_dict

        container_in["ego_status"] = []
        # # test set still has pos?
        # for frame in frame_data_dict:
        #     container_in["ego_status"].append(frame["ego_status"])

        container_in["images"] = [0,0,0]
        assert len(frame_data_dict) == 4
        for frame in frame_data_dict[-1:]:
            container_in["images"].append(frame["cameras"])

        # container_in["bev"] = bev
        return container_in

    def get_container_in_only_bev(self, idx, load_lidar=0, load_bev=0, meta_test_mini_list=None):
        container_in = {}  # first, aggregate container_in
        this_token = self.navsim._scene_loader.tokens[idx]

        only_lidar_root = os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", ""), "navsim_bev_64_64")
        only_lidar_dir = os.path.join(only_lidar_root, 'mini')

        all_exist = 1
        for frame_idx in range(12):
            if not os.path.exists(f"{only_lidar_dir}/{this_token}-{frame_idx}.pkl"):
                all_exist = 0
                break
        if all_exist:
            res = (None, this_token)
            return res


        if self.mode == 'test' and not this_token in meta_test_mini_list:
            return None
        
        synthetic = False

        try:
            scene_dict_list = self.navsim._scene_loader.scene_frames_dicts[this_token]
            sensor_blobs_path = self.navsim._scene_loader._original_sensor_path
        except:
            # print(f'loading synthetic:')
            file_path = self.navsim._scene_loader.synthetic_scenes[this_token][0]
            with open(file_path, "rb") as f:
                scene_dict_list = pickle.load(f)
            sensor_blobs_path = self.navsim._scene_loader._synthetic_sensor_path
            synthetic = True
        num_history_frames, num_future_frames = self.navsim._scene_loader._scene_filter.num_history_frames, self.navsim._scene_loader._scene_filter.num_future_frames
        sensor_config = self.navsim._scene_loader._sensor_config

        if not synthetic:
            this_scene_metadata = SceneMetadata(
                log_name=scene_dict_list[num_history_frames - 1]["log_name"],
                scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
                map_name=scene_dict_list[num_history_frames - 1]["map_location"],
                initial_token=scene_dict_list[num_history_frames - 1]["token"],
                num_history_frames=num_history_frames,
                num_future_frames=num_future_frames,
            )
        else:
            this_scene_metadata = SceneMetadata(**scene_dict_list["scene_metadata"])

            # only use frame
            scene_dict_list = scene_dict_list['frames']

        container_in["token"] = this_token
        container_in["scene_metadata"] = this_scene_metadata

        if load_bev:
            scene = self.navsim._scene_loader.get_scene_from_token(this_token)
            ann = scene_dict_list[3]['anns']
            # add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            bevs = []
            for frame_idx in range(12):
                fig, ax = plot_bev_with_agent_ori(scene, frame_idx)

                gt_bev = figax_to_bev500(fig, ax, out_hw=(500, 500), rgb=True)

                bev = gt_bev
                bevs.append(gt_bev)
        else:
            bev = None

        frame_data_dict = []
        for frame_idx in range(len(scene_dict_list)):

            # if frame_idx !=3:
            #     continue

            if load_lidar == 1 and frame_idx <12 and frame_idx == 3:
                if synthetic:
                    pass
                    
                lidar_path = scene_dict_list[frame_idx]["lidar_path"]
                lidar = Lidar.from_paths(sensor_blobs_path, lidar_path, ['lidar_pc']).lidar_pc

            # if synthetic:
            #     global_ego_status = scene_dict_list[frame_idx]['ego_status']
            #     annotations = scene_dict_list[frame_idx]['annotations']
            # else:
            #     global_ego_status = Scene._build_ego_status(
            #         scene_dict_list[frame_idx])
            #     annotations = Scene._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            if len(sensor_names) > 0:
                sensor_names = sensor_names[:-1]  # DROP lidar_pc

            this_frame_cameras = {}
            if not synthetic:
                camera_dict = scene_dict_list[frame_idx]["cams"]
            else:
                camera_dict = scene_dict_list[frame_idx]['camera_dict']
            data_dict = {}
            for camera_name in camera_dict.keys():
                camera_identifier = camera_name.lower()
                if camera_identifier in sensor_names:
                    image_path = sensor_blobs_path / \
                        camera_dict[camera_name]["data_path"]
                    data_dict[camera_identifier] = {
                        "image_path": str(image_path),
                        "sensor2lidar_rotation": camera_dict[camera_name]["sensor2lidar_rotation"],
                        "sensor2lidar_translation": camera_dict[camera_name]["sensor2lidar_translation"],
                        "intrinsics": camera_dict[camera_name]["cam_intrinsic"],
                        "distortion": camera_dict[camera_name]["distortion"],
                    }

                else:
                    data_dict[camera_identifier] = {}  # empty camera
            this_frame_cameras = data_dict  # rename it


            frame_data_dict.append({
                "token": scene_dict_list[frame_idx]["token"],
                "timestamp": scene_dict_list[frame_idx]["timestamp"],
                # "ego_status": global_ego_status if not synthetic else EasyDict(global_ego_status),
                # "annotations": annotations if not synthetic else EasyDict(annotations),
                "cameras": this_frame_cameras,
                # :5
                # "lidar": lidar,
                # "ego2global": ego2global,
                # "dynamic_mask": dynamic_masks
            })
        container_in["frame_data"] = frame_data_dict

        container_in["ego_status"] = []
        # # test set still has pos?
        # for frame in frame_data_dict:
        #     container_in["ego_status"].append(frame["ego_status"])

        container_in['images'] = []
        for frame in frame_data_dict:
            container_in["images"].append(frame["cameras"])

        container_in["bev"] = bevs
        return container_in

    def cache_queries(self, query_filename, pipeline, max_len=8000):
        assert query_filename.endswith(".json")
        # cache all queries
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"

        all_queries = []
        for idx in tqdm(range(len(self)), desc="Caching queries"):
            batch = self.get_container_in(idx)
            container_out = {}
            # container_out = self.prompt_metadata(container_out, batch)
            for self_pipeline in self.pipelines:
                if self_pipeline != pipeline:
                    container_out = self_pipeline(container_out, batch)
                else:
                    break  # for caching

                if self_pipeline.cache_response_filename is not None and 'navsim_meta_planning.json' in self_pipeline.cache_response_filename:
                    break  # stop it!

            query = pipeline.cache_construct_query(container_out, batch)
            all_queries.append(query)

        # divide the queries into chunks, and save them
        if len(all_queries) > max_len:
            num_chunks = math.ceil(len(all_queries) / max_len)
            chunk_size = math.ceil(len(all_queries) / num_chunks)
            for i in range(num_chunks):
                chunk_queries = all_queries[i *
                                            chunk_size: (i + 1) * chunk_size]
                chunk_filename = query_filename.replace(
                    ".json", "_{}.json".format(i))
                with open(chunk_filename, "w") as f:
                    json.dump(chunk_queries, f)
        else:
            with open(query_filename, "w") as f:
                json.dump(all_queries, f)

    def cache_responses(self, response_filename, pipeline, max_len=8000):
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"
        assert response_filename.endswith(".jsonl")
        all_responses = []

        if len(self) > max_len:
            num_chunks = math.ceil(len(self) / max_len)
            for i in range(num_chunks):
                chunk_filename = response_filename.replace(
                    ".jsonl", "_{}.jsonl".format(i))
                with open(chunk_filename, "r") as f:
                    for line in f:
                        all_responses.append(json.loads(line))
        else:
            with open(response_filename, "r") as f:
                for line in f:
                    all_responses.append(json.loads(line))
        assert len(all_responses) == len(
            self), "Length of responses and dataset do not match: {} vs {}".format(len(all_responses), len(self))
        for idx, response in enumerate(tqdm(all_responses)):
            batch = self.get_container_in(idx)
            container_out = {}
            for self_pipeline in self.pipelines:
                if self_pipeline != pipeline:
                    container_out = self_pipeline(container_out, batch)
                else:
                    break
            pipeline.cache_from_response(response, container_out, batch)
        pipeline.cleanup()
        return len(all_responses)

    def __getitem__(self, idx):
        batch = self.get_container_in(idx)
        batch["idx"] = idx
        # Let's forward the pipelines with container_in!

        # pipeline
        container_out = {}
        for pipeline in self.pipelines:
            container_out = pipeline(container_out, batch)
        # pipeline
        for key in self.container_out_key_comb:
            assert key in container_out["buffer_container"], "Key {} not found in container_out.buffer_container".format(
                key)
            container_out["messages"][1]["content"] += container_out["buffer_container"][key]

        container_out.pop("buffer_container")

        return container_out


def generate_batch_dataset():
    experiments = [
        {"id": "camera_ego_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["planning"]},

        {"id": "camera_ego_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["meta_planning", "planning"]},

        {"id": "camera_ego_road_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "road_agent_analysis"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["road_agent_analysis", "meta_planning", "planning"]},

        {"id": "camera_ego_scene_road_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "scene_description"},
            {"type": "road_agent_analysis"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["scene_description", "road_agent_analysis", "meta_planning", "planning"]}
    ]

    for experiment in experiments:
        v0_pipelines = experiment["pipelines"]
        v0_container_out_key_comb = experiment["container_out_key_comb"]

        dataset = VLMNavsim(mode="test", pipelines=v0_pipelines,
                            container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"navsim_test_{experiment['id']}.json")

        dataset = VLMNavsim(mode="train", pipelines=v0_pipelines,
                            container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"navsim_train_{experiment['id']}.json")


if __name__ == "__main__":
    generate_batch_dataset()

    # Load the config
    # dataset = VLMNavsim(mode="test", pipelines=v0_pipelines, container_out_key_comb=v0_container_out_key_comb)
    # batch = dataset[240]

    # dataset.cache_data("navsim_test_v00.json")
    # dataset.cache_queries("data/navsim/navsim_test_mp_queries.json", dataset.pipelines[3])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_mp_queries.jsonl", dataset.pipelines[3])
    # dataset.cache_queries("data/navsim/navsim_test_ra_queries.json", dataset.pipelines[-2])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_ra_queries.jsonl", dataset.pipelines[-2])
    # dataset.cache_queries("data/navsim/navsim_test_sd_queries.json", dataset.pipelines[-1])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_sd_queries.jsonl", dataset.pipelines[-1])
    # dataset.cache_data("navsim_test_full.json")