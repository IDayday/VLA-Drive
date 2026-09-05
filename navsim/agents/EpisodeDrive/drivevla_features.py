from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory, Annotations
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .drivevla_backbone import DriveVLABackbone
from .utils.internvl_preprocess import load_image
from .layers.world_model.future_image_io import encode_path_tensor


from enum import IntEnum
import cv2
import numpy.typing as npt

import torch

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
# from .bevformer.bev_feature_build import _get_bev_feature

from PIL import Image
from scipy.interpolate import CubicSpline

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class DriveVLAFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False,
                 return_vision: bool = False):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
        """
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode
        self.return_vision=return_vision

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = DriveVLABackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            
            path_as_ordinals = [ord(char) for char in image_path]
            
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            
            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
        else:
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            
            pixel_values = load_image(str(cameras[-1].cam_f0.image),max_num=12).unsqueeze(0)

            pixel_values_squeezed = pixel_values.squeeze(1)
            num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
            pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

            navigation_commands = ['turn left', 'go straight', 'turn right',"unknown"]
            # command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
            command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1))
            history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
            
            prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
            output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
            questions = [f"{prompt}{output_requirements}"]

            outputs = self.backbone(pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list,return_vision=self.return_vision)
            if self.return_vision:
                last_hidden_state = outputs.hidden_states[-1]
                print("-------last_hiddent_state.shape",last_hidden_state.shape)
            else:
                last_hidden_state = outputs.hidden_states[-1]

            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                "status_feature": status_feature.cpu(),
            }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    FUTURE_OFFSETS = (1, 3, 6)
    FUTURE_HORIZONS_SEC = (0.5, 1.5, 3.0)
    MAX_PATH_BYTES = 1024

    def __init__(self, config: Dict, world_model_config=None):
        self._config = config
        self._world_model_config = world_model_config
        self._task_future_lite = getattr(world_model_config, "mode", "legacy_register_prediction") == "task_future_lite"
        self._future_supervision_enabled = bool(
            world_model_config is not None
            and getattr(world_model_config, "enabled", False)
        )
        if self._future_supervision_enabled:
            horizons = tuple(
                float(value)
                for value in getattr(
                    world_model_config,
                    "horizons_sec",
                    self.FUTURE_HORIZONS_SEC,
                )
            )
            if horizons != self.FUTURE_HORIZONS_SEC:
                raise ValueError(
                    "PlanReg-WM-V1 future horizons must be [0.5,1.5,3.0], "
                    f"got {horizons}"
                )

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        if self._future_supervision_enabled and self._task_future_lite:
            return "trajectory_target_task_future_lite_v1"
        return (
            "trajectory_target_planreg_wm_v1"
            if self._future_supervision_enabled
            else "trajectory_target"
        )

    def _build_future_image_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        path_tensors = torch.zeros(
            len(self.FUTURE_OFFSETS),
            self.MAX_PATH_BYTES,
            dtype=torch.uint8,
        )
        path_lengths = torch.zeros(len(self.FUTURE_OFFSETS), dtype=torch.long)
        valid_mask = torch.zeros(len(self.FUTURE_OFFSETS), dtype=torch.bool)
        current_idx = int(scene.scene_metadata.num_history_frames) - 1

        for horizon_index, frame_offset in enumerate(self.FUTURE_OFFSETS):
            frame_index = current_idx + frame_offset
            if frame_index >= len(scene.frames):
                continue
            image = scene.frames[frame_index].cameras.cam_f0.image
            if image is None:
                continue
            if not isinstance(image, (str, Path)):
                raise RuntimeError(
                    "PlanReg-WM-V1 future supervision requires camera images "
                    "to be loaded as paths. Set load_image_path=true."
                )
            encoded, length = encode_path_tensor(
                image, max_bytes=self.MAX_PATH_BYTES
            )
            path_tensors[horizon_index] = encoded
            path_lengths[horizon_index] = length
            valid_mask[horizon_index] = Path(image).is_file()

        return {
            "future_image_paths": path_tensors,
            "future_image_path_lengths": path_lengths,
            "future_valid_mask": valid_mask,
        }

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._config.trajectory_sampling.num_poses
            ).poses
        )
        # frame_idx = scene.scene_metadata.num_history_frames - 1
        # annotations = scene.frames[frame_idx].annotations
        # ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        # agent_states, agent_labels = self._compute_agent_targets(annotations)
        # bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        if self._config.long_trajectory_additional_poses > 0:
            trajectory_long = scene.get_future_trajectory(
                    num_trajectory_frames=self._config.trajectory_sampling.num_poses + self._config.long_trajectory_additional_poses
                ).poses
            x = np.arange(trajectory_long.shape[0], dtype=np.float32)
            alpha = 2 * self._config.long_trajectory_additional_poses / (self._config.trajectory_sampling.num_poses*(self._config.trajectory_sampling.num_poses+1))
            x_new = np.arange(trajectory.shape[0], dtype=np.float32)
            off_sets = np.cumsum((x_new+1)*alpha)
            x_new += off_sets
            traj_ = []
            for i in range(3):
                y = trajectory_long[:,i]
                cs = CubicSpline(x, y)
                traj_.append(cs(x_new))
            trajectory_long = np.stack(traj_, axis=1)

            trajectory_long = torch.tensor(trajectory_long)
            targets = {
                "trajectory": trajectory,
                "trajectory_long": trajectory_long,
                "token":scene.scene_metadata.initial_token
            }
        else:
            targets = {
                "trajectory": trajectory,
                # "agent_states": agent_states,
                # "agent_labels": agent_labels,
                # "bev_semantic_map": bev_semantic_map,
                "token":scene.scene_metadata.initial_token
            }
        if self._future_supervision_enabled:
            targets.update(self._build_future_image_targets(scene))
            if self._task_future_lite:
                from .layers.world_model.logged_future_pose import build_logged_future_pose_metadata
                targets.update(build_logged_future_pose_metadata(scene))
        return targets
