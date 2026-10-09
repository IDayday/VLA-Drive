"""NAVSIM inference Agent: current images and legal ego history only."""
from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .contracts import require
from .export import load_bundle


def observation_from_agent_input(agent_input):
    require(len(agent_input.ego_statuses) == 4 and len(agent_input.cameras) == 4, "original S0 requires four legal ego-history frames")
    poses = np.asarray([e.ego_pose for e in agent_input.ego_statuses], dtype=np.float64)
    require(poses.shape == (4, 3) and np.isfinite(poses).all(), "invalid legal ego history")
    require(len({e.in_global_frame for e in agent_input.ego_statuses}) == 1, "mixed ego coordinate frames")
    previous, current = poses[2:4]
    c, s = np.cos(current[2]), np.sin(current[2])
    xy = (current[:2] - previous[:2]) @ np.array([[c, -s], [s, c]])
    yaw = (current[2] - previous[2] + np.pi) % (2 * np.pi) - np.pi
    from starVLA.dataloader.foresight_dataset import encode_ego
    state = encode_ego(np.array([[*xy, yaw]]))
    command = np.asarray(agent_input.ego_statuses[-1].driving_command)
    require(command.shape == (4,) and np.isfinite(command).all() and command.sum() == 1 and np.all((command == 0) | (command == 1)), "navigation command must be original legal four-way one-hot")
    navigation = int(command.argmax())
    images = []
    for name in ("cam_f0", "cam_l0", "cam_r0"):
        pixels = getattr(agent_input.cameras[-1], name).image
        require(pixels is not None and pixels.dtype == np.uint8, "original uint8 camera observation required")
        img = Image.fromarray(pixels).convert("RGB")
        w, h = img.size
        if w / h > 16 / 9:
            cw = int(h * 16 / 9); img = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h))
        elif w / h < 16 / 9:
            ch = int(w * 9 / 16); img = img.crop((0, (h - ch) // 2, w, (h + ch) // 2))
        images.append(img.resize((1024, 576), Image.Resampling.LANCZOS))
    text = ("turn left", "keep straight", "turn right", "unknown")[navigation]
    return {"image": images, "state": state, "lang": f"You are an autonomous driving agent. The navigation command for the current timestep is {text}. Your task is to plan future actions based on the understanding of the driving scene.", "token": "online"}


class IQEAgent(AbstractAgent):
    def __init__(self, bundle_path, device="cuda", allow_smoke=False):
        super().__init__(requires_scene=False)
        self.bundle_path, self.device, self.allow_smoke = bundle_path, device, allow_smoke
        self.model, self.rule, self.metadata = None, None, None

    def name(self):
        return "iqe_v1"

    def get_sensor_config(self):
        return SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False, cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def initialize(self):
        self.model, self.rule, self.metadata = load_bundle(self.bundle_path, self.device)
        require(self.allow_smoke or self.metadata["mode"] == "full", "smoke bundle is diagnostic only")

    def forward(self, observations):
        require(self.model is not None, "initialize Agent before inference")
        if self.metadata["selector_type"] == "scene_router":
            trajectory, ids = self.model.forward_prerouted(observations, self.rule)
        else:
            trajectory, ids, _ = self.model.predict(observations, self.rule)
        return {"trajectory": trajectory, "selected_expert_ids": ids}

    @torch.no_grad()
    def compute_trajectory(self, agent_input):
        output = self.forward([observation_from_agent_input(agent_input)])
        return Trajectory(output["trajectory"][0].cpu().numpy(), TrajectorySampling(num_poses=self.metadata["contract"]["trajectory"]["horizon"], interval_length=self.metadata["contract"]["trajectory"]["dt"]))
