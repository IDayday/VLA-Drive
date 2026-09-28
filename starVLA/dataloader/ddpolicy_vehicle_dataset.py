"""Strict data adapter retaining DDP camera/ego/video processing.

No original loader fallback to the previous scene, no driving hidden caches,
and no label-derived graph inputs. GT targets and generic depth are separate.
"""
import copy
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from starVLA.dataloader.navsim_dataset import NavSimDataset
from starVLA.model.modules.vehicle_joint.targets import SCHEMA
from tools.ddpolicy_vehicle.prepare_data import load_trusted


class DDPVehicleDataset(Dataset):
    def __init__(self, tokens_file, processed_root, vehicle_root, depth_root, config,
                 sensor_roots=(), training=True):
        if os.environ.get("NAVSIM_FEATURE_CACHE_ROOT"):
            raise ValueError("Old feature cache environment must be unset for from-scratch training")
        if config.enable_image_aug or config.doing_s2 or config.vit_pre:
            raise ValueError("Use the fixed original input protocol without driving proxies/augmentation")
        self.tokens = json.loads(Path(tokens_file).read_text())
        if not isinstance(self.tokens, list) or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Dataset requires a unique explicit token manifest")
        self.processed_root, self.vehicle_root = Path(processed_root), Path(vehicle_root)
        self.depth_root = Path(depth_root) if depth_root else None
        self.training = training
        self.sensor_roots = [Path(x) for x in sensor_roots]
        self.identity = json.loads((self.vehicle_root/"identity.json").read_text())["identity"]
        # Reuse only the actual sample transformations; bypass its __getitem__
        # that catches exceptions and substitutes an earlier scene.
        local = OmegaConf.create(OmegaConf.to_container(config))
        local.w_depth = 0
        if not training: local.datasets.video_data.load_2d_data = 0
        self.recipe = NavSimDataset(tokens_file, split="train", video_data_cfg=local.datasets.video_data,
            gs_data_cfg=local.datasets.gs_data, reward_data_cfg=local.datasets.reward_data,
            ver_1225=1, dataset_cfg=local.datasets.vla_data, all_cfg=local,
            data_root=str(self.processed_root.parent.parent))
        self.recipe.video_source = "images"
        self.use_depth = bool(config.w_depth) and training
        self.depth_identity = None
        if self.use_depth:
            from starVLA.model.modules.vehicle_joint.initialization import identity_hash
            self.depth_identity = identity_hash(json.loads((self.depth_root/"identity.json").read_text()))

    def __len__(self): return len(self.tokens)

    def resolve_image(self, path):
        path = Path(path)
        if path.is_file(): return str(path)
        relative = Path(*path.parts[-3:])
        for root in self.sensor_roots:
            candidate = root/relative
            if candidate.is_file(): return str(candidate)
        raise FileNotFoundError(f"Required camera frame is missing: {relative}")

    def __getitem__(self, idx):
        token = self.tokens[idx]
        raw = load_trusted(self.processed_root/(token+".pkl"))
        for cam in ("cam_f0", "cam_l0", "cam_r0"):
            paths = raw["glo_images"][cam]["image_paths"]
            # Deployment reads only t0; future paths are not resolved or opened.
            for t in (range(3, 12) if self.training else (3,)):
                paths[t] = self.resolve_image(paths[t])
        sample = (self.recipe._get_sample(raw, token, None, cached_features={}) if self.training
                  else self.current_sample(raw, token))
        with np.load(self.vehicle_root/"observations"/(token+".npz")) as obs:
            if str(obs["identity"]) != self.identity or str(obs["schema"]) != SCHEMA:
                raise ValueError("Current observation identity mismatch")
            sample["current_calibration"] = {k: obs[k].copy() for k in ("intrinsics", "extrinsics", "distortion")}
        sample["ego_speed"] = float(np.linalg.norm(np.asarray(raw["glo_status"]["velocities"])[3, :2]))
        sample["navigation"] = int(np.asarray(raw["glo_status"]["commands"])[3].argmax())
        if self.training:
            # Initial v1 records stored np.str_ track audit IDs. Allow only the
            # narrow NumPy scalar/string constructors, retaining weights_only.
            with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype,
                                                    type(np.dtype("U16")), np.str_]):
                payload = torch.load(self.vehicle_root/"targets"/(token+".pt"), map_location="cpu", weights_only=True)
            payload["targets"]["track_ids"] = tuple(str(x) for x in payload["targets"]["track_ids"])
            if payload["schema"] != SCHEMA or payload["identity"] != self.identity:
                raise ValueError("Vehicle target identity mismatch")
            sample["vehicle_targets"] = payload["targets"]
            sample["vehicle_population"] = payload["counts"]
        if self.use_depth:
            with np.load(self.depth_root/(token+".npz")) as depth:
                if str(depth["identity"]) != self.depth_identity: raise ValueError("Generic depth identity mismatch")
                maps = depth["depth"][[1, 0, 2]]
            maps = np.stack([cv2.resize(m, (256, 144), interpolation=cv2.INTER_NEAREST) for m in maps])
            rgbs = np.stack([np.asarray(sample["image"][i].resize((256, 144), Image.Resampling.LANCZOS), dtype=np.float32)/255.
                             for i in (1, 0, 2)])
            sample["depth_data"] = {"image": torch.from_numpy(rgbs).permute(0, 3, 1, 2),
                                     "depth": torch.from_numpy(maps)[:, None],
                                     "mask": torch.from_numpy(maps > .1)[:, None]}
        if not self.training:
            sample = {k: sample[k] for k in ("image", "state", "lang", "token",
                                            "current_calibration", "ego_speed", "navigation")}
        return sample

    def current_sample(self, raw, token):
        """Exact DDP current prompt/history, without reading future pose labels."""
        from starVLA.dataloader.navsim_dataset import x_mean, x_std, y_mean, y_std
        history = np.asarray(raw["glo_status"]["global_poses"])[:4]
        if history.shape != (4, 3) or not np.isfinite(history).all():
            raise ValueError("Invalid allowed ego pose history")
        previous, current = history[2], history[3]
        c, s = np.cos(current[2]), np.sin(current[2])
        dx, dy = (current[:2]-previous[:2]) @ np.array([[c, -s], [s, c]])
        yaw = (current[2]-previous[2]+np.pi) % (2*np.pi)-np.pi
        state = np.array([[(dx-x_mean)/x_std, (dy-y_mean)/y_std, np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
        navigation = int(np.asarray(raw["glo_status"]["commands"])[3].argmax())
        command = ("turn left", "keep straight", "turn right", "unknown")[navigation]
        prompt = (f"You are an autonomous driving agent. The navigation command for the current timestep is {command}. "
                  "Your task is to plan future actions based on the understanding of the driving scene.")
        images = [self.recipe._load_image(raw["glo_images"][cam]["image_paths"][3])
                  for cam in ("cam_f0", "cam_l0", "cam_r0")]
        return {"image": images, "state": state, "lang": prompt, "token": token}
