"""Deployment observations: no access to labels, raw logs, or future files."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def current_example(record):
    from starVLA.dataloader.navsim_dataset import x_mean, x_std, y_mean, y_std
    allowed = {"identity", "token", "log", "timestamp", "image_paths", "global_pose_history",
               "ego_speed", "navigation", "current_calibration"}
    if set(record) != allowed: raise ValueError("Current input whitelist mismatch")
    if record["navigation"] not in (0, 1, 2, 3) or not np.isfinite(record["ego_speed"]) or record["ego_speed"] < 0:
        raise ValueError("Invalid current navigation/speed")
    poses = np.asarray(record["global_pose_history"], dtype=np.float64)
    if poses.shape != (4, 3) or not np.isfinite(poses).all(): raise ValueError("Invalid ego history")
    prev, now = poses[2], poses[3]
    c, s = np.cos(now[2]), np.sin(now[2])
    dx, dy = (now[:2]-prev[:2])@np.array([[c, -s], [s, c]])
    yaw = (now[2]-prev[2]+np.pi) % (2*np.pi)-np.pi
    state = np.array([[(dx-x_mean)/x_std, (dy-y_mean)/y_std, np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
    images = []
    if len(record["image_paths"]) != 3: raise ValueError("Expected exactly three current front cameras")
    for path in record["image_paths"]:
        with Image.open(path) as source:
            image = source.convert("RGB"); w, h = image.size
            if w/h > 16/9:
                cw = int(h*16/9); image = image.crop(((w-cw)//2, 0, (w+cw)//2, h))
            elif w/h < 16/9:
                ch = int(w*9/16); image = image.crop((0, (h-ch)//2, w, (h+ch)//2))
            images.append(image.resize((1024, 576), Image.Resampling.LANCZOS))
    command = ("turn left", "keep straight", "turn right", "unknown")[record["navigation"]]
    prompt = (f"You are an autonomous driving agent. The navigation command for the current timestep is {command}. "
              "Your task is to plan future actions based on the understanding of the driving scene.")
    calibration = {k:np.asarray(v) for k,v in record["current_calibration"].items()}
    expected = {"intrinsics": (3, 3, 3), "extrinsics": (3, 4, 4), "distortion": (3, 5)}
    if set(calibration) != set(expected) or any(calibration[k].shape != shape or not np.isfinite(calibration[k]).all() for k,shape in expected.items()):
        raise ValueError("Invalid current camera calibration")
    return {"image": images, "state": state, "lang": prompt, "token": record["token"],
            "ego_speed": record["ego_speed"], "navigation": record["navigation"], "current_calibration": calibration}


class CurrentCameraDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        identity = json.loads((self.root/"identity.json").read_text())
        if identity["schema"] != "ddpolicy_current_cameras_v1": raise ValueError("Invalid current-only schema")
        self.identity = identity["identity"]
        self.metadata = identity
        self.index = json.loads((self.root/"index.json").read_text())
        from starVLA.model.modules.vehicle_joint.initialization import identity_hash
        if identity_hash(self.index) != identity["index_sha256"]: raise ValueError("Current index changed")
    def __len__(self): return len(self.index)
    def __getitem__(self, index):
        row = self.index[index]
        record = json.loads((self.root/"current"/(row["token"]+".json")).read_text())
        if record["identity"] != self.identity: raise ValueError("Current observation cache identity mismatch")
        return current_example(record)
