"""Export allowed CURRENT camera observations only, with no annotation fields."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from .prepare_data import load_trusted, atomic_json, camera_path, CAMERAS
from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ("index", "raw-log-root", "sensor-root", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--fallback-sensor-root", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--split", choices=("train", "dev", "navtest"), required=True)
    a = p.parse_args()
    # Index may contain metric-cache paths, but they are NEVER read or exported.
    index = [{k:r[k] for k in ("token", "log")} for r in json.loads(Path(a.index).read_text())]
    if len({r["token"] for r in index}) != len(index): raise ValueError("Duplicate scene identity")
    identity = {"schema": "ddpolicy_current_cameras_v1", "split": a.split, "index_sha256": identity_hash(index),
                "raw_log_root": str(Path(a.raw_log_root).resolve()), "sensor_root": a.sensor_root,
                "fallback_sensor_root": a.fallback_sensor_root, "cameras": CAMERAS,
                "image_size": [1024, 576], "history_poses": 4, "history_images": 0,
                "annotations_in_cache": False, "future_in_cache": False,
                "writer_sha256": file_sha256(__file__)}
    sha = identity_hash(identity)
    out = Path(a.output)
    if out.exists():
        if not a.resume or json.loads((out/"identity.json").read_text())["identity"] != sha:
            raise FileExistsError("Current cache exists with a different identity or without --resume")
    else:
        (out/"current").mkdir(parents=True)
        atomic_json(out/"identity.json", {"identity": sha, **identity})
        atomic_json(out/"index.json", index)
    groups = defaultdict(list)
    for row in index: groups[row["log"]].append(row["token"])
    failures = []; completed = 0
    for log, tokens in groups.items():
        frames = load_trusted(Path(a.raw_log_root)/(log+".pkl"))
        locations = {f["token"]:i for i,f in enumerate(frames)}
        for token in tokens:
            destination = out/"current"/(token+".json")
            if destination.exists():
                existing = json.loads(destination.read_text())
                if existing["identity"] != sha or existing["token"] != token or existing["log"] != log:
                    raise ValueError("Existing current observation has changed identity")
                completed += 1; continue
            try:
                at = locations[token]
                if at < 3: raise ValueError("Insufficient allowed ego history")
                current = frames[at]
                poses = []
                for frame in frames[at-3:at+1]:
                    if frame["timestamp"] > current["timestamp"]: raise ValueError("Future history input")
                    xy = frame["ego2global_translation"][:2]
                    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
                    poses.append([float(xy[0]), float(xy[1]), float(yaw)])
                images, ks, es, distortions = [], [], [], []
                for cam in CAMERAS:
                    c = current["cams"][cam]
                    image = camera_path(c["data_path"], vars(a))
                    with Image.open(image) as im: k, _ = crop_resize_intrinsics(c["cam_intrinsic"], im.size, (1024, 576))
                    e = np.eye(4); e[:3,:3] = c["sensor2lidar_rotation"]; e[:3,3] = c["sensor2lidar_translation"]
                    e = np.asarray(current.get("lidar2ego", np.eye(4)))@e
                    images.append(str(image)); ks.append(k.tolist()); es.append(e.tolist()); distortions.append(np.asarray(c["distortion"]).tolist())
                payload = {"identity": sha, "token": token, "log": log, "timestamp": int(current["timestamp"]),
                    "image_paths": images, "global_pose_history": poses,
                    "ego_speed": float(np.linalg.norm(current["ego_dynamic_state"][:2])),
                    "navigation": int(np.asarray(current["driving_command"]).argmax()),
                    "current_calibration": {"intrinsics": ks, "extrinsics": es, "distortion": distortions}}
                atomic_json(destination, payload); completed += 1
            except Exception as error: failures.append({"token": token, "log": log, "error": repr(error)})
        atomic_json(out/"progress.json", {"completed": completed, "requested": len(index), "failed": len(failures)})
    atomic_json(out/"audit.json", {"completed": completed, "requested": len(index), "failures": failures, "identity": sha})
    if failures: raise RuntimeError("Current-only cache failures retained")


if __name__ == "__main__": main()
