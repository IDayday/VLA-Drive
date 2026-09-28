"""Build versioned vehicle labels and a separate current-observation whitelist.

Trusted local pickle inputs only. No model-generated cache is read. No test split
is used here. Per-log output is atomic and resumable by an exact identity.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch

from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics, geometric_fov
from starVLA.model.modules.vehicle_joint.targets import make_vehicle_targets, SCHEMA
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")


class TrustedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_trusted(path):
    with Path(path).open("rb") as stream:
        return TrustedNumpyUnpickler(stream).load()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(tmp, path)


def process_log(task):
    log, tokens, config = task
    out = Path(config["output"])
    report_path = out / "logs" / (log + ".json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != config["identity"]:
            raise ValueError("Existing per-log output has a different identity")
        for row in report["records"]:
            if "error" not in row:
                for subdir, suffix in (("targets", ".pt"), ("observations", ".npz")):
                    if not (out / subdir / (row["token"] + suffix)).is_file():
                        raise FileNotFoundError("Completed record is missing its artifact")
        return report
    raw_path = Path(config["raw_log_root"]) / (log + ".pkl")
    frames = load_trusted(raw_path)
    index = {f["token"]: i for i, f in enumerate(frames)}
    if len(index) != len(frames):
        raise ValueError("Duplicate raw frame identity")
    rows = []
    for token in tokens:
        row = {"token": token, "log": log}
        try:
            i = index[token]
            current = frames[i]
            if not np.allclose(current["lidar2ego"], np.eye(4), atol=1e-6):
                raise ValueError("Current annotation frame adapter required for nonidentity lidar2ego")
            meta = load_trusted(Path(config["processed_root"]) / (token + ".pkl"))
            ks, es, affine, distortions, paths = [], [], [], [], []
            # Metadata/observation alignment is checked against the raw decision
            # frame, never inferred from an adjacent frame or future bbox.
            for cam in CAMERAS:
                raw_cam = current["cams"][cam]
                p = Path(config["sensor_root"]) / raw_cam["data_path"]
                expected = meta["glo_images"][cam.lower()]["image_paths"][3]
                if Path(expected).name != p.name:
                    raise ValueError("Current image timestamp mismatch")
                with Image.open(p) as image:
                    k, a = crop_resize_intrinsics(raw_cam["cam_intrinsic"], image.size, (1024, 576))
                e = np.eye(4)
                e[:3, :3] = raw_cam["sensor2lidar_rotation"]
                e[:3, 3] = raw_cam["sensor2lidar_translation"]
                ks.append(k); es.append(e); affine.append(a)
                distortions.append(raw_cam["distortion"]); paths.append(str(p))
            for frame in frames[i:i + 9]:
                for cam in CAMERAS:
                    if not (Path(config["sensor_root"]) / frame["cams"][cam]["data_path"]).is_file():
                        raise FileNotFoundError("Missing original-recipe current/future camera frame")
            if len(frames[i:i + 9]) != 9:
                raise ValueError("Insufficient future frames for original video/ego labels")
            boxes = np.asarray(current["anns"]["gt_boxes"]) if current.get("anns") else np.empty((0, 7))
            support = geometric_fov(boxes[:, :3], np.array(ks), np.array(es), np.array(distortions))
            targets, counts = make_vehicle_targets(current, frames[i+1:i+9],
                                                   capacity=config["capacity"], current_eligibility=support)
            xx, yy = np.meshgrid(np.arange(1.5, 50), np.arange(-19.5, 20), indexing="ij")
            coverage = np.zeros(xx.shape, dtype=bool)
            for z in (0., 1., 2.):
                points = np.stack((xx, yy, np.full_like(xx, z)), -1).reshape(-1, 3)
                coverage |= geometric_fov(points, np.array(ks), np.array(es), np.array(distortions)).reshape(xx.shape)
            targets.supervision_grid = torch.from_numpy(coverage)
            payload = {"schema": SCHEMA, "identity": config["identity"], "targets": asdict(targets)}
            # Label-only raw current vehicles preserve low-support/capacity
            # population for coverage audits; not formal prediction graph inputs.
            names = np.asarray(current["anns"]["gt_names"])
            vehicle = names == "vehicle"
            payload["source_vehicle_boxes"] = torch.from_numpy(boxes[vehicle].copy())
            payload["source_vehicle_support"] = torch.from_numpy(support[vehicle].copy())
            payload["counts"] = counts
            dst = out / "targets" / (token + ".pt")
            tmp = dst.with_suffix(".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, dst)
            # Whitelist: no annotations, IDs, label counts or future paths.
            obs = out / "observations" / (token + ".npz")
            tmp = obs.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez(stream, intrinsics=np.array(ks), extrinsics=np.array(es),
                         image_transforms=np.array(affine), distortion=np.array(distortions),
                         timestamp=np.array(current["timestamp"]), camera_names=np.array(CAMERAS),
                         image_paths=np.array(paths), schema=np.array(SCHEMA), identity=np.array(config["identity"]))
            os.replace(tmp, obs)
            row.update(counts)
            row["low_support_vehicles"] = int((~support[vehicle]).sum())
            row["target_sha256"] = file_sha256(dst)
            row["observation_sha256"] = file_sha256(obs)
        except Exception as error:
            row["error"] = repr(error)
        rows.append(row)
    report = {"identity": config["identity"], "raw_log_sha256": file_sha256(raw_path), "records": rows}
    atomic_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("split-manifest", "processed-root", "raw-log-root", "sensor-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0 means full fixed train+dev; nonzero is a separate smoke cache")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    split = json.loads(Path(a.split_manifest).read_text())
    train, dev = split["train_tokens"], split["dev_tokens"]
    if len(set(train)) != len(train) or len(set(dev)) != len(dev) or set(train) & set(dev):
        raise ValueError("Invalid or overlapping fixed token split")
    logs = split["token_logs"]
    if {logs[t] for t in train} & {logs[t] for t in dev}:
        raise ValueError("Training/dev log overlap")
    tokens = train + dev
    if a.limit: tokens = tokens[:a.limit]
    source = {"schema": SCHEMA, "split_sha256": file_sha256(a.split_manifest),
              "capacity": a.capacity, "limit": a.limit, "cameras": CAMERAS,
              "roi": [1, -20, 50, 20], "time_step": .5, "steps": 8,
              "raw_log_root": str(Path(a.raw_log_root).resolve()),
              "processed_root": str(Path(a.processed_root).resolve()),
              "sensor_root": str(Path(a.sensor_root).resolve()),
              "builder_sha256": file_sha256(__file__),
              "target_code_sha256": file_sha256(Path(__file__).parents[2] / "starVLA/model/modules/vehicle_joint/targets.py")}
    ident = identity_hash(source)
    out = Path(a.output)
    if out.exists():
        if not a.resume or json.loads((out / "identity.json").read_text())["identity"] != ident:
            raise FileExistsError("Use a new schema directory or exact --resume")
    else:
        out.mkdir(parents=True)
        for directory in ("logs", "targets", "observations"): (out / directory).mkdir()
        atomic_json(out / "identity.json", {"identity": ident, **source})
        token_set = set(tokens)
        atomic_json(out / "split.json", {"train_tokens": [t for t in train if t in token_set],
                                         "dev_tokens": [t for t in dev if t in token_set]})
    grouped = {}
    for token in tokens: grouped.setdefault(logs[token], []).append(token)
    config = vars(a) | {"identity": ident}
    counters, failures, completed = Counter(), [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for report in pool.map(process_log, [(log, values, config) for log, values in grouped.items()]):
            for row in report["records"]:
                if "error" in row: failures.append(row)
                else:
                    completed += 1
                    for key, value in row.items():
                        if isinstance(value, int) and not isinstance(value, bool): counters[key] += value
            atomic_json(out / "progress.json", {"requested": len(tokens), "completed": completed,
                                                "failures": len(failures), "counts": dict(counters)})
    atomic_json(out / "audit.json", {"identity": ident, "requested": len(tokens), "completed": completed,
                "failures": failures, "counts": dict(counters), "source_population": "raw current log annotations",
                "input_population": "camera-only; no GT graph", "raw_log_population_available": True,
                "support_limitation": "Geometric FOV only, not occlusion visibility",
                "train_count": len(train), "dev_count": len(dev), "token_overlap": 0, "log_overlap": 0})
    print(json.dumps({"completed": completed, "failures": len(failures), "identity": ident}), flush=True)
    if failures: raise RuntimeError("Data failures retained; training cannot silently skip these scenes")


if __name__ == "__main__": main()
