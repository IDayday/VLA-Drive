"""Prepare full official navtrain without modifying shared data or NumPy installs.

Metadata is independently checked against each official raw scene window. Only
current image PATHS change to the verified alternate root when necessary.
Future labels remain supervision/reward data. No temporal/candidate fallback.
"""

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import signal
import traceback
import json
import pickle
import time
import numpy as np
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.contracts import digest


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace("numpy._core", "numpy.core"), name)


def prepare_metadata(root, workers):
    official = Path(
        "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    )
    cfg = OmegaConf.load(official)
    requested = set(cfg.tokens)
    raw = Path("/mnt/project/DriveDreamer-Policy/navsim_raw")
    source = Path("/mnt/navsim/navsim_dataset/meta/train")
    missing = json.loads(
        Path("reports/ddp_flow_grpo_paired/missing_train_images.json").read_text()
    )
    if missing["recoverable_in_alternate_root"] != missing["missing_references"]:
        raise ValueError("full target images are not available")
    alternatives = {
        (row["token"], row["view"]): row["alternate"] for row in missing["rows"]
    }
    dest = root / "dataset/meta/train"
    dest.mkdir(parents=True, exist_ok=False)
    (root / "dataset/meta/test").symlink_to(
        "/mnt/project/DriveDreamer-Policy/navsim_dataset/meta/test",
        target_is_directory=True,
    )

    def log_job(log):
        with (raw / "navsim_logs/trainval" / f"{log}.pkl").open("rb") as f:
            frames = pickle.load(f)
        poses = np.asarray(
            [
                [
                    f["ego2global_translation"][0],
                    f["ego2global_translation"][1],
                    Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0],
                ]
                for f in frames
            ],
            dtype=np.float64,
        )
        result = []
        for start in range(0, len(frames), cfg.frame_interval):
            if start + cfg.num_history_frames + cfg.num_future_frames > len(frames):
                continue
            token = frames[start + cfg.num_history_frames - 1]["token"]
            if token not in requested:
                continue
            with (source / f"{token}.pkl").open("rb") as f:
                metadata = NumpyCompatUnpickler(f).load()
            target = poses[start : start + 14]
            actual = metadata["glo_status"]["global_poses"]
            if actual.shape != target.shape or not np.allclose(
                actual, target, atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"metadata does not match original official window: {token}"
                )
            commands = np.asarray(
                [f["driving_command"] for f in frames[start : start + 14]]
            )
            if not np.array_equal(metadata["glo_status"]["commands"], commands):
                raise ValueError(f"driving command mismatch: {token}")
            current = frames[start + 3]
            images = []
            for lower in ("cam_f0", "cam_l0", "cam_r0"):
                upper = lower.upper()
                relative = current["cams"][upper]["data_path"]
                original = metadata["glo_images"][lower]["image_paths"][3]
                if Path(original).name != Path(relative).name:
                    raise ValueError(f"current image/window mismatch: {token}/{lower}")
                image = alternatives.get(
                    (token, upper), str(raw / "sensor_blobs/trainval" / relative)
                )
                metadata["glo_images"][lower]["image_paths"][3] = image
                images.append(image)
            path = dest / f"{token}.pkl"
            temporary = path.with_suffix(".partial")
            with temporary.open("xb") as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
            result.append(
                {
                    "token": token,
                    "log": log,
                    "metadata": str(path.resolve()),
                    "images": images,
                    "pose_max_abs": float(np.max(np.abs(actual - target))),
                }
            )
        return result

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(log_job, list(cfg.log_names))):
            records.extend(rows)
            if i % 50 == 0:
                print(json.dumps({"logs": i + 1, "scenes": len(records)}), flush=True)
    if {r["token"] for r in records} != requested or len(records) != len(requested):
        raise ValueError("full prepared token set differs from official navtrain")
    (root / "metadata_records.json").write_text(json.dumps(records))
    (root / "navtrain_tokens.json").write_text(json.dumps(sorted(requested)))
    groups = defaultdict(list)
    for row in records:
        groups[row["log"]].append(row["token"])
    prior = json.loads(
        Path("reports/ddp_flow_grpo_paired/data/split_manifest.json").read_text()
    )
    calibration_logs = {prior["token_logs"][t] for t in prior["calibration_tokens"]}
    ordered = sorted(groups, key=lambda log: digest(["rl_log_holdout_v1", log]))
    dev, dev_logs = [], []
    for log in ordered:
        if log in calibration_logs:
            continue
        dev_logs.append(log)
        dev.extend(sorted(groups[log]))
        if len(dev) >= 512 and len(dev_logs) >= 16:
            break
    train = sorted(requested - set(dev), key=digest)
    manifest = {
        "schema": 1,
        "selection": "whole logs, sha256(rl_log_holdout_v1,log), >=512 scenes and >=16 logs; exclude prior numerical calibration logs",
        "experiment_scope": "full official navtrain target with whole-log RL dev holdout",
        "official_filter_sha256": file_sha(official),
        "available_train_scenes": len(requested),
        "available_train_logs": len(groups),
        "train_tokens": train,
        "dev_tokens": dev,
        "train_scenes": len(train),
        "dev_scenes": len(dev),
        "dev_log_count": len(dev_logs),
        "dev_logs": dev_logs,
        "train_logs": sorted(set(groups) - set(dev_logs)),
        "token_logs": {r["token"]: r["log"] for r in records},
        "calibration_tokens": prior["calibration_tokens"],
        "sft_unseen": False,
    }
    assert set(manifest["calibration_tokens"]) <= set(train)
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "dev_tokens.json").write_text(json.dumps(dev))
    (root / "rl_train_tokens.json").write_text(json.dumps(train))
    (root / "METADATA_COMPLETE").write_text(
        json.dumps(
            {
                "scenes": len(records),
                "pose_max_abs": max(r["pose_max_abs"] for r in records),
                "source_metadata": str(source),
                "numpy_compatibility": "local trusted-data Unpickler only; no global patch/upgrade",
            }
        )
    )
    print((root / "METADATA_COMPLETE").read_text(), flush=True)


def finalize(root, workers=8):
    from starVLA.rl.flow_grpo.asset_publication import finalize_assets

    if not (root / "METADATA_COMPLETE").is_file():
        raise ValueError("metadata incomplete")
    templates = {
        v: f"configs/flow_grpo/paired_fp32_partition_{v}.yaml"
        for v in ("frozen_visual", "unfrozen_visual")
    }
    result = finalize_assets(
        root,
        templates,
        raw_root="/mnt/project/DriveDreamer-Policy/navsim_raw",
        workers=workers,
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["metadata", "finalize", "watch", "validate"])
    p.add_argument("--root", default="runs/paired_full_assets_v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--validation-output")
    p.add_argument("--wait-hours", type=float, default=6)
    args = p.parse_args()
    root = Path(args.root)
    if args.phase == "validate":
        from starVLA.rl.flow_grpo.asset_publication import (
            expected_cache_records,
            cache_builder_status,
            validate_cache_assets,
        )
        from starVLA.rl.flow_grpo.transactions import atomic_json

        if not args.validation_output:
            p.error("validate requires --validation-output (new report path)")
        records = expected_cache_records(
            root, "/mnt/project/DriveDreamer-Policy/navsim_raw"
        )
        state = cache_builder_status(root, len(records))
        result = (
            validate_cache_assets(root, records, args.workers)
            if state["finished"]
            else {"status": "NOT_RUN_BUILDING"}
        )
        result["builder_status"] = state
        destination = Path(args.validation_output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, result)
        if result["status"] != "PASS":
            raise RuntimeError(f"full assets not ready: {destination}")
        return
    if args.phase == "metadata":
        return prepare_metadata(root, args.workers)
    if args.phase == "watch":
        deadline = time.monotonic() + args.wait_hours * 3600
        while time.monotonic() < deadline:
            log = (root / "cache_build.log").read_text(errors="replace")
            metadata_log = root / "metadata_preparation.log"
            metadata_errors = (
                metadata_log.read_text(errors="replace")
                if metadata_log.is_file()
                else ""
            )
            if (
                "Traceback (most recent call last)" in log
                or "Traceback (most recent call last)" in metadata_errors
            ):
                raise RuntimeError("official cache build failed; log preserved")
            if (
                root / "METADATA_COMPLETE"
            ).is_file() and "Completed dataset caching! All" in log:
                break
            time.sleep(30)
        else:
            job = json.loads((root / "cache_job.json").read_text())
            pid = job["pid"]
            proc = Path(f"/proc/{pid}/cmdline")
            if (
                proc.exists()
                and job["cache"] in proc.read_bytes().decode(errors="replace")
                and os.getpgid(pid) == pid
            ):
                os.killpg(pid, signal.SIGTERM)
                time.sleep(2)
                if proc.exists():
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            raise TimeoutError(
                "bounded full asset preparation wait expired; only owned cache process group stopped"
            )
    finalize(root, args.workers)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Preserve traceback and partial assets. No distributed barrier or cleanup
        # of any unrelated training/scoring process.
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        raise
