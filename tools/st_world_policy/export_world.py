"""Export one original ten-step Flow trajectory from a complete ST checkpoint.

Only fixed train/dev observations are admitted. No teacher, future label, metric
cache, optimizer or scorer is read by the inference loop. Shards are resumable
only under their original checkpoint, observation and runtime identities.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.dataloader.st_world_observation import STObservationDataset
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.foresight.checkpoints import scene_noise
from tools.st_world_policy.base_binding import immutable_json
from tools.st_world_policy.checkpoints import checkpoint_identity, load_world_checkpoint
from tools.st_world_policy.training_config import environment_record


def validate_prediction(raw, physical):
    if (raw.shape != (8, 4) or physical.shape != (8, 3)
            or not torch.isfinite(raw).all() or not torch.isfinite(physical).all()):
        raise ValueError("INVALID_PREDICTION: original finite 8x4 raw and 8x3 physical trajectory required")


def validate_cached_prediction(path, meta, *, identity, token, log):
    if (meta.get("identity") != identity or meta.get("token") != token or meta.get("log") != log
            or meta.get("status") != "VALID_PREDICTION" or file_sha256(path) != meta["sha256"]):
        raise ValueError("Cached prediction identity/checksum/status differs")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"raw", "trajectory"}:
            raise ValueError("Unexpected inference payload")
        validate_prediction(torch.from_numpy(arrays["raw"]), torch.from_numpy(arrays["trajectory"]))


def run(args):
    if (type(args.max_scenes) is not int or args.max_scenes <= 0 or not 0 <= args.rank < args.world_size
            or args.world_size <= 0 or args.seed < 0 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("Explicit positive scene/time and valid shard budgets required")
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("Commit clean inference source before execution")
    cfg, _, checkpoint = checkpoint_identity(args.registration, args.checkpoint_tag)
    data = STObservationDataset(args.current_root, args.manifest, history_root=args.history_root)
    if len(data) != args.max_scenes:
        raise ValueError("Scene budget must equal the complete preselected manifest; no implicit prefix")
    if cfg["execution"]["scope"] != "FORMAL" and len(data) > 64:
        raise ValueError("Diagnostic model export is limited to 64 fixed scenes")
    if bool(args.history_root) != bool(cfg["data"]["history_root"]):
        raise ValueError("Checkpoint current/history input contract changed")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("Use an allocated real CUDA device for this image deployment audit")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    signature = {"schema": "st_single_trajectory_export_v1", "checkpoint": checkpoint,
        "observations": data.identity, "environment": environment_record(),
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "seed": args.seed, "solver": "original_Euler_10", "precision": "FP32_master_eval_TF32_off",
        "scene_noise": "foresight-action-v1", "candidates_per_scene": 1,
        "targets_read": False, "teacher_cache_read": False, "scorer": None,
        "world_size": args.world_size, "requested_scenes": len(data),
        "selected_scenes_sha256": identity_hash(data.rows)}
    signature["identity"] = identity_hash(signature)
    out = Path(args.output); (out / "predictions").mkdir(parents=True, exist_ok=True)
    immutable_json(out / "IDENTITY.json", signature)
    immutable_json(out / "OBSERVATIONS.json", {"scenes": data.rows, "identity": data.identity["identity"]})
    with (out / f"rank_{args.rank}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        call_id = str(time.time_ns()); attempt = out / "attempts" / call_id
        immutable_json(attempt / "INVOCATION.json", {"arguments": vars(args), "host": socket.gethostname(),
            "pid": os.getpid(), "started_unix": time.time(), "identity": signature["identity"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
        assigned = list(range(args.rank, len(data), args.world_size))
        completed, failures, pending = [], [], []
        for index in assigned:
            row = data.rows[index]; target = out / "predictions" / (row["token"] + ".npz")
            receipt = target.with_suffix(".json")
            if receipt.exists():
                validate_cached_prediction(target, json.loads(receipt.read_text()), identity=signature["identity"],
                                           token=row["token"], log=row["log"])
                completed.append({"token": row["token"], "receipt_sha256": file_sha256(receipt)})
            else:
                pending.append(index)
        started = time.perf_counter(); load_seconds = 0.; model = None
        calls = {"visual": 0, "language": 0, "DiT": 0}; handles = []
        try:
            if pending:
                model, loaded = load_world_checkpoint(args.registration, args.checkpoint_tag, device=args.device)
                if loaded != checkpoint:
                    raise ValueError("Checkpoint changed during deployment load")
                load_seconds = time.perf_counter() - started
                def count(name):
                    def hook(*_):
                        calls[name] += 1
                    return hook
                handles = [module.register_forward_hook(count(name)) for name, module in (
                    ("visual", model.base.qwen_vl_interface.model.model.visual),
                    ("language", model.base.qwen_vl_interface.model.model.language_model),
                    ("DiT", model.action_model.model))]
            for index in pending:
                if time.perf_counter() - started >= args.max_seconds:
                    break
                row = data.rows[index]; token = row["token"]
                before = dict(calls); begin = time.perf_counter()
                try:
                    observation = data[index]
                    read_seconds = time.perf_counter() - begin
                    noise = scene_noise(token, args.seed, args.device)
                    torch.cuda.synchronize(); inference_start = time.perf_counter()
                    with torch.no_grad():
                        raw = model.predict_raw([observation], initial_noise=noise)[0].float()
                        physical = decode_ego(raw.unsqueeze(0))[0]
                    torch.cuda.synchronize(); inference_seconds = time.perf_counter() - inference_start
                    validate_prediction(raw, physical)
                    target = out / "predictions" / (token + ".npz")
                    temporary = target.with_suffix("." + call_id + ".tmp")
                    with temporary.open("xb") as stream:
                        np.savez(stream, raw=raw.cpu().numpy(), trajectory=physical.cpu().numpy())
                        stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, target)
                    meta = {"status": "VALID_PREDICTION", "identity": signature["identity"],
                        "token": token, "log": row["log"], "sha256": file_sha256(target),
                        "image_read_seconds": read_seconds, "model_seconds": inference_seconds,
                        "image_to_trajectory_seconds": time.perf_counter() - begin,
                        "calls": {k: calls[k] - before[k] for k in calls}}
                    immutable_json(target.with_suffix(".json"), meta)
                    completed.append({"token": token, "receipt_sha256": file_sha256(target.with_suffix(".json"))})
                except Exception as error:
                    failure = {"token": token, "log": row["log"], "error": repr(error),
                               "identity": signature["identity"], "valid_prediction": False}
                    immutable_json(attempt / (token + ".failure.json"), failure)
                    failures.append(failure)
            result = {"status": "COMPLETE" if len(completed) == len(assigned) else "PARTIAL",
                "identity": signature["identity"], "rank": args.rank, "requested": len(assigned),
                "completed": len(completed), "failed_this_attempt": len(failures), "receipts": completed,
                "load_seconds": load_seconds, "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_scope": checkpoint["scope"], "scientific_result": False}
            immutable_json(attempt / "RESULT.json", result)
            if result["status"] == "COMPLETE":
                immutable_json(out / f"COMPLETE_rank{args.rank}.json", {
                    "identity": signature["identity"], "rank": args.rank, "requested": len(assigned),
                    "receipts": sorted(completed, key=lambda x: x["token"])})
            return result
        finally:
            for handle in handles:
                handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("registration", "checkpoint-tag", "current-root", "manifest", "output", "device"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--history-root")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-scenes", type=int, required=True)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": repr(exc)}), flush=True)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "receipts"}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
