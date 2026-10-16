"""Bounded real-image original S0 load/input audit; no optimizer or evaluation split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset, decode_ego
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash, tensor_hash
from tools.foresight.checkpoints import load_student, scene_noise
from tools.st_world_policy.base_binding import immutable_json, read_binding


def state_hash(model):
    return identity_hash({name: tensor_hash(tensor) for name, tensor in model.state_dict().items()})


def run(args):
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise ValueError("This integration receipt requires the explicitly allocated real GPU")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Commit clean qualification source before running")
    binding = read_binding(args.binding)
    manifest = json.loads(Path(args.manifest).read_text())
    expected_fields = {"schema", "role", "data_root", "index_sha256", "indices", "tokens", "seed"}
    if set(manifest) != expected_fields or manifest["schema"] != "st_s0_diagnostic_manifest_v1":
        raise ValueError("An explicit diagnostic manifest is required")
    if manifest["role"] != "train" or not 1 <= len(manifest["indices"]) <= 64:
        raise ValueError("Qualification is bounded to at most64 fixed training scenes")
    dataset = ForesightCurrentDataset(manifest["data_root"])
    if dataset.identity["split"] != "train":
        raise ValueError("Only the registered training split is admitted")
    if file_sha256(Path(manifest["data_root"]) / "index.json") != manifest["index_sha256"]:
        raise ValueError("Current observation population changed")
    indices = manifest["indices"]
    if len(set(indices)) != len(indices) or [dataset.index[i]["token"] for i in indices] != manifest["tokens"]:
        raise ValueError("Manifest has duplicated, changed or out-of-range scenes")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Keep prior qualification attempts; use a new output namespace")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    immutable_json(output / "REGISTRATION.json", {
        "schema": "st_s0_input_qualification_v1", "scope": "SMOKE", "optimizer_updates": 0,
        "maximum_scenes": 64, "requested_scenes": len(indices),
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "binding_sha256": binding["binding_sha256"], "manifest_sha256": file_sha256(args.manifest),
        "device": args.device, "hostname": os.uname().nodename, "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "canonical FP32 master / TF32 off / deterministic",
        "formal_initialization_from_this_job": False, "started_unix": time.time()})
    model = load_student(binding["run"], binding["tag"], binding["parent_identity"],
                         device=args.device, precision="fp32", strip=False)
    loaded_seconds = time.perf_counter() - started
    before = state_hash(model)
    calls = {"visual": 0, "language": 0, "action": 0}

    def hook(name):
        def count(*_):
            calls[name] += 1
        return count

    handles = [module.register_forward_hook(hook(name)) for name, module in (
        ("visual", model.qwen_vl_interface.model.model.visual),
        ("language", model.qwen_vl_interface.model.model.language_model),
        ("action", model.action_model.model))]
    records = []
    try:
        with torch.no_grad():
            for i in indices:
                obs = dataset[i]
                calls_before = dict(calls)
                a = model.encode_current([obs])
                after_first = dict(calls)
                poisoned = dict(obs, ego_target="FORBIDDEN_LABEL_SENTINEL",
                                future_actors="FORBIDDEN_LABEL_SENTINEL", score=float("nan"))
                b = model.encode_current([poisoned])
                if any(not torch.equal(a[k], b[k]) for k in ("W", "action_queries")):
                    raise AssertionError("Labels affected the original current-only representation")
                initial = scene_noise(obs["token"], manifest["seed"], args.device)
                with model.amp():
                    raw = model.action_model.predict_action(model.build_planner_condition(a), initial_noise=initial)
                    repeat = model.action_model.predict_action(model.build_planner_condition(b), initial_noise=initial.clone())
                physical = decode_ego(raw.float())
                if not torch.isfinite(raw).all() or not torch.isfinite(physical).all():
                    raise FloatingPointError("Original S0 prediction is nonfinite")
                if not torch.equal(raw, repeat):
                    raise AssertionError("Fixed-noise deterministic S0 replay differs")
                count = {k: after_first[k] - calls_before[k] for k in calls}
                if count["visual"] != 1 or count["language"] != 1:
                    raise AssertionError("Original representation does not use one shared visual/language call")
                records.append({"scene_id": obs["token"], "W_shape": list(a["W"].shape),
                                "action_shape": list(a["action_queries"].shape),
                                "raw_sha256": tensor_hash(raw), "physical_sha256": tensor_hash(physical),
                                "label_independence_exact": True, "fixed_noise_repeat_exact": True,
                                "single_encoder_calls": count})
                torch.save({"raw": raw.cpu(), "physical": physical.cpu()}, output / (obs["token"] + ".pt"))
                print(json.dumps({"status": "SCENE_VERIFIED", "completed": len(records),
                                  "requested": len(indices)}), flush=True)
        after = state_hash(model)
        if before != after:
            raise AssertionError("No-update input audit mutated model parameters/buffers")
        torch.cuda.synchronize()
        receipt = {"status": "VERIFIED", "strict_load": True, "scenes": records,
                   "parameter_buffer_sha256_before": before, "parameter_buffer_sha256_after": after,
                   "load_seconds": loaded_seconds, "total_seconds": time.perf_counter() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "hardware": torch.cuda.get_device_name(), "optimizer_updates": 0,
                   "scientific_gain": "NOT_TESTED", "formal_launch_approval": False}
        immutable_json(output / "COMPLETE.json", receipt)
        return receipt
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        failure = {"status": "FAILED", "error": repr(error), "optimizer_updates": 0}
        directory = Path(args.output)
        if directory.exists() and not (directory / "FAILURE.json").exists():
            immutable_json(directory / "FAILURE.json", failure)
        print(json.dumps(failure), flush=True)
        raise
    print(json.dumps({"status": receipt["status"], "scenes": len(receipt["scenes"]),
                      "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
