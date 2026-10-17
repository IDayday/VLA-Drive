"""Bounded one-shot ST loss/shared-gradient measurement, without optimizer updates.

validate reads controls only. measure needs a separate Lead-owned GPU admission;
this tool neither launches workers nor allocates a cache/checkpoint bank.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time

from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy.base_binding import immutable_json

SCHEMA = "st_temporal_scale_calibration_v1"
GROUPS = "last_language_block_and_world_queries_v1"
RUNTIME_ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def runtime_preflight(cfg, training):
    """Require the registered environment before the first CUDA context.

    Never repair this in-process: PyTorch caches its cuBLAS determinism check.
    The owned launcher sets the child environment before starting Python.
    """
    if cfg.get("runtime_environment") != RUNTIME_ENVIRONMENT:
        raise ValueError("Exact registered deterministic runtime environment required")
    if training["runtime"]["deterministic"] is not True:
        raise ValueError("Calibration must retain deterministic=True")
    observed = {key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT}
    if observed != RUNTIME_ENVIRONMENT:
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the calibration process")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("Calibration runtime preflight requires an uninitialized CUDA context")
    return {"schema": "st_calibration_runtime_preflight_v1", "runtime_environment": observed,
            "deterministic_algorithms": True, "cuda_initialized_at_check": False}


def exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " has missing or unknown fields")


def pinned(pin):
    exact(pin, {"path", "sha256"}, "Content pin")
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError("Missing/oversize/nonabsolute control artifact")
    if file_sha256(path) != pin["sha256"]:
        raise ValueError("Control checksum mismatch: " + str(path))
    return json.loads(path.read_text())


def validate(cfg):
    exact(cfg, {"schema", "purpose", "source", "training_config", "data_manifest", "teacher_manifest",
                "teacher_requests", "batch", "parameter_groups", "output_root", "runtime_environment"}, "Calibration config")
    if cfg["runtime_environment"] != RUNTIME_ENVIRONMENT:
        raise ValueError("Exact registered deterministic runtime environment required")
    if cfg["schema"] != SCHEMA or cfg["purpose"] not in {"DIAGNOSTIC", "SCIENTIFIC_CALIBRATION"}:
        raise ValueError("Explicit calibration purpose required")
    source = cfg["source"]
    if source.get("kind") == "common_s0":
        exact(source, {"kind", "binding", "initialization_reference"}, "Common S0 source")
        if source["initialization_reference"] != "COMMON_S0_WITH_REGISTERED_INITIALIZATION":
            raise ValueError("Common S0 requires the explicitly registered new-module initialization")
        pinned(source["binding"])
    elif source.get("kind") == "native_st":
        exact(source, {"kind", "registration", "tag", "receipt"}, "Native ST source")
        reg, receipt = pinned(source["registration"]), pinned(source["receipt"])
        parent_cfg = pinned(reg["artifacts"]["config"])
        if parent_cfg["execution"]["scope"] in {"SMOKE", "PROFILE"} or reg.get("weights_usable_for_formal_init") is not True:
            raise ValueError("SMOKE/PROFILE learned weights cannot supply scientific calibration")
        if not source["tag"] or Path(source["tag"]).name != source["tag"] or source["tag"] in {".", ".."}:
            raise ValueError("Unsafe native tag")
        expected = Path(parent_cfg["runtime"]["output_root"]) / reg["run_id"] / "checkpoints" / source["tag"] / "COMPLETE.json"
        if Path(source["receipt"]["path"]).resolve() != expected.resolve() or receipt["state"]["completed"] <= 0:
            raise ValueError("Native source receipt/tag mismatch")
    else:
        raise ValueError("No arbitrary checkpoint loaders")
    from tools.st_world_policy.training_config import validate_config, official_runtime_policy
    training = pinned(cfg["training_config"])
    if training["runtime"]["deterministic"] is not True:
        raise ValueError("Calibration must retain deterministic=True")
    st = validate_config(training)
    if st.cell != "T4" or not all((st.current, st.future, st.delta)) or official_runtime_policy(training) is None:
        raise ValueError("All four original objectives and explicit official execution are required")
    if any(getattr(st, name + "_weight") != 1. for name in ("current", "future", "delta")):
        raise ValueError("Probe coefficients must be explicit ones; actual unweighted losses are measured")
    if "continuation" in training or "teacher_provider" in training["data"]:
        raise ValueError("Calibration uses immutable existing real64 targets, no trainer continuation/provider")
    if training["runtime"]["gradient_checkpointing"]:
        raise ValueError("This autograd.grad diagnostic requires the registered no-recompute runtime")
    if training["execution"]["world_size"] != 1:
        raise ValueError("Calibration is one process; no distributed optimizer parity claim")
    manifest = pinned(cfg["data_manifest"])
    teacher = pinned(cfg["teacher_manifest"])
    if manifest.get("schema") != "st_fixed_scene_list_v1" or manifest.get("split") != "train":
        raise ValueError("Only fixed training scenes are allowed, never dev/cal/test")
    if len(manifest["scenes"]) > 64 or not manifest["scenes"]:
        raise ValueError("Existing diagnostic fixture is bounded at 64 unique scenes")
    from starVLA.dataloader.st_world_targets import validate_target_manifest
    requests = validate_target_manifest(pinned(cfg["teacher_requests"]))
    if (requests["split"] != "train" or requests["times_s"] != [0., .5, 1., 2., 4.] or
            requests["identity"] != teacher["request_manifest_identity"]):
        raise ValueError("Frozen current/all-future training teacher required")
    if (Path(cfg["teacher_manifest"]["path"]).resolve() != (Path(training["data"]["teacher_root"]) / "identity.json").resolve() or
            Path(cfg["teacher_requests"]["path"]).resolve() != (Path(training["data"]["teacher_root"]) / "requests.json").resolve()):
        raise ValueError("Registered teacher root differs")
    batch = cfg["batch"]
    exact(batch, {"scene_indices", "scene_tokens", "micro_batch", "patch_seed", "horizon_seed",
                  "noise_seed", "time_seed", "dropout_seed"}, "Fixed batch")
    ids = batch["scene_indices"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids) or
            any(type(i) is not int or not 0 <= i < len(manifest["scenes"]) for i in ids)):
        raise ValueError("Unique in-range fixed batch indices required")
    if batch["scene_tokens"] != [manifest["scenes"][i]["token"] for i in ids]:
        raise ValueError("Fixed batch token/index mismatch")
    if len(set(batch["scene_tokens"])) != len(ids):
        raise ValueError("Duplicate scene identity")
    if type(batch["micro_batch"]) is not int or not 1 <= batch["micro_batch"] <= len(ids):
        raise ValueError("Invalid microbatch")
    if (training["execution"]["global_batch"] != len(ids) or
            training["execution"]["micro_batch"] != batch["micro_batch"]):
        raise ValueError("Reference batch topology differs")
    if any(type(batch[k]) is not int or not 0 <= batch[k] < 2**63 for k in batch if k.endswith("seed")):
        raise ValueError("Resolved RNG seeds required")
    if cfg["parameter_groups"] != GROUPS or not Path(cfg["output_root"]).is_absolute():
        raise ValueError("Fixed shared subset and isolated absolute output required")
    return training, st, manifest, teacher


def admission_check(path, sha, cfg_path, cfg):
    admission = pinned({"path": str(Path(path).resolve()), "sha256": sha})
    required = {"schema", "status", "reviewer", "config", "source_sha", "host", "cuda_visible_devices",
                "gpu_uuids", "valid_from_unix", "valid_until_unix", "max_seconds", "max_scene_exposures",
                "external_resource_controller_required", "controller_token"}
    exact(admission, required, "Calibration admission")
    if (admission["schema"] != "st_loss_scale_resource_admission_v1" or
            admission["status"] != "PASS_BOUNDED_MEASUREMENT_ONLY" or not admission["reviewer"] or
            admission["external_resource_controller_required"] is not True):
        raise ValueError("Independent bounded resource admission required")
    if admission["config"] != {"path": str(Path(cfg_path).resolve()), "sha256": file_sha256(cfg_path)}:
        raise ValueError("Admission refers to another config")
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != admission["source_sha"] or subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo):
        raise ValueError("Calibration source must be the exact clean reviewed commit")
    for key in ("valid_from_unix", "valid_until_unix", "max_seconds"):
        if type(admission[key]) not in (int, float) or not math.isfinite(admission[key]):
            raise ValueError("Nonfinite resource window")
    if (not 0 < admission["max_seconds"] <= 3600 or
            not admission["valid_from_unix"] <= time.time() < admission["valid_until_unix"] or
            time.time() + admission["max_seconds"] > admission["valid_until_unix"]):
        raise ValueError("Insufficient current lease for bounded diagnostic")
    if (admission["host"] != socket.gethostname() or
            os.environ.get("CUDA_VISIBLE_DEVICES") != admission["cuda_visible_devices"] or
            len(admission["gpu_uuids"]) != 1):
        raise ValueError("Actual single-GPU placement differs")
    if os.environ.get("ST_CALIBRATION_CONTROLLER_TOKEN") != admission["controller_token"] or not admission["controller_token"]:
        raise ValueError("Lead resource controller must own this execution; this is not self-admission")
    uuids = subprocess.check_output(["nvidia-smi", "--id=" + admission["cuda_visible_devices"],
                                    "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if uuids != admission["gpu_uuids"] or admission["max_scene_exposures"] != len(cfg["batch"]["scene_indices"]):
        raise ValueError("GPU or scene budget mismatch")
    return admission


@contextmanager
def wall_limit(seconds):
    def expired(*_):
        raise TimeoutError("Calibration wall budget exhausted")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def execute(cfg, training, st, teacher):
    preflight = runtime_preflight(cfg, training)
    import numpy as np
    import torch
    from starVLA.dataloader.st_world_dataset import STWorldTrainingDataset, collate_st_world
    from starVLA.model.framework.st_world_model import SpaceTimeStudent
    from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future
    from starVLA.model.modules.st_world.losses import optimizer_batch_counts
    from tools.st_world_policy.base_binding import read_binding
    from tools.st_world_policy.training_config import official_runtime_policy
    from tools.st_world_policy.official_runtime import apply_official_model_runtime
    from tools.st_world_policy.calibration_measurement import measure, shared_groups, preserved_rng
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.distributed.is_initialized():
        raise ValueError("Single-process measurement only")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(training["runtime"]["deterministic"])
    seeds = cfg["batch"]
    generators = {name: torch.Generator(device="cpu" if name in ("patch", "horizon") else "cuda").manual_seed(
        seeds[name + "_seed"]) for name in ("patch", "horizon", "noise", "time")}
    with preserved_rng(generators):
        seed = training["execution"]["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        source = cfg["source"]
        if source["kind"] == "common_s0":
            from tools.foresight.checkpoints import load_student
            from starVLA.model.modules.st_world.history_adapter import build_history_encoder
            binding = read_binding(source["binding"]["path"], verify_files=True)
            parent = load_student(binding["run"], binding["tag"], binding["parent_identity"],
                                  device="cpu", precision="fp32", strip=False)
            model = SpaceTimeStudent(parent, st, initialization_seed_value=seed,
                history_encoder=build_history_encoder(parent, initialization_seed=seed + 3700))
            source_identity = {"kind": "common_s0", "binding": binding["binding_sha256"],
                               "scope": cfg["purpose"], "initialization_reference": source["initialization_reference"],
                               "initialization_seed": seed, "migration": model.migration}
        else:
            from tools.st_world_policy.checkpoints import load_world_checkpoint
            model, source_identity = load_world_checkpoint(source["registration"]["path"], source["tag"],
                                                           device="cpu", strip=False)
            prior = model.st_config.to_dict()
            wanted = st.to_dict()
            temporal = {"cell", "spatial_parent", "future", "delta", "current_weight", "future_weight", "delta_weight"}
            if {k: v for k, v in prior.items() if k not in temporal} != {k: v for k, v in wanted.items() if k not in temporal}:
                raise ValueError("Calibration cannot silently change native W/readout/teacher/history architecture")
            if source_identity["native_receipt_sha256"] != source["receipt"]["sha256"]:
                raise ValueError("Native receipt changed")
            model.st_config = st  # Enables the existing returned unweighted losses, no new parameters.
        model.base.inference_fp32 = False
        policy = official_runtime_policy(training)
        runtime = apply_official_model_runtime(vlm=model.base.qwen_vl_interface.model,
            language=model.base.qwen_vl_interface.model.model.language_model,
            action_transformer=model.base.action_model.model, policy=policy, training_copy=True)
        runtime = {**runtime, "calibration_preflight": preflight}
        # Match DS's BF16 execution casting, without creating Adam/ZeRO states.
        # This is a gradient probe, not a backend throughput/resume qualification.
        model.to(device="cuda", dtype=torch.bfloat16).train()
        data = STWorldTrainingDataset(training["data"]["root"], cfg["data_manifest"]["path"],
            training["data"]["teacher_root"], teacher["identity"], st.teacher,
            history_root=training["data"]["history_root"], pca_path=training["data"]["pca_path"],
            require_split="train", need_future=True)
        observations, targets = collate_st_world([data[i] for i in seeds["scene_indices"]])
        request = sample_requests(len(observations), len(st.teacher.view_names), st.teacher.output_grid_hw,
            st.sampled_positions, patch_generator=generators["patch"], future=True,
            horizon_generator=generators["horizon"])
        _, cmask = gather_current(targets["current_dense_features"], targets["current_valid"], request)
        _, fmask = gather_future(targets["future_dense_features"], targets["future_valid"], request)
        counts = {key: int(value.item()) for key, value in optimizer_batch_counts(st.teacher.channels,
            current_valid=cmask.cuda(), future_valid=fmask.cuda(), delta=True, distributed=False).items()}
        counts["ego_scenes"] = len(observations)
        if any(value <= 0 for value in counts.values()):
            raise ValueError("UNCALIBRATABLE: missing valid current/future/delta observations")
        random.seed(seeds["dropout_seed"]); np.random.seed(seeds["dropout_seed"] % 2**32)
        torch.manual_seed(seeds["dropout_seed"])
        def forwards():
            for start in range(0, len(observations), seeds["micro_batch"]):
                end = min(start + seeds["micro_batch"], len(observations))
                def forward(start=start, end=end):
                    return model(observations[start:end], {k: v[start:end] for k, v in targets.items()},
                        requests=request.slice(start, end), global_counts=counts,
                        noise_generator=generators["noise"], time_generator=generators["time"])
                yield forward
        result = measure(model, forwards(), groups=shared_groups(model), counts=counts,
            generators=generators, max_microbatches=math.ceil(len(observations) / seeds["micro_batch"]))
        result.update(source=source_identity, runtime=runtime, dataset=data.identity,
            batch=seeds, patch_indices=request.patch_indices.tolist(), future_pair=request.future_pair.tolist(),
            requests_sha256=identity_hash({"patch": request.patch_indices.tolist(), "pair": request.future_pair.tolist()}),
            teacher_instantiated=False, new_teacher_tensor_bytes=0,
            precision_note="BF16 execution and native per-task BF16 accumulation; no optimizer, ZeRO communication or clipping",
            output_purpose=cfg["purpose"], full_shared_model_gradient_measured=False,
            pair_scene_counts={str(pair): int((request.future_pair[:, 0] == pair).sum()) for pair in range(3)})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "measure"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    args = parser.parse_args()
    try:
        cfg = json.loads(Path(args.config).read_text())
        training, st, _, teacher = validate(cfg)
        if args.action == "validate":
            if args.admission or args.admission_sha256:
                raise ValueError("Validate is metadata-only and accepts no launch admission")
            print(json.dumps({"status": "CONFIG_VALID_ONLY", "config_sha256": file_sha256(args.config),
                              "runtime_environment": cfg["runtime_environment"],
                              "payload_reads": 0, "GPU_calls": 0, "optimizer_steps": 0}))
            return 0
        if not args.admission or not args.admission_sha256:
            raise ValueError("measure requires independent resource admission and its checksum")
        preflight = runtime_preflight(cfg, training)
        admission = admission_check(args.admission, args.admission_sha256, args.config, cfg)
        output = Path(cfg["output_root"])
        output.mkdir(parents=True, exist_ok=False)  # one attempt; failed evidence is never overwritten
        immutable_json(output / "RESOLVED_CONFIG.json", cfg)
        immutable_json(output / "RUNTIME_PREFLIGHT.json", preflight)
        started = time.monotonic()
        try:
            with wall_limit(admission["max_seconds"]):
                result = execute(cfg, training, st, teacher)
            result.update(status="MEASURED" if result["proposal"]["status"] == "PROPOSAL_ONLY" else "UNCALIBRATABLE",
                          config_sha256=file_sha256(args.config), source_sha=admission["source_sha"],
                          elapsed_seconds=time.monotonic() - started)
            immutable_json(output / "MEASUREMENT.json", result)
            print(json.dumps({"status": result["status"], "output": str(output), "optimizer_steps": 0}))
            return 0 if result["status"] == "MEASURED" else 3
        except Exception as error:
            immutable_json(output / "FAILED.json", {"status": "FAILED", "error": repr(error),
                "runtime_preflight_sha256": identity_hash(preflight),
                "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0})
            raise
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, TimeoutError) as error:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
