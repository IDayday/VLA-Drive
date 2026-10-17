"""Resolved, deliberately narrow configuration for original S0 ST continuation."""
from __future__ import annotations

import importlib.metadata
import math
import platform

from starVLA.model.modules.st_world.config import SpaceTimeConfig
from tools.st_world_policy.check_formal_approval import validate_execution


PRECISION = "S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off"


def exact_fields(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(description + " has missing or unknown fields")


def official_runtime_policy(cfg):
    """Explicit opt-in; old registered experiments retain their actual backend."""
    if "official_execution" not in cfg["runtime"]:
        return None
    from tools.st_world_policy.official_runtime import resolve_official_execution
    return resolve_official_execution(cfg["runtime"]["official_execution"],
        execution=cfg["execution"], optimizer=cfg["optimizer"],
        gradient_checkpointing=cfg["runtime"]["gradient_checkpointing"])


def environment_record():
    import torch
    return {"schema": "st_training_environment_v1", "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "deepspeed", "timm", "safetensors", "numpy")},
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def validate_config(cfg):
    fields = {"schema", "execution", "space_time", "optimizer", "runtime", "data"}
    exact_fields(cfg, fields | ({"continuation"} if "continuation" in cfg else set()), "ST run")
    if "continuation" in cfg:
        from tools.st_world_policy.budget_extension import validate_extension_config
        validate_extension_config(cfg["continuation"])
    if cfg["schema"] != "st_training_v1":
        raise ValueError("Unknown ST training configuration")
    validate_execution(cfg["execution"])
    execution = cfg["execution"]
    if execution["precision_policy"] != PRECISION:
        raise ValueError("This runner preserves the original S0 training precision")
    st = SpaceTimeConfig.from_dict(cfg["space_time"])
    if execution["family"] != ("temporal" if st.cell.startswith("T") else "spatial"):
        raise ValueError("Cell and registered experiment family disagree")
    opt = cfg["optimizer"]
    exact_fields(opt, {"lr", "minimum_lr", "warmup", "betas", "eps", "weight_decay", "clip"}, "Optimizer")
    if any(isinstance(opt[k], bool) or not isinstance(opt[k], (int, float)) or not math.isfinite(opt[k])
           for k in ("lr", "minimum_lr", "eps", "weight_decay", "clip")):
        raise ValueError("Optimizer constants must be finite numbers")
    if not (0 < opt["minimum_lr"] <= opt["lr"] and opt["eps"] > 0 and opt["weight_decay"] >= 0
            and opt["clip"] == 1. and type(opt["warmup"]) is int
            and 0 <= opt["warmup"] < execution["max_updates"]):
        raise ValueError("Invalid fixed cosine/AdamW recipe")
    if (not isinstance(opt["betas"], list) or len(opt["betas"]) != 2
            or any(not isinstance(x, (float, int)) or isinstance(x, bool) or not 0 <= x < 1 for x in opt["betas"])):
        raise ValueError("Adam betas must be resolved")
    runtime = cfg["runtime"]
    runtime_keys = {"output_root", "loader_workers", "save_every", "max_seconds",
                    "deterministic", "gradient_checkpointing"}
    optional = {key for key in ("profile_mode", "official_execution") if key in runtime}
    exact_fields(runtime, runtime_keys | optional, "Runtime")
    profile = runtime.get("profile_mode", "none")
    if profile not in ("none", "timing", "trace"):
        raise ValueError("Unknown profiling mode")
    if profile != "none" and execution["scope"] not in ("SMOKE", "PROFILE"):
        raise ValueError("Profiling must be a separately registered diagnostic run")
    if profile == "timing" and (execution["scope"] != "PROFILE" or execution["stop_after"] != 240):
        raise ValueError("Complete timing requires the two registered 20+100 update windows")
    if profile == "trace" and execution["stop_after"] > 4:
        raise ValueError("Trace diagnostics are limited to four optimizer updates")
    if type(runtime["loader_workers"]) is not int or not 0 <= runtime["loader_workers"] <= 8:
        raise ValueError("Bounded loader worker count required")
    if type(runtime["save_every"]) is not int or runtime["save_every"] < 1:
        raise ValueError("Positive checkpoint interval required")
    if (not isinstance(runtime["max_seconds"], (int, float)) or isinstance(runtime["max_seconds"], bool)
            or not math.isfinite(runtime["max_seconds"]) or runtime["max_seconds"] <= 0):
        raise ValueError("Explicit finite invocation wall budget required")
    if any(type(runtime[k]) is not bool for k in ("deterministic", "gradient_checkpointing")):
        raise ValueError("Backend flags must be booleans")
    official_runtime_policy(cfg)
    from pathlib import Path
    if not isinstance(runtime["output_root"], str) or not Path(runtime["output_root"]).is_absolute():
        raise ValueError("An isolated absolute run output root is required")
    provider = cfg["data"].get("teacher_provider")
    exact_fields(cfg["data"], {"root", "teacher_root", "history_root", "pca_path"} |
                 ({"teacher_provider"} if "teacher_provider" in cfg["data"] else set()), "Data")
    if "teacher_provider" in cfg["data"]:
        from tools.st_world_policy.teacher_provider import validate_option
        validate_option(provider, loader_workers=runtime["loader_workers"],
            teacher_root=cfg["data"]["teacher_root"], pca_path=cfg["data"]["pca_path"], scope=execution["scope"])
    for name, value in cfg["data"].items():
        if name == "teacher_provider" or (provider is not None and name == "teacher_root"):
            continue
        if value is None and name in ("history_root", "pca_path"):
            continue
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Real absolute data dependency required: " + name)
    if st.history != (cfg["data"]["history_root"] is not None):
        raise ValueError("History input configuration and actual data dependency disagree")
    return st
