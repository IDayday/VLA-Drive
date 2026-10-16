"""Executable official DriveDreamer backend, with explicit research adaptations.

This module never loads weights, launches workers, changes losses, or selects data.
The caller owns native checkpoint/master initialization and global loss counts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch


SCHEMA = "st_official_drivedreamer_execution_v1"
PRESET = "drivedreamer_f35fe783"
OFFICIAL_COMMIT = "f35fe7836625434dc4f67176a3967bf42b0d446d"
REFERENCE_RECIPE = {
    "world_size": 8, "micro_batch": 4, "global_batch": 32, "max_updates": 100000,
    "warmup": 5000, "lr": 1e-5, "minimum_lr": 5e-7, "weight_decay": 1e-3,
    "betas": [0.9, 0.95], "eps": 1e-8, "clip": 1., "gradient_checkpointing": False,
}
BACKEND = {
    "execution_dtype": "bfloat16", "gradient_accumulation_dtype": "bfloat16",
    "master_dtype": "float32", "moment_dtype": "float32", "zero_stage": 2,
    "optimizer": "torch.optim.AdamW", "optimizer_offload": False,
    "parameter_offload": False, "attention": "flash_attention_2",
    "reduce_bucket_elements": 500000000, "allgather_bucket_elements": 500000000,
}
PINNED_PACKAGES = {"torch": "2.5.1", "deepspeed": "0.16.9",
                   "transformers": "4.57.0", "flash-attn": "2.7.4.post1"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _exact(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(label + " has missing or unknown fields")


def _number(value, label, *, positive=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError("Invalid " + label)


def resolve_official_execution(request, *, execution, optimizer, gradient_checkpointing):
    """Bind existing recipe/budgets; never replace them with the reference 100k.

    ``request`` is cfg.runtime.official_execution. Existing cfg.optimizer fields
    and execution topology are passed unchanged. Declared overrides must equal
    actual reference differences, including checkpointing's effective value.
    """
    _exact(request, {"schema", "preset", "scope", "adaptation_reason", "declared_recipe_overrides"},
           "Official execution request")
    if request["schema"] != SCHEMA or request["preset"] != PRESET:
        raise ValueError("Unknown official preset")
    if request["scope"] not in {"official_reference", "research_stage"}:
        raise ValueError("Unknown recipe scope")
    _exact(optimizer, {"lr", "minimum_lr", "warmup", "betas", "eps", "weight_decay", "clip"}, "Optimizer")
    topology = {}
    for key in ("world_size", "micro_batch", "global_batch", "max_updates"):
        value = execution.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError("Positive integer required: " + key)
        topology[key] = value
    divisor = topology["world_size"] * topology["micro_batch"]
    if topology["global_batch"] % divisor:
        raise ValueError("Global batch is not an integral microbatch accumulation")
    for key in ("lr", "minimum_lr", "eps", "clip", "weight_decay"):
        _number(optimizer[key], key, positive=key != "weight_decay")
    if optimizer["minimum_lr"] > optimizer["lr"]:
        raise ValueError("minimum_lr exceeds lr")
    if type(optimizer["warmup"]) is not int or not 0 <= optimizer["warmup"] < topology["max_updates"]:
        raise ValueError("Warmup must fit the unchanged schedule horizon")
    if (not isinstance(optimizer["betas"], (list, tuple)) or len(optimizer["betas"]) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1
                   for v in optimizer["betas"])):
        raise ValueError("Invalid Adam betas")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("Checkpointing must be an effective boolean")
    effective = {**topology, **deepcopy(optimizer), "gradient_checkpointing": gradient_checkpointing}
    effective["betas"] = list(effective["betas"])
    changes = {k: {"reference": v, "effective": effective[k]}
               for k, v in REFERENCE_RECIPE.items() if effective[k] != v}
    declared = request["declared_recipe_overrides"]
    if (not isinstance(declared, list) or any(not isinstance(v, str) for v in declared)
            or len(set(declared)) != len(declared) or set(declared) != set(changes)):
        raise ValueError("Declared recipe overrides differ from actual changes: " + str(sorted(changes)))
    reason = request["adaptation_reason"]
    if not isinstance(reason, str) or (request["scope"] == "research_stage" and not reason.strip()):
        raise ValueError("Research adaptation needs a nonempty reason")
    if request["scope"] == "official_reference" and (changes or reason):
        raise ValueError("Official reference cannot silently adapt a recipe")
    result = {"schema": SCHEMA, "preset": PRESET, "official_commit": OFFICIAL_COMMIT,
              "scope": request["scope"], "reference_recipe": deepcopy(REFERENCE_RECIPE),
              "effective_recipe": effective, "recipe_overrides": changes, "adaptation_reason": reason,
              "backend": deepcopy(BACKEND), "accumulation_steps": topology["global_batch"] // divisor,
              "checkpointing_evidence": {"official_yaml_flag": True,
                  "official_effective": False, "flag_consumed_by_official_selected_path": False,
                  "source": "cfg_yaw_1225.yaml:244; train_starvla/TrainerUtils/QwenOFT/QWen3; DiT defaults false"},
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu_qualified": False}
    result["identity"] = _hash(result)
    return result


def _policy(policy):
    if (not isinstance(policy, dict) or policy.get("schema") != SCHEMA
            or policy.get("backend") != BACKEND or policy.get("official_commit") != OFFICIAL_COMMIT
            or policy.get("reference_recipe") != REFERENCE_RECIPE
            or policy.get("implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or policy.get("identity") != _hash({k: v for k, v in policy.items() if k != "identity"})):
        raise ValueError("Modified, stale, or foreign resolved official policy")
    return policy["effective_recipe"]


def official_deepspeed_config(policy, *, accumulation_owner):
    """Return actual DS input. No optimizer section: pass the Torch client.

    Current ST trainers already normalize by global valid counts and explicitly
    mark update boundaries. Native GAS=1 avoids a second loss division. This is
    recorded separately from actual global batch/accumulation in the policy.
    """
    recipe = _policy(policy)
    if accumulation_owner not in {"deepspeed", "caller_global_valid_counts"}:
        raise ValueError("Explicit accumulation owner required")
    gas = policy["accumulation_steps"] if accumulation_owner == "deepspeed" else 1
    return {"fp16": {"enabled": False}, "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": recipe["micro_batch"], "gradient_accumulation_steps": gas,
        "train_batch_size": recipe["micro_batch"] * recipe["world_size"] * gas,
        "zero_optimization": {"stage": 2, "allgather_partitions": True,
            "allgather_bucket_size": 500000000, "reduce_scatter": True, "reduce_bucket_size": 500000000,
            "overlap_comm": True, "contiguous_gradients": True, "cpu_offload": False},
        "gradient_clipping": recipe["clip"], "steps_per_print": 10,
        # Accelerate 1.5.2 accelerator.py:1835 adds the same flag for a real client.
        "zero_allow_untested_optimizer": True}


def build_official_optimizer(model, parameter_groups, policy):
    """Construct Torch AdamW without sorting, regrouping, or changing group rates."""
    recipe = _policy(policy)
    named = {id(p): (name, p) for name, p in model.named_parameters() if p.requires_grad}
    if not named:
        raise ValueError("No trainable parameters")
    groups, records, seen = [], [], set()
    for original in parameter_groups:
        if not isinstance(original, dict) or not set(original) <= {"params", "name", "lr", "betas", "eps", "weight_decay"}:
            raise ValueError("Unsupported optimizer group options")
        group = {k: deepcopy(v) for k, v in original.items() if k != "params"}
        params = list(original.get("params", []))
        if not params:
            raise ValueError("Empty parameter group")
        rows = []
        for p in params:
            if id(p) not in named or id(p) in seen:
                raise ValueError("Foreign, frozen, or duplicate optimizer parameter")
            if p.dtype != torch.float32:
                raise ValueError("Construct the optimizer from canonical FP32 sources before DS casting")
            seen.add(id(p))
            rows.append({"name": named[id(p)][0], "shape": list(p.shape), "dtype": str(p.dtype)})
        for key in ("lr", "eps", "weight_decay"):
            _number(group.get(key, recipe[key]), "group " + key, positive=key != "weight_decay")
        betas = group.get("betas", recipe["betas"])
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1 for v in betas)):
            raise ValueError("Invalid group betas")
        if "betas" in group:
            group["betas"] = tuple(betas)
        groups.append({**group, "params": params})
        records.append({"parameters": rows, "explicit_group_options": group})
    if seen != set(named):
        raise ValueError("Optimizer groups must cover exactly all trainable parameters")
    optimizer = torch.optim.AdamW(groups, lr=recipe["lr"], betas=tuple(recipe["betas"]),
        eps=recipe["eps"], weight_decay=recipe["weight_decay"])
    receipt = {"schema": "st_official_optimizer_layout_v1", "policy_identity": policy["identity"],
               "implementation": "torch.optim.AdamW", "groups": records,
               "fused": optimizer.defaults["fused"], "foreach": optimizer.defaults["foreach"]}
    receipt["identity"] = _hash(receipt)
    optimizer._st_official_optimizer_identity = receipt["identity"]
    return optimizer, receipt


def check_flash_attention2_available():
    """Read actual installed capability; no fallback or package installation."""
    versions = {}
    for name, expected in PINNED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Official runtime dependency missing: " + name) from exc
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Unsupported official runtime dependency {name}: {actual}, expected {expected}")
        versions[name] = actual
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("FlashAttention2 requires an available CUDA device")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Official BF16 FlashAttention2 requires Ampere or newer with BF16 support")
    try:
        fa = importlib.import_module("flash_attn")
        if not callable(getattr(fa, "flash_attn_func", None)) or not callable(getattr(fa, "flash_attn_varlen_func", None)):
            raise ImportError("Missing FlashAttention2 entrypoints")
    except (ImportError, OSError) as exc:
        raise RuntimeError("FlashAttention2 extension failed to load; no SDPA fallback") from exc
    return {"packages": versions, "cuda_runtime": torch.version.cuda,
            "device_capability": list(capability), "kernel_execution_tested": False}


def apply_official_model_runtime(*, vlm, language, action_transformer, policy, training_copy):
    """Apply to the training instance only; bound canonical evaluation is unchanged."""
    recipe = _policy(policy)
    if training_copy is not True:
        raise ValueError("Explicit training copy required; do not mutate canonical evaluation")
    environment = check_flash_attention2_available()
    if (not callable(getattr(vlm, "set_attn_implementation", None))
            or not (getattr(vlm, "_supports_flash_attn", False) or getattr(vlm, "_supports_flash_attn_2", False))):
        raise ValueError("VLM does not support explicit FlashAttention2 switching")
    for name in ("gradient_checkpointing_enable", "gradient_checkpointing_disable"):
        if not callable(getattr(language, name, None)):
            raise ValueError("Language checkpointing cannot be applied and verified")
    if not hasattr(action_transformer, "gradient_checkpointing"):
        raise ValueError("Action transformer has no explicit checkpointing flag")
    visual = getattr(vlm, "visual", None)
    if visual is None:
        visual = getattr(getattr(vlm, "model", None), "visual", None)
    modules = {id(module): name or "<root>" for name, module in vlm.named_modules()}
    for label, module in (("root", vlm), ("language", language), ("visual", visual)):
        if id(module) not in modules or not hasattr(getattr(module, "config", None), "_attn_implementation"):
            raise ValueError("Missing verifiable Qwen " + label + " attention configuration")
    vlm.set_attn_implementation("flash_attention_2")
    attention = {}
    for name, module in vlm.named_modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            actual = config._attn_implementation
            if actual != "flash_attention_2":
                raise RuntimeError("Attention fallback rejected in submodule " + name + ": " + str(actual))
            attention[name or "<root>"] = actual
    if not attention:
        raise RuntimeError("No verifiable attention configuration")
    enabled = recipe["gradient_checkpointing"]
    getattr(language, "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable")()
    action_transformer.gradient_checkpointing = enabled
    actual = getattr(language, "is_gradient_checkpointing", None)
    if type(actual) is not bool or actual != enabled:
        raise RuntimeError("Language checkpointing setting did not take effect")
    if action_transformer.gradient_checkpointing is not enabled:
        raise RuntimeError("Action checkpointing setting did not take effect")
    return {"schema": "st_official_model_runtime_v1", "policy_identity": policy["identity"],
            "attention_modules": attention, "effective_gradient_checkpointing": enabled,
            "environment": environment, "weights_or_trainable_set_changed": False,
            "gpu_qualified": False}


def verify_official_engine(engine, optimizer, policy, *, require_moments=False):
    """Call after DS initialize/master restoration, and after the first update."""
    _policy(policy)
    zero = engine.optimizer
    if (type(optimizer) is not torch.optim.AdamW or zero.optimizer is not optimizer
            or not getattr(optimizer, "_st_official_optimizer_identity", None)):
        raise ValueError("Actual engine did not retain this preset's Torch AdamW client")
    if (engine.zero_optimization_stage() != 2 or not engine.bfloat16_enabled()
            or zero.cpu_offload or zero.dtype != torch.bfloat16
            or zero.gradient_accumulation_dtype != torch.bfloat16):
        raise ValueError("Actual ZeRO/BF16/offload contract differs")
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in engine.module.parameters()):
        raise ValueError("Execution parameter dtype differs from BF16")
    masters = zero.single_partition_of_fp32_groups
    if not masters or any(p.dtype != torch.float32 for p in masters):
        raise ValueError("Missing or wrong FP32 master partitions")
    if (len(optimizer.param_groups) != len(masters)
            or any(len(group["params"]) != 1 or group["params"][0] is not master
                   for group, master in zip(optimizer.param_groups, masters))):
        raise ValueError("Torch optimizer parameter groups are not the ordered ZeRO master partitions")
    if (zero.reduce_bucket_size != BACKEND["reduce_bucket_elements"]
            or zero.allgather_bucket_size != BACKEND["allgather_bucket_elements"]):
        raise ValueError("Actual ZeRO communication buckets differ from official values")
    moments = 0
    for master in masters:
        state = optimizer.state.get(master, {})
        if not state and not require_moments:
            continue
        for name in ("exp_avg", "exp_avg_sq"):
            if name not in state or state[name].dtype != torch.float32 or state[name].shape != master.shape:
                raise ValueError("Missing or wrong FP32 Adam moment: " + name)
            moments += state[name].numel()
    return {"schema": "st_official_engine_check_v1", "policy_identity": policy["identity"],
            "optimizer_layout_identity": optimizer._st_official_optimizer_identity,
            "master_elements": sum(p.numel() for p in masters), "moment_elements": moments,
            "moments_required": require_moments, "execution_dtype": "bfloat16",
            "gradient_accumulation_dtype": "bfloat16", "master_dtype": "float32",
            "optimizer": "torch.optim.AdamW"}


def main():
    parser = argparse.ArgumentParser(description="Resolve official execution without launching training")
    parser.add_argument("--runtime-policy", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import yaml
    request = yaml.safe_load(Path(args.runtime_policy).read_text())
    cfg = yaml.safe_load(Path(args.training_config).read_text())
    resolved = resolve_official_execution(request, execution=cfg["execution"], optimizer=cfg["optimizer"],
                                         gradient_checkpointing=cfg["runtime"]["gradient_checkpointing"])
    from starVLA.dataloader.st_world_targets import atomic_json
    atomic_json(Path(args.output), resolved, immutable=True)
    print(json.dumps({"resolved_identity": resolved["identity"], "gpu_qualified": False}))


if __name__ == "__main__":
    main()
