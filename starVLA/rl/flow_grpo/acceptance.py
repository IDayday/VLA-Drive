"""Fail closed: bounded diagnostics cannot silently become formal experiments."""
from pathlib import Path
import json
import os
from .audit import source_fingerprints
from .config import config_hash
from .contracts import digest
from .loading import file_sha

GATES = (
    "parameter_contract",
    "own_visual_unchanged",
    "rl_sft_reference_gradients",
    "fp32_chunk_mathematics",
    "production_optimizer_update",
    "fixed_noise_ode",
    "production_repeat",
    "distributed_update_scaling",
    "checkpointing_accumulation",
    "exact_resume",
    "export_original_protocol",
    "inner_epochs_2",
    "distributed_faults",
    "global_metrics",
    "evaluation_protocol",
    "clean_checkout",
)


def executable_identity():
    files = source_fingerprints()
    for root in ("tests/flow_grpo", "scripts/flow_grpo"):
        for path in sorted(Path(root).rglob("*")):
            if path.suffix in {".py", ".sh"}:
                files[str(path)] = file_sha(path)
    return digest(files)


def acceptance_context(cfg, resume_identity):
    return {
        "executable_sha256": executable_identity(),
        "config_sha256": config_hash(cfg),
        "checkpoint_sha256": cfg["checkpoint_contract"]["sha256"],
        "resume_identity": resume_identity,
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
    }


def enforce_training_budget(cfg, context=None):
    runtime = cfg["runtime"]
    mode = runtime.get("run_mode", "diagnostic")
    maximum = runtime["max_updates"]
    if mode == "diagnostic":
        if maximum > 8:
            raise ValueError(
                "unreleased diagnostic mode is bounded to 8 optimizer updates"
            )
        if cfg["sampling"]["candidate_chunk_size"] > 1 and not runtime.get(
            "experimental_candidate_chunks", False
        ):
            raise ValueError(
                "chunk>=2 needs explicit experimental_candidate_chunks diagnostic opt-in"
            )
        return
    if mode not in {"paired_short", "formal"}:
        raise ValueError("unknown run_mode")
    if maximum > (100 if mode == "paired_short" else 2000):
        raise ValueError("pre-registered experiment budget exceeded")
    if cfg["sampling"]["candidate_chunk_size"] != 1:
        raise ValueError(
            "candidate_chunk>=2 is experimental; not accepted for paired training"
        )
    world = int(os.getenv("WORLD_SIZE", "1"))
    if runtime["accumulation_steps"] * world != 16:
        raise ValueError("paired global scene batch must be 16")
    required = {
        "sampling": {
            "group_size": 8,
            "num_steps": 10,
            "noise_level": 0.1,
            "transition_chunk_size": 1,
        },
        "algorithm": {
            "inner_epochs": 2,
            "ppo_clip_range": 0.02,
            "advantage_epsilon": 1e-6,
            "advantage_clip": 5.0,
            "reference_kl_coefficient": 0.01,
            "logprob_reduction": "flow_grpo_dimension_mean",
        },
        "retention": {"original_sft_coefficient": 0.1},
        "runtime": {
            "seed": 42,
            "deepspeed_stage": 2,
            "scene_microbatch": 1,
            "replay_microbatch": 1,
            "numerical_profile": "bf16_zero2_fp32_accum_v1",
        },
    }
    for section, fields in required.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError(f"paired recipe changed: {section}.{key}")
    path = runtime.get("acceptance_record")
    if not path or not Path(path).is_file():
        raise ValueError("NOT_READY: missing production profile acceptance record")
    record = json.loads(Path(path).read_text())
    if record.get("status") != "READY_FOR_THIS_PROFILE" or (
        context is not None and record.get("context") != context
    ):
        raise ValueError(
            "NOT_READY: acceptance record does not match current code/config/profile/assets"
        )
    for gate in GATES:
        evidence = record.get("gates", {}).get(gate, {})
        if evidence.get("status") != "PASS" or not evidence.get("path"):
            raise ValueError(f"NOT_READY: gate {gate}")
        if file_sha(evidence["path"]) != evidence.get("sha256"):
            raise ValueError(f"acceptance evidence changed: {gate}")
