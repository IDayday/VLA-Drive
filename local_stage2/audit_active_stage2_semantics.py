#!/usr/bin/env python3
"""Snapshot training-critical semantics from a live Stage-2 rank-zero process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


RELEASE_EXACT_FILES = (
    "navsim/agents/EpisodeDrive/action_decoder.py",
    "navsim/agents/EpisodeDrive/layers/losses/episode_drive_loss.py",
    "navsim/agents/EpisodeDrive/drivevla_features.py",
)

EXPECTED_OVERRIDES = {
    "seed": "2",
    "agent.checkpoint_path": "null",
    "agent.cache_data": "false",
    "agent.vlm_config.freeze_backbone": "true",
    "agent.vlm_config.cache_hidden_state": "false",
    "agent.vlm_config.cache_mode": "false",
    "agent.vlm_config.initialize_from_config": "true",
    "agent.vlm_config.use_flash_attn": "false",
    "agent.vlm_config.frozen_backbone_mode": "train",
    "agent.action_head_config.long_trajectory_additional_poses": "2",
    "agent.batch_size": "1",
    "agent.num_gpus": "16",
    "agent.lr_args.name": "AdamW",
    "agent.lr_args.base_lr": "1e-4",
    "agent.lr_args.base_batch_size": "16",
    "agent.lr_args.effective_global_batch_size": "16",
    "agent.lr_args.decay_norm_and_bias": "true",
    "dataloader.params.batch_size": "1",
    "trainer.params.devices": "8",
    "trainer.params.num_nodes": "2",
    "trainer.params.precision": "bf16-mixed",
    "trainer.params.accumulate_grad_batches": "1",
    "trainer.params.max_epochs": "27",
    "official_stage2_sampler": "true",
    "official_stage2_reference_global_batch_size": "16",
}

SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "DRIVEVLA_FUSE_VALIDATION_SCORING",
    "DRIVEVLA_SCORE_PROCESSES",
    "DRIVEVLA_SCORE_PARTITIONS",
    "DRIVEVLA_SCORE_START_METHOD",
    "DRIVEVLA_SYNC_TRAIN_METRICS",
    "DRIVEVLA_TRAIN_LOG_INTERVAL",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def parse_overrides(arguments: list[str]) -> dict[str, str]:
    """Parse Hydra ``key=value`` command arguments without interpreting values."""

    overrides = {}
    for argument in arguments:
        if "=" not in argument:
            continue
        key, value = argument.split("=", 1)
        overrides[key.lstrip("+")] = value
    return overrides


def _read_process(pid: int) -> tuple[list[str], dict[str, str]]:
    proc = Path("/proc") / str(pid)
    arguments = [
        item.decode("utf-8", "replace")
        for item in (proc / "cmdline").read_bytes().split(b"\0")
        if item
    ]
    environment = {}
    for item in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8", "replace")
        if decoded_key in SAFE_ENVIRONMENT_KEYS:
            environment[decoded_key] = value.decode("utf-8", "replace")
    return arguments, environment


def audit(repo: Path, pid: int, release_ref: str) -> dict[str, Any]:
    arguments, environment = _read_process(pid)
    overrides = parse_overrides(arguments[2:])
    source_identity = {}
    for relative_path in RELEASE_EXACT_FILES:
        current = (repo / relative_path).read_bytes()
        released = _git(repo, "show", f"{release_ref}:{relative_path}")
        source_identity[relative_path] = {
            "current_sha256": _sha256(current),
            "release_sha256": _sha256(released),
            "byte_exact": current == released,
        }

    checks = {
        key: {
            "expected": expected,
            "actual": overrides.get(key),
            "matches": overrides.get(key) == expected,
        }
        for key, expected in EXPECTED_OVERRIDES.items()
    }
    stage1_path = overrides.get("agent.stage1_checkpoint_path")
    scheduler = overrides.get("agent.scheduler_args")
    extra_checks = {
        "rank_zero_command_is_training": "run_training_full.py" in arguments[1],
        "stage1_checkpoint_present": bool(
            stage1_path and Path(stage1_path).is_file()
        ),
        "long_target_cache_selected": overrides.get("cache_path", "").endswith(
            "feature_cache_navtrain_long2"
        ),
        "source_scheduler_has_expected_shape": bool(
            scheduler
            and "dataset_size:103288" in scheduler
            and "num_epochs:27" in scheduler
            and "warmup_ratio:0.1" in scheduler
            and "start_lr_ratio:1e-6" in scheduler
        ),
        "release_trajectory_and_loss_sources_are_exact": all(
            item["byte_exact"] for item in source_identity.values()
        ),
    }

    return {
        "audit": "live_stage2_training_semantic_lock",
        "pid": pid,
        "executable": os.readlink(f"/proc/{pid}/exe"),
        "repository": str(repo.resolve()),
        "git_head": _git(repo, "rev-parse", "HEAD").decode().strip(),
        "git_branch": _git(repo, "branch", "--show-current").decode().strip(),
        "release_reference": release_ref,
        "command_arguments": arguments,
        "selected_environment": environment,
        "critical_overrides": checks,
        "source_identity": source_identity,
        "extra_checks": extra_checks,
        "all_critical_checks_pass": all(
            item["matches"] for item in checks.values()
        )
        and all(extra_checks.values()),
        "acceleration_scope": {
            "worker_image_preprocessing": overrides.get(
                "preprocess_images_in_workers"
            ),
            "worker_tokenization": overrides.get(
                "pretokenize_inputs_in_workers"
            ),
            "multiprocess_offline_scoring": environment.get(
                "DRIVEVLA_SCORE_PROCESSES"
            ),
            "fused_validation_scoring": environment.get(
                "DRIVEVLA_FUSE_VALIDATION_SCORING"
            ),
            "reduced_train_metric_collectives": environment.get(
                "DRIVEVLA_SYNC_TRAIN_METRICS"
            ),
            "interpretation": (
                "These switches change input preparation, offline-label throughput, "
                "logging, or validation I/O. The proposal decoder, trajectory loss, "
                "and target builder are byte-identical to the release source."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--release-ref", default="b9a4f27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo, args.pid, args.release_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_critical_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
