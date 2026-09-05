#!/usr/bin/env python3
"""Hard preflight for either formal dual-initialization PlanReg-WM run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (  # noqa: E402
    audit_vlm_checkpoint,
    sha256_file,
)
from navsim.agents.EpisodeDrive.shared_planreg_initialization import (  # noqa: E402
    state_sha256,
)
from navsim.planning.training.input_only_cache import (  # noqa: E402
    DYNAMIC_FEATURE_CACHE_KEYS,
    validate_input_only_manifest,
)


EXPECTED_DATASET_SIZE = 103288


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def validate_layout_lock(lock: Dict[str, Any]) -> None:
    required = (
        "selected_layout",
        "gpu_count",
        "per_gpu_batch_size",
        "global_batch_size",
        "num_workers_per_rank",
        "scorer_processes_per_rank",
        "scorer_partitions_per_scene",
        "gradient_checkpointing",
        "read_only_attention_backend",
        "steps_per_epoch",
        "total_steps",
    )
    missing = [name for name in required if name not in lock]
    if missing:
        raise RuntimeError(f"Layout lock is missing fields: {missing}")
    if not bool(lock.get("shared_between_base_and_vqa", False)):
        raise RuntimeError("Layout lock is not declared shared between Base and VQA")
    if int(lock.get("dataset_length", -1)) != EXPECTED_DATASET_SIZE:
        raise RuntimeError("Layout lock does not describe the complete 103k trainval")
    if int(lock.get("dataset_epochs", -1)) != 27:
        raise RuntimeError("Layout lock must contain exactly 27 dataset epochs")
    global_batch = int(lock["global_batch_size"])
    if global_batch != int(lock["gpu_count"]) * int(lock["per_gpu_batch_size"]):
        raise RuntimeError("Layout lock global batch is internally inconsistent")
    if not isinstance(lock["gradient_checkpointing"], bool):
        raise RuntimeError("Layout lock gradient_checkpointing must be boolean")
    if int(lock["scorer_processes_per_rank"]) <= 0:
        raise RuntimeError("Layout lock scorer process count must be positive")
    if int(lock["scorer_partitions_per_scene"]) <= 0:
        raise RuntimeError("Layout lock scorer partition count must be positive")
    if lock["read_only_attention_backend"] not in {"eager", "split_sdpa"}:
        raise RuntimeError("Layout lock has an unsupported attention backend")
    steps = math.ceil(EXPECTED_DATASET_SIZE / global_batch)
    if int(lock["steps_per_epoch"]) != steps or int(lock["total_steps"]) != steps * 27:
        raise RuntimeError("Layout lock step budget is internally inconsistent")


def validate_resume_path(
    resume_checkpoint: Path, output_dir: Path, identity: Dict[str, Any]
) -> None:
    resume = resume_checkpoint.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"Explicit RESUME_CHECKPOINT does not exist: {resume}")
    try:
        resume.relative_to(output)
    except ValueError as error:
        raise RuntimeError(
            "Formal resume checkpoint must belong to the same output directory"
        ) from error
    identity_path = output / "run_metadata" / "formal_run_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("Formal resume requires the original run identity manifest")
    prior = _read_json(identity_path)
    keys = (
        "variant",
        "seed",
        "experiment_name",
        "vlm_checkpoint_sha256",
        "vlm_config_sha256",
        "shared_init_sha256",
        "layout_lock_sha256",
        "input_cache_manifest_sha256",
    )
    differences = {
        key: {"prior": prior.get(key), "current": identity.get(key)}
        for key in keys
        if prior.get(key) != identity.get(key)
    }
    if differences:
        raise RuntimeError(
            "Formal resume identity mismatch: " + json.dumps(differences, sort_keys=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-version', choices=('v1', 'v1p1', 'task_future_lite'), default='v1')
    parser.add_argument("--variant", choices=("base", "driving_vqa"), required=True)
    parser.add_argument("--vlm-path", required=True, type=Path)
    parser.add_argument("--vlm-audit", required=True, type=Path)
    parser.add_argument("--shared-init", required=True, type=Path)
    parser.add_argument("--layout-lock", required=True, type=Path)
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    status = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(
            "Formal training requires a clean worktree; commit generated protocol "
            f"artifacts first. Dirty entries: {status.splitlines()[:20]}"
        )
    git_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()

    layout = _read_json(args.layout_lock)
    validate_layout_lock(layout)
    metrics_root = args.layout_lock.parent / "throughput"
    for layout_name, expected_sha in layout.get(
        "benchmark_metrics_sha256", {}
    ).items():
        metrics_path = metrics_root / layout_name / "metrics.json"
        if not metrics_path.is_file() or sha256_file(metrics_path) != expected_sha:
            raise RuntimeError(
                "Layout lock benchmark evidence is missing or changed: "
                f"{metrics_path}"
            )
    manifest = validate_input_only_manifest(args.input_cache)
    if args.protocol_version in ('v1p1','task_future_lite'):
        if manifest.get('prompt_version') != 'single_front_v1p1':
            raise RuntimeError('V1.1 requires separately rebuilt single-front prompt input cache')
        if layout.get('protocol_version') != args.protocol_version or not layout.get('train_only_pilot_locked', False):
            raise RuntimeError('V1.1 requires a new layout lock with train-only pilot evidence; V1 GB128 lock is not reusable')
        if int(layout['global_batch_size']) not in (64, 128):
            raise RuntimeError('V1.1 pilot layouts must be GB64 or GB128')
        if args.protocol_version == 'task_future_lite':
            if manifest.get('protocol_version') != 'task_future_lite' or manifest.get('schema_version') != 2:
                raise RuntimeError('Lite requires a NEW schema-v2 logged-pose input cache')
            if not layout.get('full_physical_sidecar_smoke_passed',False):
                raise RuntimeError('Lite needs representative full-step physical-sidecar smoke evidence')
    if int(manifest.get("record_count", -1)) != EXPECTED_DATASET_SIZE:
        raise RuntimeError(
            "Formal input-only cache must contain exactly 103,288 records; "
            f"found {manifest.get('record_count')}"
        )
    if not bool(manifest.get("required_source_files_complete", False)):
        raise RuntimeError(
            "Formal input cache manifest does not prove complete raw input/target sources"
        )
    if not bool(manifest.get("front_camera_only", False)) or int(
        manifest.get("sensor_camera_count", -1)
    ) != 1:
        raise RuntimeError("Formal input-only cache must be front-camera-only")
    forbidden = sorted(
        set(manifest.get("cached_fields", ())).intersection(DYNAMIC_FEATURE_CACHE_KEYS)
    )
    if forbidden:
        raise RuntimeError(f"Input cache contains dynamic feature fields: {forbidden}")

    formal_audit = _read_json(args.vlm_audit)
    if not bool(formal_audit.get("pair", {}).get("formal_pair_compatible", False)):
        raise RuntimeError("Formal VLM pair audit did not pass")
    expected_vlm = formal_audit[args.variant]
    actual_vlm = audit_vlm_checkpoint(
        str(args.vlm_path), variant=args.variant, load_runtime_classes=False
    )
    checked_vlm_fields = (
        "checkpoint_path",
        "checkpoint_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_vocab_sha256",
        "vocab_size",
        "vision_block_count",
        "vision_hidden_size",
        "patch_size",
        "llm_hidden_size",
        "prompt_template",
    )
    vlm_differences = {
        key: {"audit": expected_vlm.get(key), "actual": actual_vlm.get(key)}
        for key in checked_vlm_fields
        if expected_vlm.get(key) != actual_vlm.get(key)
    }
    if vlm_differences:
        raise RuntimeError(
            "Formal VLM no longer matches its audited checkpoint: "
            + json.dumps(vlm_differences, sort_keys=True)
        )
    if bool(actual_vlm.get("forbidden_agent_state_detected", True)):
        raise RuntimeError("Formal VLM contains agent/action/scorer parameters")
    if manifest.get("tokenizer_vocab_sha256") != actual_vlm.get(
        "tokenizer_vocab_sha256"
    ):
        raise RuntimeError(
            "Input-only cache token IDs do not match the audited formal tokenizer"
        )
    marker_detected = bool(actual_vlm.get("driving_vqa_training_marker_detected"))
    if args.variant == "base" and marker_detected:
        raise RuntimeError("Base formal VLM unexpectedly has a Driving-VQA marker")
    if args.variant == "driving_vqa" and not marker_detected:
        raise RuntimeError("Driving-VQA formal VLM lacks its provenance marker")

    try:
        shared_payload = torch.load(
            args.shared_init, map_location="cpu", weights_only=True
        )
    except TypeError:  # pragma: no cover
        shared_payload = torch.load(args.shared_init, map_location="cpu")
    shared_metadata = dict(shared_payload.get("metadata", {}))
    shared_state = shared_payload.get("trainable_state_dict")
    if not isinstance(shared_state, dict):
        raise RuntimeError("Shared initialization lacks trainable_state_dict")
    if args.protocol_version in ('v1p1','task_future_lite'):
        if any(value.dtype != torch.float32 for value in shared_state.values()):
            raise RuntimeError('V1.1 shared trainable storage must be FP32')
        if not any('global_local_readout.global_queries' in key for key in shared_state):
            raise RuntimeError('V1.1 requires new global/local readout shared initialization')
        if args.protocol_version == 'task_future_lite':
            if not any(key.startswith('physical_query_decoder.') for key in shared_state):
                raise RuntimeError('Lite requires its own shared physical-head initialization artifact')
            if any(key.startswith('future_register_predictor.') for key in shared_state):
                raise RuntimeError('Legacy latent predictor must not be in Lite shared initialization')
    if state_sha256(shared_state) != shared_metadata.get("trainable_state_sha256"):
        raise RuntimeError("Shared initialization state hash is invalid")
    if int(shared_metadata.get("seed", -1)) != args.seed:
        raise RuntimeError(
            "Shared initialization seed does not match formal run seed: "
            f"artifact={shared_metadata.get('seed')} run={args.seed}"
        )
    required_groups = {
        "planning_adapter",
        "semantic_fusion",
        "action_generator",
        "scorer",
        "future_predictor",
        "semantic_qformer",
        "vision_qv_lora",
    }
    if set(shared_metadata.get("logical_modules", {})) != required_groups:
        raise RuntimeError("Shared initialization logical topology is incomplete")

    identity = {
        "schema_version": 1,
        "experiment_name": args.experiment_name,
        "variant": args.variant,
        "seed": args.seed,
        "git_commit": git_commit,
        "agent_checkpoint_loaded": False,
        "vlm_path": str(args.vlm_path.expanduser().resolve()),
        "vlm_checkpoint_sha256": actual_vlm["checkpoint_sha256"],
        "vlm_config_sha256": actual_vlm["config_sha256"],
        "tokenizer_sha256": actual_vlm["tokenizer_sha256"],
        "shared_init_path": str(args.shared_init.expanduser().resolve()),
        "shared_init_sha256": sha256_file(args.shared_init),
        "shared_trainable_state_sha256": shared_metadata["trainable_state_sha256"],
        "shared_trainable_parameter_count": shared_metadata[
            "trainable_parameter_count"
        ],
        "layout_lock_path": str(args.layout_lock.expanduser().resolve()),
        "layout_lock_sha256": sha256_file(args.layout_lock),
        "selected_layout": layout["selected_layout"],
        "input_cache_manifest": str(
            (args.input_cache / "planreg_input_only_manifest.json").resolve()
        ),
        "input_cache_manifest_sha256": sha256_file(
            args.input_cache / "planreg_input_only_manifest.json"
        ),
        "world_model_enabled": True,
        "future_mode": "correct",
        "world_model_candidate_count": 8 if args.protocol_version == 'task_future_lite' else 1,
        "world_model_mode": 'task_future_lite' if args.protocol_version == 'task_future_lite' else 'legacy_register_prediction',
        "multi_trajectory_consequence_modeling_implemented": False,
        "consequence_scope": "Training-only multi-candidate physical answers; no V2 structured future-register model or consequence-driven deployment" if args.protocol_version == 'task_future_lite' else "Legacy K=1",
    }
    if args.resume_checkpoint is not None:
        validate_resume_path(args.resume_checkpoint, args.output_dir, identity)
        identity["resume_checkpoint"] = str(args.resume_checkpoint.resolve())
    else:
        identity["resume_checkpoint"] = None

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metadata_output.with_suffix(args.metadata_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.metadata_output)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
