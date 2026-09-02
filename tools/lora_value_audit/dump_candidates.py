#!/usr/bin/env python3
"""Online FP32 export of every DrivoR proposal stage and scorer value.

Only current camera frames and ego status enter this process. Captured scene
tokens are written in bounded shards so arbitrary-union scorer tests do not
rerun the visual encoder. No metric cache, future annotation, GT, or PDM value
is imported here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import navsim
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

from .drivor_adapter import load_drivor_model, max_abs, score_proposals, stack_factor_logits
from .replay import stable_shard
from .utils import atomic_json, load_proposal_pickle, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-proposals", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--drivor-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--repeat-parity-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _completed_tokens(directory: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted(directory.glob("candidate_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            values = archive["tokens"].astype(str).tolist()
        overlap = result.intersection(values)
        if overlap:
            raise RuntimeError(f"Duplicate completed token: {sorted(overlap)[:3]}")
        result.update(values)
    return result


def _log_mapping(path: Path) -> Dict[str, str]:
    with np.load(path, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        logs = archive["log_names"].astype(str)
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate token in candidate matrix")
    return dict(zip(tokens.tolist(), logs.tolist()))


def _batch_features(entries, loader, builder, device):
    images, statuses = [], []
    for token, _log_name in entries:
        value = builder.compute_features(loader.get_agent_input_from_token(token))
        images.append(value["image"])
        statuses.append(value["ego_status"])
    return {
        "image": torch.stack(images).to(device=device, dtype=torch.float32),
        "ego_status": torch.stack(statuses).to(device=device, dtype=torch.float32),
    }


def _trajectory_error(output) -> float:
    indices = output["pdm_score"].argmax(dim=1)
    selected = output["proposals"][torch.arange(len(indices), device=indices.device), indices]
    return max_abs(selected, output["trajectory"])


def main() -> None:
    args = parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    imported = Path(navsim.__file__).resolve()
    if args.drivor_repo.resolve() not in imported.parents:
        raise RuntimeError(f"Imported navsim from {imported}, not {args.drivor_repo}")
    for path in (
        args.reference_proposals,
        args.candidate_matrix,
        args.checkpoint,
        args.config,
        args.dino_weights,
        args.log_path,
        args.sensor_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")

    references = load_proposal_pickle(args.reference_proposals)
    logs = _log_mapping(args.candidate_matrix)
    if set(references) != set(logs):
        raise RuntimeError(f"Token inventory mismatch reference={len(references)} logs={len(logs)}")
    entries = [
        (token, logs[token])
        for token in sorted(references)
        if stable_shard(token, args.shard_count) == args.shard_index
    ]
    if args.token_file is not None:
        requested = {line.strip() for line in args.token_file.read_text().splitlines() if line.strip()}
        unknown = requested.difference(references)
        if unknown:
            raise RuntimeError(f"Token file contains {len(unknown)} tokens outside navtest")
        entries = [entry for entry in entries if entry[0] in requested]
    if args.max_scenes:
        entries = entries[: args.max_scenes]
    if not entries:
        raise RuntimeError("No scenes selected")

    checkpoint_sha = sha256_file(args.checkpoint)
    lineage = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(args.drivor_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "dino_weights": str(args.dino_weights.resolve()),
        "dino_weights_sha256": sha256_file(args.dino_weights),
        "reference_proposals": str(args.reference_proposals.resolve()),
        "reference_proposals_sha256": sha256_file(args.reference_proposals),
        "candidate_matrix": str(args.candidate_matrix.resolve()),
        "candidate_matrix_sha256": sha256_file(args.candidate_matrix),
        "split": "navtest",
        "precision": "fp32",
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "metric_cache_input": False,
        "proposal_num": 64,
        "ref_num": 4,
        "scorer_ref_num": 4,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "token_file": str(args.token_file.resolve()) if args.token_file else None,
        "token_file_sha256": sha256_file(args.token_file) if args.token_file else None,
        "hydra_overrides": [],
        "log_path": str(args.log_path.resolve()),
        "sensor_root": str(args.sensor_root.resolve()),
    }
    shard_dir = args.output_dir / f"shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        if args.overwrite:
            raise RuntimeError("Refusing destructive overwrite; use a new output directory")
        if not args.resume:
            raise FileExistsError(manifest_path)
        print(json.dumps({"status": "already_complete", "manifest": str(manifest_path)}))
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = shard_dir / "lineage.json"
    if lineage_path.exists():
        previous = json.loads(lineage_path.read_text())
        left, right = dict(previous), dict(lineage)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            raise RuntimeError("Partial output lineage mismatch")
        lineage = previous
    else:
        atomic_json(lineage_path, lineage)

    completed = _completed_tokens(shard_dir)
    if not completed.issubset({token for token, _ in entries}):
        raise RuntimeError("Partial output contains token outside selected shard")
    pending = [entry for entry in entries if entry[0] not in completed]
    unique_logs = sorted({log_name for _, log_name in entries})
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        log_names=unique_logs,
        tokens=[token for token, _ in entries],
    )
    sensor_config = SensorConfig(
        cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[], cam_r0=[3],
        cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[],
    )
    loader = SceneLoader(args.log_path, args.sensor_root, scene_filter, sensor_config)
    missing_loader = {token for token, _ in entries}.difference(loader.tokens)
    if missing_loader:
        raise RuntimeError(f"SceneLoader misses {len(missing_loader)} tokens")
    model, config = load_drivor_model(args.config, args.checkpoint, args.dino_weights, device)
    builder = DrivoRFeatureBuilder(config)

    captured: Dict[str, torch.Tensor] = {}
    hook = model.image_backbone.register_forward_hook(
        lambda _module, _inputs, value: captured.__setitem__("scene_features", value)
    )
    output: Dict[str, List[object]] = {
        key: []
        for key in (
            "tokens", "log_names", "proposal_stages", "factor_logits",
            "factor_probabilities", "predicted_scores", "selected_indices",
            "trajectories", "scene_features", "ego_status",
        )
    }
    chunk_index = len(list(shard_dir.glob("candidate_*.npz")))
    repeat_remaining = max(0, args.repeat_parity_scenes)
    maxima = {
        "forward_repeat_final_proposals": 0.0,
        "forward_repeat_factor_logits": 0.0,
        "forward_repeat_predicted_scores": 0.0,
        "external_original64_factor_logits": 0.0,
        "external_original64_predicted_scores": 0.0,
        "selected_trajectory": 0.0,
        "locked_final_proposals": 0.0,
        # The pre-existing native cache deliberately stored factor logits as
        # float16.  Compare its lossless float16 round-trip representation;
        # newly exported raw logits remain float32.
        "locked_factor_logits_fp16_roundtrip": 0.0,
        "locked_predicted_scores": 0.0,
    }
    selected_index_equal = True
    failures: List[Dict[str, str]] = []

    def flush() -> None:
        nonlocal chunk_index
        if not output["tokens"]:
            return
        _atomic_npz(
            shard_dir / f"candidate_{chunk_index:06d}.npz",
            tokens=np.asarray(output["tokens"]),
            log_names=np.asarray(output["log_names"]),
            proposal_stages=np.stack(output["proposal_stages"]).astype(np.float32),
            factor_logits=np.stack(output["factor_logits"]).astype(np.float32),
            factor_probabilities=np.stack(output["factor_probabilities"]).astype(np.float32),
            predicted_scores=np.stack(output["predicted_scores"]).astype(np.float32),
            selected_indices=np.asarray(output["selected_indices"], dtype=np.int16),
            trajectories=np.stack(output["trajectories"]).astype(np.float32),
            scene_features=np.stack(output["scene_features"]).astype(np.float32),
            ego_status=np.stack(output["ego_status"]).astype(np.float32),
        )
        chunk_index += 1
        for values in output.values():
            values.clear()

    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_entries = pending[start : start + args.batch_size]
            try:
                features = _batch_features(batch_entries, loader, builder, device)
                captured.clear()
                native = model(dict(features))
                scene_features = captured["scene_features"]
                status = features["ego_status"][:, -1]
                ego_token = model.hist_encoding(status)[:, None]
                external = score_proposals(model, config, native["proposals"], scene_features, ego_token)
                native_logits = stack_factor_logits(native["pred_logit"])
                maxima["external_original64_factor_logits"] = max(
                    maxima["external_original64_factor_logits"], max_abs(native_logits, external["factor_logits"])
                )
                maxima["external_original64_predicted_scores"] = max(
                    maxima["external_original64_predicted_scores"], max_abs(native["pdm_score"], external["pdm_score"])
                )
                maxima["selected_trajectory"] = max(maxima["selected_trajectory"], _trajectory_error(native))

                if repeat_remaining:
                    repeat_count = min(repeat_remaining, len(batch_entries))
                    # Keep the native batch shape: CUDA attention kernels may
                    # choose a different numerical path at a smaller batch size.
                    # Determinism is checked on the requested prefix only.
                    repeated = model(dict(features))
                    maxima["forward_repeat_final_proposals"] = max(
                        maxima["forward_repeat_final_proposals"], max_abs(native["proposals"][:repeat_count], repeated["proposals"][:repeat_count])
                    )
                    maxima["forward_repeat_factor_logits"] = max(
                        maxima["forward_repeat_factor_logits"], max_abs(native_logits[:repeat_count], stack_factor_logits(repeated["pred_logit"])[:repeat_count])
                    )
                    maxima["forward_repeat_predicted_scores"] = max(
                        maxima["forward_repeat_predicted_scores"], max_abs(native["pdm_score"][:repeat_count], repeated["pdm_score"][:repeat_count])
                    )
                    repeat_remaining -= repeat_count

                stages = torch.stack(native["proposal_list"], dim=1)
                for row, (token, log_name) in enumerate(batch_entries):
                    reference = references[token]
                    ref_proposals = torch.as_tensor(reference["proposals"], device=device)
                    ref_logits = torch.as_tensor(reference["factor_logits"], device=device)
                    ref_scores = torch.as_tensor(reference["predicted_scores"], device=device)
                    maxima["locked_final_proposals"] = max(maxima["locked_final_proposals"], max_abs(native["proposals"][row], ref_proposals))
                    maxima["locked_factor_logits_fp16_roundtrip"] = max(
                        maxima["locked_factor_logits_fp16_roundtrip"],
                        max_abs(native_logits[row].half().float(), ref_logits),
                    )
                    maxima["locked_predicted_scores"] = max(maxima["locked_predicted_scores"], max_abs(native["pdm_score"][row], ref_scores))
                    selected_index_equal &= int(native["pdm_score"][row].argmax()) == int(ref_scores.argmax())
                    output["tokens"].append(token)
                    output["log_names"].append(log_name)
                    output["proposal_stages"].append(stages[row].cpu().numpy())
                    output["factor_logits"].append(native_logits[row].cpu().numpy())
                    output["factor_probabilities"].append(native_logits[row].sigmoid().cpu().numpy())
                    output["predicted_scores"].append(native["pdm_score"][row].cpu().numpy())
                    output["selected_indices"].append(int(native["pdm_score"][row].argmax()))
                    output["trajectories"].append(native["trajectory"][row].cpu().numpy())
                    output["scene_features"].append(scene_features[row].cpu().numpy())
                    output["ego_status"].append(status[row].cpu().numpy())
                if len(output["tokens"]) >= args.chunk_size:
                    flush()
                print(json.dumps({"processed": min(start + len(batch_entries), len(pending)), "pending": len(pending), "shard": args.shard_index}), flush=True)
            except Exception as error:
                failures.extend({"token": token, "error": repr(error)} for token, _ in batch_entries)
                with (shard_dir / "failed_tokens.jsonl").open("a") as stream:
                    for token, _ in batch_entries:
                        stream.write(json.dumps({"token": token, "error": repr(error)}) + "\n")
    hook.remove()
    flush()
    completed_after = _completed_tokens(shard_dir)
    expected = {token for token, _ in entries}
    missing = sorted(expected.difference(completed_after))
    extra = sorted(completed_after.difference(expected))
    failure_rate = len(failures) / len(entries)
    repeat_count = args.repeat_parity_scenes - repeat_remaining
    parity_passed = (
        max(maxima.values()) <= args.parity_tolerance
        and selected_index_equal
        and not missing
        and not extra
        and repeat_count == min(args.repeat_parity_scenes, len(entries))
        and failure_rate <= 0.001
    )
    manifest = {
        "lineage": lineage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(completed_after),
        "expected_scene_count": len(entries),
        "missing_token_count": len(missing),
        "extra_token_count": len(extra),
        "failed_token_count": len(failures),
        "failure_rate": failure_rate,
        "candidate_count": int(config.proposal_num),
        "stage_count": int(config.ref_num) + 1,
        "pose_count": int(config.num_poses),
        "repeat_parity_scene_count": repeat_count,
        "selected_index_equal": selected_index_equal,
        "max_abs_error": maxima,
        "parity_tolerance": args.parity_tolerance,
        "parity_passed": parity_passed,
        "chunk_count": chunk_index,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not parity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
