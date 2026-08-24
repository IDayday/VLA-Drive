#!/usr/bin/env python3
"""Build fixed, planning-matched and geometry-far GP-SQ3D-Mix donors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.cache.navsim_feature_cache import NavsimFeatureCacheReader
from starVLA.gp_sq3dmix_v2 import (
    HARD_NEGATIVE_CONTRACT,
    descriptor_projection,
    hard_negative_contract_sha256,
    navsim_log_id,
    navsim_navigation_command,
    navsim_training_action,
    pooled_scene_descriptor,
    sha256_bytes,
    sha256_file,
    token_order_sha256,
)
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_geometry_per_view,
)


SCHEMA_VERSION = 2
CROSS_LOG_TEMPORAL_DISTANCE_SENTINEL = 1.0e30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-datalist", required=True)
    parser.add_argument("--source-cache-root", required=True)
    parser.add_argument("--source-processed-root", required=True)
    parser.add_argument("--source-descriptors", required=True)
    parser.add_argument("--source-stats-manifest", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--source-log-index")
    parser.add_argument("--target-datalist", required=True)
    parser.add_argument("--target-cache-root", required=True)
    parser.add_argument("--target-processed-root", required=True)
    parser.add_argument("--target-split", choices=("train", "test", "mini"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--act-norm", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def read_tokens(path: Path) -> list[str]:
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(token, str) and token for token in tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError(f"Invalid unique-token datalist: {path}")
    return tokens


def _processed_path(root: Path, split: str, token: str) -> Path:
    return root / "meta" / split / f"{token}.pkl"


def _read_planning_record(arguments: tuple[str, str, str, bool]) -> dict:
    root_string, split, token, act_norm = arguments
    path = _processed_path(Path(root_string), split, token)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    return {
        "token": token,
        "action": navsim_training_action(raw, act_norm=act_norm),
        "command": navsim_navigation_command(raw),
        "log_id": navsim_log_id(raw),
    }


def _frame_indices(
    tokens: list[str], log_ids: list[str], log_index_path: Path | None
) -> tuple[list[int], str]:
    if log_index_path is not None:
        if not log_index_path.is_file():
            raise FileNotFoundError(log_index_path)
        mapping = json.loads(log_index_path.read_text(encoding="utf-8"))
        positions = {}
        for log_id, values in mapping.items():
            if not isinstance(values, list):
                raise ValueError("NAVSIM log-index entries must be token lists")
            for index, token in enumerate(values):
                if token in positions:
                    raise ValueError("NAVSIM log-index contains a duplicate token")
                positions[token] = (str(log_id), index)
        result = []
        for token, log_id in zip(tokens, log_ids):
            if token not in positions:
                raise RuntimeError(f"Token is absent from NAVSIM log index: {token}")
            indexed_log, frame_index = positions[token]
            if indexed_log != log_id:
                raise RuntimeError(
                    f"Processed log id disagrees with NAVSIM log index for {token}"
                )
            result.append(int(frame_index))
        return result, sha256_file(log_index_path)

    counts: Counter[str] = Counter()
    result = []
    for log_id in log_ids:
        result.append(counts[log_id])
        counts[log_id] += 1
    identity = sha256_bytes(
        json.dumps(
            list(zip(tokens, log_ids, result)),
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return result, identity


def _planning_content_sha(
    tokens: list[str], actions: torch.Tensor, commands: list[str], log_ids: list[str]
) -> str:
    digest = hashlib.sha256()
    digest.update(token_order_sha256(tokens).encode("ascii"))
    digest.update(actions.contiguous().numpy().tobytes())
    digest.update(
        json.dumps(
            [commands, log_ids], separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
    return digest.hexdigest()


def load_or_build_source_index(
    *,
    path: Path,
    datalist: Path,
    processed_root: Path,
    tokens: list[str],
    log_index_path: Path | None,
    num_workers: int,
    act_norm: bool,
) -> dict:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_datalist_sha256": sha256_file(datalist),
        "token_order_sha256": token_order_sha256(tokens),
        "sample_count": len(tokens),
        "act_norm": bool(act_norm),
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError("Source planning index is not a dictionary")
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"Source planning index mismatch for {key}")
        if payload.get("tokens") != tokens:
            raise RuntimeError("Source planning index token order mismatch")
        actions = payload.get("actions")
        if actions.shape != (len(tokens), 8, 4) or actions.dtype != torch.float32:
            raise RuntimeError("Source planning index action tensor mismatch")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(processed_root), "train", token, bool(act_norm)) for token in tokens
    ]
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        records = list(pool.map(_read_planning_record, jobs, chunksize=64))
    actions = torch.from_numpy(np.stack([record["action"] for record in records]))
    commands = [record["command"] for record in records]
    log_ids = [record["log_id"] for record in records]
    frame_indices, log_index_sha = _frame_indices(
        tokens, log_ids, log_index_path
    )
    payload = {
        **expected,
        "tokens": tokens,
        "actions": actions,
        "commands": commands,
        "log_ids": log_ids,
        # The processed public schema exposes log identity but no finer episode
        # id.  Treating a log as the episode is conservative because all same-log
        # candidates are forbidden before temporal filtering.
        "episode_ids": log_ids,
        "frame_indices": frame_indices,
        "frame_interval_seconds": 0.5,
        "log_index_sha256": log_index_sha,
        "processed_data_root_identity": _planning_content_sha(
            tokens, actions, commands, log_ids
        ),
    }
    atomic_torch_save(path, payload)
    return payload


def _validate_descriptor_asset(
    descriptor_path: Path, stats_manifest_path: Path, source_tokens: list[str]
) -> torch.Tensor:
    stats_manifest = json.loads(stats_manifest_path.read_text(encoding="utf-8"))
    if stats_manifest.get("complete") is not True:
        raise RuntimeError("Slot-stat descriptor manifest is incomplete")
    if stats_manifest.get("descriptor_file_sha256") != sha256_file(descriptor_path):
        raise RuntimeError("Descriptor file SHA256 does not match slot-stat manifest")
    payload = torch.load(descriptor_path, map_location="cpu", weights_only=True)
    if payload.get("tokens") != source_tokens:
        raise RuntimeError("Descriptor token order does not match source datalist")
    descriptors = payload.get("descriptors")
    if descriptors.shape != (len(source_tokens), 128) or descriptors.dtype != torch.float16:
        raise RuntimeError("Descriptor asset must contain float16[N,128]")
    if not torch.isfinite(descriptors).all():
        raise RuntimeError("Descriptor asset contains non-finite values")
    return descriptors.float()


def _target_records(
    target_tokens: list[str], processed_root: Path, split: str, workers: int, act_norm: bool
) -> list[dict]:
    jobs = [
        (str(processed_root), split, token, bool(act_norm))
        for token in target_tokens
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_read_planning_record, jobs, chunksize=64))


def _target_descriptors(
    target_tokens: list[str],
    source_token_to_index: dict[str, int],
    source_descriptors: torch.Tensor,
    target_cache_root: Path,
) -> torch.Tensor:
    result = torch.empty((len(target_tokens), 128), dtype=torch.float32)
    missing = []
    for index, token in enumerate(target_tokens):
        source_index = source_token_to_index.get(token)
        if source_index is None:
            missing.append((index, token))
        else:
            result[index] = source_descriptors[source_index]
    if missing:
        reader = NavsimFeatureCacheReader(
            target_cache_root, components=("vggt_dense",), strict=True
        )
        projection = descriptor_projection()
        for completed, (target_index, token) in enumerate(missing, 1):
            payload = reader.get("vggt_dense", target_index, token)
            if payload is None:
                raise RuntimeError(f"Target dense-cache miss for {token}")
            pooled = pool_dense_vggt_geometry_per_view(
                [payload], device="cpu", dtype=torch.float32
            )["features"][0]
            result[target_index] = pooled_scene_descriptor(pooled, projection)
            if completed % 250 == 0:
                print(
                    f"[hard-negative] target descriptors {completed}/{len(missing)}",
                    flush=True,
                )
    return torch.nn.functional.normalize(result, dim=-1, eps=1e-12)


def _candidate_rows(
    target_actions: np.ndarray,
    target_commands: list[str],
    source_actions: np.ndarray,
    source_commands: list[str],
) -> list[list[tuple[int, float, int]]]:
    command_indices: dict[str, np.ndarray] = {}
    command_trees: dict[str, cKDTree] = {}
    for command in sorted(set(source_commands)):
        indices = np.asarray(
            [index for index, value in enumerate(source_commands) if value == command],
            dtype=np.int64,
        )
        if len(indices) < 193:
            raise RuntimeError(
                f"Command {command!r} has only {len(indices)} source donors"
            )
        command_indices[command] = indices
        command_trees[command] = cKDTree(source_actions[indices])
    rows = []
    for target_action, command in zip(target_actions, target_commands):
        if command not in command_trees:
            raise RuntimeError(f"No source donors for navigation command {command!r}")
        indices = command_indices[command]
        k = min(HARD_NEGATIVE_CONTRACT["action_top_k"], len(indices))
        distances, local_indices = command_trees[command].query(target_action, k=k)
        distances = np.atleast_1d(distances)
        local_indices = np.atleast_1d(local_indices)
        rows.append(
            [
                (int(indices[int(local)]), float(distance), rank)
                for rank, (local, distance) in enumerate(
                    zip(local_indices, distances), start=1
                )
            ]
        )
    return rows


def build_map(
    *,
    target_tokens: list[str],
    target_records: list[dict],
    target_descriptors: torch.Tensor,
    source: dict,
    source_descriptors: torch.Tensor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[list[dict], dict]:
    source_tokens = source["tokens"]
    source_actions_raw = source["actions"].numpy()
    target_actions_raw = np.stack([record["action"] for record in target_records])
    source_actions = ((source_actions_raw - action_mean) / action_std).reshape(
        len(source_tokens), -1
    )
    target_actions = ((target_actions_raw - action_mean) / action_std).reshape(
        len(target_tokens), -1
    )
    source_commands = list(source["commands"])
    target_commands = [record["command"] for record in target_records]
    candidate_rows = _candidate_rows(
        target_actions, target_commands, source_actions, source_commands
    )
    reuse: Counter[int] = Counter()
    source_log_ids = source["log_ids"]
    source_episode_ids = source["episode_ids"]
    source_token_to_index = {token: index for index, token in enumerate(source_tokens)}
    output = []
    fallback_counts: Counter[int] = Counter()
    for target_index, (token, record, candidates) in enumerate(
        zip(target_tokens, target_records, candidate_rows)
    ):
        target_source_index = source_token_to_index.get(token)
        selected = None
        for fallback_level, minimum_rank, maximum_rank, capacity in (
            (0, 9, 128, 16),
            (1, 5, 192, 16),
            (2, 5, 192, 32),
        ):
            prelim = []
            for donor_index, action_distance, rank in candidates:
                if not minimum_rank <= rank <= maximum_rank:
                    continue
                if donor_index == target_source_index or source_tokens[donor_index] == token:
                    continue
                if source_commands[donor_index] != record["command"]:
                    continue
                if source_log_ids[donor_index] == record["log_id"]:
                    continue
                if source_episode_ids[donor_index] == record["log_id"]:
                    continue
                if reuse[donor_index] >= capacity:
                    continue
                prelim.append((donor_index, action_distance, rank))
            valid = []
            if prelim:
                donor_indices = torch.as_tensor(
                    [value[0] for value in prelim], dtype=torch.long
                )
                geometry_distances = 1.0 - torch.mv(
                    source_descriptors.index_select(0, donor_indices),
                    target_descriptors[target_index],
                )
                for (
                    donor_index,
                    action_distance,
                    rank,
                ), geometry_distance_tensor in zip(prelim, geometry_distances):
                    geometry_distance = float(geometry_distance_tensor)
                    valid.append(
                    (
                        -geometry_distance,
                        source_tokens[donor_index],
                        donor_index,
                        action_distance,
                        rank,
                        geometry_distance,
                    )
                    )
            if valid:
                selected = (fallback_level, min(valid))
                break
        if selected is None:
            raise RuntimeError(
                f"No legal fixed hard donor for target {token}; random/batch fallback forbidden"
            )
        fallback_level, candidate = selected
        _, donor_token, donor_index, action_distance, rank, geometry_distance = candidate
        reuse[donor_index] += 1
        fallback_counts[fallback_level] += 1
        output.append(
            {
                "target_token": token,
                "donor_token": donor_token,
                "target_index": target_index,
                "donor_index": int(donor_index),
                "command": record["command"],
                "action_distance": float(action_distance),
                "action_neighbor_rank": int(rank),
                "geometry_cosine_distance": float(geometry_distance),
                "same_log": False,
                "temporal_distance": CROSS_LOG_TEMPORAL_DISTANCE_SENTINEL,
                "fallback_level": int(fallback_level),
            }
        )
    fallback_rate = sum(
        count for level, count in fallback_counts.items() if level > 0
    ) / len(output)
    self_donor_count = sum(
        row["target_token"] == row["donor_token"] for row in output
    )
    same_log_violations = sum(bool(row["same_log"]) for row in output)
    if fallback_rate > 0.01:
        raise RuntimeError(
            f"Hard-negative fallback rate {fallback_rate:.6f} exceeds 1%"
        )
    if self_donor_count or same_log_violations:
        raise RuntimeError("Hard-negative map violates self/same-log gates")
    return output, {
        "fallback_rate": fallback_rate,
        "fallback_counts": {str(key): value for key, value in sorted(fallback_counts.items())},
        "donor_reuse_histogram": {
            str(key): value for key, value in sorted(Counter(reuse.values()).items())
        },
        "maximum_donor_reuse": max(reuse.values(), default=0),
        "same_log_violation_count": same_log_violations,
        "self_donor_count": self_donor_count,
    }


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    source_datalist = Path(args.source_datalist).resolve()
    source_cache_root = Path(args.source_cache_root).resolve()
    source_processed_root = Path(args.source_processed_root).resolve()
    descriptor_path = Path(args.source_descriptors).resolve()
    stats_manifest_path = Path(args.source_stats_manifest).resolve()
    source_index_path = Path(args.source_index).resolve()
    target_datalist = Path(args.target_datalist).resolve()
    target_cache_root = Path(args.target_cache_root).resolve()
    target_processed_root = Path(args.target_processed_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_log_index = (
        Path(args.source_log_index).resolve() if args.source_log_index else None
    )
    source_cache_manifest = source_cache_root / "vggt_dense" / "manifest.json"
    target_cache_manifest = target_cache_root / "vggt_dense" / "manifest.json"
    required = (
        source_datalist,
        descriptor_path,
        stats_manifest_path,
        target_datalist,
        source_cache_manifest,
        target_cache_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_tokens = read_tokens(source_datalist)
    target_tokens = read_tokens(target_datalist)
    source_cache_metadata = json.loads(source_cache_manifest.read_text(encoding="utf-8"))
    target_cache_metadata = json.loads(target_cache_manifest.read_text(encoding="utf-8"))
    if source_cache_metadata.get("datalist_sha256") != sha256_file(source_datalist):
        raise RuntimeError("Source dense cache does not match source datalist")
    if int(source_cache_metadata.get("sample_count", -1)) != len(source_tokens):
        raise RuntimeError("Source dense cache sample count mismatch")
    target_sha = sha256_file(target_datalist)
    target_is_source_subset = set(target_tokens).issubset(set(source_tokens))
    if not target_is_source_subset:
        if target_cache_metadata.get("datalist_sha256") == target_sha:
            expected_target_count = len(target_tokens)
        elif int(target_cache_metadata.get("sample_count", -1)) >= len(target_tokens):
            # Fixed navtest-2k is a subset of the immutable full-navtest cache.
            expected_target_count = None
        else:
            raise RuntimeError("Target dense cache cannot cover target datalist")
        if expected_target_count is not None and int(
            target_cache_metadata.get("sample_count", -1)
        ) != expected_target_count:
            raise RuntimeError("Target dense cache sample count mismatch")
    resolved = {
        "source_sample_count": len(source_tokens),
        "target_sample_count": len(target_tokens),
        "source_datalist_sha256": sha256_file(source_datalist),
        "target_split_sha256": target_sha,
        "descriptor_file_sha256": sha256_file(descriptor_path),
        "source_cache_manifest_sha256": sha256_file(source_cache_manifest),
        "target_cache_manifest_sha256": sha256_file(target_cache_manifest),
        "hard_negative_contract_sha256": hard_negative_contract_sha256(),
        "source_index": str(source_index_path),
        "source_index_exists": source_index_path.is_file(),
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        if output_dir.exists():
            raise FileExistsError(f"Refusing existing hard-negative output: {output_dir}")
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite hard-negative output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = load_or_build_source_index(
            path=source_index_path,
            datalist=source_datalist,
            processed_root=source_processed_root,
            tokens=source_tokens,
            log_index_path=source_log_index,
            num_workers=args.num_workers,
            act_norm=bool(args.act_norm),
        )
        source_descriptors = _validate_descriptor_asset(
            descriptor_path, stats_manifest_path, source_tokens
        )
        target_records = _target_records(
            target_tokens,
            target_processed_root,
            args.target_split,
            args.num_workers,
            bool(args.act_norm),
        )
        source_token_to_index = {
            token: index for index, token in enumerate(source_tokens)
        }
        target_descriptors = _target_descriptors(
            target_tokens,
            source_token_to_index,
            source_descriptors,
            target_cache_root,
        )
        source_actions = source["actions"].numpy().astype(np.float64)
        action_mean = source_actions.mean(axis=(0, 1), keepdims=True)
        action_std = source_actions.std(axis=(0, 1), keepdims=True)
        if np.any(action_std < 1e-8):
            raise RuntimeError("Training action channel has near-zero standard deviation")
        rows, statistics = build_map(
            target_tokens=target_tokens,
            target_records=target_records,
            target_descriptors=target_descriptors,
            source=source,
            source_descriptors=source_descriptors,
            action_mean=action_mean,
            action_std=action_std,
        )
        map_path = output_dir / "hard_negative_map.json"
        atomic_json(map_path, rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "source_datalist_sha256": sha256_file(source_datalist),
            "descriptor_file_sha256": sha256_file(descriptor_path),
            "dense_cache_manifest_sha256": sha256_file(target_cache_manifest),
            "source_dense_cache_manifest_sha256": sha256_file(source_cache_manifest),
            "processed_data_root_identity": source["processed_data_root_identity"],
            "target_split_sha256": target_sha,
            "target_sample_count": len(target_tokens),
            "source_sample_count": len(source_tokens),
            "algorithm_parameters": HARD_NEGATIVE_CONTRACT,
            "hard_negative_contract_sha256": hard_negative_contract_sha256(),
            "action_channel_mean": action_mean.reshape(-1).tolist(),
            "action_channel_std": action_std.reshape(-1).tolist(),
            "code_commit": git_commit(),
            "map_file": map_path.name,
            "map_file_sha256": sha256_file(map_path),
            "source_index_file_sha256": sha256_file(source_index_path),
            "cross_log_temporal_distance_sentinel": CROSS_LOG_TEMPORAL_DISTANCE_SENTINEL,
            **statistics,
        }
        atomic_json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException:
        # A directory without a complete manifest must never be mistaken for a
        # usable map, but keep it for diagnosis instead of deleting user data.
        atomic_json(
            output_dir / "FAILED.json",
            {"status": "failed", "code_commit": git_commit()},
        )
        raise


if __name__ == "__main__":
    main()
