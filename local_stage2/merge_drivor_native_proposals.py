#!/usr/bin/env python3
"""Merge sharded native DrivOR proposal exports into the scorer cache schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


EXPECTED_CANDIDATES = 64
EXPECTED_POSES = 8
EXPECTED_STATE_SIZE = 3


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(dict(payload), stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _comparable_lineage(lineage: Mapping[str, object]) -> Dict[str, object]:
    value = dict(lineage)
    value.pop("created_utc", None)
    value.pop("shard_index", None)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=12_146)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    manifests = sorted(args.input_root.glob("shard_*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed shard manifests under {args.input_root}")

    manifest_payloads = [json.loads(path.read_text()) for path in manifests]
    lineage = manifest_payloads[0]["lineage"]
    if not bool(lineage.get("native_proposals", False)):
        raise RuntimeError("Input lineage is not a native DrivOR proposal export")
    expected_shards = int(lineage["shard_count"])
    if len(manifests) != expected_shards:
        raise RuntimeError(
            f"Incomplete shard set: {len(manifests)} / {expected_shards}"
        )
    comparable = _comparable_lineage(lineage)
    for manifest in manifest_payloads:
        if not bool(manifest.get("native_proposals", False)):
            raise RuntimeError("A shard manifest is not marked native")
        if _comparable_lineage(manifest["lineage"]) != comparable:
            raise RuntimeError("Native proposal shard lineage mismatch")

    merged: Dict[str, Dict[str, np.ndarray]] = {}
    log_names: Dict[str, str] = {}
    for chunk_path in sorted(args.input_root.glob("shard_*/chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu")
        required = {
            "tokens",
            "log_names",
            "proposals",
            "scores",
            "factor_logits",
        }
        if not required.issubset(chunk):
            raise RuntimeError(f"Malformed native proposal chunk: {chunk_path}")
        tokens = [str(value) for value in chunk["tokens"]]
        logs = [str(value) for value in chunk["log_names"]]
        proposals = chunk["proposals"].float().numpy()
        scores = chunk["scores"].float().numpy()
        factors = chunk["factor_logits"].float().numpy()
        expected_proposal_shape = (
            len(tokens),
            EXPECTED_CANDIDATES,
            EXPECTED_POSES,
            EXPECTED_STATE_SIZE,
        )
        if proposals.shape != expected_proposal_shape:
            raise RuntimeError(
                f"Proposal shape mismatch in {chunk_path}: {proposals.shape}"
            )
        if scores.shape != (len(tokens), EXPECTED_CANDIDATES):
            raise RuntimeError(f"Score shape mismatch in {chunk_path}: {scores.shape}")
        if factors.shape != (len(tokens), EXPECTED_CANDIDATES, 6):
            raise RuntimeError(
                f"Factor-logit shape mismatch in {chunk_path}: {factors.shape}"
            )
        if not (
            np.isfinite(proposals).all()
            and np.isfinite(scores).all()
            and np.isfinite(factors).all()
        ):
            raise RuntimeError(f"Non-finite native output in {chunk_path}")
        for row, (token, log_name) in enumerate(zip(tokens, logs)):
            if token in merged:
                raise RuntimeError(f"Duplicate native proposal token: {token}")
            merged[token] = {
                "proposals": proposals[row],
                "predicted_scores": scores[row],
                "factor_logits": factors[row],
            }
            log_names[token] = log_name

    if len(merged) != args.expected_scenes:
        raise RuntimeError(
            f"Unexpected native proposal coverage: {len(merged)} / {args.expected_scenes}"
        )
    if args.candidate_matrix is not None:
        with np.load(args.candidate_matrix, allow_pickle=False) as archive:
            expected_tokens = set(archive["tokens"].astype(str).tolist())
        if set(merged) != expected_tokens:
            raise RuntimeError(
                "Native proposal token set differs from locked candidate matrix: "
                f"missing={len(expected_tokens - set(merged))}, "
                f"extra={len(set(merged) - expected_tokens)}"
            )

    _atomic_pickle(merged, args.output)
    output_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "agent_target": "navsim.agents.drivoR.drivor_agent.DrivoRAgent",
        "model_target": "navsim.agents.drivoR.drivor_model.DrivoRModel",
        "checkpoint": lineage["drivor_checkpoint"],
        "checkpoint_sha256": lineage["drivor_checkpoint_sha256"],
        "config": lineage["drivor_config"],
        "config_sha256": lineage["drivor_config_sha256"],
        "precision": "fp32",
        "scene_count": len(merged),
        "log_count": len(set(log_names.values())),
        "candidate_count": EXPECTED_CANDIDATES,
        "pose_count": EXPECTED_POSES,
        "proposal_predictions_path": str(args.output.resolve()),
        "proposal_predictions_sha256": _sha256(args.output),
        "shard_manifest_sha256": {
            str(path.relative_to(args.input_root)): _sha256(path)
            for path in manifests
        },
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_input": False,
        "external_proposal_input": False,
    }
    manifest_path = args.output.with_name("proposal_cache_manifest.json")
    _atomic_json(output_manifest, manifest_path)
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
