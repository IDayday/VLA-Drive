#!/usr/bin/env python3
"""Validate the custom DrivOR scorer against its online external path.

The external exporter first proves full DrivOR forward == proposal-adapter
forward on real current observations.  This script closes the second half of
the chain by proving adapter scores == the custom ranker's scores when the
same FP32 scene registers, ego status and proposals are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

_DETERMINISTIC_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _DETERMINISTIC_CUBLAS:
    raise RuntimeError(
        "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python"
    )

import torch

from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    pdms_factor_log_utility,
)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _chunk_paths(root: Path) -> List[Path]:
    paths = sorted(root.glob("**/chunk_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no chunk_*.pt files under {root}")
    return paths


def _candidate_external_rows(
    root: Path, maximum_candidates: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    rows: Dict[str, Dict[str, torch.Tensor]] = {}
    for path in _chunk_paths(root):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = ("tokens", "scores", "factor_logits", "selected_indices")
        if not all(key in payload for key in required):
            continue
        for index, raw_token in enumerate(payload["tokens"]):
            token = str(raw_token)
            if token in rows:
                raise RuntimeError(f"duplicate external score token: {token}")
            rows[token] = {
                "scores": payload["scores"][index].float(),
                "factor_logits": payload["factor_logits"][index].float(),
                "selected_indices": torch.as_tensor(
                    payload["selected_indices"][index]
                ).long(),
            }
            if len(rows) >= maximum_candidates:
                return rows
    if not rows:
        raise RuntimeError("external score cache contains no compatible rows")
    return rows


def _load_chunk_rows(
    root: Path,
    wanted: set[str],
    tensor_keys: Iterable[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    rows: Dict[str, Dict[str, torch.Tensor]] = {}
    for path in _chunk_paths(root):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "tokens" not in payload or not all(key in payload for key in tensor_keys):
            continue
        for index, raw_token in enumerate(payload["tokens"]):
            token = str(raw_token)
            if token not in wanted:
                continue
            if token in rows:
                raise RuntimeError(f"duplicate token under {root}: {token}")
            rows[token] = {
                key: torch.as_tensor(payload[key][index]) for key in tensor_keys
            }
        if wanted.issubset(rows):
            break
    return rows


def _load_pickle_proposals(
    path: Path, wanted: set[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return {
        token: {"proposals": torch.as_tensor(payload[token]["proposals"])}
        for token in wanted
        if token in payload
    }


def _manifests(root: Path) -> List[Mapping[str, object]]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("**/manifest.json"))]


def _online_self_parity(root: Path) -> List[Mapping[str, object]]:
    results: List[Mapping[str, object]] = []
    for path in sorted(root.glob("**/self_parity.json")):
        payload = json.loads(path.read_text())
        results.extend(payload.get("scenes", []))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drivor-checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposal-root", type=Path)
    source.add_argument("--proposal-pickle", type=Path)
    parser.add_argument("--register-root", type=Path, required=True)
    parser.add_argument("--external-score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--search-scenes", type=int, default=256)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--factor-cache-tolerance", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.num_scenes < 4:
        raise ValueError("at least four scenes are required")
    if args.search_scenes < args.num_scenes:
        raise ValueError("search-scenes must be >= num-scenes")
    for path in (
        args.drivor_checkpoint,
        args.register_root,
        args.external_score_root,
        args.proposal_root or args.proposal_pickle,
    ):
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    register_manifests = _manifests(args.register_root)
    if not register_manifests:
        raise RuntimeError("register cache has no completed manifests")
    precisions = {str(value.get("precision")) for value in register_manifests}
    if precisions != {"fp32_compute_float32_cache"}:
        raise RuntimeError(
            f"benchmark parity requires an FP32 register cache, got {precisions}"
        )
    online_parity = _online_self_parity(args.external_score_root)
    if len(online_parity) < args.num_scenes or not all(
        bool(value.get("passed")) for value in online_parity
    ):
        raise RuntimeError("external online scorer self-parity is incomplete")

    external = _candidate_external_rows(
        args.external_score_root, args.search_scenes
    )
    wanted = set(external)
    registers = _load_chunk_rows(
        args.register_root,
        wanted,
        ("visual_tokens", "visual_valid_mask", "status_feature"),
    )
    if args.proposal_root is not None:
        proposals = _load_chunk_rows(
            args.proposal_root, wanted, ("proposals",)
        )
    else:
        assert args.proposal_pickle is not None
        proposals = _load_pickle_proposals(args.proposal_pickle, wanted)
    tokens = sorted(set(external).intersection(registers, proposals))[: args.num_scenes]
    if len(tokens) != args.num_scenes:
        raise RuntimeError(
            f"only {len(tokens)} common rows found; need {args.num_scenes}"
        )

    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = DrivORInitializedProposalRanker(DrivORRankerConfig())
    load_audit = model.load_drivor_checkpoint(args.drivor_checkpoint)
    model.to(device).eval()
    scene = torch.stack(
        [registers[token]["visual_tokens"].float() for token in tokens]
    ).to(device)
    mask = torch.stack(
        [registers[token]["visual_valid_mask"].bool() for token in tokens]
    ).to(device)
    status = torch.stack(
        [registers[token]["status_feature"].float() for token in tokens]
    ).to(device)
    proposal_tensor = torch.stack(
        [proposals[token]["proposals"].float() for token in tokens]
    ).to(device)
    with torch.inference_mode():
        output = model(scene, status, proposal_tensor, scene_valid_mask=mask)
        scores = pdms_factor_log_utility(output["factor_logits"])
    reference_scores = torch.stack(
        [external[token]["scores"] for token in tokens]
    ).to(device)
    reference_factors = torch.stack(
        [external[token]["factor_logits"] for token in tokens]
    ).to(device)
    reference_indices = torch.stack(
        [external[token]["selected_indices"] for token in tokens]
    ).to(device)
    score_errors = (scores - reference_scores).abs()
    factor_errors = (output["factor_logits"] - reference_factors).abs()
    selected = scores.argmax(dim=1)
    selected_equal = selected.eq(reference_indices)
    score_max = float(score_errors.max())
    factor_max = float(factor_errors.max())
    passed = (
        score_max <= args.score_tolerance
        and factor_max <= args.factor_cache_tolerance
        and bool(selected_equal.all())
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "scene_count": len(tokens),
        "tokens": tokens,
        "device": str(device),
        "checkpoint": str(args.drivor_checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.drivor_checkpoint),
        "register_root": str(args.register_root.resolve()),
        "register_precision": sorted(precisions),
        "external_score_root": str(args.external_score_root.resolve()),
        "online_self_parity_scene_count": len(online_parity),
        "online_self_parity_max_score_error": max(
            float(value["score_max_abs_error"]) for value in online_parity
        ),
        "custom_cache_score_max_abs_error": score_max,
        "custom_cache_factor_logit_max_abs_error_vs_fp16_archive": factor_max,
        "selected_index_equal": bool(selected_equal.all()),
        "score_tolerance": args.score_tolerance,
        "factor_cache_tolerance": args.factor_cache_tolerance,
        "load_audit": load_audit,
        "future_or_evaluator_input": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("DrivOR online/cache parity failed")


if __name__ == "__main__":
    main()
