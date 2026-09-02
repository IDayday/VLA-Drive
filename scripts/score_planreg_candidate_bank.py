#!/usr/bin/env python3
"""Score one immutable formal 64-candidate bank with resumable NAVSIM shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (  # noqa: E402
    get_sub_score,
)
from navsim.common.dataloader import MetricCacheLoader  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _score_one(task: Tuple[str, str, np.ndarray, str]) -> Tuple[str, np.ndarray]:
    token, metric_cache_path, proposals, shard_path_text = task
    shard_path = Path(shard_path_text)
    proposal_sha = _array_sha256(proposals)
    if shard_path.is_file():
        with np.load(shard_path, allow_pickle=False) as cached:
            cached_token = str(cached["token"].item())
            cached_sha = str(cached["proposal_sha256"].item())
            scores = np.asarray(cached["candidate_scores"], dtype=np.float32)
        if cached_token != token or cached_sha != proposal_sha or scores.shape != (64, 7):
            raise RuntimeError(f"Stale candidate score shard: {shard_path}")
        return token, scores

    scores = np.asarray(
        get_sub_score(metric_cache_path, proposals, True)[0], dtype=np.float32
    )
    if scores.shape != (64, 7) or not np.isfinite(scores).all():
        raise RuntimeError(
            f"Invalid official candidate scores for {token}: {scores.shape}"
        )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            token=np.asarray(token),
            proposal_sha256=np.asarray(proposal_sha),
            candidate_scores=scores,
        )
    temporary.replace(shard_path)
    return token, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", required=True, type=Path)
    parser.add_argument("--metric-cache", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument(
        "--start-method", choices=("spawn", "forkserver"), default="forkserver"
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    candidate_sha = _sha256(args.candidate_bank)
    with np.load(args.candidate_bank, allow_pickle=False) as bank:
        tokens = [str(value) for value in bank["tokens"].tolist()]
        proposals = np.asarray(bank["proposals"], dtype=np.float32)
    if proposals.shape != (len(tokens), 64, 8, 3):
        raise RuntimeError(
            f"Candidate bank must be [scenes,64,8,3], got {proposals.shape}"
        )
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Candidate bank contains duplicate scene tokens")

    metric_loader = MetricCacheLoader(args.metric_cache)
    missing = sorted(set(tokens) - set(metric_loader.metric_cache_paths))
    if missing:
        raise RuntimeError(f"Metric cache is missing {len(missing)} tokens: {missing[:20]}")
    shard_root = args.work_dir / "score_shards"
    tasks = [
        (
            token,
            str(metric_loader.metric_cache_paths[token]),
            proposals[index],
            str(shard_root / f"{token}.npz"),
        )
        for index, token in enumerate(tokens)
    ]
    context = mp.get_context(args.start_method)
    score_by_token: Dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        for token, scores in tqdm(
            pool.map(_score_one, tasks, chunksize=1),
            total=len(tasks),
            desc="Official NAVSIM candidate scoring",
        ):
            score_by_token[token] = scores
    if len(score_by_token) != len(tokens):
        raise RuntimeError("Candidate scoring ended with missing tokens")
    ordered_scores = np.stack([score_by_token[token] for token in tokens])

    with np.load(args.candidate_bank, allow_pickle=False) as bank:
        arrays = {name: bank[name] for name in bank.files}
    arrays["candidate_scores"] = ordered_scores
    arrays["pdm_scores"] = ordered_scores
    arrays["official_component_names"] = np.asarray(
        (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
            "pdm_score",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1,
        "source_candidate_bank": str(args.candidate_bank.resolve()),
        "source_candidate_bank_sha256": candidate_sha,
        "scored_candidate_bank": str(args.output.resolve()),
        "scored_candidate_bank_sha256": _sha256(args.output),
        "metric_cache": str(args.metric_cache.resolve()),
        "scene_count": len(tokens),
        "candidate_count": 64,
        "all_shards_complete": True,
        "scorer_reads_predicted_future_registers": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
