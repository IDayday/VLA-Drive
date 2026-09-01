"""Build compact candidate-relative temporal consequence labels on Navtrain.

The public EpisodeDrive scorer predicts only aggregate PDM factors.  Its source
tree also contains dormant ``agent_pred`` and ``area_pred`` branches, while the
released loss sets all three corresponding losses to zero.  This script keeps
the immutable inference cache and existing factor-label cache untouched and
materializes the missing training-only supervision in a separate tree.

Only the Base top-K proposals are rescored.  For every proposal we retain eight
0.5 s horizons of:

* collision/TTC event-by-horizon targets;
* the relevant logged-future actor box, expressed in the candidate ego frame;
* non-drivable/oncoming area occupancy.

No tensor written here is a deployment input.  At inference a learned head must
predict these consequences from the current observation and candidate only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (
    config as pdm_scorer_config,
    get_sub_score_from_metric_cache,
    proposal_sampling,
)
from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import PDMScorer
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)


HORIZON_SECONDS = np.arange(0.5, 4.0 + 1e-6, 0.5, dtype=np.float32)
# PDM uses 0.1 s states after the initial state.  Array index 4 is 0.5 s.
HORIZON_ARRAY_INDICES = np.rint(HORIZON_SECONDS / 0.1).astype(np.int64) - 1
HORIZON_STATE_INDICES = HORIZON_ARRAY_INDICES + 1
KEY_AGENT_KINDS = ("collision", "ttc")
KEY_AGENT_STATE_FIELDS = (
    "relative_x",
    "relative_y",
    "length",
    "width",
    "sin_2_relative_axis_heading",
    "cos_2_relative_axis_heading",
)
EGO_AREA_FIELDS = ("non_drivable_area", "oncoming_traffic")

_METRIC_PATHS: Dict[str, str] = {}
_SIMULATOR: Optional[PDMSimulator] = None
_SCORER: Optional[PDMScorer] = None


def _atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _base_topk_indices(base_scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return deterministic Base-score order, including the deployed argmax."""

    value = np.asarray(base_scores)
    if value.ndim != 2:
        raise ValueError("base_scores must have shape [B, K]")
    if not 0 < top_k <= value.shape[1]:
        raise ValueError("top_k must be in [1, candidate_count]")
    return np.argsort(-value, axis=1, kind="stable")[:, :top_k]


def _event_by_horizon(
    event_indices: np.ndarray,
    horizon_state_indices: Sequence[int] = HORIZON_STATE_INDICES,
) -> np.ndarray:
    """Convert PDM event indices into cumulative event-by-horizon targets."""

    event = np.asarray(event_indices, dtype=np.float64)
    if event.ndim != 2 or event.shape[-1] != len(KEY_AGENT_KINDS):
        raise ValueError("event_indices must have shape [K, 2]")
    horizon = np.asarray(horizon_state_indices, dtype=np.float64)
    return np.isfinite(event[..., None]) & (event[..., None] <= horizon)


def _candidate_relative_box_state(
    corners: np.ndarray,
    valid: np.ndarray,
    proposals: np.ndarray,
) -> np.ndarray:
    """Encode logged-future actor rectangles in each candidate ego frame.

    ``corners`` is ``[K, H, 2, 4, 2]`` in the current-ego frame and
    ``proposals`` is ``[K, H, 3]`` in the same frame.  Rectangle orientation is
    represented modulo pi through ``sin(2 theta), cos(2 theta)`` because raw
    polygon corners do not identify the actor's forward direction reliably.
    """

    corners = np.asarray(corners, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    proposals = np.asarray(proposals, dtype=np.float64)
    if corners.ndim != 5 or corners.shape[-2:] != (4, 2):
        raise ValueError("corners must have shape [K, H, 2, 4, 2]")
    if valid.shape != corners.shape[:-2]:
        raise ValueError("valid mask shape does not match corners")
    if proposals.shape != (*corners.shape[:2], 3):
        raise ValueError("proposals must have shape [K, H, 3]")

    delta = corners - proposals[:, :, None, None, :2]
    heading = proposals[:, :, None, None, 2]
    cosine = np.cos(heading)
    sine = np.sin(heading)
    local_x = delta[..., 0] * cosine + delta[..., 1] * sine
    local_y = -delta[..., 0] * sine + delta[..., 1] * cosine
    local_corners = np.stack((local_x, local_y), axis=-1)

    center = local_corners.mean(axis=-2)
    edge_a = local_corners[..., 1, :] - local_corners[..., 0, :]
    edge_b = local_corners[..., 2, :] - local_corners[..., 1, :]
    norm_a = np.linalg.norm(edge_a, axis=-1)
    norm_b = np.linalg.norm(edge_b, axis=-1)
    choose_a = norm_a >= norm_b
    long_edge = np.where(choose_a[..., None], edge_a, edge_b)
    length = np.maximum(norm_a, norm_b)
    width = np.minimum(norm_a, norm_b)
    axis_heading = np.arctan2(long_edge[..., 1], long_edge[..., 0])

    state = np.stack(
        (
            center[..., 0],
            center[..., 1],
            length,
            width,
            np.sin(2.0 * axis_heading),
            np.cos(2.0 * axis_heading),
        ),
        axis=-1,
    )
    state[~valid] = 0.0
    if not np.isfinite(state).all():
        raise RuntimeError("Non-finite candidate-relative box state")
    return state.astype(np.float32)


def _worker_init(metric_cache_path: str) -> None:
    global _METRIC_PATHS, _SIMULATOR, _SCORER
    torch.set_num_threads(1)
    loader = MetricCacheLoader(Path(metric_cache_path))
    _METRIC_PATHS = {
        str(token): str(path) for token, path in loader.metric_cache_paths.items()
    }
    _SIMULATOR = PDMSimulator(proposal_sampling)
    _SCORER = PDMScorer(proposal_sampling, pdm_scorer_config)


def _score_one(
    token: str,
    proposals: np.ndarray,
    expected_factors: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    if _SIMULATOR is None or _SCORER is None:
        raise RuntimeError("PDM consequence worker was not initialized")
    path = _METRIC_PATHS.get(token)
    if path is None:
        raise KeyError(f"Metric cache missing token {token}")
    with lzma.open(path, "rb") as file:
        metric_cache = pickle.load(file)

    scores, corners, key_valid, ego_areas = get_sub_score_from_metric_cache(
        metric_cache,
        proposals,
        False,
        simulator_instance=_SIMULATOR,
        scorer_instance=_SCORER,
    )
    scores = np.asarray(scores, dtype=np.float32)
    expected = np.asarray(expected_factors, dtype=np.float32)
    if scores.shape != expected.shape:
        raise RuntimeError(f"Factor parity shape mismatch: {scores.shape} vs {expected.shape}")
    factor_max_abs_error = float(np.max(np.abs(scores - expected)))
    if factor_max_abs_error > 1e-6:
        raise RuntimeError(
            f"Factor parity failed for {token}: max_abs={factor_max_abs_error}"
        )

    sampled_corners = np.asarray(corners)[..., HORIZON_ARRAY_INDICES, :, :, :]
    sampled_valid = np.asarray(key_valid)[..., HORIZON_ARRAY_INDICES, :]
    sampled_areas = np.asarray(ego_areas)[..., HORIZON_ARRAY_INDICES, :]
    actor_state = _candidate_relative_box_state(
        sampled_corners,
        sampled_valid,
        proposals,
    )
    event_indices = np.stack(
        (
            np.asarray(_SCORER._collision_time_idcs),
            np.asarray(_SCORER._ttc_time_idcs),
        ),
        axis=-1,
    )
    event_by_horizon = _event_by_horizon(event_indices)
    finite_event_indices = np.where(np.isfinite(event_indices), event_indices, -1)
    if not np.all(
        (finite_event_indices == -1)
        | ((finite_event_indices >= 0) & (finite_event_indices <= 40))
    ):
        raise RuntimeError(f"Unexpected PDM event index for {token}")

    return {
        "key_actor_state": actor_state,
        "key_actor_valid": sampled_valid.astype(bool),
        "ego_area_violation": sampled_areas.astype(bool),
        "risk_event_by_horizon": event_by_horizon.transpose(0, 2, 1).astype(bool),
        "risk_event_index": finite_event_indices.astype(np.int16),
        "factor_max_abs_error": factor_max_abs_error,
    }


def _score_chunk(task: Tuple[str, str, str, str, int]) -> Dict[str, object]:
    source_path = Path(task[0])
    factor_path = Path(task[1])
    output_path = Path(task[2])
    relative_path = task[3]
    top_k = int(task[4])
    if output_path.is_file():
        return {"relative_path": relative_path, "status": "already_complete"}
    try:
        source = torch.load(source_path, map_location="cpu")
        factors = torch.load(factor_path, map_location="cpu")
        tokens = [str(token) for token in source["tokens"]]
        if tokens != [str(token) for token in factors["tokens"]]:
            raise RuntimeError(f"Source/factor token mismatch: {relative_path}")
        proposals = source["proposals"].float().numpy()
        base_scores = source["base_scores"].float().numpy()
        target_factors = factors["target_factors"].float().numpy()
        topk_indices = _base_topk_indices(base_scores, top_k)
        row = np.arange(len(tokens))[:, None]
        selected_proposals = proposals[row, topk_indices]
        selected_factors = target_factors[row, topk_indices]

        results = [
            _score_one(token, selected_proposals[index], selected_factors[index])
            for index, token in enumerate(tokens)
        ]
        max_factor_error = max(float(value["factor_max_abs_error"]) for value in results)
        payload = {
            "schema_version": 1,
            "source_relative_path": relative_path,
            "tokens": tokens,
            "log_names": [str(value) for value in source["log_names"]],
            "candidate_indices": torch.from_numpy(topk_indices.astype(np.int16)),
            "horizon_seconds": torch.from_numpy(HORIZON_SECONDS.copy()),
            "key_agent_kinds": KEY_AGENT_KINDS,
            "key_agent_state_fields": KEY_AGENT_STATE_FIELDS,
            "ego_area_fields": EGO_AREA_FIELDS,
            "key_actor_state": torch.from_numpy(
                np.stack([value["key_actor_state"] for value in results])
            ).to(torch.float16),
            "key_actor_valid": torch.from_numpy(
                np.stack([value["key_actor_valid"] for value in results])
            ),
            "ego_area_violation": torch.from_numpy(
                np.stack([value["ego_area_violation"] for value in results])
            ),
            "risk_event_by_horizon": torch.from_numpy(
                np.stack([value["risk_event_by_horizon"] for value in results])
            ),
            "risk_event_index": torch.from_numpy(
                np.stack([value["risk_event_index"] for value in results])
            ),
            "factor_parity_max_abs_error": max_factor_error,
            "training_labels_only": True,
        }
        _atomic_torch_save(payload, output_path)
        return {
            "relative_path": relative_path,
            "status": "scored",
            "scene_count": len(tokens),
            "factor_parity_max_abs_error": max_factor_error,
        }
    except Exception as error:
        return {
            "relative_path": relative_path,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _belongs_to_worker(relative_path: str, count: int, index: int) -> bool:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count == index


def _discover_tasks(
    source_root: Path,
    factor_root: Path,
    output_root: Path,
    worker_shard_count: int,
    worker_shard_index: int,
    top_k: int,
) -> List[Tuple[str, str, str, str, int]]:
    tasks: List[Tuple[str, str, str, str, int]] = []
    for source_path in sorted(source_root.glob("*_shard_*-of-*/chunk_*.pt")):
        relative = str(source_path.relative_to(source_root))
        if not _belongs_to_worker(relative, worker_shard_count, worker_shard_index):
            continue
        factor_path = factor_root / relative
        if not factor_path.is_file():
            continue
        output_path = output_root / relative
        if not output_path.is_file():
            tasks.append(
                (str(source_path), str(factor_path), str(output_path), relative, top_k)
            )
    return tasks


def _source_complete(source_root: Path) -> bool:
    shard_dirs = sorted(source_root.glob("*_shard_*-of-*"))
    if not shard_dirs or not all((path / "manifest.json").is_file() for path in shard_dirs):
        return False
    manifests = [json.loads((path / "manifest.json").read_text()) for path in shard_dirs]
    shard_count = int(manifests[0]["shard_count"])
    return {int(value["shard_index"]) for value in manifests} == set(range(shard_count))


def _worker_complete(
    source_root: Path,
    factor_root: Path,
    output_root: Path,
    worker_shard_count: int,
    worker_shard_index: int,
) -> bool:
    for source_path in source_root.glob("*_shard_*-of-*/chunk_*.pt"):
        relative = str(source_path.relative_to(source_root))
        if not _belongs_to_worker(relative, worker_shard_count, worker_shard_index):
            continue
        if not (factor_root / relative).is_file() or not (output_root / relative).is_file():
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--worker-shard-count", type=int, default=1)
    parser.add_argument("--worker-shard-index", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.source_root, args.factor_root, args.metric_cache):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_shard_index < args.worker_shard_count:
        raise ValueError("worker shard index is outside worker shard count")
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed: List[Dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(args.metric_cache),),
    ) as pool:
        while True:
            tasks = _discover_tasks(
                args.source_root,
                args.factor_root,
                args.output_root,
                args.worker_shard_count,
                args.worker_shard_index,
                args.top_k,
            )
            if tasks:
                futures = [pool.submit(_score_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    processed += 1
                    if result["status"] == "failed":
                        failed.append(result)
                    print(
                        "PDM_CONSEQUENCE "
                        f"worker_shard={args.worker_shard_index}/{args.worker_shard_count} "
                        f"chunks={processed} result={json.dumps(result, sort_keys=True)}",
                        flush=True,
                    )
                if failed:
                    break

            source_done = _source_complete(args.source_root)
            worker_done = _worker_complete(
                args.source_root,
                args.factor_root,
                args.output_root,
                args.worker_shard_count,
                args.worker_shard_index,
            )
            if source_done and worker_done:
                break
            if not args.watch:
                break
            time.sleep(args.poll_seconds)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "factor_root": str(args.factor_root.resolve()),
        "metric_cache": str(args.metric_cache.resolve()),
        "top_k": args.top_k,
        "horizon_seconds": HORIZON_SECONDS.tolist(),
        "key_agent_kinds": KEY_AGENT_KINDS,
        "key_agent_state_fields": KEY_AGENT_STATE_FIELDS,
        "ego_area_fields": EGO_AREA_FIELDS,
        "worker_shard_count": args.worker_shard_count,
        "worker_shard_index": args.worker_shard_index,
        "processed_chunks_this_run": processed,
        "failed_chunk_count": len(failed),
        "source_complete": _source_complete(args.source_root),
        "worker_complete": _worker_complete(
            args.source_root,
            args.factor_root,
            args.output_root,
            args.worker_shard_count,
            args.worker_shard_index,
        ),
        "offline_training_labels_only": True,
        "inference_inputs_added": False,
    }
    _atomic_json_dump(
        manifest,
        args.output_root
        / f"worker_manifest_{args.worker_shard_index:03d}-of-{args.worker_shard_count:03d}.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} PDM consequence chunks failed")


if __name__ == "__main__":
    main()
