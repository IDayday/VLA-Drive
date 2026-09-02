#!/usr/bin/env python3
"""Resumable per-log true NAVSIM PDM scoring for an arbitrary candidate bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (
    get_sub_score_from_metric_cache,
)
from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import (
    PDMScorer as FixedProgressPDMScorer,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer as OfficialPDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)

from .schema import FACTOR_NAMES
from .utils import atomic_json, load_proposal_pickle, sha256_file, token_index


_BANK: Dict[str, Dict[str, object]] = {}
_METRIC_PATHS: Dict[str, Path] = {}
_LOG_TOKENS: Dict[str, List[str]] = {}
_OUTPUT_DIR: Path | None = None
_BANK_SHA = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _derive_fixed_progress(metric_cache, simulator, scorer) -> None:
    if hasattr(metric_cache, "pdm_progress"):
        return
    states = get_trajectory_as_array(
        metric_cache.trajectory,
        simulator.proposal_sampling,
        metric_cache.ego_state.time_point,
    )
    simulated = simulator.simulate_proposals(states[None], metric_cache.ego_state)
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    metric_cache.pdm_progress = float((scorer._progress_raw * scorer._multi_metrics.prod(axis=0))[0])


def _log_dir(log_name: str) -> Path:
    assert _OUTPUT_DIR is not None
    digest = hashlib.sha256(log_name.encode()).hexdigest()[:16]
    return _OUTPUT_DIR / "scored_logs" / digest


def _complete(log_name: str) -> bool:
    directory = _log_dir(log_name)
    path = directory / "manifest.json"
    if not path.exists() or not (directory / "scores.npz").exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return value.get("log_name") == log_name and value.get("candidate_bank_sha256") == _BANK_SHA


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _score_log(log_name: str) -> Dict[str, object]:
    if _complete(log_name):
        return {"log_name": log_name, "status": "already_complete"}
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    official = OfficialPDMScorer(sampling)
    fixed = FixedProgressPDMScorer(sampling)
    valid_tokens: List[str] = []
    factors: List[np.ndarray] = []
    failures: List[Dict[str, str]] = []
    candidate_count = None
    for token in _LOG_TOKENS[log_name]:
        try:
            proposals = np.asarray(_BANK[token]["proposals"], dtype=np.float32)
            if proposals.ndim != 3 or proposals.shape[1:] != (8, 3):
                raise RuntimeError(f"invalid proposal shape {proposals.shape}")
            if not np.isfinite(proposals).all():
                raise RuntimeError("non-finite proposal")
            if candidate_count is None:
                candidate_count = len(proposals)
            elif candidate_count != len(proposals):
                raise RuntimeError("candidate count varies within bank")
            with lzma.open(_METRIC_PATHS[token], "rb") as stream:
                metric_cache = pickle.load(stream)
            _derive_fixed_progress(metric_cache, simulator, official)
            values, *_ = get_sub_score_from_metric_cache(
                metric_cache,
                proposals,
                True,
                simulator_instance=simulator,
                scorer_instance=fixed,
            )
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (len(proposals), 7) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid evaluator output {values.shape}")
            if values.min() < -1e-7 or values.max() > 1.0 + 1e-7:
                raise RuntimeError(f"true score outside [0,1]: {values.min()} {values.max()}")
            valid_tokens.append(token)
            factors.append(values.astype(np.float32))
        except Exception as error:
            failures.append({"token": token, "log_name": log_name, "error": repr(error)})
    directory = _log_dir(log_name)
    directory.mkdir(parents=True, exist_ok=True)
    if valid_tokens:
        stacked = np.stack(factors)
    else:
        stacked = np.empty((0, candidate_count or 0, 7), dtype=np.float32)
    _atomic_npz(
        directory / "scores.npz",
        tokens=np.asarray(valid_tokens),
        candidate_factors=stacked,
        candidate_factor_names=np.asarray(FACTOR_NAMES),
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "log_name": log_name,
        "candidate_bank_sha256": _BANK_SHA,
        "expected_scene_count": len(_LOG_TOKENS[log_name]),
        "valid_scene_count": len(valid_tokens),
        "failed_scene_count": len(failures),
        "candidate_count": int(candidate_count or 0),
        "failed_tokens": failures,
    }
    atomic_json(directory / "manifest.json", manifest)
    return {"log_name": log_name, "status": "scored", **manifest}


def main() -> None:
    global _BANK, _METRIC_PATHS, _LOG_TOKENS, _OUTPUT_DIR, _BANK_SHA
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and args.resume:
        print(summary_path.read_text(), end="")
        return
    _BANK = load_proposal_pickle(args.candidate_bank)
    _BANK_SHA = sha256_file(args.candidate_bank)
    with np.load(args.base_matrix, allow_pickle=False) as archive:
        all_tokens = archive["tokens"].astype(str)
        all_logs = archive["log_names"].astype(str)
    index = token_index(all_tokens)
    unknown = set(_BANK).difference(index)
    if unknown:
        raise RuntimeError(f"Candidate bank has {len(unknown)} tokens outside base split")
    ordered = [token for token in all_tokens if token in _BANK]
    if args.token_file is not None:
        requested = {line.strip() for line in args.token_file.read_text().splitlines() if line.strip()}
        unknown = requested.difference(ordered)
        if unknown:
            raise RuntimeError(f"Token file contains {len(unknown)} tokens outside candidate bank")
        ordered = [token for token in ordered if token in requested]
    if args.max_scenes:
        ordered = ordered[: args.max_scenes]
        _BANK = {token: _BANK[token] for token in ordered}
    counts = {len(np.asarray(_BANK[token]["proposals"])) for token in ordered}
    if len(counts) != 1:
        raise RuntimeError(f"Candidate counts vary across scenes: {sorted(counts)}")
    metric_loader = MetricCacheLoader(args.metric_cache)
    missing_cache = set(ordered).difference(metric_loader.metric_cache_paths)
    if missing_cache:
        raise RuntimeError(f"Metric cache misses {len(missing_cache)} bank tokens")
    _METRIC_PATHS = {token: metric_loader.metric_cache_paths[token] for token in ordered}
    _LOG_TOKENS = {}
    for token in ordered:
        _LOG_TOKENS.setdefault(str(all_logs[index[token]]), []).append(token)
    _OUTPUT_DIR = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_bank": str(args.candidate_bank.resolve()),
        "candidate_bank_sha256": _BANK_SHA,
        "base_matrix": str(args.base_matrix.resolve()),
        "base_matrix_sha256": sha256_file(args.base_matrix),
        "metric_cache": str(args.metric_cache.resolve()),
        "candidate_count": next(iter(counts)),
        "expected_scene_count": len(ordered),
        "token_file": str(args.token_file.resolve()) if args.token_file else None,
        "token_file_sha256": sha256_file(args.token_file) if args.token_file else None,
        "log_count": len(_LOG_TOKENS),
        "evaluator": "true NAVSIM PDM fixed-reference-progress path",
        "official_single_candidate_parity_gate": "tools.lora_value_audit.exact_candidate_scorer",
    }
    lineage_path = args.output_dir / "lineage.json"
    if lineage_path.exists():
        old = json.loads(lineage_path.read_text()); old.pop("created_utc", None)
        check = dict(lineage); check.pop("created_utc", None)
        if old != check:
            raise RuntimeError("Existing output lineage mismatch")
    else:
        atomic_json(lineage_path, lineage)

    pending_logs = [name for name in sorted(_LOG_TOKENS) if not _complete(name)]
    print(json.dumps({"pending_logs": len(pending_logs), "total_logs": len(_LOG_TOKENS), "scenes": len(ordered)}), flush=True)
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=context) as pool:
        futures = {pool.submit(_score_log, name): name for name in pending_logs}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed_index % 10 == 0 or completed_index == len(futures):
                print(json.dumps({"completed_logs": completed_index, "pending_at_start": len(futures)}), flush=True)

    arrays = {}
    failures = []
    for log_name in sorted(_LOG_TOKENS):
        directory = _log_dir(log_name)
        manifest = json.loads((directory / "manifest.json").read_text())
        failures.extend(manifest["failed_tokens"])
        with np.load(directory / "scores.npz", allow_pickle=False) as archive:
            tokens = archive["tokens"].astype(str)
            values = archive["candidate_factors"].astype(np.float32)
        for row, token in enumerate(tokens):
            if token in arrays:
                raise RuntimeError(f"Duplicate scored token {token}")
            arrays[token] = values[row]
    expected = set(ordered)
    if set(arrays) | {row["token"] for row in failures} != expected:
        raise RuntimeError("Scored/failed token accounting mismatch")
    valid_tokens = [token for token in ordered if token in arrays]
    stacked = np.stack([arrays[token] for token in valid_tokens])
    scores = stacked[..., -1]
    oracle_indices = scores.argmax(axis=1).astype(np.int16)
    _atomic_npz(
        args.output_dir / "candidate_scores.npz",
        tokens=np.asarray(valid_tokens),
        log_names=np.asarray([str(all_logs[index[token]]) for token in valid_tokens]),
        candidate_scores=scores.astype(np.float32),
        oracle_indices=oracle_indices,
        candidate_factors=stacked,
        candidate_factor_names=np.asarray(FACTOR_NAMES),
    )
    with (args.output_dir / "failed_tokens.jsonl").open("w") as stream:
        for row in failures:
            stream.write(json.dumps(row) + "\n")
    failure_rate = len(failures) / len(ordered)
    summary = {
        **lineage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid_scene_count": len(valid_tokens),
        "failed_scene_count": len(failures),
        "failure_rate": failure_rate,
        "mean_candidate_pdms": float(scores.mean()),
        "mean_oracle_pdms": float(scores.max(axis=1).mean()),
        "minimum_pdms": float(scores.min()),
        "maximum_pdms": float(scores.max()),
        "full_result_usable": bool(failure_rate <= 0.001 and len(valid_tokens) == len(ordered)),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failure_rate > 0.001 or len(valid_tokens) != len(ordered):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
