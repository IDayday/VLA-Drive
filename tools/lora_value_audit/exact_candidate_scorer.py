#!/usr/bin/env python3
"""Strict official-single-trajectory parity for vectorized candidate PDM.

The efficient path scores several candidates together only after deriving the
fixed PDM-reference progress scalar.  Every audited candidate is also scored in
an isolated official ``pdm_score`` call, where the proposal set is exactly
``[PDM reference, candidate]``.  No score from DrivoR's neural scorer enters
this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (
    get_sub_score_from_metric_cache,
)
from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import (
    PDMScorer as FixedProgressPDMScorer,
)
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer as OfficialPDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)

from .schema import FACTOR_NAMES
from .utils import atomic_json, load_proposal_pickle, sha256_file, token_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-pickle", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=32)
    parser.add_argument("--pairs-per-scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _derive_fixed_progress(metric_cache, simulator, scorer) -> float:
    """Derive the reference's multiplicatively-gated raw progress in memory."""

    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        simulator.proposal_sampling,
        metric_cache.ego_state.time_point,
    )
    simulated = simulator.simulate_proposals(pdm_states[None], metric_cache.ego_state)
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    multiplicative = scorer._multi_metrics.prod(axis=0)
    value = float((scorer._progress_raw * multiplicative)[0])
    metric_cache.pdm_progress = value
    return value


def _candidate_indices(
    selected: int,
    oracle: int,
    candidate_count: int,
    count: int,
    rng: np.random.Generator,
) -> List[int]:
    if count < 2:
        raise ValueError("pairs-per-scene must be at least two")
    result: List[int] = []
    for value in (selected, oracle):
        if value not in result:
            result.append(int(value))
    for value in rng.permutation(candidate_count).tolist():
        if value not in result:
            result.append(int(value))
        if len(result) == count:
            break
    if len(result) != count:
        raise RuntimeError("Could not construct enough unique candidate indices")
    return result


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "pdm_candidate_parity.json"
    if result_path.exists() and not args.overwrite:
        print(result_path.read_text(), end="")
        return
    for path in (args.proposal_pickle, args.candidate_matrix, args.metric_cache):
        if not path.exists():
            raise FileNotFoundError(path)

    proposals = load_proposal_pickle(args.proposal_pickle)
    with np.load(args.candidate_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        true_scores = archive["candidate_scores"].astype(np.float64)
        selected_indices = archive["selected_indices"].astype(np.int64)
        oracle_indices = archive["oracle_indices"].astype(np.int64)
        stored_factors = archive["candidate_factors"].astype(np.float64)
        stored_names = archive["candidate_factor_names"].astype(str).tolist()
    if stored_names != list(FACTOR_NAMES):
        raise RuntimeError(f"Unexpected factor order: {stored_names}")
    if len(tokens) != len(proposals) or set(tokens) != set(proposals):
        raise RuntimeError("Proposal and candidate-score token inventories differ")
    if true_scores.shape != (len(tokens), 64) or stored_factors.shape != (len(tokens), 64, 7):
        raise RuntimeError("Expected a [N,64] score matrix and [N,64,7] factors")

    rng = np.random.default_rng(args.seed)
    if args.scene_count > len(tokens):
        raise ValueError("scene-count exceeds available tokens")
    chosen_rows = np.sort(rng.choice(len(tokens), size=args.scene_count, replace=False))
    row_for_token = token_index(tokens)
    metric_loader = MetricCacheLoader(args.metric_cache)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    official_scorer = OfficialPDMScorer(sampling)
    fixed_scorer = FixedProgressPDMScorer(sampling)
    rows: List[Dict[str, object]] = []
    max_errors = {name: 0.0 for name in FACTOR_NAMES}
    max_stored_errors = {name: 0.0 for name in FACTOR_NAMES}
    minimum_reproduction = None

    for matrix_row in chosen_rows:
        token = str(tokens[matrix_row])
        indices = _candidate_indices(
            int(selected_indices[matrix_row]),
            int(oracle_indices[matrix_row]),
            true_scores.shape[1],
            args.pairs_per_scene,
            rng,
        )
        with lzma.open(metric_loader.metric_cache_paths[token], "rb") as stream:
            metric_cache = pickle.load(stream)
        pdm_progress = _derive_fixed_progress(metric_cache, simulator, official_scorer)
        candidate_poses = np.asarray(proposals[token]["proposals"], dtype=np.float32)[indices]
        efficient, *_ = get_sub_score_from_metric_cache(
            metric_cache,
            candidate_poses,
            True,
            simulator_instance=simulator,
            scorer_instance=fixed_scorer,
        )
        efficient = np.asarray(efficient, dtype=np.float64)

        for local_row, candidate_index in enumerate(indices):
            official = asdict(
                pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=Trajectory(candidate_poses[local_row]),
                    future_sampling=sampling,
                    simulator=simulator,
                    scorer=official_scorer,
                )
            )
            official_values = np.asarray(
                [float(official[name]) for name in FACTOR_NAMES], dtype=np.float64
            )
            efficient_values = efficient[local_row]
            stored_values = stored_factors[matrix_row, candidate_index]
            errors = np.abs(efficient_values - official_values)
            stored_errors = np.abs(stored_values - official_values)
            for factor_index, name in enumerate(FACTOR_NAMES):
                max_errors[name] = max(max_errors[name], float(errors[factor_index]))
                max_stored_errors[name] = max(
                    max_stored_errors[name], float(stored_errors[factor_index])
                )
            failed = bool(np.max(errors) > args.tolerance)
            if failed and minimum_reproduction is None:
                minimum_reproduction = {
                    "token": token,
                    "candidate_index": int(candidate_index),
                    "pdm_reference_progress": pdm_progress,
                    "official": dict(zip(FACTOR_NAMES, official_values.tolist())),
                    "efficient": dict(zip(FACTOR_NAMES, efficient_values.tolist())),
                    "abs_error": dict(zip(FACTOR_NAMES, errors.tolist())),
                }
            row: Dict[str, object] = {
                "token": token,
                "candidate_index": int(candidate_index),
                "is_selected": candidate_index == int(selected_indices[matrix_row]),
                "is_oracle": candidate_index == int(oracle_indices[matrix_row]),
                "pdm_reference_progress": pdm_progress,
            }
            for factor_index, name in enumerate(FACTOR_NAMES):
                row[f"official_{name}"] = float(official_values[factor_index])
                row[f"efficient_{name}"] = float(efficient_values[factor_index])
                row[f"abs_error_{name}"] = float(errors[factor_index])
                row[f"stored_abs_error_{name}"] = float(stored_errors[factor_index])
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pdm_candidate_parity_pairs.csv", rows)
    overall = max(max_errors.values())
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_pickle": str(args.proposal_pickle.resolve()),
        "proposal_pickle_sha256": sha256_file(args.proposal_pickle),
        "candidate_matrix": str(args.candidate_matrix.resolve()),
        "candidate_matrix_sha256": sha256_file(args.candidate_matrix),
        "metric_cache": str(args.metric_cache.resolve()),
        "selection": "seeded random scenes; selected + oracle + unique random candidates",
        "seed": args.seed,
        "scene_count": int(args.scene_count),
        "pair_count": len(rows),
        "pairs_per_scene": args.pairs_per_scene,
        "official_protocol": "one PDM reference plus one candidate per pdm_score call",
        "efficient_protocol": "fixed PDM-reference progress, vectorized candidates",
        "factor_order": list(FACTOR_NAMES),
        "max_abs_error_efficient_vs_official": max_errors,
        "overall_max_abs_error_efficient_vs_official": overall,
        "max_abs_error_stored_float32_vs_official": max_stored_errors,
        "tolerance": args.tolerance,
        "minimum_reproduction": minimum_reproduction,
        "parity_passed": bool(overall <= args.tolerance and len(rows) >= 128),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["parity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
