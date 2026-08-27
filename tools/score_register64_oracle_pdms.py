#!/usr/bin/env python3
"""Select the exact NAVSIM-v1.1 PDMS oracle from Register64 candidates.

Each scene is simulated once as a 64-candidate pool.  NAVSIM-v1.1 normalizes
progress against the PDM reference trajectory, so the final score is
recomputed per candidate with the same two-proposal denominator used by the
official evaluator.  The resulting Oracle@64 trajectories can therefore be
fed back to the unmodified official scorer as a parity check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = 1
_WORKER: dict[str, Any] = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--prediction-manifest")
    parser.add_argument("--metric-cache-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-num", type=int, default=64)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Development-only prefix limit; zero evaluates the full datalist.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _load_tokens(path: Path, max_scenes: int) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"datalist must contain a non-empty JSON list: {path}")
    if any(not isinstance(token, str) or not token for token in raw):
        raise TypeError("datalist entries must be non-empty strings")
    if len(raw) != len(set(raw)):
        raise ValueError("datalist contains duplicate tokens")
    if max_scenes < 0:
        raise ValueError("--max-scenes must be non-negative")
    return raw[:max_scenes] if max_scenes else raw


def _load_candidate(path: Path, proposal_num: int) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        proposals = np.asarray(payload["proposals"], dtype=np.float64)
        selected_index = np.asarray(payload["selected_index"])
    if proposals.shape != (proposal_num, 8, 3):
        raise ValueError(
            f"candidate archive {path} has shape {proposals.shape}, "
            f"expected {(proposal_num, 8, 3)}"
        )
    if not np.isfinite(proposals).all():
        raise ValueError(f"candidate archive contains NaN or Inf: {path}")
    if selected_index.shape != ():
        raise ValueError(f"selected_index must be scalar: {path}")
    selected = int(selected_index)
    if not 0 <= selected < proposal_num:
        raise IndexError(f"selected_index {selected} is out of range: {path}")
    return proposals, selected


def _metric_cache_token_hash(root: Path) -> tuple[str, int]:
    metadata_dir = root / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"metric-cache metadata is missing: {metadata_dir}")
    tokens: set[str] = set()
    for path in metadata_files:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            value = line.strip().split(",")[-1].strip().strip('"')
            if value:
                tokens.add(Path(value).parent.name)
    if not tokens:
        raise RuntimeError(f"metric-cache metadata contains no tokens: {metadata_dir}")
    payload = "\n".join(sorted(tokens)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(tokens)


def _validate_candidate_manifest(
    manifest_path: Path,
    candidate_dir: Path,
    datalist_path: Path,
    tokens: list[str],
    proposal_num: int,
    *,
    full_datalist_size: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    export = manifest.get("candidate_export")
    if not isinstance(export, dict):
        raise RuntimeError("prediction manifest does not describe a candidate export")
    if export.get("proposal_shape") != [proposal_num, 8, 3]:
        raise RuntimeError(
            "candidate proposal shape does not match the requested Oracle protocol"
        )
    if manifest.get("datalist_sha256") != _sha256(datalist_path):
        raise RuntimeError("candidate manifest datalist hash mismatch")
    if int(manifest.get("num_predictions", -1)) != full_datalist_size:
        raise RuntimeError("candidate manifest prediction count mismatch")
    expected_dir = manifest_path.parent / str(export["relative_directory"])
    if expected_dir.resolve() != candidate_dir.resolve():
        raise RuntimeError(
            f"candidate directory does not match manifest: {candidate_dir} != {expected_dir}"
        )
    missing = [token for token in tokens if not (candidate_dir / f"{token}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"candidate export is incomplete: missing={len(missing)} first={missing[0]}"
        )
    return manifest


def _individual_equivalent_scores(
    multi_metrics: np.ndarray,
    weighted_metrics: np.ndarray,
    raw_progress: np.ndarray,
    metric_weights: np.ndarray,
    *,
    progress_index: int,
    progress_distance_threshold: float,
) -> np.ndarray:
    """Recover official one-candidate scores after one vectorized pool pass."""

    multi_metrics = np.asarray(multi_metrics, dtype=np.float64)
    weighted_metrics = np.asarray(weighted_metrics, dtype=np.float64)
    raw_progress = np.asarray(raw_progress, dtype=np.float64)
    metric_weights = np.asarray(metric_weights, dtype=np.float64)
    if multi_metrics.ndim != 2 or weighted_metrics.ndim != 2:
        raise ValueError("PDM metric arrays must be rank two")
    if multi_metrics.shape[1] != weighted_metrics.shape[1]:
        raise ValueError("PDM metric arrays disagree on proposal count")
    if raw_progress.shape != (multi_metrics.shape[1],):
        raise ValueError("raw progress has the wrong proposal count")
    if metric_weights.shape != (weighted_metrics.shape[0],):
        raise ValueError("PDM metric weights have the wrong shape")
    if multi_metrics.shape[1] < 2:
        raise ValueError("PDM pool must contain the reference and one candidate")

    multiplicative = multi_metrics.prod(axis=0)
    gated_progress = raw_progress * multiplicative
    candidate_denominator = np.maximum(gated_progress[0], gated_progress[1:])
    normalized_progress = np.divide(
        gated_progress[1:],
        candidate_denominator,
        out=np.zeros_like(candidate_denominator),
        where=candidate_denominator > progress_distance_threshold,
    )
    below_threshold = candidate_denominator <= progress_distance_threshold
    normalized_progress[below_threshold] = (
        multiplicative[1:][below_threshold] != 0.0
    ).astype(np.float64)

    candidate_weighted = weighted_metrics[:, 1:].copy()
    candidate_weighted[progress_index] = normalized_progress
    weight_sum = metric_weights.sum()
    if weight_sum <= 0:
        raise ValueError("PDM metric weights must have a positive sum")
    weighted_score = (
        candidate_weighted * metric_weights[:, None]
    ).sum(axis=0) / weight_sum
    result = multiplicative[1:] * weighted_score
    if not np.isfinite(result).all():
        raise FloatingPointError("PDM candidate scores contain NaN or Inf")
    return result


def _initialize_worker(metric_cache_root: str) -> None:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )

    sampling_40 = TrajectorySampling(num_poses=40, interval_length=0.1)
    _WORKER.update(
        {
            "loader": MetricCacheLoader(Path(metric_cache_root)),
            "sampling_8": TrajectorySampling(time_horizon=4.0, interval_length=0.5),
            "sampling_40": sampling_40,
            "simulator": PDMSimulator(sampling_40),
            "scorer": PDMScorer(sampling_40),
        }
    )


def _score_candidate_pool(task: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import (
        get_trajectory_as_array,
        transform_trajectory,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        WeightedMetricIndex,
    )

    token, proposals = task
    loader = _WORKER["loader"]
    sampling_8 = _WORKER["sampling_8"]
    sampling_40 = _WORKER["sampling_40"]
    simulator = _WORKER["simulator"]
    scorer = _WORKER["scorer"]
    metric_cache = loader.get_from_token(token)
    initial_ego_state = metric_cache.ego_state
    states = [
        get_trajectory_as_array(
            metric_cache.trajectory,
            sampling_40,
            initial_ego_state.time_point,
        )
    ]
    for proposal in proposals:
        trajectory = Trajectory(proposal, sampling_8)
        absolute = transform_trajectory(trajectory, initial_ego_state)
        states.append(
            get_trajectory_as_array(
                absolute,
                sampling_40,
                initial_ego_state.time_point,
            )
        )
    simulated = simulator.simulate_proposals(
        np.stack(states, axis=0), initial_ego_state
    )
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    scores = _individual_equivalent_scores(
        scorer._multi_metrics,
        scorer._weighted_metrics,
        scorer._progress_raw,
        scorer._config.weighted_metrics_array,
        progress_index=int(WeightedMetricIndex.PROGRESS),
        progress_distance_threshold=float(
            scorer._config.progress_distance_threshold
        ),
    )
    return token, scores


def _open_database(path: Path, identity: Mapping[str, Any], resume: bool) -> sqlite3.Connection:
    existed = path.exists()
    if resume and not existed:
        raise FileNotFoundError(f"cannot resume without score database: {path}")
    if not resume and existed:
        raise FileExistsError(f"refusing to overwrite score database: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            token TEXT PRIMARY KEY,
            score_vector BLOB NOT NULL,
            oracle_index INTEGER NOT NULL,
            drivor_index INTEGER NOT NULL,
            proposal0_score REAL NOT NULL,
            random_score REAL NOT NULL,
            drivor_score REAL NOT NULL,
            oracle_score REAL NOT NULL
        )
        """
    )
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='identity'"
    ).fetchone()
    if existing is not None and existing[0] != encoded:
        raise RuntimeError("refusing to resume Oracle scores with a different identity")
    if existing is None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('identity', ?)", (encoded,)
        )
    connection.commit()
    return connection


def _random_index(token: str, proposal_num: int) -> int:
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8], byteorder="little"
    ) % proposal_num


def _store_result(
    connection: sqlite3.Connection,
    *,
    token: str,
    scores: np.ndarray,
    drivor_index: int,
) -> int:
    oracle_index = int(np.argmax(scores))
    random_index = _random_index(token, scores.shape[0])
    connection.execute(
        """
        INSERT OR REPLACE INTO scores(
            token, score_vector, oracle_index, drivor_index, proposal0_score,
            random_score, drivor_score, oracle_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token,
            np.asarray(scores, dtype=np.float64).tobytes(),
            oracle_index,
            drivor_index,
            float(scores[0]),
            float(scores[random_index]),
            float(scores[drivor_index]),
            float(scores[oracle_index]),
        ),
    )
    return oracle_index


def _submit_bounded(
    executor: ProcessPoolExecutor,
    tokens: Iterable[str],
    candidate_dir: Path,
    proposal_num: int,
    max_pending: int,
):
    token_iterator = iter(tokens)
    pending: dict[Any, tuple[str, int, np.ndarray]] = {}

    def fill() -> None:
        while len(pending) < max_pending:
            try:
                token = next(token_iterator)
            except StopIteration:
                return
            proposals, selected_index = _load_candidate(
                candidate_dir / f"{token}.npz", proposal_num
            )
            future = executor.submit(_score_candidate_pool, (token, proposals))
            pending[future] = (token, selected_index, proposals)

    fill()
    while pending:
        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            expected_token, selected_index, proposals = pending.pop(future)
            token, scores = future.result()
            if token != expected_token or scores.shape != (proposal_num,):
                raise RuntimeError("Oracle worker returned an invalid result contract")
            yield token, scores, selected_index, proposals
        fill()


def _write_report(
    connection: sqlite3.Connection,
    *,
    tokens: list[str],
    proposal_num: int,
    identity: Mapping[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    rows = {
        row[0]: row[1:]
        for row in connection.execute(
            """
            SELECT token, oracle_index, drivor_index, proposal0_score,
                   random_score, drivor_score, oracle_score
            FROM scores
            """
        )
    }
    missing = [token for token in tokens if token not in rows]
    if missing or len(rows) != len(tokens):
        raise RuntimeError(
            f"Oracle score database is incomplete: rows={len(rows)} "
            f"expected={len(tokens)} missing={len(missing)}"
        )
    ordered = [rows[token] for token in tokens]
    oracle_indices = np.asarray([row[0] for row in ordered], dtype=np.int64)
    drivor_indices = np.asarray([row[1] for row in ordered], dtype=np.int64)
    proposal0 = np.asarray([row[2] for row in ordered], dtype=np.float64)
    random_scores = np.asarray([row[3] for row in ordered], dtype=np.float64)
    drivor_scores = np.asarray([row[4] for row in ordered], dtype=np.float64)
    oracle_scores = np.asarray([row[5] for row in ordered], dtype=np.float64)
    histogram = np.bincount(oracle_indices, minlength=proposal_num)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "navsim_v1.1_pdms_oracle_at_64",
        "identity_hash": _stable_hash(identity),
        "num_scenes": len(tokens),
        "proposal_num": proposal_num,
        "mean_oracle_at_64_pdms": float(oracle_scores.mean()),
        "median_oracle_at_64_pdms": float(np.median(oracle_scores)),
        "mean_drivor_selected_pdms": float(drivor_scores.mean()),
        "mean_proposal0_pdms": float(proposal0.mean()),
        "mean_deterministic_random_pdms": float(random_scores.mean()),
        "oracle_gain_over_drivor": float((oracle_scores - drivor_scores).mean()),
        "oracle_gain_over_proposal0": float((oracle_scores - proposal0).mean()),
        "drivor_exact_oracle_rate": float(
            np.equal(oracle_indices, drivor_indices).mean()
        ),
        "oracle_register_histogram": histogram.tolist(),
        "active_oracle_register_ratio": float((histogram > 0).mean()),
        "score_finite_fraction": float(
            np.isfinite(
                np.concatenate((proposal0, random_scores, drivor_scores, oracle_scores))
            ).mean()
        ),
        "wall_time_seconds": float(wall_time_seconds),
        "scenes_per_second": len(tokens) / max(wall_time_seconds, 1e-9),
        "oracle_prediction_dir": "oracle_predictions/test",
    }
    return report


def main() -> None:
    args = _parse_args()
    if args.proposal_num <= 1:
        raise ValueError("--proposal-num must be greater than one")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    manifest_path = (
        Path(args.prediction_manifest).expanduser().resolve()
        if args.prediction_manifest
        else candidate_dir.parent / "prediction_manifest.json"
    )
    cache_root = Path(args.metric_cache_root).expanduser().resolve()
    datalist_path = Path(args.datalist).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    full_tokens = _load_tokens(datalist_path, 0)
    tokens = _load_tokens(datalist_path, args.max_scenes)
    manifest = _validate_candidate_manifest(
        manifest_path,
        candidate_dir,
        datalist_path,
        tokens,
        args.proposal_num,
        full_datalist_size=len(full_tokens),
    )
    cache_token_hash, cache_token_count = _metric_cache_token_hash(cache_root)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "navsim_v1.1_pdms_oracle_at_64",
        "repository_commit": _repository_commit(),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "candidate_identity_hash": manifest.get("identity_hash"),
        "datalist_sha256": _sha256(datalist_path),
        "num_scenes": len(tokens),
        "full_datalist_size": len(full_tokens),
        "max_scenes": int(args.max_scenes),
        "proposal_num": int(args.proposal_num),
        "metric_cache_token_hash": cache_token_hash,
        "metric_cache_token_count": cache_token_count,
        "score_semantics": "official_two_proposal_progress_normalization",
    }

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite Oracle output: {output_dir}; use --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "oracle_predictions" / "test"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "oracle_scores.sqlite3"
    connection = _open_database(database_path, identity, args.resume)
    existing_tokens = {
        row[0] for row in connection.execute("SELECT token FROM scores")
    }
    unexpected = existing_tokens.difference(tokens)
    if unexpected:
        raise RuntimeError(
            f"Oracle score database contains {len(unexpected)} unexpected tokens"
        )
    pending_tokens = [token for token in tokens if token not in existing_tokens]
    print(
        f"[oracle64] scenes={len(tokens)} completed={len(existing_tokens)} "
        f"pending={len(pending_tokens)} workers={args.workers}",
        flush=True,
    )

    started = time.perf_counter()
    completed = len(existing_tokens)
    if pending_tokens:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(str(cache_root),),
        )
        try:
            for token, scores, selected_index, proposals in _submit_bounded(
                executor,
                pending_tokens,
                candidate_dir,
                args.proposal_num,
                max_pending=max(args.workers * 2, 1),
            ):
                oracle_index = _store_result(
                    connection,
                    token=token,
                    scores=scores,
                    drivor_index=selected_index,
                )
                _atomic_numpy(
                    prediction_dir / f"{token}.npy",
                    proposals[oracle_index].astype(np.float32, copy=False),
                )
                completed += 1
                if completed % 32 == 0:
                    connection.commit()
                if completed % 100 == 0 or completed == len(tokens):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[oracle64] progress={completed}/{len(tokens)} "
                        f"rate={max(completed - len(existing_tokens), 0) / max(elapsed, 1e-9):.2f} scenes/s",
                        flush=True,
                    )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        connection.commit()

    # A database row is authoritative on resume; repair only its exact derived
    # official prediction when a previous process stopped between the two writes.
    for token, oracle_index in connection.execute(
        "SELECT token, oracle_index FROM scores"
    ):
        path = prediction_dir / f"{token}.npy"
        valid = False
        if path.is_file():
            try:
                value = np.load(path, allow_pickle=False)
                valid = value.shape == (8, 3) and np.isfinite(value).all()
            except Exception:
                valid = False
        if not valid:
            proposals, _ = _load_candidate(
                candidate_dir / f"{token}.npz", args.proposal_num
            )
            _atomic_numpy(
                path, proposals[int(oracle_index)].astype(np.float32, copy=False)
            )

    actual_predictions = {path.stem for path in prediction_dir.glob("*.npy")}
    if actual_predictions != set(tokens):
        raise RuntimeError(
            "Oracle prediction token set mismatch: "
            f"actual={len(actual_predictions)} expected={len(tokens)}"
        )
    wall_time = time.perf_counter() - started
    report = _write_report(
        connection,
        tokens=tokens,
        proposal_num=args.proposal_num,
        identity=identity,
        wall_time_seconds=wall_time,
    )
    connection.close()
    selection_manifest = {
        **identity,
        "identity_hash": _stable_hash(identity),
        "status": "complete",
        "report": report,
    }
    _atomic_json(output_dir / "oracle_selection_manifest.json", selection_manifest)
    _atomic_json(output_dir / "oracle_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
