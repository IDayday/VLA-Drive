"""Resumable, multi-process scoring for cached Navtest proposal banks.

GPU inference is deliberately separated from CPU PDM simulation.  Work is
partitioned by complete log, persisted atomically per log, and can additionally
be sharded across hosts that share the output directory.  This makes repeated
scorer audits cheap and lets interrupted scoring resume without another VLM
forward pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker

from local_stage2.run_navtest_proposal_audit import (
    FACTOR_NAMES,
    _sha256,
    score_candidate_partition,
)


CONFIG_PATH = "../navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def _log_directory(root: Path, log_name: str) -> Path:
    digest = hashlib.sha256(log_name.encode("utf-8")).hexdigest()[:16]
    return root / "scored_logs" / digest


def _completed_log(root: Path, log_name: str) -> bool:
    directory = _log_directory(root, log_name)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("log_name") == log_name
        and (directory / "rows.csv").is_file()
        and (directory / "scores.npz").is_file()
    )


def _persist_log_rows(root: Path, log_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    directory = _log_directory(root, log_name)
    directory.mkdir(parents=True, exist_ok=True)
    scalar_rows: List[Dict[str, Any]] = []
    score_by_token: Dict[str, np.ndarray] = {}
    prediction_by_token: Dict[str, np.ndarray] = {}
    for source in rows:
        row = dict(source)
        if row.get("valid"):
            token = str(row["token"])
            score_by_token[token] = np.asarray(row.pop("candidate_scores"), dtype=np.float32)
            prediction_by_token[token] = np.asarray(row.pop("predicted_scores"), dtype=np.float32)
        scalar_rows.append(row)

    valid_tokens = sorted(score_by_token)
    if valid_tokens:
        candidate_scores = np.stack([score_by_token[token] for token in valid_tokens])
        predicted_scores = np.stack([prediction_by_token[token] for token in valid_tokens])
    else:
        candidate_scores = np.empty((0, 0), dtype=np.float32)
        predicted_scores = np.empty((0, 0), dtype=np.float32)

    _atomic_write_csv(directory / "rows.csv", pd.DataFrame(scalar_rows))
    _atomic_write_npz(
        directory / "scores.npz",
        tokens=np.asarray(valid_tokens),
        candidate_scores=candidate_scores,
        predicted_scores=predicted_scores,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_name": log_name,
        "scene_count": len(rows),
        "valid_scene_count": len(valid_tokens),
        "invalid_scene_count": len(rows) - len(valid_tokens),
    }
    _atomic_write_text(
        directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def score_and_persist_partition(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ray worker body; each finished log is durable before the next starts."""

    completed: List[Dict[str, Any]] = []
    for item in args:
        output_dir = Path(item["score_output_dir"])
        log_name = str(item["log_name"])
        if _completed_log(output_dir, log_name):
            completed.append({"log_name": log_name, "status": "already_complete"})
            continue
        rows = score_candidate_partition([item])
        manifest = _persist_log_rows(output_dir, log_name, rows)
        completed.append({"log_name": log_name, "status": "scored", **manifest})
    return completed


def _load_predictions(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with path.open("rb") as file:
        predictions = pickle.load(file)
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError(f"No token-indexed predictions in {path}")
    for token, prediction in predictions.items():
        if set(prediction) < {"proposals", "predicted_scores"}:
            raise ValueError(f"Malformed prediction for token {token}")
    return predictions


def _predictions_by_log(
    cfg: DictConfig,
    predictions: Dict[str, Dict[str, np.ndarray]],
) -> tuple[Dict[str, Dict[str, Any]], set[str]]:
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    expected_tokens = set(scene_loader.tokens) & set(metric_loader.tokens)
    prediction_tokens = set(predictions)
    unknown = prediction_tokens - expected_tokens
    missing = expected_tokens - prediction_tokens
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} tokens outside split/cache")
    allow_partial = bool(cfg.get("proposal_score_allow_partial", False))
    if missing and not allow_partial:
        raise ValueError(
            f"Predictions omit {len(missing)} split/cache tokens; "
            "set +proposal_score_allow_partial=true only for smoke audits"
        )

    log_for_token = {
        token: log_name
        for log_name, tokens in scene_loader.get_tokens_list_per_log().items()
        for token in tokens
    }
    grouped: Dict[str, Dict[str, Any]] = {}
    for token, prediction in predictions.items():
        grouped.setdefault(log_for_token[token], {})[token] = prediction
    return grouped, prediction_tokens


def _read_persisted_rows(root: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for manifest_path in sorted((root / "scored_logs").glob("*/manifest.json")):
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        frame = pd.read_csv(directory / "rows.csv")
        if set(frame["log_name"].astype(str)) != {str(manifest["log_name"])}:
            raise ValueError(f"Log mismatch in {directory}")
        rows.extend(frame.to_dict(orient="records"))
        with np.load(directory / "scores.npz") as archive:
            tokens = archive["tokens"].astype(str)
            scores = archive["candidate_scores"]
            predictions = archive["predicted_scores"]
        if len(tokens) != len(scores) or len(tokens) != len(predictions):
            raise ValueError(f"Array length mismatch in {directory}")
        for index, token in enumerate(tokens):
            if token in arrays:
                raise ValueError(f"Duplicate persisted token {token}")
            arrays[token] = (scores[index], predictions[index])
    return rows, arrays


def aggregate_persisted_logs(
    output_dir: Path,
    expected_tokens: set[str],
    prediction_path: Path,
    cfg: DictConfig,
) -> Dict[str, Any]:
    rows, arrays = _read_persisted_rows(output_dir)
    row_by_token = {str(row["token"]): row for row in rows}
    if len(row_by_token) != len(rows):
        raise ValueError("Duplicate token across persisted log rows")
    missing = expected_tokens - set(row_by_token)
    extra = set(row_by_token) - expected_tokens
    if missing or extra:
        raise RuntimeError(
            f"Scoring shards incomplete: missing={len(missing)}, extra={len(extra)}"
        )
    rows = [row_by_token[token] for token in sorted(row_by_token)]
    valid_rows = [row for row in rows if bool(row["valid"])]
    valid_tokens = [str(row["token"]) for row in valid_rows]
    if set(valid_tokens) != set(arrays):
        raise ValueError("Persisted scalar rows and score arrays disagree")

    candidate_scores = np.stack([arrays[token][0] for token in valid_tokens])
    predicted_scores = np.stack([arrays[token][1] for token in valid_tokens])
    selected_indices = np.asarray(
        [int(row["selected_index"]) for row in valid_rows], dtype=np.int16
    )
    oracle_indices = np.asarray(
        [int(row["oracle_index"]) for row in valid_rows], dtype=np.int16
    )
    _atomic_write_npz(
        output_dir / "candidate_scores.npz",
        tokens=np.asarray(valid_tokens),
        log_names=np.asarray([str(row["log_name"]) for row in valid_rows]),
        candidate_scores=candidate_scores,
        predicted_scores=predicted_scores,
        selected_indices=selected_indices,
        oracle_indices=oracle_indices,
    )
    _atomic_write_csv(output_dir / "per_scene_candidate_quality.csv", pd.DataFrame(rows))

    numeric_keys = (
        "selected_pdms",
        "standard_selected_pdms",
        "selected_score_parity_abs",
        "best_of_64_pdms",
        "scorer_regret",
        "mean_candidate_pdms",
        "median_candidate_pdms",
        "top5_oracle_mean_pdms",
        "fraction_candidates_pdms_ge_0_9",
        "fraction_candidates_pdms_ge_0_8",
        "unique_candidate_count",
        "mean_pairwise_endpoint_distance_m",
        "mean_pairwise_ade_m",
    ) + tuple(
        f"{prefix}_{factor}"
        for prefix in ("selected", "oracle")
        for factor in FACTOR_NAMES[:-1]
    )
    checkpoint = cfg.get("proposal_score_checkpoint_path")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "navtest",
        "scene_count": len(rows),
        "valid_scene_count": len(valid_rows),
        "invalid_scene_count": len(rows) - len(valid_rows),
        "log_count": len({str(row["log_name"]) for row in valid_rows}),
        "candidate_count": int(candidate_scores.shape[1]),
        "proposal_predictions_path": str(prediction_path.resolve()),
        "proposal_predictions_sha256": _sha256(prediction_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": (
            _sha256(Path(checkpoint)) if checkpoint and Path(checkpoint).is_file() else None
        ),
        "metrics": {
            key: float(np.mean([float(row[key]) for row in valid_rows]))
            for key in numeric_keys
        },
        "max_selected_score_parity_abs": float(
            max(float(row["selected_score_parity_abs"]) for row in valid_rows)
        ),
        "execution": {
            "resumable_per_log": True,
            "host_shard_count": int(cfg.get("proposal_score_shard_count", 1)),
            "cpu_workers_per_host": cfg.worker.get("threads_per_node"),
            "blas_threads_per_worker": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
        },
    }
    _atomic_write_text(
        output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(cfg.proposal_predictions_path)
    predictions = _load_predictions(prediction_path)
    grouped, expected_tokens = _predictions_by_log(cfg, predictions)

    aggregate_only = bool(cfg.get("proposal_score_aggregate_only", False))
    if not aggregate_only:
        shard_count = int(cfg.get("proposal_score_shard_count", 1))
        shard_index = int(cfg.get("proposal_score_shard_index", 0))
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Invalid proposal-score shard index/count")
        selected_logs = [
            log_name
            for index, log_name in enumerate(sorted(grouped))
            if index % shard_count == shard_index
        ]
        pending = [
            {
                "cfg": cfg,
                "log_name": log_name,
                "predictions": grouped[log_name],
                "score_output_dir": str(output_dir),
            }
            for log_name in selected_logs
            if not _completed_log(output_dir, log_name)
        ]
        print(
            json.dumps(
                {
                    "host_shard": f"{shard_index}/{shard_count}",
                    "selected_logs": len(selected_logs),
                    "already_complete_logs": len(selected_logs) - len(pending),
                    "pending_logs": len(pending),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if pending:
            worker = build_worker(cfg)
            # Submit exactly one log per Ray task.  The generic worker_map
            # pre-chunks 136 logs into ``num_workers`` bundles, which leaves a
            # long tail when a bundle happens to contain two large logs.
            worker.map(
                Task(fn=score_and_persist_partition),
                [[item] for item in pending],
            )

    if bool(cfg.get("proposal_score_aggregate_when_complete", True)):
        summary = aggregate_persisted_logs(output_dir, expected_tokens, prediction_path, cfg)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
