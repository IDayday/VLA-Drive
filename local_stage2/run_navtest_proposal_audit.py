"""Evaluate selected and all EpisodeDrive proposals on the NAVSIM Navtest split.

The released evaluator only serializes the selected trajectory.  This audit keeps
the model forward path unchanged, exports the final proposal bank, and scores all
proposals with the same fixed-PDM-progress scorer used by Stage-2 validation.  A
second, standard single-trajectory PDM call audits score parity for every scene.
"""

from __future__ import annotations

import hashlib
import json
import logging
import lzma
import os
import pickle
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (
    get_sub_score_from_metric_cache,
)
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset


logger = logging.getLogger(__name__)
CONFIG_PATH = "../navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"
FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
SCORER_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class ProposalExportModule(AgentLightningModule):
    """Prediction-only wrapper that exports final proposals without PDM input."""

    def predict_step(self, batch, batch_idx: int):
        features, _targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            prediction = self.agent.forward(features)

        proposals = prediction["proposals"].detach().float().cpu().numpy()
        predicted_scores = prediction["pdm_score"].detach().float().cpu().numpy()
        if proposals.ndim != 4 or proposals.shape[:2] != predicted_scores.shape:
            raise RuntimeError(
                "Unexpected proposal/prediction shapes: "
                f"{proposals.shape} versus {predicted_scores.shape}"
            )

        factor_logits = None
        if "scorer_candidate_features" in prediction:
            factor_logits = torch.stack(
                [prediction["pred_logit"][key] for key in SCORER_FACTOR_KEYS],
                dim=-1,
            )
        result = {}
        for index, token in enumerate(tokens):
            token_result = {
                "proposals": proposals[index],
                "predicted_scores": predicted_scores[index],
            }
            if factor_logits is not None:
                token_result.update(
                    {
                        # These tensors are current-observation inference outputs.
                        # Keep FP32 so offline and online residual scorer decisions
                        # can be checked at numerical precision.
                        "base_factor_logits": factor_logits[index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        "candidate_features": prediction[
                            "scorer_candidate_features"
                        ][index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                    }
                )
            if "language_feature" in prediction:
                token_result["scene_features"] = (
                    prediction["language_feature"][index]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            if "ego_feature" in prediction:
                token_result["ego_features"] = (
                    prediction["ego_feature"][index]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            result[str(token)] = token_result
        return result


def _compare_prediction_banks(
    candidate: Dict[str, Dict[str, np.ndarray]],
    reference: Dict[str, Dict[str, np.ndarray]],
    *,
    allow_reference_superset: bool = False,
) -> Dict[str, Any]:
    """Strictly compare the proposal/base-score portion of two caches."""

    candidate_tokens = set(candidate)
    reference_tokens = set(reference)
    token_sets_valid = (
        candidate_tokens.issubset(reference_tokens)
        if allow_reference_superset
        else candidate_tokens == reference_tokens
    )
    if not token_sets_valid:
        raise RuntimeError(
            "Prediction token mismatch: "
            f"missing={len(reference_tokens - candidate_tokens)}, "
            f"extra={len(candidate_tokens - reference_tokens)}"
        )
    maximum = {"proposals": 0.0, "predicted_scores": 0.0}
    for token in sorted(candidate_tokens):
        for key in maximum:
            left = np.asarray(candidate[token][key])
            right = np.asarray(reference[token][key])
            if left.shape != right.shape:
                raise RuntimeError(
                    f"Shape mismatch for {token}/{key}: {left.shape} != {right.shape}"
                )
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise RuntimeError(f"Non-finite values in {token}/{key}")
            maximum[key] = max(
                maximum[key], float(np.max(np.abs(left.astype(np.float64) - right)))
            )
    maximum["overall"] = max(maximum.values())
    return {
        "scene_count": len(candidate_tokens),
        "reference_scene_count": len(reference_tokens),
        "reference_superset_allowed": bool(allow_reference_superset),
        "max_abs": maximum,
        "passes_1e_8": bool(maximum["overall"] <= 1e-8),
    }


def _candidate_geometry(proposals: np.ndarray) -> Dict[str, float]:
    rounded = np.round(proposals.reshape(len(proposals), -1), decimals=6)
    unique_count = len(np.unique(rounded, axis=0))
    left, right = np.triu_indices(len(proposals), k=1)
    if len(left):
        endpoint_distance = np.linalg.norm(
            proposals[left, -1, :2] - proposals[right, -1, :2], axis=-1
        )
        ade_distance = np.linalg.norm(
            proposals[left, :, :2] - proposals[right, :, :2], axis=-1
        ).mean(axis=-1)
        mean_endpoint_distance = float(endpoint_distance.mean())
        mean_pairwise_ade = float(ade_distance.mean())
    else:
        mean_endpoint_distance = 0.0
        mean_pairwise_ade = 0.0
    return {
        "unique_candidate_count": float(unique_count),
        "mean_pairwise_endpoint_distance_m": mean_endpoint_distance,
        "mean_pairwise_ade_m": mean_pairwise_ade,
    }


def _ensure_fixed_pdm_progress(metric_cache, simulator, scorer) -> None:
    """Backfill the train-scorer normalization scalar for official caches.

    The official Navtest ``MetricCache`` contains the PDM reference trajectory,
    while the Stage-2 train cache stores only its precomputed raw progress.  The
    values are equivalent; derive the latter in memory without changing cache
    files so every candidate is normalized against the same PDM reference.
    """

    if hasattr(metric_cache, "pdm_progress"):
        return
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        simulator.proposal_sampling,
        metric_cache.ego_state.time_point,
    )
    simulated_states = simulator.simulate_proposals(
        pdm_states[None], metric_cache.ego_state
    )
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    multiplicative = scorer._multi_metrics.prod(axis=0)
    metric_cache.pdm_progress = scorer._progress_raw * multiplicative


def score_candidate_partition(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score all candidates for a collection of whole-log work items."""

    rows: List[Dict[str, Any]] = []
    for item in args:
        cfg: DictConfig = item["cfg"]
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        simulator = instantiate(cfg.simulator)
        scorer = instantiate(cfg.scorer)
        log_name = item["log_name"]

        for token, prediction in item["predictions"].items():
            row: Dict[str, Any] = {
                "token": token,
                "log_name": log_name,
                "valid": True,
            }
            try:
                metric_path = metric_cache_loader.metric_cache_paths[token]
                with lzma.open(metric_path, "rb") as file:
                    metric_cache = pickle.load(file)

                proposals = np.asarray(prediction["proposals"], dtype=np.float32)
                predicted_scores = np.asarray(
                    prediction["predicted_scores"], dtype=np.float32
                )
                if proposals.shape[0] != predicted_scores.shape[0]:
                    raise RuntimeError("Candidate count differs from predicted scores")

                _ensure_fixed_pdm_progress(metric_cache, simulator, scorer)

                target_factors, *_ = get_sub_score_from_metric_cache(
                    metric_cache,
                    proposals,
                    True,
                )
                target_factors = np.asarray(target_factors, dtype=np.float64)
                target_scores = target_factors[:, -1]
                selected_index = int(np.argmax(predicted_scores))
                oracle_index = int(np.argmax(target_scores))

                # The benchmark scorer evaluates one prediction together with the
                # PDM reference trajectory.  Verify that vectorized candidate
                # scoring uses the identical fixed reference normalization.
                standard = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=Trajectory(proposals[selected_index]),
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                standard_fields = asdict(standard)
                standard_score = float(standard_fields["score"])
                selected_score = float(target_scores[selected_index])

                top_count = min(5, len(target_scores))
                top_indices = np.argpartition(target_scores, -top_count)[-top_count:]
                row.update(
                    {
                        "candidate_count": int(len(target_scores)),
                        "selected_index": selected_index,
                        "oracle_index": oracle_index,
                        "selected_pdms": selected_score,
                        "standard_selected_pdms": standard_score,
                        "selected_score_parity_abs": abs(
                            selected_score - standard_score
                        ),
                        "best_of_64_pdms": float(target_scores[oracle_index]),
                        "scorer_regret": float(
                            target_scores[oracle_index] - selected_score
                        ),
                        "mean_candidate_pdms": float(target_scores.mean()),
                        "median_candidate_pdms": float(np.median(target_scores)),
                        "top5_oracle_mean_pdms": float(
                            target_scores[top_indices].mean()
                        ),
                        "fraction_candidates_pdms_ge_0_9": float(
                            np.mean(target_scores >= 0.9)
                        ),
                        "fraction_candidates_pdms_ge_0_8": float(
                            np.mean(target_scores >= 0.8)
                        ),
                        "candidate_scores": target_scores.astype(np.float32),
                        "candidate_factors": target_factors.astype(np.float32),
                        "predicted_scores": predicted_scores.astype(np.float32),
                        **_candidate_geometry(proposals),
                    }
                )
                for factor_index, factor_name in enumerate(FACTOR_NAMES[:-1]):
                    row[f"selected_{factor_name}"] = float(
                        target_factors[selected_index, factor_index]
                    )
                    row[f"oracle_{factor_name}"] = float(
                        target_factors[oracle_index, factor_index]
                    )
            except Exception:
                logger.warning("Candidate scoring failed for token %s", token)
                traceback.print_exc()
                row["valid"] = False
            rows.append(row)
    return rows


def _flatten_worker_rows(worker_rows: List[Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for value in worker_rows:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.agent.checkpoint_path)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=cfg.load_image_path,
    )
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    limit_scenes = int(cfg.get("proposal_audit_limit_scenes", 0))
    if limit_scenes > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(limit_scenes, len(dataset))))

    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(
        ProposalExportModule(agent=agent),
        dataloader,
        return_predictions=True,
    )

    # Do not all_gather Python proposal objects through NCCL.  That path copies
    # the complete ~76 MB bank to every rank and keeps GPUs busy after inference.
    # Each rank instead publishes one atomic shard to the shared output path;
    # rank zero merges them after a barrier without changing array values.
    world_size = dist.get_world_size() if _dist_ready() else 1
    rank = dist.get_rank() if _dist_ready() else 0
    local_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for batch_predictions in predictions:
        overlap = set(local_predictions).intersection(batch_predictions)
        if overlap:
            raise RuntimeError(f"Duplicate local prediction tokens: {sorted(overlap)[:3]}")
        local_predictions.update(batch_predictions)
    rank_shard_path = output_dir / f"proposal_predictions.rank{rank:03d}-of-{world_size:03d}.pkl"
    temporary_shard_path = rank_shard_path.with_name(
        f".{rank_shard_path.name}.{os.getpid()}.tmp"
    )
    with temporary_shard_path.open("wb") as file:
        pickle.dump(local_predictions, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_shard_path, rank_shard_path)

    if _dist_ready():
        dist.barrier()
    if rank != 0:
        return

    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for shard_rank in range(world_size):
        shard_path = output_dir / (
            f"proposal_predictions.rank{shard_rank:03d}-of-{world_size:03d}.pkl"
        )
        with shard_path.open("rb") as file:
            rank_predictions = pickle.load(file)
        overlap = set(merged).intersection(rank_predictions)
        if overlap:
            raise RuntimeError(f"Duplicate prediction tokens: {sorted(overlap)[:3]}")
        merged.update(rank_predictions)

    # Persist inference separately so a scorer-side failure can be resumed
    # without rerunning the multi-billion-parameter visual backbone.
    with (output_dir / "proposal_predictions.pkl").open("wb") as file:
        pickle.dump(merged, file, protocol=pickle.HIGHEST_PROTOCOL)

    prediction_path = output_dir / "proposal_predictions.pkl"
    sample = next(iter(merged.values()))
    cache_manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "agent_target": str(cfg.agent._target_),
        "precision": str(cfg.trainer.params.precision),
        "scene_count": len(merged),
        "world_size": world_size,
        "proposal_predictions_path": str(prediction_path.resolve()),
        "proposal_predictions_sha256": _sha256(prediction_path),
        "fields": {
            key: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in sample.items()
        },
        "inference_inputs_only": True,
        "future_target_present": False,
        "official_score_input_present": False,
    }
    reference_path_value = cfg.get("proposal_audit_reference_predictions_path")
    if reference_path_value:
        reference_path = Path(str(reference_path_value))
        with reference_path.open("rb") as file:
            reference_predictions = pickle.load(file)
        cache_manifest["reference_predictions_path"] = str(reference_path.resolve())
        cache_manifest["reference_predictions_sha256"] = _sha256(reference_path)
        cache_manifest["reference_parity"] = _compare_prediction_banks(
            merged,
            reference_predictions,
            allow_reference_superset=bool(limit_scenes > 0),
        )
        if not cache_manifest["reference_parity"]["passes_1e_8"]:
            raise RuntimeError(
                "Exported proposal bank differs from the locked FP32 public bank: "
                f"{cache_manifest['reference_parity']}"
            )
    (output_dir / "proposal_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )
    if bool(cfg.get("proposal_audit_skip_cpu_scoring", False)):
        print(json.dumps(cache_manifest, indent=2, sort_keys=True), flush=True)
        return

    log_for_token = {
        token: log_name
        for log_name, tokens in scene_loader.get_tokens_list_per_log().items()
        for token in tokens
    }
    predictions_by_log: Dict[str, Dict[str, Any]] = {}
    for token, prediction in merged.items():
        log_name = log_for_token[token]
        predictions_by_log.setdefault(log_name, {})[token] = prediction

    data_points = [
        {
            "cfg": cfg,
            "log_name": log_name,
            "predictions": log_predictions,
        }
        for log_name, log_predictions in sorted(predictions_by_log.items())
    ]
    worker = build_worker(cfg)
    # One log per Ray task avoids the long tail caused by pre-bundling two
    # large logs into a single worker invocation.
    worker_rows = worker.map(
        Task(fn=score_candidate_partition),
        [[item] for item in data_points],
    )
    rows = _flatten_worker_rows(worker_rows)
    rows.sort(key=lambda row: row["token"])

    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        raise RuntimeError("No valid proposal audit rows were produced")
    candidate_scores = np.stack(
        [row.pop("candidate_scores") for row in valid_rows], axis=0
    )
    candidate_factors = np.stack(
        [row.pop("candidate_factors") for row in valid_rows], axis=0
    )
    predicted_scores = np.stack(
        [row.pop("predicted_scores") for row in valid_rows], axis=0
    )
    np.savez_compressed(
        output_dir / "candidate_scores.npz",
        tokens=np.asarray([row["token"] for row in valid_rows]),
        log_names=np.asarray([row["log_name"] for row in valid_rows]),
        candidate_scores=candidate_scores,
        candidate_factors=candidate_factors,
        candidate_factor_names=np.asarray(FACTOR_NAMES),
        predicted_scores=predicted_scores,
        selected_indices=np.asarray(
            [row["selected_index"] for row in valid_rows], dtype=np.int16
        ),
        oracle_indices=np.asarray(
            [row["oracle_index"] for row in valid_rows], dtype=np.int16
        ),
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_scene_candidate_quality.csv", index=False)
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
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": str(cfg.experiment_name),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "agent_target": str(cfg.agent._target_),
        "split": "navtest",
        "scene_count": len(rows),
        "valid_scene_count": len(valid_rows),
        "invalid_scene_count": len(rows) - len(valid_rows),
        "log_count": len({row["log_name"] for row in valid_rows}),
        "candidate_count": int(candidate_scores.shape[1]),
        "metrics": {
            key: float(np.mean([row[key] for row in valid_rows]))
            for key in numeric_keys
        },
        "max_selected_score_parity_abs": float(
            max(row["selected_score_parity_abs"] for row in valid_rows)
        ),
        "artifacts": {
            "per_scene_csv": str(output_dir / "per_scene_candidate_quality.csv"),
            "candidate_scores_npz": str(output_dir / "candidate_scores.npz"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
