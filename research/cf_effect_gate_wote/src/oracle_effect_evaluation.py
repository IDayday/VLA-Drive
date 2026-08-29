"""Train and evaluate the pre-registered Gate2O v2 probe family.

This module stops at oracle-effect representation value.  It contains no
forward effect predictor, inverse dynamics, trajectory refinement, or policy
distillation path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from torch import Tensor

from .effect_tokenizer import MODEL_VARIANTS
from .feature_store import FeatureShardReader, atomic_write_json
from .metrics import candidate_ranks, paired_scene_bootstrap, pairwise_ranking_accuracy
from .models.structured_six_factor_probe import (
    CHECKPOINT_SCHEMA,
    SIX_FACTOR_ORDER,
    StructuredSixFactorProbe,
    checkpoint_payload,
    load_v2_checkpoint,
    six_factor_probe_loss,
    trainable_parameter_count,
)
from .oracle_effect_data import RawProbeBatch, iter_raw_batches


WEIGHT_DECAY = 1.0e-4
BOOTSTRAP_SEED = 20260827
INTERVENTIONS = (
    "none",
    "full_effect_swap",
    "actor_only_swap",
    "static_only_swap",
    "scene_mean_effect",
)


class OracleEffectEvaluationError(RuntimeError):
    """The v2 training/evaluation contract failed closed."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise OracleEffectEvaluationError("CUDA requested but unavailable")
    return result


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move_batch(batch: RawProbeBatch, device: torch.device) -> RawProbeBatch:
    return RawProbeBatch(
        tokens=batch.tokens,
        trajectory=batch.trajectory.to(device),
        ego_status=batch.ego_status.to(device),
        current_bev_tokens=batch.current_bev_tokens.to(device),
        auxiliary_tokens=batch.auxiliary_tokens.to(device),
        factor_labels=batch.factor_labels.to(device),
        score_labels=batch.score_labels.to(device),
        candidate_indices=batch.candidate_indices.to(device),
        pair_indices=batch.pair_indices.to(device),
        wote_selected_indices=batch.wote_selected_indices,
    )


def _forward(model: StructuredSixFactorProbe, batch: RawProbeBatch) -> Mapping[str, Tensor]:
    return model(
        batch.trajectory,
        batch.ego_status,
        batch.current_bev_tokens,
        batch.auxiliary_tokens,
    )


@torch.inference_mode()
def validation_metrics(
    model: StructuredSixFactorProbe,
    *,
    frozen_root: Path,
    effect_root: Path,
    label_root: Path,
    model_type: str,
    batch_scenes: int,
    device: torch.device,
    scene_limit: int | None = None,
) -> tuple[float, float]:
    model.eval()
    selected_sum = 0.0
    regret_sum = 0.0
    scenes = 0
    for raw in iter_raw_batches(
        frozen_root=frozen_root,
        effect_root=effect_root,
        label_root=label_root,
        model_type=model_type,
        batch_scenes=batch_scenes,
        seed=0,
        epoch=0,
        full_candidates=True,
        scene_limit=scene_limit,
    ):
        batch = _move_batch(raw, device)
        predicted = _forward(model, batch)["score"]
        selected = predicted.argmax(dim=1)
        rows = torch.arange(len(selected), device=device)
        selected_true = batch.score_labels[rows, selected]
        oracle = batch.score_labels.max(dim=1).values
        selected_sum += float(selected_true.sum().cpu())
        regret_sum += float((oracle - selected_true).sum().cpu())
        scenes += len(selected)
    if scenes == 0:
        raise OracleEffectEvaluationError("validation split is empty")
    return regret_sum / scenes, selected_sum / scenes


def train_trial(
    *,
    train_cache: Path,
    val_cache: Path,
    train_effects: Path,
    val_effects: Path,
    train_labels: Path,
    val_labels: Path,
    model_type: str,
    seed: int,
    learning_rate: float,
    pairwise_weight: float,
    max_epochs: int,
    patience: int,
    batch_scenes: int,
    device: torch.device,
    output: Path,
    train_scene_limit: int | None = None,
    val_scene_limit: int | None = None,
) -> Mapping[str, Any]:
    if model_type not in MODEL_VARIANTS:
        raise OracleEffectEvaluationError(f"unknown model type: {model_type}")
    _seed_everything(seed)
    model = StructuredSixFactorProbe().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    best_regret = float("inf")
    best_selected = -float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    steps = 0
    history: list[dict[str, float]] = []
    peak_memory = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(max_epochs):
        model.train()
        loss_sum = 0.0
        factor_sum = 0.0
        pair_sum = 0.0
        epoch_steps = 0
        for raw in iter_raw_batches(
            frozen_root=train_cache,
            effect_root=train_effects,
            label_root=train_labels,
            model_type=model_type,
            batch_scenes=batch_scenes,
            seed=seed,
            epoch=epoch,
            full_candidates=False,
            scene_limit=train_scene_limit,
        ):
            batch = _move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            losses = six_factor_probe_loss(
                prediction["logits"],
                batch.factor_labels,
                prediction["score"],
                batch.score_labels,
                batch.pair_indices,
                pairwise_weight,
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    f"{model_type} seed={seed} epoch={epoch}: non-finite loss"
                )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(losses["total"].detach().cpu())
            factor_sum += float(losses["factor"].detach().cpu())
            pair_sum += float(losses["pairwise"].detach().cpu())
            epoch_steps += 1
            steps += 1
        if epoch_steps == 0:
            raise OracleEffectEvaluationError("training split is empty")
        val_regret, val_selected = validation_metrics(
            model,
            frozen_root=val_cache,
            effect_root=val_effects,
            label_root=val_labels,
            model_type=model_type,
            batch_scenes=max(1, min(batch_scenes, 2)),
            device=device,
            scene_limit=val_scene_limit,
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": loss_sum / epoch_steps,
                "factor_loss": factor_sum / epoch_steps,
                "pairwise_loss": pair_sum / epoch_steps,
                "validation_regret": val_regret,
                "validation_selected_score": val_selected,
            }
        )
        improved = val_regret < best_regret - 1.0e-9 or (
            abs(val_regret - best_regret) <= 1.0e-9 and val_selected > best_selected
        )
        if improved:
            best_regret = val_regret
            best_selected = val_selected
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise OracleEffectEvaluationError("early stopping captured no checkpoint")
    model.load_state_dict(best_state, strict=True)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    metadata = {
        "learning_rate": float(learning_rate),
        "pairwise_weight": float(pairwise_weight),
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best_epoch),
        "training_steps": int(steps),
        "validation_regret": float(best_regret),
        "validation_selected_score": float(best_selected),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "batch_scenes": int(batch_scenes),
        "train_candidates_per_scene": 64,
        "train_scene_limit": train_scene_limit,
        "val_scene_limit": val_scene_limit,
        "peak_gpu_memory_bytes": peak_memory,
        "history": history,
    }
    payload = checkpoint_payload(
        model, model_type=model_type, seed=seed, metadata=metadata
    )
    _atomic_torch_save(output, payload)
    return {
        "checkpoint": str(output),
        "schema_version": CHECKPOINT_SCHEMA,
        "model_type": model_type,
        "seed": seed,
        "learning_rate": learning_rate,
        "pairwise_weight": pairwise_weight,
        "best_epoch": best_epoch,
        "training_steps": steps,
        "validation_regret": best_regret,
        "validation_selected_score": best_selected,
        "trainable_parameter_count": trainable_parameter_count(model),
        "peak_gpu_memory_bytes": peak_memory,
    }


def two_scene_overfit(args: argparse.Namespace) -> Mapping[str, Any]:
    device = _device(args.device)
    results: list[dict[str, Any]] = []
    for model_type in ("direct_current", "full_primitive_action_effect"):
        _seed_everything(0)
        raw = next(
            iter_raw_batches(
                frozen_root=args.train_cache,
                effect_root=args.train_effects,
                label_root=args.train_labels,
                model_type=model_type,
                batch_scenes=2,
                seed=0,
                epoch=0,
                full_candidates=True,
                scene_limit=2,
            )
        )
        batch = _move_batch(raw, device)
        model = StructuredSixFactorProbe().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=WEIGHT_DECAY)
        losses: list[float] = []
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            objective = six_factor_probe_loss(
                prediction["logits"],
                batch.factor_labels,
                prediction["score"],
                batch.score_labels,
                batch.pair_indices,
                0.5,
            )["total"]
            if not torch.isfinite(objective):
                raise FloatingPointError("two-scene overfit produced NaN/Inf")
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach().cpu()))
        results.append(
            {
                "model_type": model_type,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_decreased": losses[-1] < losses[0],
                "output_shape": list(_forward(model, batch)["factors"].shape),
                "finite": bool(np.isfinite(losses).all()),
            }
        )
    status = all(
        row["loss_decreased"]
        and row["finite"]
        and row["output_shape"] == [2, 256, 6]
        for row in results
    )
    payload = {"status": "PASS" if status else "FAIL", "models": results}
    atomic_write_json(args.output, payload)
    if not status:
        raise OracleEffectEvaluationError("two-scene overfit smoke failed")
    return payload


def hyperparameter_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing existing pilot output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = args.output.parent / "hyperparameter_pilot_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    learning_rates = [float(value) for value in config["pilot"]["learning_rates"]]
    pairwise_weights = [float(value) for value in config["pilot"]["pairwise_weights"]]
    device = _device(args.device)
    trials: list[Mapping[str, Any]] = []
    for learning_rate, pairwise_weight, model_type in itertools.product(
        learning_rates,
        pairwise_weights,
        ("direct_current", "full_primitive_action_effect"),
    ):
        checkpoint = (
            artifact_root
            / "checkpoints"
            / f"{model_type}-lr-{learning_rate:g}-pair-{pairwise_weight:g}.pt"
        )
        if checkpoint.exists():
            existing = _validate_existing_checkpoint(
                checkpoint, model_type, 0, learning_rate, pairwise_weight
            )
            metadata = existing["metadata"]
            result = {
                "checkpoint": str(checkpoint),
                "schema_version": CHECKPOINT_SCHEMA,
                "model_type": model_type,
                "seed": 0,
                "learning_rate": learning_rate,
                "pairwise_weight": pairwise_weight,
                "best_epoch": int(metadata["best_epoch"]),
                "training_steps": int(metadata["training_steps"]),
                "validation_regret": float(metadata["validation_regret"]),
                "validation_selected_score": float(
                    metadata["validation_selected_score"]
                ),
                "trainable_parameter_count": int(
                    existing["trainable_parameter_count"]
                ),
                "peak_gpu_memory_bytes": int(
                    metadata.get("peak_gpu_memory_bytes", 0)
                ),
                "resumed": True,
            }
        else:
            result = train_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=args.train_effects,
                val_effects=args.val_effects,
                train_labels=args.train_labels,
                val_labels=args.val_labels,
                model_type=model_type,
                seed=0,
                learning_rate=learning_rate,
                pairwise_weight=pairwise_weight,
                max_epochs=int(config["pilot"]["max_epochs"]),
                patience=int(config["pilot"]["patience"]),
                batch_scenes=int(config["training"]["batch_scenes"]),
                device=device,
                output=checkpoint,
                train_scene_limit=256,
                val_scene_limit=64,
            )
        trials.append(result)
    grouped: list[dict[str, Any]] = []
    for learning_rate, pairwise_weight in itertools.product(learning_rates, pairwise_weights):
        matched = [
            row
            for row in trials
            if float(row["learning_rate"]) == learning_rate
            and float(row["pairwise_weight"]) == pairwise_weight
        ]
        if {str(row["model_type"]) for row in matched} != {
            "direct_current",
            "full_primitive_action_effect",
        }:
            raise OracleEffectEvaluationError("pilot grid is incomplete")
        grouped.append(
            {
                "learning_rate": learning_rate,
                "pairwise_weight": pairwise_weight,
                "mean_validation_regret": float(
                    np.mean([float(row["validation_regret"]) for row in matched])
                ),
                "direct_validation_regret": float(
                    next(
                        row["validation_regret"]
                        for row in matched
                        if row["model_type"] == "direct_current"
                    )
                ),
                "full_validation_regret": float(
                    next(
                        row["validation_regret"]
                        for row in matched
                        if row["model_type"] == "full_primitive_action_effect"
                    )
                ),
            }
        )
    selected = min(
        grouped,
        key=lambda row: (
            row["mean_validation_regret"],
            row["learning_rate"],
            row["pairwise_weight"],
        ),
    )
    payload = {
        "schema_version": "oracle_effect_global_hyperparameters.v2",
        "selection_data": {"train_scenes": 256, "val_scenes": 64, "seed": 0},
        "models": ["direct_current", "full_primitive_action_effect"],
        "grid": grouped,
        "selected": selected,
        "test_metrics_used": False,
        "tie_break": ["lower_learning_rate", "lower_pairwise_weight"],
        "trials": trials,
    }
    atomic_write_json(args.output, payload)
    return payload


def _validate_existing_checkpoint(
    path: Path, model_type: str, seed: int, learning_rate: float, pairwise_weight: float
) -> Mapping[str, Any]:
    _, payload = load_v2_checkpoint(path)
    metadata = payload["metadata"]
    expected = {
        "model_type": model_type,
        "seed": seed,
        "learning_rate": learning_rate,
        "pairwise_weight": pairwise_weight,
    }
    actual = {
        "model_type": payload["model_type"],
        "seed": int(payload["seed"]),
        "learning_rate": float(metadata["learning_rate"]),
        "pairwise_weight": float(metadata["pairwise_weight"]),
    }
    if actual != expected:
        raise OracleEffectEvaluationError(
            f"existing checkpoint identity mismatch: {path}: {actual} != {expected}"
        )
    return payload


def train_main(args: argparse.Namespace) -> Mapping[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.hyperparameters.read_text(encoding="utf-8"))["selected"]
    learning_rate = float(selection["learning_rate"])
    pairwise_weight = float(selection["pairwise_weight"])
    seeds = [int(value) for value in config["run"]["seeds"]]
    models = tuple(args.models or config["probe"]["models"])
    if set(models) - set(MODEL_VARIANTS):
        raise OracleEffectEvaluationError("main training includes an unregistered model")
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    trials: list[Mapping[str, Any]] = []
    for model_type, seed in itertools.product(models, seeds):
        checkpoint = args.output / "checkpoints" / model_type / f"seed-{seed}.pt"
        if checkpoint.exists():
            payload = _validate_existing_checkpoint(
                checkpoint, model_type, seed, learning_rate, pairwise_weight
            )
            metadata = payload["metadata"]
            trials.append(
                {
                    "checkpoint": str(checkpoint),
                    "schema_version": CHECKPOINT_SCHEMA,
                    "model_type": model_type,
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "pairwise_weight": pairwise_weight,
                    "best_epoch": int(metadata["best_epoch"]),
                    "training_steps": int(metadata["training_steps"]),
                    "validation_regret": float(metadata["validation_regret"]),
                    "validation_selected_score": float(metadata["validation_selected_score"]),
                    "trainable_parameter_count": int(payload["trainable_parameter_count"]),
                    "resumed": True,
                }
            )
            continue
        trials.append(
            train_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=args.train_effects,
                val_effects=args.val_effects,
                train_labels=args.train_labels,
                val_labels=args.val_labels,
                model_type=model_type,
                seed=seed,
                learning_rate=learning_rate,
                pairwise_weight=pairwise_weight,
                max_epochs=int(config["training"]["max_epochs"]),
                patience=int(config["training"]["patience"]),
                batch_scenes=int(config["training"]["batch_scenes"]),
                device=device,
                output=checkpoint,
            )
        )
    counts = {int(row["trainable_parameter_count"]) for row in trials}
    if len(counts) != 1:
        raise OracleEffectEvaluationError(f"A-L parameter counts differ: {counts}")
    manifest = {
        "schema_version": "oracle_effect_training.v2",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "models": list(models),
        "seeds": seeds,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "pairwise_weight": pairwise_weight,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": int(config["training"]["max_epochs"]),
        "patience": int(config["training"]["patience"]),
        "early_stopping_metric": "validation_top1_regret",
        "train_candidates_per_scene": 64,
        "validation_candidates_per_scene": 256,
        "trainable_parameter_count": next(iter(counts)),
        "trials": trials,
    }
    manifest_path = args.output / "training_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    else:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise OracleEffectEvaluationError("existing training manifest differs")
    return manifest


@dataclass(frozen=True)
class EvaluationArrays:
    tokens: tuple[str, ...]
    predicted_scores: npt.NDArray[np.float32]
    predicted_factors: npt.NDArray[np.float32]
    true_scores: npt.NDArray[np.float32]
    true_factors: npt.NDArray[np.float32]
    wote_selected: npt.NDArray[np.int64]


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    *,
    test_cache: Path,
    test_effects: Path,
    test_labels: Path,
    device: torch.device,
    intervention: str = "none",
    forced_model_type: str | None = None,
) -> tuple[EvaluationArrays, Mapping[str, Any]]:
    model, payload = load_v2_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    model_type = forced_model_type or str(payload["model_type"])
    tokens: list[str] = []
    predicted_scores: list[np.ndarray] = []
    predicted_factors: list[np.ndarray] = []
    true_scores: list[np.ndarray] = []
    true_factors: list[np.ndarray] = []
    selected: list[int] = []
    for raw in iter_raw_batches(
        frozen_root=test_cache,
        effect_root=test_effects,
        label_root=test_labels,
        model_type=model_type,
        batch_scenes=1,
        seed=int(payload["seed"]),
        epoch=0,
        full_candidates=True,
        intervention=intervention,
    ):
        batch = _move_batch(raw, device)
        output = _forward(model, batch)
        tokens.extend(raw.tokens)
        predicted_scores.append(output["score"].float().cpu().numpy())
        predicted_factors.append(output["factors"].float().cpu().numpy())
        true_scores.append(raw.score_labels.float().numpy())
        true_factors.append(raw.factor_labels.float().numpy())
        selected.extend(int(value) for value in raw.wote_selected_indices)
    arrays = EvaluationArrays(
        tokens=tuple(tokens),
        predicted_scores=np.concatenate(predicted_scores, axis=0),
        predicted_factors=np.concatenate(predicted_factors, axis=0),
        true_scores=np.concatenate(true_scores, axis=0),
        true_factors=np.concatenate(true_factors, axis=0),
        wote_selected=np.asarray(selected, dtype=np.int64),
    )
    if arrays.predicted_scores.shape != (512, 256):
        raise OracleEffectEvaluationError(
            f"test evaluation expected [512,256], got {arrays.predicted_scores.shape}"
        )
    return arrays, payload


def _safe_auc(target: npt.ArrayLike, prediction: npt.ArrayLike) -> float:
    truth = np.asarray(target, dtype=np.int64)
    values = np.asarray(prediction, dtype=np.float64)
    return float(roc_auc_score(truth, values)) if len(np.unique(truth)) == 2 else float("nan")


def _kendall_mean(predicted: np.ndarray, target: np.ndarray) -> float:
    values = [
        float(kendalltau(p, t, nan_policy="raise").statistic)
        for p, t in zip(predicted, target)
    ]
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _model_rows(
    arrays: EvaluationArrays,
    model_type: str,
    seed: int,
    intervention: str,
) -> tuple[pd.DataFrame, Mapping[str, Any], list[dict[str, Any]]]:
    predicted = arrays.predicted_scores
    target = arrays.true_scores
    factors = arrays.true_factors
    selected = predicted.argmax(axis=1)
    rows = np.arange(len(selected))
    selected_score = target[rows, selected]
    selected_factor = factors[rows, selected]
    oracle = target.max(axis=1)
    ranks = candidate_ranks(target)[rows, selected]
    wote_score = target[rows, arrays.wote_selected]
    gap = oracle - wote_score
    capture = np.full(len(target), np.nan, dtype=np.float64)
    applicable = gap >= 1.0e-6
    capture[applicable] = (
        selected_score[applicable] - wote_score[applicable]
    ) / gap[applicable]
    applicable_capture = capture[applicable]
    hard_false_safe = (
        (selected_factor[:, 0] == 0)
        | (selected_factor[:, 1] == 0)
        | (selected_factor[:, 2] == 0)
        | (selected_factor[:, 4] == 0)
    )
    zero_wote = wote_score == 0
    outcomes = pd.DataFrame(
        {
            "scene_token": arrays.tokens,
            "model_type": model_type,
            "seed": seed,
            "intervention": intervention,
            "selected_index": selected,
            "wote_selected_index": arrays.wote_selected,
            "selected_score": selected_score,
            "oracle_score": oracle,
            "wote_selected_score": wote_score,
            "regret": oracle - selected_score,
            "candidate_rank": ranks,
            "oracle_capture": capture,
            "hard_false_safe": hard_false_safe,
            "direction_non_compliance": selected_factor[:, 2] < 1.0,
            "ddc_half": selected_factor[:, 2] == 0.5,
            "ddc_zero": selected_factor[:, 2] == 0.0,
            "zero_score_selection": selected_score == 0.0,
            "wote_failure_recovery": zero_wote & (selected_score > 0),
            **{
                f"selected_{name}": selected_factor[:, index]
                for index, name in enumerate(SIX_FACTOR_ORDER)
            },
        }
    )
    metric = {
        "model_type": model_type,
        "seed": seed,
        "intervention": intervention,
        "selected_score": float(selected_score.mean()),
        "top1_regret": float((oracle - selected_score).mean()),
        "mean_selected_candidate_rank": float(ranks.mean()),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(predicted, target),
        "kendall_tau": _kendall_mean(predicted, target),
        "oracle_capture_mean": (
            float(np.mean(applicable_capture)) if len(applicable_capture) else float("nan")
        ),
        "oracle_capture_median": (
            float(np.median(applicable_capture)) if len(applicable_capture) else float("nan")
        ),
        "oracle_capture_q25": (
            float(np.quantile(applicable_capture, 0.25))
            if len(applicable_capture)
            else float("nan")
        ),
        "oracle_capture_q75": (
            float(np.quantile(applicable_capture, 0.75))
            if len(applicable_capture)
            else float("nan")
        ),
        "oracle_capture_fraction_gt_0": (
            float(np.mean(applicable_capture > 0))
            if len(applicable_capture)
            else float("nan")
        ),
        "oracle_capture_fraction_ge_0_5": (
            float(np.mean(applicable_capture >= 0.5))
            if len(applicable_capture)
            else float("nan")
        ),
        "hard_false_safe_rate": float(hard_false_safe.mean()),
        "direction_non_compliance_rate": float((selected_factor[:, 2] < 1).mean()),
        "ddc_half_rate": float((selected_factor[:, 2] == 0.5).mean()),
        "ddc_zero_rate": float((selected_factor[:, 2] == 0).mean()),
        "zero_score_selection_rate": float((selected_score == 0).mean()),
        "wote_failure_recovery_rate": float(
            (zero_wote & (selected_score > 0)).sum() / max(int(zero_wote.sum()), 1)
        ),
    }
    factor_rows: list[dict[str, Any]] = []
    for factor_index, factor_name in enumerate(SIX_FACTOR_ORDER):
        truth = factors[..., factor_index].reshape(-1)
        prediction = arrays.predicted_factors[..., factor_index].reshape(-1)
        row: dict[str, Any] = {
            "model_type": model_type,
            "seed": seed,
            "factor": factor_name,
            "mae": float(np.mean(np.abs(prediction - truth))),
            "brier": float(np.mean((prediction - truth) ** 2)),
            "auc": float("nan"),
            "strict_compliance_auc": float("nan"),
            "violation_auc": float("nan"),
        }
        if factor_name in {"NC", "DAC", "TTC", "Comfort"}:
            row["auc"] = _safe_auc(truth == 1.0, prediction)
        if factor_name == "DDC":
            row["strict_compliance_auc"] = _safe_auc(truth == 1.0, prediction)
            row["violation_auc"] = _safe_auc(truth < 1.0, -prediction)
        factor_rows.append(row)
    return outcomes, metric, factor_rows


def _wote_baseline(arrays: EvaluationArrays) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    rows = np.arange(len(arrays.tokens))
    selected = arrays.wote_selected
    selected_score = arrays.true_scores[rows, selected]
    oracle = arrays.true_scores.max(axis=1)
    selected_factor = arrays.true_factors[rows, selected]
    ranks = candidate_ranks(arrays.true_scores)[rows, selected]
    outcome = pd.DataFrame(
        {
            "scene_token": arrays.tokens,
            "model_type": "wote_base_selector",
            "seed": -1,
            "intervention": "none",
            "selected_index": selected,
            "wote_selected_index": selected,
            "selected_score": selected_score,
            "oracle_score": oracle,
            "wote_selected_score": selected_score,
            "regret": oracle - selected_score,
            "candidate_rank": ranks,
            "oracle_capture": np.where(oracle - selected_score >= 1e-6, 0.0, np.nan),
            "hard_false_safe": (
                (selected_factor[:, 0] == 0)
                | (selected_factor[:, 1] == 0)
                | (selected_factor[:, 2] == 0)
                | (selected_factor[:, 4] == 0)
            ),
            "direction_non_compliance": selected_factor[:, 2] < 1,
            "ddc_half": selected_factor[:, 2] == 0.5,
            "ddc_zero": selected_factor[:, 2] == 0,
            "zero_score_selection": selected_score == 0,
            "wote_failure_recovery": False,
            **{
                f"selected_{name}": selected_factor[:, index]
                for index, name in enumerate(SIX_FACTOR_ORDER)
            },
        }
    )
    metric = {
        "model_type": "wote_base_selector",
        "seed": -1,
        "intervention": "none",
        "selected_score": float(selected_score.mean()),
        "top1_regret": float((oracle - selected_score).mean()),
        "mean_selected_candidate_rank": float(ranks.mean()),
        "pairwise_ranking_accuracy": float("nan"),
        "kendall_tau": float("nan"),
        "oracle_capture_mean": 0.0,
        "oracle_capture_median": 0.0,
        "oracle_capture_q25": 0.0,
        "oracle_capture_q75": 0.0,
        "oracle_capture_fraction_gt_0": 0.0,
        "oracle_capture_fraction_ge_0_5": 0.0,
        "hard_false_safe_rate": float(outcome["hard_false_safe"].mean()),
        "direction_non_compliance_rate": float((selected_factor[:, 2] < 1).mean()),
        "ddc_half_rate": float((selected_factor[:, 2] == 0.5).mean()),
        "ddc_zero_rate": float((selected_factor[:, 2] == 0).mean()),
        "zero_score_selection_rate": float((selected_score == 0).mean()),
        "wote_failure_recovery_rate": 0.0,
    }
    return outcome, metric


def _aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in per_seed.columns
        if column not in {"model_type", "seed", "intervention"}
    ]
    rows: list[dict[str, Any]] = []
    for (model_type, intervention), group in per_seed.groupby(
        ["model_type", "intervention"], sort=False
    ):
        row: dict[str, Any] = {
            "model_type": model_type,
            "intervention": intervention,
            "seeds": int(group["seed"].nunique()),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{column}_std"] = (
                float(values.std(ddof=0)) if values.notna().any() else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _interaction_rich_flags(effect_root: Path) -> Mapping[str, bool]:
    reader = FeatureShardReader(effect_root, verify_shard_hashes=False)
    flags: dict[str, bool] = {}
    for sidecar, arrays in reader.iter_shards(
        ("primitive_actor_mask", "primitive_interaction_mask")
    ):
        for index, record in enumerate(sidecar["records"]):
            validity = np.asarray(arrays["primitive_actor_mask"][index], dtype=bool)
            interaction = np.asarray(
                arrays["primitive_interaction_mask"][index], dtype=bool
            )
            if validity.shape != (256, 8, 16) or interaction.shape != validity.shape:
                raise OracleEffectEvaluationError("invalid interaction subset masks")
            count = (validity & interaction).sum(axis=(1, 2))
            flags[str(record["scene_token"])] = bool((count >= 2).mean() >= 0.10)
    return flags


def _interaction_subset_table(scene_frame: pd.DataFrame) -> pd.DataFrame:
    base = scene_frame[
        (scene_frame["intervention"] == "none")
        & scene_frame["model_type"].isin(
            [
                "direct_current",
                "static_primitive_effect",
                "full_primitive_action_effect",
            ]
        )
    ]
    averaged = (
        base.groupby(["scene_token", "model_type", "interaction_rich"], as_index=False)[
            "selected_score"
        ]
        .mean()
        .pivot(index=["scene_token", "interaction_rich"], columns="model_type", values="selected_score")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for subset, selected in (
        ("all", averaged),
        ("interaction_rich", averaged[averaged["interaction_rich"]]),
        ("non_interaction", averaged[~averaged["interaction_rich"]]),
    ):
        if len(selected) == 0:
            rows.append({"subset": subset, "scenes": 0})
            continue
        full = selected["full_primitive_action_effect"].to_numpy(dtype=np.float64)
        static = selected["static_primitive_effect"].to_numpy(dtype=np.float64)
        direct = selected["direct_current"].to_numpy(dtype=np.float64)
        full_static = paired_scene_bootstrap(
            full, static, samples=5000, confidence=0.95, seed=BOOTSTRAP_SEED
        )
        full_direct = paired_scene_bootstrap(
            full, direct, samples=5000, confidence=0.95, seed=BOOTSTRAP_SEED
        )
        rows.append(
            {
                "subset": subset,
                "scenes": len(selected),
                "full_score": float(full.mean()),
                "static_score": float(static.mean()),
                "direct_score": float(direct.mean()),
                "full_vs_static_gain": float((full - static).mean()),
                "full_vs_static_ci_lower": full_static.lower,
                "full_vs_static_ci_upper": full_static.upper,
                "full_vs_direct_gain": float((full - direct).mean()),
                "full_vs_direct_ci_lower": full_direct.lower,
                "full_vs_direct_ci_upper": full_direct.upper,
                "definition": "fraction(interaction_count>=2)>=0.10",
                "label_free": True,
            }
        )
    return pd.DataFrame(rows)


def evaluate_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    device = _device(args.device)
    manifest = json.loads((args.training / "training_manifest.json").read_text(encoding="utf-8"))
    trials = manifest["trials"]
    outcomes: list[pd.DataFrame] = []
    metrics: list[Mapping[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    reference_arrays: EvaluationArrays | None = None
    for trial in trials:
        checkpoint = Path(trial["checkpoint"])
        arrays, payload = evaluate_checkpoint(
            checkpoint,
            test_cache=args.test_cache,
            test_effects=args.test_effects,
            test_labels=args.test_labels,
            device=device,
        )
        if reference_arrays is None:
            reference_arrays = arrays
        elif arrays.tokens != reference_arrays.tokens or not np.array_equal(
            arrays.true_scores, reference_arrays.true_scores
        ):
            raise OracleEffectEvaluationError("test identity changed across models")
        frame, metric, factors = _model_rows(
            arrays, str(payload["model_type"]), int(payload["seed"]), "none"
        )
        outcomes.append(frame)
        metrics.append(metric)
        factor_rows.extend(factors)

    if reference_arrays is None:
        raise OracleEffectEvaluationError("training manifest had no trials")
    baseline_frame, baseline_metric = _wote_baseline(reference_arrays)
    outcomes.append(baseline_frame)
    metrics.append(baseline_metric)

    # Candidate-specificity interventions use the trained G checkpoint only.
    full_trials = [
        row for row in trials if row["model_type"] == "full_primitive_action_effect"
    ]
    if len(full_trials) != 3:
        raise OracleEffectEvaluationError("expected three full-primitive checkpoints")
    intervention_metrics: list[Mapping[str, Any]] = []
    intervention_outcomes: list[pd.DataFrame] = []
    for trial in full_trials:
        for intervention in INTERVENTIONS[1:]:
            arrays, payload = evaluate_checkpoint(
                Path(trial["checkpoint"]),
                test_cache=args.test_cache,
                test_effects=args.test_effects,
                test_labels=args.test_labels,
                device=device,
                intervention=intervention,
            )
            frame, metric, _ = _model_rows(
                arrays,
                "full_primitive_action_effect",
                int(payload["seed"]),
                intervention,
            )
            intervention_outcomes.append(frame)
            intervention_metrics.append(metric)
    outcomes.extend(intervention_outcomes)
    metrics.extend(intervention_metrics)

    scene_frame = pd.concat(outcomes, ignore_index=True)
    interaction_flags = _interaction_rich_flags(args.test_effects)
    scene_frame["interaction_rich"] = scene_frame["scene_token"].map(interaction_flags)
    if scene_frame["interaction_rich"].isna().any():
        raise OracleEffectEvaluationError("interaction subset missing test scene tokens")
    per_seed = pd.DataFrame(metrics)
    aggregate = _aggregate_metrics(per_seed)
    factor_frame = pd.DataFrame(factor_rows)
    scene_frame.to_parquet(args.output / "scene_level_results.parquet", index=False)
    per_seed.to_csv(args.output / "probe_metrics_per_seed.csv", index=False)
    aggregate.to_csv(args.output / "probe_metrics_aggregate.csv", index=False)
    factor_frame.to_csv(args.output / "factor_metrics.csv", index=False)

    intervention_table = aggregate[
        (aggregate["model_type"] == "full_primitive_action_effect")
        & (aggregate["intervention"].isin(INTERVENTIONS))
    ].copy()
    full_base = float(
        aggregate.loc[
            (aggregate["model_type"] == "full_primitive_action_effect")
            & (aggregate["intervention"] == "none"),
            "selected_score",
        ].iloc[0]
    )
    full_regret = float(
        aggregate.loc[
            (aggregate["model_type"] == "full_primitive_action_effect")
            & (aggregate["intervention"] == "none"),
            "top1_regret",
        ].iloc[0]
    )
    intervention_table["drop_vs_full"] = full_base - intervention_table["selected_score"]
    intervention_table["regret_increase"] = intervention_table["top1_regret"] - full_regret
    # H and I are formal checkpoints, never an ad-hoc mutation of G.
    for model_type, label in (
        ("full_primitive_no_interaction_mask", "no_interaction_mask"),
        ("interaction_mask_only", "interaction_mask_only"),
    ):
        row = aggregate[
            (aggregate["model_type"] == model_type)
            & (aggregate["intervention"] == "none")
        ].copy()
        row.loc[:, "intervention"] = label
        row.loc[:, "drop_vs_full"] = full_base - row["selected_score"]
        row.loc[:, "regret_increase"] = row["top1_regret"] - full_regret
        intervention_table = pd.concat([intervention_table, row], ignore_index=True)
    intervention_table.to_csv(args.output / "intervention_ablation.csv", index=False)
    _interaction_subset_table(scene_frame).to_csv(
        args.output / "interaction_subset.csv", index=False
    )

    oracle_capture = aggregate[
        [
            "model_type",
            "intervention",
            "oracle_capture_mean",
            "oracle_capture_median",
            "oracle_capture_q25",
            "oracle_capture_q75",
            "oracle_capture_fraction_gt_0",
            "oracle_capture_fraction_ge_0_5",
        ]
    ]
    oracle_capture.to_csv(args.output / "oracle_capture.csv", index=False)
    ddc = aggregate[
        [
            "model_type",
            "intervention",
            "direction_non_compliance_rate",
            "ddc_half_rate",
            "ddc_zero_rate",
        ]
    ]
    ddc.to_csv(args.output / "ddc_diagnostics.csv", index=False)
    failure = scene_frame[
        (scene_frame["intervention"] == "none")
        & (scene_frame["model_type"] != "wote_base_selector")
        & (scene_frame["regret"] >= 0.25)
    ].sort_values("regret", ascending=False)
    failure.head(500).to_csv(args.output / "failure_cases.csv", index=False)
    atomic_write_json(
        args.output / "evaluation_manifest.json",
        {
            "schema_version": "oracle_effect_evaluation.v2",
            "test_scenes": len(reference_arrays.tokens),
            "candidate_count": int(reference_arrays.true_scores.shape[1]),
            "trained_trials": len(trials),
            "intervention_trials": len(intervention_metrics),
            "model_types": sorted({str(row["model_type"]) for row in trials}),
            "seeds": sorted({int(row["seed"]) for row in trials}),
            "status": "PASS",
        },
    )


def _common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-effects", type=Path, required=True)
    parser.add_argument("--val-effects", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--device", default="cuda")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    overfit = commands.add_parser("overfit-smoke")
    overfit.add_argument("--train-cache", type=Path, required=True)
    overfit.add_argument("--train-effects", type=Path, required=True)
    overfit.add_argument("--train-labels", type=Path, required=True)
    overfit.add_argument("--device", default="cuda")
    overfit.add_argument("--steps", type=int, default=30)
    overfit.add_argument("--output", type=Path, required=True)
    pilot = commands.add_parser("pilot")
    _common_training_arguments(pilot)
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train-main")
    _common_training_arguments(train)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--hyperparameters", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--models", nargs="+", choices=MODEL_VARIANTS)
    evaluate = commands.add_parser("evaluate-suite")
    evaluate.add_argument("--training", type=Path, required=True)
    evaluate.add_argument("--test-cache", type=Path, required=True)
    evaluate.add_argument("--test-effects", type=Path, required=True)
    evaluate.add_argument("--test-labels", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "overfit-smoke":
        two_scene_overfit(args)
    elif args.command == "pilot":
        hyperparameter_pilot(args)
    elif args.command == "train-main":
        train_main(args)
    elif args.command == "evaluate-suite":
        evaluate_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
