"""Train and evaluate G4 inverse identifiability and frozen planning gates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
from torch import Tensor

from .evaluate_probe import evaluate_checkpoint
from .feature_store import atomic_write_json
from .metrics import paired_scene_bootstrap, pdms_from_factors
from .models.inverse_probe import (
    DELTA_SCALES,
    INVERSE_MODES,
    InverseProbe,
    farthest_point_candidates,
    inverse_training_loss,
    pack_inverse_effect,
    pack_trajectory,
    trajectory_delta_descriptors,
)
from .train_probe import _seed_everything, iter_probe_scenes


@dataclass(frozen=True)
class InverseBatch:
    tokens: tuple[str, ...]
    effects: Tensor
    trajectories: Tensor
    pair_first: Tensor
    pair_second: Tensor
    delta_targets: Tensor


@dataclass(frozen=True)
class InverseDataset:
    tokens: tuple[str, ...]
    effects: npt.NDArray[np.float32]
    trajectories: npt.NDArray[np.float32]
    pair_first: npt.NDArray[np.int64]
    pair_second: npt.NDArray[np.int64]
    delta_targets: npt.NDArray[np.float32]


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite inverse checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pairs(candidate_count: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    first, second = np.triu_indices(candidate_count, k=1)
    return first.astype(np.int64), second.astype(np.int64)


def _inverse_scene(
    scene: Any,
    mode: str,
    retrieval_candidates: int,
    time_permutation: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[Any], ...]:
    if scene.effects is None:
        raise ValueError(f"{scene.token}: inverse requires replay effects")
    trajectory_all = np.asarray(scene.frozen["trajectory"], dtype=np.float32)
    selected = farthest_point_candidates(trajectory_all, retrieval_candidates)
    effect_all = pack_inverse_effect(
        scene.effects, mode, time_permutation=time_permutation
    )
    trajectory = trajectory_all[selected]
    first, second = _pairs(retrieval_candidates)
    delta = trajectory_delta_descriptors(trajectory, first, second)
    return effect_all[selected], pack_trajectory(trajectory), first, second, delta


def _stack_inverse_batch(
    values: Sequence[tuple[str, tuple[npt.NDArray[Any], ...]]]
) -> InverseBatch:
    return InverseBatch(
        tokens=tuple(item[0] for item in values),
        effects=torch.from_numpy(np.stack([item[1][0] for item in values])),
        trajectories=torch.from_numpy(np.stack([item[1][1] for item in values])),
        pair_first=torch.from_numpy(np.stack([item[1][2] for item in values])),
        pair_second=torch.from_numpy(np.stack([item[1][3] for item in values])),
        delta_targets=torch.from_numpy(np.stack([item[1][4] for item in values])),
    )


def iter_inverse_batches(
    frozen_cache: Path,
    effect_cache: Path,
    mode: str,
    retrieval_candidates: int,
    batch_scenes: int,
) -> Iterator[InverseBatch]:
    pending: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene in iter_probe_scenes(frozen_cache, effect_cache):
        pending.append((scene.token, _inverse_scene(scene, mode, retrieval_candidates)))
        if len(pending) == batch_scenes:
            yield _stack_inverse_batch(pending)
            pending.clear()
    if pending:
        yield _stack_inverse_batch(pending)


def load_inverse_dataset(
    frozen_cache: Path,
    effect_cache: Path,
    mode: str,
    retrieval_candidates: int,
    *,
    time_shuffle_seed: int | None = None,
) -> InverseDataset:
    values: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene in iter_probe_scenes(frozen_cache, effect_cache):
        permutation = None
        if time_shuffle_seed is not None:
            digest = hashlib.sha256(
                f"time:{scene.token}:{time_shuffle_seed}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            permutation = rng.permutation(8).astype(np.int64)
            if np.array_equal(permutation, np.arange(8)):
                permutation = np.roll(permutation, 1)
        values.append(
            (
                scene.token,
                _inverse_scene(scene, mode, retrieval_candidates, permutation),
            )
        )
    if not values:
        raise ValueError("inverse dataset is empty")
    batch = _stack_inverse_batch(values)
    return InverseDataset(
        tokens=batch.tokens,
        effects=batch.effects.numpy().astype(np.float32),
        trajectories=batch.trajectories.numpy().astype(np.float32),
        pair_first=batch.pair_first.numpy().astype(np.int64),
        pair_second=batch.pair_second.numpy().astype(np.int64),
        delta_targets=batch.delta_targets.numpy().astype(np.float32),
    )


@torch.inference_mode()
def _quick_validation(
    model: InverseProbe,
    frozen_cache: Path,
    effect_cache: Path,
    mode: str,
    retrieval_candidates: int,
    batch_scenes: int,
    temperature: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    sign_correct = 0
    sign_total = 0
    for batch in iter_inverse_batches(
        frozen_cache, effect_cache, mode, retrieval_candidates, batch_scenes
    ):
        effects = batch.effects.to(device)
        trajectories = batch.trajectories.to(device)
        logits = model.retrieval_logits(effects, trajectories, temperature)
        labels = torch.arange(retrieval_candidates, device=device)[None]
        correct += int((logits.argmax(dim=-1) == labels).sum().cpu())
        total += logits.shape[0] * retrieval_candidates
        rows = torch.arange(len(effects), device=device)[:, None]
        first = batch.pair_first.to(device)
        second = batch.pair_second.to(device)
        predicted = model.predict_delta(effects[rows, first], effects[rows, second])
        target = batch.delta_targets.to(device)
        valid = target.abs() > 1.0e-3
        sign_correct += int((torch.sign(predicted[valid]) == torch.sign(target[valid])).sum().cpu())
        sign_total += int(valid.sum().cpu())
    if not total or not sign_total:
        raise ValueError("inverse validation has no retrievals/deltas")
    return correct / total, sign_correct / sign_total


def _train_inverse_trial(
    *,
    train_cache: Path,
    val_cache: Path,
    train_effects: Path,
    val_effects: Path,
    mode: str,
    seed: int,
    learning_rate: float,
    config: Mapping[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = InverseProbe(embedding_dim=int(config["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    retrieval_candidates = int(config["retrieval_candidates"])
    batch_scenes = int(config["batch_scenes"])
    temperature = float(config["temperature"])
    delta_weight = float(config["delta_weight"])
    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    best_top1 = -1.0
    best_delta_sign = -1.0
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    steps = 0
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        retrieval_loss = 0.0
        delta_loss = 0.0
        epoch_steps = 0
        for batch in iter_inverse_batches(
            train_cache, train_effects, mode, retrieval_candidates, batch_scenes
        ):
            effects = batch.effects.to(device)
            trajectories = batch.trajectories.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = inverse_training_loss(
                model,
                effects,
                trajectories,
                batch.pair_first.to(device),
                batch.pair_second.to(device),
                batch.delta_targets.to(device),
                temperature,
                delta_weight,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(losses.total.detach().cpu())
            retrieval_loss += float(losses.retrieval.detach().cpu())
            delta_loss += float(losses.delta.detach().cpu())
            epoch_steps += 1
            steps += 1
        if epoch_steps == 0:
            raise ValueError("inverse training cache is empty")
        val_top1, val_delta_sign = _quick_validation(
            model,
            val_cache,
            val_effects,
            mode,
            retrieval_candidates,
            batch_scenes,
            temperature,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / epoch_steps,
                "retrieval_loss": retrieval_loss / epoch_steps,
                "delta_loss": delta_loss / epoch_steps,
                "validation_top1": val_top1,
                "validation_delta_sign_accuracy": val_delta_sign,
            }
        )
        improved = val_top1 > best_top1 + 1.0e-9 or (
            abs(val_top1 - best_top1) <= 1.0e-9
            and val_delta_sign > best_delta_sign + 1.0e-9
        )
        if improved:
            best_top1 = val_top1
            best_delta_sign = val_delta_sign
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("inverse training never produced a checkpoint")
    trainable = sum(parameter.numel() for parameter in model.parameters())
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    payload = {
        "schema_version": "inverse_probe.v1",
        "mode": mode,
        "seed": seed,
        "learning_rate": learning_rate,
        "embedding_dim": int(config["embedding_dim"]),
        "temperature": temperature,
        "retrieval_candidates": retrieval_candidates,
        "effect_input_schema": (
            ["ego_kinematics_without_absolute_pose"]
            if mode == "ego_only"
            else ["map_relative", "logged_actor_relative", "masks"]
            if mode == "environment_only"
            else [
                "ego_kinematics_without_absolute_pose",
                "map_relative",
                "logged_actor_relative",
                "masks",
            ]
        ),
        "forbidden_effect_inputs": [
            "trajectory",
            "candidate_index",
            "selected_index",
            "score",
            "future_ego_absolute_coordinates",
        ],
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "validation_top1": best_top1,
        "validation_delta_sign_accuracy": best_delta_sign,
        "training_steps": steps,
        "peak_gpu_memory_bytes": peak_memory,
        "model_state_dict": best_state,
        "history": history,
    }
    _atomic_torch_save(output, payload)
    return {
        key: payload[key]
        for key in (
            "mode",
            "seed",
            "learning_rate",
            "trainable_parameters",
            "best_epoch",
            "validation_top1",
            "validation_delta_sign_accuracy",
            "training_steps",
            "peak_gpu_memory_bytes",
            "effect_input_schema",
            "forbidden_effect_inputs",
        )
    } | {"checkpoint": str(output)}


def train_inverse_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing inverse training output: {args.output}")
    full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = full_config["inverse"]
    seeds = [int(value) for value in full_config["run"]["seeds"]]
    learning_rates = [float(value) for value in config["learning_rates"]]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    trials: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for mode, seed in itertools.product(INVERSE_MODES, seeds):
        mode_trials: list[dict[str, Any]] = []
        for learning_rate in learning_rates:
            checkpoint = args.output / "search" / mode / f"seed-{seed}-lr-{learning_rate:g}.pt"
            result = _train_inverse_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=args.train_effects,
                val_effects=args.val_effects,
                mode=mode,
                seed=seed,
                learning_rate=learning_rate,
                config=config,
                device=device,
                output=checkpoint,
            )
            mode_trials.append(result)
            trials.append(result)
        selected.append(
            max(
                mode_trials,
                key=lambda row: (
                    float(row["validation_top1"]),
                    float(row["validation_delta_sign_accuracy"]),
                    -float(row["learning_rate"]),
                ),
            )
        )
    counts = {int(trial["trainable_parameters"]) for trial in selected}
    if len(counts) != 1:
        raise RuntimeError(f"inverse mode capacities differ: {counts}")
    atomic_write_json(
        args.output / "inverse_training_manifest.json",
        {
            "schema_version": "inverse_training.v1",
            "config": str(args.config),
            "seeds": seeds,
            "modes": list(INVERSE_MODES),
            "learning_rate_grid": learning_rates,
            "candidate_selection": "trajectory_geometry_farthest_point_sampling",
            "score_used_for_candidate_selection": False,
            "trainable_parameter_count": next(iter(counts)),
            "selected_trials": selected,
            "all_trials": trials,
        },
    )


def _load_inverse_model(
    checkpoint: Path, device: torch.device
) -> tuple[InverseProbe, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "inverse_probe.v1":
        raise ValueError(f"unsupported inverse checkpoint: {checkpoint}")
    forbidden = set(payload.get("forbidden_effect_inputs", ()))
    required_forbidden = {
        "trajectory",
        "candidate_index",
        "selected_index",
        "score",
        "future_ego_absolute_coordinates",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("inverse checkpoint does not enforce leakage contract")
    model = InverseProbe(embedding_dim=int(payload["embedding_dim"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _within_scene_permutation(tokens: Sequence[str], candidates: int, seed: int) -> npt.NDArray[np.int64]:
    output = np.empty((len(tokens), candidates), dtype=np.int64)
    for scene_index, token in enumerate(tokens):
        digest = hashlib.sha256(f"inverse-shuffle:{token}:{seed}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        output[scene_index] = rng.permutation(candidates)
        if np.array_equal(output[scene_index], np.arange(candidates)):
            output[scene_index] = np.roll(output[scene_index], 1)
    return output


@torch.inference_mode()
def evaluate_inverse_checkpoint(
    checkpoint: Path,
    frozen_cache: Path,
    effect_cache: Path,
    device: torch.device,
    batch_scenes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model, payload = _load_inverse_model(checkpoint, device)
    mode = str(payload["mode"])
    seed = int(payload["seed"])
    candidates = int(payload["retrieval_candidates"])
    temperature = float(payload["temperature"])
    dataset = load_inverse_dataset(frozen_cache, effect_cache, mode, candidates)
    time_dataset = load_inverse_dataset(
        frozen_cache, effect_cache, mode, candidates, time_shuffle_seed=seed
    )
    permutation = _within_scene_permutation(dataset.tokens, candidates, seed)
    within_effects = np.take_along_axis(
        dataset.effects, permutation[..., None], axis=1
    )
    cross_effects = np.roll(dataset.effects, 1, axis=0)
    variants = {
        "ordinary": dataset.effects,
        "within_scene_shuffle": within_effects,
        "cross_scene_shuffle": cross_effects,
        "time_order_shuffle": time_dataset.effects,
    }
    retrieval: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for name, effect_values in variants.items():
        logits_parts: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(dataset.tokens), batch_scenes):
            stop = start + batch_scenes
            logits = model.retrieval_logits(
                torch.from_numpy(effect_values[start:stop]).to(device),
                torch.from_numpy(dataset.trajectories[start:stop]).to(device),
                temperature,
            )
            logits_parts.append(logits.cpu().numpy().astype(np.float32))
        logits_all = np.concatenate(logits_parts, axis=0)
        order = np.argsort(-logits_all, axis=-1)
        labels = np.arange(candidates)[None, :, None]
        rank = np.argmax(order == labels, axis=-1) + 1
        retrieval[name] = {
            "top1_scene": (rank == 1).mean(axis=1),
            "top3_scene": (rank <= 3).mean(axis=1),
            "mrr_scene": (1.0 / rank).mean(axis=1),
        }

    rows = np.arange(len(dataset.tokens))[:, None]
    ordinary_delta: list[npt.NDArray[np.float32]] = []
    shuffled_delta: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(dataset.tokens), batch_scenes):
        stop = start + batch_scenes
        effects = torch.from_numpy(dataset.effects[start:stop]).to(device)
        shuffled = torch.from_numpy(within_effects[start:stop]).to(device)
        first = torch.from_numpy(dataset.pair_first[start:stop]).to(device)
        second = torch.from_numpy(dataset.pair_second[start:stop]).to(device)
        local_rows = torch.arange(len(effects), device=device)[:, None]
        ordinary_delta.append(
            model.predict_delta(effects[local_rows, first], effects[local_rows, second])
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        shuffled_delta.append(
            model.predict_delta(shuffled[local_rows, first], shuffled[local_rows, second])
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    ordinary_prediction = np.concatenate(ordinary_delta, axis=0)
    shuffled_prediction = np.concatenate(shuffled_delta, axis=0)
    target = dataset.delta_targets
    valid = np.abs(target) > 1.0e-3

    def sign_by_scene(prediction: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        correct = (np.sign(prediction) == np.sign(target)) & valid
        return correct.sum(axis=(1, 2)) / valid.sum(axis=(1, 2))

    sign_scene = sign_by_scene(ordinary_prediction)
    shuffled_sign_scene = sign_by_scene(shuffled_prediction)
    normalized_mae_scene = np.abs(ordinary_prediction - target).mean(axis=(1, 2))
    residual = np.square(ordinary_prediction - target).sum()
    total = np.square(target - target.mean(axis=(0, 1), keepdims=True)).sum()
    r_squared = float(1.0 - residual / total) if total > 0 else float("nan")
    scene_frame = pd.DataFrame(
        {
            "scene_token": dataset.tokens,
            "mode": mode,
            "seed": seed,
            "top1_retrieval": retrieval["ordinary"]["top1_scene"],
            "top3_retrieval": retrieval["ordinary"]["top3_scene"],
            "mrr": retrieval["ordinary"]["mrr_scene"],
            "within_shuffle_top1": retrieval["within_scene_shuffle"]["top1_scene"],
            "cross_shuffle_top1": retrieval["cross_scene_shuffle"]["top1_scene"],
            "time_shuffle_top1": retrieval["time_order_shuffle"]["top1_scene"],
            "delta_sign_accuracy": sign_scene,
            "shuffled_delta_sign_accuracy": shuffled_sign_scene,
            "delta_normalized_mae": normalized_mae_scene,
        }
    )
    metric = {
        "effect_input": mode,
        "seed": seed,
        "retrieval_candidates": candidates,
        "random_top1": 1.0 / candidates,
        "top1_retrieval": float(scene_frame["top1_retrieval"].mean()),
        "top3_retrieval": float(scene_frame["top3_retrieval"].mean()),
        "mrr": float(scene_frame["mrr"].mean()),
        "within_scene_shuffle_top1": float(scene_frame["within_shuffle_top1"].mean()),
        "cross_scene_shuffle_top1": float(scene_frame["cross_shuffle_top1"].mean()),
        "time_order_shuffle_top1": float(scene_frame["time_shuffle_top1"].mean()),
        "delta_sign_accuracy": float(sign_scene.mean()),
        "shuffled_delta_sign_accuracy": float(shuffled_sign_scene.mean()),
        "delta_normalized_mae": float(normalized_mae_scene.mean()),
        "delta_r_squared": r_squared,
        "delta_mae_endpoint_lateral_m": float(
            np.abs(ordinary_prediction[..., 0] - target[..., 0]).mean() * DELTA_SCALES[0]
        ),
        "delta_mae_endpoint_longitudinal_m": float(
            np.abs(ordinary_prediction[..., 1] - target[..., 1]).mean() * DELTA_SCALES[1]
        ),
        "trainable_parameters": int(payload["trainable_parameters"]),
        "training_steps": int(payload["training_steps"]),
        "best_validation_epoch": int(payload["best_epoch"]),
        "peak_gpu_memory_bytes": int(payload["peak_gpu_memory_bytes"]),
    }
    return metric, scene_frame


def evaluate_inverse_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing inverse evaluation output: {args.output}")
    manifest = json.loads(
        (args.training_root / "inverse_training_manifest.json").read_text()
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics: list[dict[str, Any]] = []
    scenes: list[pd.DataFrame] = []
    for trial in manifest["selected_trials"]:
        metric, scene_frame = evaluate_inverse_checkpoint(
            Path(trial["checkpoint"]),
            args.test_cache,
            args.test_effects,
            device,
            int(config["inverse"]["batch_scenes"]),
        )
        metrics.append(metric)
        scenes.append(scene_frame)
    pd.DataFrame(metrics).to_csv(args.output / "identifiability_metrics.csv", index=False)
    pd.concat(scenes, ignore_index=True).to_parquet(
        args.output / "scene_level_inverse_identifiability.parquet", index=False
    )


@torch.inference_mode()
def inverse_consistency(
    checkpoint: Path,
    frozen_cache: Path,
    predicted_effects: Path,
    device: torch.device,
    candidate_chunk: int = 64,
) -> tuple[tuple[str, ...], npt.NDArray[np.float32]]:
    """Compute matched effect/trajectory similarity for all fixed candidates."""

    model, payload = _load_inverse_model(checkpoint, device)
    mode = str(payload["mode"])
    tokens: list[str] = []
    values: list[npt.NDArray[np.float32]] = []
    for scene in iter_probe_scenes(frozen_cache, predicted_effects):
        if scene.effects is None:
            raise ValueError(f"{scene.token}: predicted effect cache is absent")
        effect = pack_inverse_effect(scene.effects, mode)
        trajectory = pack_trajectory(scene.frozen["trajectory"])
        similarities: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(effect), candidate_chunk):
            effect_embedding = model.encode_effect(
                torch.from_numpy(effect[start : start + candidate_chunk]).to(device)
            )
            trajectory_embedding = model.encode_trajectory(
                torch.from_numpy(trajectory[start : start + candidate_chunk]).to(device)
            )
            similarities.append(
                (effect_embedding * trajectory_embedding)
                .sum(dim=-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        tokens.append(scene.token)
        values.append(np.concatenate(similarities))
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("inverse consistency produced empty/duplicate scene tokens")
    return tuple(tokens), np.stack(values)


def _random_consistency(tokens: Sequence[str], candidates: int, seed: int) -> npt.NDArray[np.float32]:
    output = np.empty((len(tokens), candidates), dtype=np.float32)
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(f"random-gate:{token}:{seed}".encode("utf-8")).digest()
        output[index] = np.random.default_rng(int.from_bytes(digest[:8], "little")).normal(
            size=candidates
        )
    return output


def _shuffle_consistency(
    values: npt.NDArray[np.float32], tokens: Sequence[str], seed: int
) -> npt.NDArray[np.float32]:
    output = np.empty_like(values)
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(f"shuffled-gate:{token}:{seed}".encode("utf-8")).digest()
        permutation = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(
            values.shape[1]
        )
        output[index] = values[index, permutation]
    return output


def _normalize_consistency(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    mean = values.mean(axis=1, keepdims=True)
    standard = values.std(axis=1, keepdims=True)
    standard[standard < 1.0e-6] = 1.0
    return ((values - mean) / standard).astype(np.float32)


def _select_with_adjustment(
    base_scores: npt.NDArray[np.float32],
    consistency: npt.NDArray[np.float32],
    kind: str,
    value: float,
) -> npt.NDArray[np.int64]:
    if base_scores.shape != consistency.shape or base_scores.ndim != 2:
        raise ValueError("planning gate arrays must be matching [scene,candidate]")
    normalized = _normalize_consistency(consistency)
    if kind == "additive":
        adjusted = base_scores + float(value) * normalized
    elif kind == "reject":
        adjusted = base_scores.copy()
        reject_count = int(np.floor(base_scores.shape[1] * float(value) / 100.0))
        if reject_count:
            rejected = np.argsort(consistency, axis=1)[:, :reject_count]
            rows = np.arange(len(adjusted))[:, None]
            adjusted[rows, rejected] = -np.inf
    else:
        raise ValueError(f"unknown planning adjustment: {kind}")
    return adjusted.argmax(axis=1).astype(np.int64)


def _planning_metrics(
    selected: npt.NDArray[np.int64], factors: npt.NDArray[np.float32]
) -> tuple[dict[str, float], pd.DataFrame]:
    true_scores = pdms_from_factors(factors)
    rows = np.arange(len(selected))
    selected_true = true_scores[rows, selected]
    selected_factors = factors[rows, selected]
    oracle = true_scores.max(axis=1)
    false_safe = (
        (selected_factors[:, 0] == 0)
        | (selected_factors[:, 1] == 0)
        | (selected_factors[:, 3] == 0)
    )
    frame = pd.DataFrame(
        {
            "selected_index": selected,
            "selected_pdms": selected_true,
            "regret": oracle - selected_true,
            "false_safe": false_safe,
        }
    )
    return {
        "selected_pdms": float(selected_true.mean()),
        "top1_regret": float((oracle - selected_true).mean()),
        "false_safe_rate": float(false_safe.mean()),
    }, frame


def _choose_validation_adjustment(
    base_scores: npt.NDArray[np.float32],
    consistency: npt.NDArray[np.float32],
    factors: npt.NDArray[np.float32],
    lambda_grid: Sequence[float],
    reject_grid: Sequence[float],
) -> dict[str, Any]:
    options = [("additive", float(value)) for value in lambda_grid] + [
        ("reject", float(value)) for value in reject_grid
    ]
    evaluated: list[dict[str, Any]] = []
    for kind, value in options:
        selected = _select_with_adjustment(base_scores, consistency, kind, value)
        metrics, _ = _planning_metrics(selected, factors)
        evaluated.append({"kind": kind, "value": value, **metrics})
    return min(
        evaluated,
        key=lambda row: (
            float(row["top1_regret"]),
            -float(row["selected_pdms"]),
            float(row["false_safe_rate"]),
            0 if row["kind"] == "additive" else 1,
            float(row["value"]),
        ),
    )


def _align_consistency(
    scorer_tokens: Sequence[str],
    consistency_tokens: Sequence[str],
    values: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if tuple(scorer_tokens) == tuple(consistency_tokens):
        return values
    index = {token: position for position, token in enumerate(consistency_tokens)}
    if set(index) != set(scorer_tokens):
        raise ValueError("scorer and inverse consistency scene tokens differ")
    return values[[index[token] for token in scorer_tokens]]


def run_planning_gate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing inverse planning output: {args.output}")
    full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inverse_config = full_config["inverse"]
    inverse_manifest = json.loads(
        (args.inverse_training_root / "inverse_training_manifest.json").read_text()
    )
    scorer_manifest = json.loads((args.g3_scorer_root / "training_manifest.json").read_text())
    seeds = [int(value) for value in full_config["run"]["seeds"]]
    inverse_checkpoints = {
        (str(trial["mode"]), int(trial["seed"])): Path(trial["checkpoint"])
        for trial in inverse_manifest["selected_trials"]
    }
    scorer_checkpoints = {
        int(trial["seed"]): Path(trial["checkpoint"])
        for trial in scorer_manifest["selected_trials"]
        if trial["model_type"] == "predicted_replay_effect"
    }
    if set(scorer_checkpoints) != set(seeds):
        raise ValueError("predicted effect scorer checkpoints do not cover configured seeds")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    metric_rows: list[dict[str, Any]] = []
    scene_frames: list[pd.DataFrame] = []
    variants = ("random_scalar", "ego_only_inverse", "environment_only_inverse", "shuffled_inverse")
    for seed in seeds:
        split_data: dict[str, dict[str, Any]] = {}
        for split, frozen_cache in (("val", args.val_cache), ("test", args.test_cache)):
            predicted_effects = args.predicted_effect_root / f"seed-{seed}" / split
            scorer = evaluate_checkpoint(
                scorer_checkpoints[seed],
                frozen_cache,
                predicted_effects,
                int(full_config["probe"]["batch_scenes"]),
                device,
            )
            ego_tokens, ego_consistency = inverse_consistency(
                inverse_checkpoints[("ego_only", seed)],
                frozen_cache,
                predicted_effects,
                device,
            )
            env_tokens, env_consistency = inverse_consistency(
                inverse_checkpoints[("environment_only", seed)],
                frozen_cache,
                predicted_effects,
                device,
            )
            ego_consistency = _align_consistency(scorer.tokens, ego_tokens, ego_consistency)
            env_consistency = _align_consistency(scorer.tokens, env_tokens, env_consistency)
            split_data[split] = {
                "tokens": scorer.tokens,
                "base_scores": scorer.predicted_scores,
                "factors": scorer.true_factors,
                "consistency": {
                    "random_scalar": _random_consistency(scorer.tokens, 256, seed),
                    "ego_only_inverse": ego_consistency,
                    "environment_only_inverse": env_consistency,
                    "shuffled_inverse": _shuffle_consistency(env_consistency, scorer.tokens, seed),
                },
            }
        base_selected = split_data["test"]["base_scores"].argmax(axis=1)
        base_metrics, base_scene = _planning_metrics(
            base_selected, split_data["test"]["factors"]
        )
        metric_rows.append(
            {
                "planning_variant": "effect_scorer",
                "seed": seed,
                "selection_kind": "none",
                "selection_value": 0.0,
                **base_metrics,
            }
        )
        base_scene.insert(0, "scene_token", split_data["test"]["tokens"])
        base_scene["planning_variant"] = "effect_scorer"
        base_scene["seed"] = seed
        scene_frames.append(base_scene)
        for variant in variants:
            chosen = _choose_validation_adjustment(
                split_data["val"]["base_scores"],
                split_data["val"]["consistency"][variant],
                split_data["val"]["factors"],
                inverse_config["lambda_grid"],
                inverse_config["reject_bottom_percent_grid"],
            )
            selected = _select_with_adjustment(
                split_data["test"]["base_scores"],
                split_data["test"]["consistency"][variant],
                str(chosen["kind"]),
                float(chosen["value"]),
            )
            metrics, scene = _planning_metrics(selected, split_data["test"]["factors"])
            metric_rows.append(
                {
                    "planning_variant": variant,
                    "seed": seed,
                    "selection_kind": chosen["kind"],
                    "selection_value": chosen["value"],
                    "validation_selected_pdms": chosen["selected_pdms"],
                    "validation_top1_regret": chosen["top1_regret"],
                    "validation_false_safe_rate": chosen["false_safe_rate"],
                    **metrics,
                }
            )
            scene.insert(0, "scene_token", split_data["test"]["tokens"])
            scene["planning_variant"] = variant
            scene["seed"] = seed
            scene_frames.append(scene)
    pd.DataFrame(metric_rows).to_csv(args.output / "planning_gate_metrics.csv", index=False)
    pd.concat(scene_frames, ignore_index=True).to_parquet(
        args.output / "scene_level_inverse_planning.parquet", index=False
    )


def finalize_g4(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing G4 summary output: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    identity_metrics = pd.read_csv(args.identifiability_root / "identifiability_metrics.csv")
    identity_scenes = pd.read_parquet(
        args.identifiability_root / "scene_level_inverse_identifiability.parquet"
    )
    planning_metrics = pd.read_csv(args.planning_root / "planning_gate_metrics.csv")
    planning_scenes = pd.read_parquet(
        args.planning_root / "scene_level_inverse_planning.parquet"
    )
    averaged_identity = identity_metrics.groupby("effect_input").mean(numeric_only=True)
    environment = averaged_identity.loc["environment_only"]
    random_top1 = float(environment["random_top1"])
    retrieval_pass = float(environment["top1_retrieval"]) >= 0.1875
    shuffles_near_random = all(
        float(environment[key]) <= random_top1 + 0.05
        for key in (
            "within_scene_shuffle_top1",
            "cross_scene_shuffle_top1",
            "time_order_shuffle_top1",
        )
    )
    delta_pass = float(environment["delta_sign_accuracy"]) >= 0.65

    environment_scene = (
        identity_scenes[identity_scenes["mode"] == "environment_only"]
        .groupby("scene_token", sort=True)
        .agg(
            delta_sign_accuracy=("delta_sign_accuracy", "mean"),
            shuffled_delta_sign_accuracy=("shuffled_delta_sign_accuracy", "mean"),
        )
    )
    delta_ci = paired_scene_bootstrap(
        environment_scene["delta_sign_accuracy"].to_numpy(),
        environment_scene["shuffled_delta_sign_accuracy"].to_numpy(),
        samples=int(config["bootstrap"]["samples"]),
        confidence=float(config["bootstrap"]["confidence"]),
        seed=int(config["bootstrap"]["seed"]) + 47,
    )
    delta_above_shuffle = delta_ci.lower > 0

    averaged_planning = planning_metrics.groupby("planning_variant").mean(numeric_only=True)
    base = averaged_planning.loc["effect_scorer"]
    env_gate = averaged_planning.loc["environment_only_inverse"]
    ego_gate = averaged_planning.loc["ego_only_inverse"]
    shuffled_gate = averaged_planning.loc["shuffled_inverse"]
    random_gate = averaged_planning.loc["random_scalar"]
    env_pdms_gain = float(env_gate["selected_pdms"] - base["selected_pdms"])
    env_false_safe_reduction = (
        float((base["false_safe_rate"] - env_gate["false_safe_rate"]) / base["false_safe_rate"])
        if float(base["false_safe_rate"]) > 0
        else 0.0
    )
    planning_margin = env_pdms_gain >= 0.002 or (
        env_false_safe_reduction >= 0.10
        and float(env_gate["selected_pdms"]) >= float(base["selected_pdms"])
    )
    per_seed: list[dict[str, Any]] = []
    for seed in sorted(planning_metrics["seed"].unique()):
        rows = planning_metrics[planning_metrics["seed"] == seed].set_index(
            "planning_variant"
        )
        gain = float(
            rows.loc["environment_only_inverse", "selected_pdms"]
            - rows.loc["effect_scorer", "selected_pdms"]
        )
        base_false = float(rows.loc["effect_scorer", "false_safe_rate"])
        env_false = float(rows.loc["environment_only_inverse", "false_safe_rate"])
        reduction = (base_false - env_false) / base_false if base_false > 0 else 0.0
        per_seed.append(
            {
                "seed": int(seed),
                "environment_pdms_gain": gain,
                "environment_false_safe_reduction_fraction": reduction,
                "shuffled_pdms_gain": float(
                    rows.loc["shuffled_inverse", "selected_pdms"]
                    - rows.loc["effect_scorer", "selected_pdms"]
                ),
                "ego_pdms_gain": float(
                    rows.loc["ego_only_inverse", "selected_pdms"]
                    - rows.loc["effect_scorer", "selected_pdms"]
                ),
            }
        )
    direction_consistent = all(
        row["environment_pdms_gain"] >= 0
        or row["environment_false_safe_reduction_fraction"] > 0
        for row in per_seed
    )
    shuffled_not_equal = env_pdms_gain > float(
        shuffled_gate["selected_pdms"] - base["selected_pdms"]
    ) + 1.0e-6
    random_not_equal = env_pdms_gain > float(
        random_gate["selected_pdms"] - base["selected_pdms"]
    ) + 1.0e-6
    ego_gain = float(ego_gate["selected_pdms"] - base["selected_pdms"])
    ego_does_not_explain = env_pdms_gain > ego_gain + 1.0e-6 or (
        env_false_safe_reduction >= 0.10
        and float(env_gate["false_safe_rate"]) < float(ego_gate["false_safe_rate"])
    )
    base_scene = (
        planning_scenes[planning_scenes["planning_variant"] == "effect_scorer"]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    env_scene = (
        planning_scenes[
            planning_scenes["planning_variant"] == "environment_only_inverse"
        ]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    if not base_scene.index.equals(env_scene.index):
        raise ValueError("G4 planning scene identities differ")
    planning_ci = paired_scene_bootstrap(
        env_scene.to_numpy(),
        base_scene.to_numpy(),
        samples=int(config["bootstrap"]["samples"]),
        confidence=float(config["bootstrap"]["confidence"]),
        seed=int(config["bootstrap"]["seed"]) + 53,
    )
    conditions = {
        "environment_retrieval_top1_at_least_18_75pct": retrieval_pass,
        "retrieval_shuffles_near_random": shuffles_near_random,
        "environment_delta_sign_at_least_65pct": delta_pass,
        "delta_sign_significantly_above_shuffle": delta_above_shuffle,
        "planning_improves_pdms_or_false_safe": planning_margin,
        "three_seed_planning_direction_consistent": direction_consistent,
        "shuffled_inverse_has_no_equal_benefit": shuffled_not_equal,
        "random_scalar_has_no_equal_benefit": random_not_equal,
        "ego_only_inverse_does_not_explain_benefit": ego_does_not_explain,
    }
    gate_pass = all(conditions.values())
    summary = {
        "schema_version": "gate_g4.v1",
        "gate_g4_pass": gate_pass,
        "conditions": conditions,
        "environment_top1_retrieval": float(environment["top1_retrieval"]),
        "environment_delta_sign_accuracy": float(environment["delta_sign_accuracy"]),
        "environment_pdms_gain_raw": env_pdms_gain,
        "environment_pdms_gain_points": env_pdms_gain * 100.0,
        "environment_false_safe_reduction_fraction": env_false_safe_reduction,
        "paired_scene_bootstrap_environment_minus_base": asdict(planning_ci),
        "paired_scene_bootstrap_delta_minus_shuffle": asdict(delta_ci),
        "per_seed": per_seed,
        "inverse_role": "method_core" if gate_pass else "diagnostic_only",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    inverse_metrics = identity_metrics.copy()
    inverse_metrics["pdms_with_gate"] = np.nan
    inverse_metrics["false_safe"] = np.nan
    planning_map = {
        "ego_only": "ego_only_inverse",
        "environment_only": "environment_only_inverse",
    }
    for row_index, row in inverse_metrics.iterrows():
        planning_variant = planning_map.get(str(row["effect_input"]))
        if planning_variant is None:
            continue
        matching = planning_metrics[
            (planning_metrics["planning_variant"] == planning_variant)
            & (planning_metrics["seed"] == row["seed"])
        ]
        if len(matching) != 1:
            raise ValueError("inverse/planning metrics do not join one-to-one")
        inverse_metrics.loc[row_index, "pdms_with_gate"] = matching.iloc[0][
            "selected_pdms"
        ]
        inverse_metrics.loc[row_index, "false_safe"] = matching.iloc[0][
            "false_safe_rate"
        ]
    inverse_metrics.to_csv(args.output / "inverse_metrics.csv", index=False)
    combined_scenes = pd.concat(
        [
            identity_scenes.assign(result_family="identifiability"),
            planning_scenes.assign(result_family="planning"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_scenes.to_parquet(args.output / "scene_level_g4.parquet", index=False)
    atomic_write_json(args.output / "g4_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-inverse")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--val-cache", type=Path, required=True)
    train.add_argument("--train-effects", type=Path, required=True)
    train.add_argument("--val-effects", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")

    evaluate = subparsers.add_parser("evaluate-identifiability")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--training-root", type=Path, required=True)
    evaluate.add_argument("--test-cache", type=Path, required=True)
    evaluate.add_argument("--test-effects", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")

    planning = subparsers.add_parser("planning-gate")
    planning.add_argument("--config", type=Path, required=True)
    planning.add_argument("--inverse-training-root", type=Path, required=True)
    planning.add_argument("--g3-scorer-root", type=Path, required=True)
    planning.add_argument("--predicted-effect-root", type=Path, required=True)
    planning.add_argument("--val-cache", type=Path, required=True)
    planning.add_argument("--test-cache", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--device", default="cuda")

    finalize = subparsers.add_parser("finalize-g4")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--identifiability-root", type=Path, required=True)
    finalize.add_argument("--planning-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train-inverse":
        train_inverse_suite(args)
    elif args.command == "evaluate-identifiability":
        evaluate_inverse_suite(args)
    elif args.command == "planning-gate":
        run_planning_gate(args)
    elif args.command == "finalize-g4":
        finalize_g4(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
