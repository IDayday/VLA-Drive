"""Train, cache, score, and gate the lightweight forward effect predictor."""

from __future__ import annotations

import argparse
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

from .evaluate_probe import aggregate_metrics, evaluate_checkpoint, scene_outcomes
from .feature_store import (
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    atomic_write_json,
)
from .models.effect_predictor import (
    CandidateEffectPredictor,
    EffectLossWeights,
    count_trainable_parameters,
    decode_transformed_effect,
    effect_prediction_loss,
)
from .metrics import paired_scene_bootstrap
from .train_probe import (
    ProbeDataError,
    _candidate_subset,
    _records_match,
    _seed_everything,
    _train_trial,
    iter_probe_scenes,
)


PREDICTOR_INPUT_SCHEMA = (
    "current_bev_tokens",
    "ego_status_feature",
    "candidate_trajectory",
)
PREDICTOR_TARGET_SCHEMA = (
    "ego_effect",
    "map_effect",
    "actor_effect",
    "actor_mask",
    "interaction_mask",
)


@dataclass(frozen=True)
class EffectBatch:
    tokens: tuple[str, ...]
    current_bev_tokens: Tensor
    ego_status: Tensor
    trajectory: Tensor
    targets: Mapping[str, Tensor]


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictor checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _effect_raw_for_indices(
    scene: Any, indices: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[Any], ...]:
    if scene.effects is None:
        raise ProbeDataError(f"{scene.token}: predictor training requires oracle replay effects")
    frozen = scene.frozen
    required_frozen = PREDICTOR_INPUT_SCHEMA[:2] + ("trajectory",)
    missing = [key for key in required_frozen if key not in frozen]
    if missing:
        raise ProbeDataError(f"{scene.token}: predictor input missing {missing}")
    current = np.asarray(frozen["current_bev_tokens"], dtype=np.float32)
    ego = np.asarray(frozen["ego_status_feature"], dtype=np.float32).reshape(-1)
    trajectory = np.asarray(frozen["trajectory"], dtype=np.float32)[indices]
    if current.shape != (64, 256) or trajectory.shape[1:] != (8, 3):
        raise ProbeDataError(
            f"{scene.token}: predictor input shapes {current.shape}/{trajectory.shape}"
        )
    targets = tuple(np.asarray(scene.effects[key])[indices] for key in PREDICTOR_TARGET_SCHEMA)
    return (current, ego, trajectory, *targets)


def _stack_effect_batch(
    values: Sequence[tuple[str, tuple[npt.NDArray[Any], ...]]]
) -> EffectBatch:
    candidate_counts = {item[1][2].shape[0] for item in values}
    if len(candidate_counts) != 1:
        raise ProbeDataError(f"effect batch candidate counts differ: {candidate_counts}")
    target_values = {
        key: torch.from_numpy(np.stack([item[1][index + 3] for item in values]))
        for index, key in enumerate(PREDICTOR_TARGET_SCHEMA)
    }
    return EffectBatch(
        tokens=tuple(item[0] for item in values),
        current_bev_tokens=torch.from_numpy(np.stack([item[1][0] for item in values])),
        ego_status=torch.from_numpy(np.stack([item[1][1] for item in values])),
        trajectory=torch.from_numpy(np.stack([item[1][2] for item in values])),
        targets=target_values,
    )


def iter_effect_batches(
    frozen_cache: Path,
    oracle_effects: Path,
    *,
    batch_scenes: int,
    candidate_chunk: int,
    train_candidates: int,
    seed: int,
    epoch: int,
    full_candidates: bool,
) -> Iterator[EffectBatch]:
    if batch_scenes <= 0 or candidate_chunk <= 0:
        raise ValueError("effect batch sizes must be positive")
    pending: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene in iter_probe_scenes(frozen_cache, oracle_effects):
        indices = (
            np.arange(256, dtype=np.int64)
            if full_candidates
            else _candidate_subset(scene.token, 256, train_candidates, seed, epoch)
        )
        for start in range(0, len(indices), candidate_chunk):
            chunk = indices[start : start + candidate_chunk]
            if len(chunk) != candidate_chunk and pending:
                yield _stack_effect_batch(pending)
                pending.clear()
            raw = _effect_raw_for_indices(scene, chunk)
            pending.append((scene.token, raw))
            if len(pending) == batch_scenes:
                yield _stack_effect_batch(pending)
                pending.clear()
    if pending:
        yield _stack_effect_batch(pending)


def _targets_to_device(targets: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


@torch.inference_mode()
def _validation_loss(
    model: CandidateEffectPredictor,
    frozen_cache: Path,
    oracle_effects: Path,
    batch_scenes: int,
    candidate_chunk: int,
    device: torch.device,
    weights: EffectLossWeights,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for batch in iter_effect_batches(
        frozen_cache,
        oracle_effects,
        batch_scenes=batch_scenes,
        candidate_chunk=candidate_chunk,
        train_candidates=256,
        seed=0,
        epoch=0,
        full_candidates=True,
    ):
        prediction = model(
            batch.current_bev_tokens.to(device),
            batch.ego_status.to(device),
            batch.trajectory.to(device),
        )
        loss, _ = effect_prediction_loss(
            prediction, _targets_to_device(batch.targets, device), weights
        )
        total += float(loss.cpu())
        batches += 1
    if batches == 0:
        raise ProbeDataError("validation effect cache is empty")
    return total / batches


def _train_predictor_trial(
    *,
    train_cache: Path,
    val_cache: Path,
    train_effects: Path,
    val_effects: Path,
    seed: int,
    learning_rate: float,
    config: Mapping[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = CandidateEffectPredictor(
        hidden_dim=int(config["hidden_dim"]),
        decoder_layers=int(config["decoder_layers"]),
        attention_heads=int(config["attention_heads"]),
        actor_slots=int(config["actor_slots"]),
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    max_parameters = int(config["max_parameters"])
    if parameter_count > max_parameters:
        raise RuntimeError(
            f"effect predictor has {parameter_count} parameters; limit is {max_parameters}"
        )
    weights = EffectLossWeights(
        ego=float(config.get("ego_weight", 1.0)),
        map=float(config.get("map_weight", 1.0)),
        actor=float(config.get("actor_weight", 1.0)),
        actor_presence=float(config.get("actor_presence_weight", 0.25)),
        interaction=float(config.get("interaction_weight", 0.5)),
        temporal_consistency=float(config["temporal_consistency_weight"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    batch_scenes = int(config["batch_scenes"])
    candidate_chunk = int(config["candidate_chunk"])
    train_candidates = int(config["train_candidates_per_scene"])
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(max_epochs):
        model.train()
        component_sums: dict[str, float] = {}
        training_loss = 0.0
        epoch_steps = 0
        for batch in iter_effect_batches(
            train_cache,
            train_effects,
            batch_scenes=batch_scenes,
            candidate_chunk=candidate_chunk,
            train_candidates=train_candidates,
            seed=seed,
            epoch=epoch,
            full_candidates=False,
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.current_bev_tokens.to(device),
                batch.ego_status.to(device),
                batch.trajectory.to(device),
            )
            loss, components = effect_prediction_loss(
                prediction, _targets_to_device(batch.targets, device), weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            training_loss += float(loss.detach().cpu())
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu())
            epoch_steps += 1
            steps += 1
        if epoch_steps == 0:
            raise ProbeDataError("effect predictor training cache is empty")
        validation_loss = _validation_loss(
            model,
            val_cache,
            val_effects,
            batch_scenes,
            candidate_chunk,
            device,
            weights,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / epoch_steps,
                "validation_loss": validation_loss,
                "components": {
                    key: value / epoch_steps for key, value in component_sums.items()
                },
            }
        )
        if validation_loss < best_loss - 1.0e-9:
            best_loss = validation_loss
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
        raise RuntimeError("effect predictor did not produce a finite validation checkpoint")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    payload = {
        "schema_version": "candidate_effect_predictor.v1",
        "seed": seed,
        "learning_rate": learning_rate,
        "model_config": {
            key: config[key]
            for key in ("hidden_dim", "decoder_layers", "attention_heads", "actor_slots")
        },
        "input_schema": list(PREDICTOR_INPUT_SCHEMA),
        "target_schema": list(PREDICTOR_TARGET_SCHEMA),
        "loss_weights": asdict(weights),
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_steps": steps,
        "peak_gpu_memory_bytes": peak_memory,
        "model_state_dict": best_state,
        "history": history,
    }
    _atomic_torch_save(output, payload)
    return {
        key: payload[key]
        for key in (
            "seed",
            "learning_rate",
            "trainable_parameters",
            "best_epoch",
            "best_validation_loss",
            "training_steps",
            "peak_gpu_memory_bytes",
            "input_schema",
            "target_schema",
        )
    } | {"checkpoint": str(output)}


def train_predictor_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing predictor output: {args.output}")
    full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = full_config["effect_predictor"]
    seeds = [int(value) for value in full_config["run"]["seeds"]]
    learning_rates = [float(value) for value in config["learning_rates"]]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    all_trials: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for seed in seeds:
        seed_trials: list[dict[str, Any]] = []
        for learning_rate in learning_rates:
            checkpoint = args.output / "search" / f"seed-{seed}-lr-{learning_rate:g}.pt"
            result = _train_predictor_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=args.train_effects,
                val_effects=args.val_effects,
                seed=seed,
                learning_rate=learning_rate,
                config=config,
                device=device,
                output=checkpoint,
            )
            seed_trials.append(result)
            all_trials.append(result)
        selected.append(
            min(
                seed_trials,
                key=lambda value: (
                    float(value["best_validation_loss"]),
                    float(value["learning_rate"]),
                ),
            )
        )
    atomic_write_json(
        args.output / "predictor_manifest.json",
        {
            "schema_version": "effect_predictor_training.v1",
            "config": str(args.config),
            "seeds": seeds,
            "learning_rate_grid": learning_rates,
            "optimizer": "AdamW",
            "early_stopping_metric": "validation_effect_loss",
            "selected_trials": selected,
            "all_trials": all_trials,
        },
    )


def _load_predictor(checkpoint: Path, device: torch.device) -> tuple[CandidateEffectPredictor, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "candidate_effect_predictor.v1":
        raise ValueError(f"unsupported effect predictor checkpoint: {checkpoint}")
    if tuple(payload.get("input_schema", ())) != PREDICTOR_INPUT_SCHEMA:
        raise ValueError("effect predictor input schema permits unexpected information")
    model = CandidateEffectPredictor(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _joined_shards(
    frozen_root: Path, oracle_effect_root: Path
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    frozen = FeatureShardReader(frozen_root)
    effects = FeatureShardReader(oracle_effect_root)
    frozen_iterator = frozen.iter_shards()
    effect_iterator = effects.iter_shards()
    while True:
        try:
            frozen_sidecar, frozen_arrays = next(frozen_iterator)
        except StopIteration:
            try:
                next(effect_iterator)
            except StopIteration:
                return
            raise ProbeDataError("oracle effect cache has more shards than frozen cache")
        try:
            effect_sidecar, effect_arrays = next(effect_iterator)
        except StopIteration as error:
            raise ProbeDataError("oracle effect cache has fewer shards than frozen cache") from error
        if len(frozen_sidecar["records"]) != len(effect_sidecar["records"]):
            raise ProbeDataError("frozen/oracle-effect shard lengths differ")
        for left, right in zip(frozen_sidecar["records"], effect_sidecar["records"]):
            if not _records_match(left, right):
                raise ProbeDataError("frozen/oracle-effect record identity differs")
        yield frozen_sidecar, frozen_arrays, effect_arrays


@torch.inference_mode()
def write_predicted_effect_cache(
    *,
    checkpoint: Path,
    frozen_root: Path,
    oracle_effect_root: Path,
    output: Path,
    device: torch.device,
    candidate_chunk: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing predicted-effect cache: {output}")
    model, payload = _load_predictor(checkpoint, device)
    frozen_reader = FeatureShardReader(frozen_root)
    frozen_identity = frozen_reader.manifest["identity"]
    identity = CacheIdentity(
        run_id=f"{frozen_identity['run_id']}-predicted-effect-seed-{payload['seed']}",
        split=str(frozen_identity["split"]),
        checkpoint_sha256=str(frozen_identity["checkpoint_sha256"]),
        wote_commit_sha=str(frozen_identity["wote_commit_sha"]),
        feature_schema_version="predicted_replay_effect.v1",
    )
    writer = FeatureShardWriter(output, identity)
    for shard_index, (sidecar, frozen, _) in enumerate(
        _joined_shards(frozen_root, oracle_effect_root)
    ):
        shard_values: dict[str, list[npt.NDArray[Any]]] = {}
        records: list[SceneCacheRecord] = []
        for scene_index, record in enumerate(sidecar["records"]):
            current = torch.from_numpy(
                np.asarray(frozen["current_bev_tokens"][scene_index], dtype=np.float32)
            )[None].to(device)
            ego = torch.from_numpy(
                np.asarray(frozen["ego_status_feature"][scene_index], dtype=np.float32)
            )[None].to(device)
            trajectory = np.asarray(frozen["trajectory"][scene_index], dtype=np.float32)
            collected: dict[str, list[Tensor]] = {
                "ego_effect": [],
                "map_effect": [],
                "actor_effect": [],
                "actor_presence_probability": [],
                "interaction_probability": [],
            }
            for start in range(0, 256, candidate_chunk):
                chunk = torch.from_numpy(trajectory[start : start + candidate_chunk])[None].to(device)
                prediction = model(current, ego, chunk)
                collected["ego_effect"].append(
                    decode_transformed_effect(prediction["ego_effect_transformed"])[0].cpu()
                )
                collected["map_effect"].append(
                    decode_transformed_effect(prediction["map_effect_transformed"])[0].cpu()
                )
                collected["actor_effect"].append(
                    decode_transformed_effect(prediction["actor_effect_transformed"])[0].cpu()
                )
                collected["actor_presence_probability"].append(
                    torch.sigmoid(prediction["actor_presence_logits"])[0].cpu()
                )
                collected["interaction_probability"].append(
                    torch.sigmoid(prediction["interaction_logits"])[0].cpu()
                )
            values = {
                key: torch.cat(chunks, dim=0).numpy().astype(np.float32)
                for key, chunks in collected.items()
            }
            actor_mask = values["actor_presence_probability"] >= 0.5
            interaction_mask = (values["interaction_probability"] >= 0.5) & actor_mask
            values["actor_effect"][~actor_mask] = 0.0
            values["actor_mask"] = actor_mask
            values["interaction_mask"] = interaction_mask
            for key, value in values.items():
                if value.shape[0] != 256 or (
                    np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
                ):
                    raise ProbeDataError(
                        f"{record['scene_token']}: invalid predicted effect {key} {value.shape}"
                    )
                shard_values.setdefault(key, []).append(value)
            records.append(
                SceneCacheRecord(
                    scene_token=str(record["scene_token"]),
                    candidate_indices=tuple(record["candidate_indices"]),
                    trajectory_hash=str(record["trajectory_hash"]),
                    label_hash=str(record["label_hash"]),
                )
            )
        writer.write_shard(
            shard_index,
            {key: np.stack(values, axis=0) for key, values in shard_values.items()},
            records,
        )
    writer.finalize()


def _confusion(
    truth: npt.NDArray[np.bool_], predicted: npt.NDArray[np.bool_]
) -> tuple[int, int, int, int]:
    return (
        int((truth & predicted).sum()),
        int((~truth & predicted).sum()),
        int((truth & ~predicted).sum()),
        int((~truth & ~predicted).sum()),
    )


def compare_effect_caches(
    oracle_root: Path, predicted_root: Path, seed: int, split: str
) -> dict[str, Any]:
    oracle = FeatureShardReader(oracle_root)
    predicted = FeatureShardReader(predicted_root)
    pred_iterator = predicted.iter_shards()
    sums = {"ego": 0.0, "map": 0.0, "actor": 0.0, "temporal": 0.0}
    counts = {"ego": 0, "map": 0, "actor": 0, "temporal": 0}
    interaction_confusion = np.zeros(4, dtype=np.int64)
    presence_confusion = np.zeros(4, dtype=np.int64)
    scenes = 0
    for oracle_sidecar, oracle_arrays in oracle.iter_shards():
        try:
            predicted_sidecar, predicted_arrays = next(pred_iterator)
        except StopIteration as error:
            raise ProbeDataError("predicted effect cache has fewer shards") from error
        if oracle_sidecar["records"] != predicted_sidecar["records"]:
            raise ProbeDataError("oracle/predicted effect record mismatch")
        ego_delta = np.abs(
            oracle_arrays["ego_effect"].astype(np.float32)
            - predicted_arrays["ego_effect"].astype(np.float32)
        )
        map_delta = np.abs(
            oracle_arrays["map_effect"].astype(np.float32)
            - predicted_arrays["map_effect"].astype(np.float32)
        )
        actor_delta = np.abs(
            oracle_arrays["actor_effect"].astype(np.float32)
            - predicted_arrays["actor_effect"].astype(np.float32)
        )
        actor_valid = oracle_arrays["actor_mask"].astype(bool)
        sums["ego"] += float(ego_delta.sum())
        counts["ego"] += ego_delta.size
        sums["map"] += float(map_delta.sum())
        counts["map"] += map_delta.size
        expanded_mask = np.broadcast_to(actor_valid[..., None], actor_delta.shape)
        sums["actor"] += float(actor_delta[expanded_mask].sum())
        counts["actor"] += int(expanded_mask.sum())
        temporal_delta = np.abs(
            np.diff(predicted_arrays["ego_effect"].astype(np.float32), axis=2)
            - np.diff(oracle_arrays["ego_effect"].astype(np.float32), axis=2)
        )
        sums["temporal"] += float(temporal_delta.sum())
        counts["temporal"] += temporal_delta.size
        interaction_valid = actor_valid
        interaction_confusion += np.asarray(
            _confusion(
                oracle_arrays["interaction_mask"].astype(bool)[interaction_valid],
                predicted_arrays["interaction_mask"].astype(bool)[interaction_valid],
            )
        )
        presence_confusion += np.asarray(
            _confusion(actor_valid, predicted_arrays["actor_mask"].astype(bool))
        )
        scenes += len(oracle_sidecar["records"])
    try:
        next(pred_iterator)
    except StopIteration:
        pass
    else:
        raise ProbeDataError("predicted effect cache has more shards")
    if not scenes or any(count == 0 for count in counts.values()):
        raise ProbeDataError("effect metric cache is empty or has no valid actors")

    def f1(confusion: npt.NDArray[np.int64]) -> float:
        true_positive, false_positive, false_negative, _ = confusion
        denominator = 2 * true_positive + false_positive + false_negative
        return float(2 * true_positive / denominator) if denominator else float("nan")

    return {
        "seed": seed,
        "split": split,
        "scene_count": scenes,
        "ego_effect_mae": sums["ego"] / counts["ego"],
        "map_effect_mae": sums["map"] / counts["map"],
        "actor_effect_masked_mae": sums["actor"] / counts["actor"],
        "ego_temporal_delta_mae": sums["temporal"] / counts["temporal"],
        "interaction_mask_f1": f1(interaction_confusion),
        "actor_presence_f1": f1(presence_confusion),
        "interaction_confusion_tp_fp_fn_tn": interaction_confusion.tolist(),
        "presence_confusion_tp_fp_fn_tn": presence_confusion.tolist(),
    }


def predict_cache_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing predicted cache suite: {args.output}")
    manifest = json.loads((args.predictor_root / "predictor_manifest.json").read_text())
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["effect_predictor"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics: list[dict[str, Any]] = []
    split_inputs = {
        "train": (args.train_cache, args.train_effects),
        "val": (args.val_cache, args.val_effects),
        "test": (args.test_cache, args.test_effects),
    }
    for trial in manifest["selected_trials"]:
        seed = int(trial["seed"])
        for split, (frozen, oracle) in split_inputs.items():
            output = args.output / f"seed-{seed}" / split
            write_predicted_effect_cache(
                checkpoint=Path(trial["checkpoint"]),
                frozen_root=frozen,
                oracle_effect_root=oracle,
                output=output,
                device=device,
                candidate_chunk=int(config["candidate_chunk"]),
            )
            if split == "test":
                metrics.append(compare_effect_caches(oracle, output, seed, split))
    pd.DataFrame(metrics).to_csv(args.output / "effect_prediction_metrics.csv", index=False)


def train_g3_scorers(args: argparse.Namespace) -> None:
    """Train predicted/full/environment scorers and reuse frozen G2 controls."""

    if args.output.exists():
        raise FileExistsError(f"refusing existing G3 scorer output: {args.output}")
    full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_config = full_config["probe"]
    seeds = [int(value) for value in full_config["run"]["seeds"]]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    pairwise_weights = [
        float(value)
        for value in probe_config.get(
            "pairwise_weights", [probe_config.get("pairwise_weight", 0.5)]
        )
    ]
    g2_manifest = json.loads((args.g2_training_root / "training_manifest.json").read_text())
    reused = [
        trial
        for trial in g2_manifest["selected_trials"]
        if trial["model_type"] in {"direct_current", "oracle_replay_effect"}
    ]
    expected_reused = {
        (model, seed)
        for model in ("direct_current", "oracle_replay_effect")
        for seed in seeds
    }
    actual_reused = {(trial["model_type"], int(trial["seed"])) for trial in reused}
    if actual_reused != expected_reused:
        raise ValueError(f"G2 selected trials incomplete: {actual_reused}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    all_trials: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = list(reused)
    for model_type, seed in itertools.product(
        ("predicted_replay_effect", "wote_full_future", "wote_environment_only"),
        seeds,
    ):
        train_effects = (
            args.predicted_effect_root / f"seed-{seed}" / "train"
            if model_type == "predicted_replay_effect"
            else args.train_effects
        )
        val_effects = (
            args.predicted_effect_root / f"seed-{seed}" / "val"
            if model_type == "predicted_replay_effect"
            else args.val_effects
        )
        model_trials: list[dict[str, Any]] = []
        for learning_rate, pairwise_weight in itertools.product(
            learning_rates, pairwise_weights
        ):
            checkpoint = (
                args.output
                / "search"
                / model_type
                / f"seed-{seed}-lr-{learning_rate:g}-pair-{pairwise_weight:g}.pt"
            )
            result = _train_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=train_effects,
                val_effects=val_effects,
                model_type=model_type,
                seed=seed,
                learning_rate=learning_rate,
                pairwise_weight=pairwise_weight,
                hidden_dim=int(probe_config["hidden_dim"]),
                max_epochs=int(probe_config["max_epochs"]),
                patience=int(probe_config["patience"]),
                batch_scenes=int(probe_config["batch_scenes"]),
                train_candidates=int(probe_config["train_candidates_per_scene"]),
                device=device,
                output=checkpoint,
            )
            model_trials.append(result)
            all_trials.append(result)
        selected.append(
            min(
                model_trials,
                key=lambda row: (
                    float(row["validation_regret"]),
                    -float(row["validation_selected_pdms"]),
                    float(row["learning_rate"]),
                    float(row["pairwise_weight"]),
                ),
            )
        )
    parameter_counts = {
        int(trial["parameter_audit"]["trainable_parameters"]) for trial in selected
    }
    if len(parameter_counts) != 1:
        raise RuntimeError(f"G2/G3 scorer capacities differ: {parameter_counts}")
    atomic_write_json(
        args.output / "training_manifest.json",
        {
            "schema_version": "g3_scorer_training.v1",
            "config": str(args.config),
            "seeds": seeds,
            "optimizer": "AdamW",
            "learning_rate_grid": learning_rates,
            "pairwise_weight_grid": pairwise_weights,
            "trainable_parameter_count": next(iter(parameter_counts)),
            "reused_g2_trials": reused,
            "selected_trials": selected,
            "new_trials": all_trials,
        },
    )


def _paired_selected_arrays(
    outcomes: pd.DataFrame, left: str, right: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    left_frame = (
        outcomes[outcomes["model"] == left]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    right_frame = (
        outcomes[outcomes["model"] == right]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    if not left_frame.index.equals(right_frame.index):
        raise ValueError(f"G3 paired scene identities differ: {left}/{right}")
    return left_frame.to_numpy(), right_frame.to_numpy()


def summarize_g3(
    metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
    predictor_manifest: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "direct_current",
        "oracle_replay_effect",
        "predicted_replay_effect",
        "predicted_replay_effect_swap",
        "wote_full_future",
        "wote_environment_only",
    }
    missing = sorted(required - set(metrics["model"]))
    if missing:
        raise ValueError(f"G3 metrics missing {missing}")
    seed_sets = {
        model: set(metrics.loc[metrics["model"] == model, "seed"].astype(int))
        for model in required
    }
    if len({tuple(sorted(values)) for values in seed_sets.values()}) != 1:
        raise ValueError(f"G3 model seed sets differ: {seed_sets}")
    per_seed: list[dict[str, Any]] = []
    for seed in sorted(seed_sets["direct_current"]):
        rows = metrics[metrics["seed"] == seed].set_index("model")
        direct_regret = float(rows.loc["direct_current", "top1_regret"])
        oracle_regret = float(rows.loc["oracle_replay_effect", "top1_regret"])
        predicted_regret = float(rows.loc["predicted_replay_effect", "top1_regret"])
        direct_pdms = float(rows.loc["direct_current", "selected_pdms"])
        oracle_pdms = float(rows.loc["oracle_replay_effect", "selected_pdms"])
        predicted_pdms = float(rows.loc["predicted_replay_effect", "selected_pdms"])
        regret_denominator = direct_regret - oracle_regret
        pdms_denominator = oracle_pdms - direct_pdms
        per_seed.append(
            {
                "seed": int(seed),
                "oracle_regret_gain_recovered_fraction": (
                    (direct_regret - predicted_regret) / regret_denominator
                    if regret_denominator > 0
                    else -float("inf")
                ),
                "oracle_pdms_gain_recovered_fraction": (
                    (predicted_pdms - direct_pdms) / pdms_denominator
                    if pdms_denominator > 0
                    else -float("inf")
                ),
                "predicted_vs_direct_regret_reduction_fraction": (
                    (direct_regret - predicted_regret) / direct_regret
                    if direct_regret > 0
                    else -float("inf")
                ),
                "predicted_vs_direct_pdms_gain": predicted_pdms - direct_pdms,
                "predicted_swap_pdms_drop": predicted_pdms
                - float(rows.loc["predicted_replay_effect_swap", "selected_pdms"]),
            }
        )
    averaged = metrics.groupby("model").mean(numeric_only=True)
    direct_regret = float(averaged.loc["direct_current", "top1_regret"])
    oracle_regret = float(averaged.loc["oracle_replay_effect", "top1_regret"])
    predicted_regret = float(averaged.loc["predicted_replay_effect", "top1_regret"])
    direct_pdms = float(averaged.loc["direct_current", "selected_pdms"])
    oracle_pdms = float(averaged.loc["oracle_replay_effect", "selected_pdms"])
    predicted_pdms = float(averaged.loc["predicted_replay_effect", "selected_pdms"])
    regret_recovery = (
        (direct_regret - predicted_regret) / (direct_regret - oracle_regret)
        if direct_regret > oracle_regret
        else -float("inf")
    )
    pdms_recovery = (
        (predicted_pdms - direct_pdms) / (oracle_pdms - direct_pdms)
        if oracle_pdms > direct_pdms
        else -float("inf")
    )
    direct_regret_reduction = (
        (direct_regret - predicted_regret) / direct_regret
        if direct_regret > 0
        else -float("inf")
    )
    direct_pdms_gain = predicted_pdms - direct_pdms
    predicted_scene, swap_scene = _paired_selected_arrays(
        outcomes, "predicted_replay_effect", "predicted_replay_effect_swap"
    )
    swap_ci = paired_scene_bootstrap(
        predicted_scene,
        swap_scene,
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]) + 31,
    )
    schema_clean = all(
        tuple(trial.get("input_schema", ())) == PREDICTOR_INPUT_SCHEMA
        for trial in predictor_manifest["selected_trials"]
    )
    conditions = {
        "recovers_at_least_30pct_oracle_gain": max(regret_recovery, pdms_recovery) >= 0.30,
        "improves_direct_by_required_absolute_or_relative_margin": (
            direct_regret_reduction >= 0.10 or direct_pdms_gain >= 0.003
        ),
        "predicted_effect_swap_significantly_worse": swap_ci.lower > 0
        and all(row["predicted_swap_pdms_drop"] > 0 for row in per_seed),
        "no_full_future_ego_trajectory_input": schema_clean,
    }
    gate_pass = all(conditions.values())
    return {
        "schema_version": "gate_g3.v1",
        "gate_g3_pass": gate_pass,
        "conditions": conditions,
        "oracle_regret_gain_recovered_fraction": regret_recovery,
        "oracle_pdms_gain_recovered_fraction": pdms_recovery,
        "predicted_vs_direct_regret_reduction_fraction": direct_regret_reduction,
        "predicted_vs_direct_pdms_gain_raw": direct_pdms_gain,
        "predicted_vs_direct_pdms_gain_points": direct_pdms_gain * 100.0,
        "paired_scene_bootstrap_predicted_minus_swap": asdict(swap_ci),
        "per_seed": per_seed,
        "verdict_if_g2_passed": (
            "G3_PASS"
            if gate_pass
            else "EFFECT_TARGET_VALID_BUT_PREDICTION_BOTTLENECK"
        ),
    }


def evaluate_g3(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing G3 evaluation output: {args.output}")
    full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = json.loads((args.training_root / "training_manifest.json").read_text())
    predictor_manifest = json.loads(
        (args.predictor_root / "predictor_manifest.json").read_text()
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    metric_rows: list[dict[str, Any]] = []
    outcome_frames: list[pd.DataFrame] = []
    for trial in training["selected_trials"]:
        model_type = str(trial["model_type"])
        seed = int(trial["seed"])
        effect_root = (
            args.predicted_effect_root / f"seed-{seed}" / "test"
            if model_type == "predicted_replay_effect"
            else args.test_effects
        )
        result = evaluate_checkpoint(
            Path(trial["checkpoint"]),
            args.test_cache,
            effect_root,
            int(full_config["probe"]["batch_scenes"]),
            device,
        )
        outcomes = scene_outcomes(result)
        outcomes["gate"] = "G3"
        metrics = aggregate_metrics(result, outcomes)
        metrics["gate"] = "G3"
        metric_rows.append(metrics)
        outcome_frames.append(outcomes)
        if model_type == "predicted_replay_effect":
            swapped = evaluate_checkpoint(
                Path(trial["checkpoint"]),
                args.test_cache,
                effect_root,
                int(full_config["probe"]["batch_scenes"]),
                device,
                swap_effects=True,
            )
            swapped_outcomes = scene_outcomes(swapped)
            swapped_outcomes["gate"] = "G3"
            swapped_metrics = aggregate_metrics(swapped, swapped_outcomes)
            swapped_metrics["gate"] = "G3"
            metric_rows.append(swapped_metrics)
            outcome_frames.append(swapped_outcomes)
    metrics_frame = pd.DataFrame(metric_rows)
    outcomes_frame = pd.concat(outcome_frames, ignore_index=True)
    summary = summarize_g3(
        metrics_frame,
        outcomes_frame,
        predictor_manifest,
        full_config["bootstrap"],
    )
    metrics_frame.to_csv(args.output / "probe_metrics_g3.csv", index=False)
    outcomes_frame.to_parquet(args.output / "scene_level_g3.parquet", index=False)
    prediction_metrics = pd.read_csv(
        args.predicted_effect_root / "effect_prediction_metrics.csv"
    )
    prediction_metrics.to_csv(args.output / "effect_prediction_metrics.csv", index=False)
    atomic_write_json(args.output / "g3_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-predictor")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--val-cache", type=Path, required=True)
    train.add_argument("--train-effects", type=Path, required=True)
    train.add_argument("--val-effects", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")

    predict = subparsers.add_parser("predict-caches")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--predictor-root", type=Path, required=True)
    for split in ("train", "val", "test"):
        predict.add_argument(f"--{split}-cache", type=Path, required=True)
        predict.add_argument(f"--{split}-effects", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cuda")

    scorers = subparsers.add_parser("train-scorers")
    scorers.add_argument("--config", type=Path, required=True)
    scorers.add_argument("--g2-training-root", type=Path, required=True)
    scorers.add_argument("--predicted-effect-root", type=Path, required=True)
    scorers.add_argument("--train-cache", type=Path, required=True)
    scorers.add_argument("--val-cache", type=Path, required=True)
    scorers.add_argument("--train-effects", type=Path, required=True)
    scorers.add_argument("--val-effects", type=Path, required=True)
    scorers.add_argument("--output", type=Path, required=True)
    scorers.add_argument("--device", default="cuda")

    evaluate = subparsers.add_parser("evaluate-g3")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--training-root", type=Path, required=True)
    evaluate.add_argument("--predictor-root", type=Path, required=True)
    evaluate.add_argument("--predicted-effect-root", type=Path, required=True)
    evaluate.add_argument("--test-cache", type=Path, required=True)
    evaluate.add_argument("--test-effects", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train-predictor":
        train_predictor_suite(args)
    elif args.command == "predict-caches":
        predict_cache_suite(args)
    elif args.command == "train-scorers":
        train_g3_scorers(args)
    elif args.command == "evaluate-g3":
        evaluate_g3(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
