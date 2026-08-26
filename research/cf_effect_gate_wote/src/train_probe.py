"""Build replay-effect caches and train matched-capacity G2 factor probes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
import yaml
from torch import Tensor

from .feature_store import (
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    atomic_write_json,
)
from .models.probe_heads import (
    AUXILIARY_RAW_DIM,
    BASE_RAW_DIM,
    CURRENT_RAW_DIM,
    MatchedCapacityFactorProbe,
    MatchedInputComposer,
    audit_parameters,
    factorized_probe_loss,
    pairwise_ranking_loss,
)
from .replay_effect_builder import (
    ACTOR_EFFECT_NAMES,
    EffectBuilderConfig,
    InteractionThresholds,
    ReplayEffectTensors,
    ReplayGroundedEffectBuilder,
    context_from_navsim_scene,
)


G2_MODEL_TYPES = (
    "trajectory_only",
    "direct_current",
    "shared_logged_future",
    "oracle_replay_effect",
)
ALL_SCORER_TYPES = G2_MODEL_TYPES + (
    "predicted_replay_effect",
    "wote_full_future",
    "wote_environment_only",
)


class ProbeDataError(RuntimeError):
    """A frozen feature, effect, label, or scene identity is inconsistent."""


@dataclass(frozen=True)
class ProbeScene:
    token: str
    frozen: Mapping[str, npt.NDArray[Any]]
    effects: Mapping[str, npt.NDArray[Any]] | None


@dataclass(frozen=True)
class RawProbeBatch:
    tokens: tuple[str, ...]
    base: Tensor
    current: Tensor
    auxiliary: Tensor
    factor_labels: Tensor
    selected_indices: npt.NDArray[np.int64]


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _records_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    keys = ("scene_token", "candidate_indices", "trajectory_hash", "label_hash")
    return all(first.get(key) == second.get(key) for key in keys)


def iter_probe_scenes(
    frozen_root: Path,
    effect_root: Path | None,
) -> Iterator[ProbeScene]:
    """Join two strict sharded stores without loading an all-scenes index."""

    frozen_reader = FeatureShardReader(frozen_root)
    effect_reader = FeatureShardReader(effect_root) if effect_root is not None else None
    effect_iterator = effect_reader.iter_shards() if effect_reader is not None else None
    scene_count = 0
    for frozen_sidecar, frozen_arrays in frozen_reader.iter_shards():
        if effect_iterator is None:
            effect_sidecar = None
            effect_arrays = None
        else:
            try:
                effect_sidecar, effect_arrays = next(effect_iterator)
            except StopIteration as error:
                raise ProbeDataError("effect cache has fewer shards than frozen cache") from error
            if len(frozen_sidecar["records"]) != len(effect_sidecar["records"]):
                raise ProbeDataError("frozen/effect shard scene counts differ")
        for scene_index, frozen_record in enumerate(frozen_sidecar["records"]):
            if effect_sidecar is not None:
                effect_record = effect_sidecar["records"][scene_index]
                if not _records_match(frozen_record, effect_record):
                    raise ProbeDataError(
                        f"frozen/effect record mismatch at {frozen_record['scene_token']}"
                    )
            frozen_scene = {key: value[scene_index] for key, value in frozen_arrays.items()}
            effect_scene = (
                {key: value[scene_index] for key, value in effect_arrays.items()}
                if effect_arrays is not None
                else None
            )
            token = str(frozen_record["scene_token"])
            _validate_frozen_scene(token, frozen_scene)
            if effect_scene is not None:
                _validate_effect_scene(token, effect_scene, len(frozen_scene["trajectory"]))
            scene_count += 1
            yield ProbeScene(token=token, frozen=frozen_scene, effects=effect_scene)
    if effect_iterator is not None:
        try:
            next(effect_iterator)
        except StopIteration:
            pass
        else:
            raise ProbeDataError("effect cache has more shards than frozen cache")
    expected = int(frozen_reader.manifest["scene_count"])
    if scene_count != expected:
        raise ProbeDataError(f"read {scene_count} scenes but manifest claims {expected}")


def _validate_frozen_scene(token: str, scene: Mapping[str, npt.NDArray[Any]]) -> None:
    required = {
        "current_bev_pool",
        "ego_status_feature",
        "trajectory",
        "factor_labels",
        "selected_index",
    }
    missing = sorted(required - scene.keys())
    if missing:
        raise ProbeDataError(f"{token}: frozen cache missing {missing}")
    trajectory = np.asarray(scene["trajectory"])
    factors = np.asarray(scene["factor_labels"])
    if trajectory.shape != (256, 8, 3) or factors.shape != (256, 5):
        raise ProbeDataError(
            f"{token}: trajectory/factor shapes are {trajectory.shape}/{factors.shape}"
        )
    if not np.isfinite(trajectory).all() or not np.isfinite(factors).all():
        raise ProbeDataError(f"{token}: frozen cache contains NaN/Inf")
    selected = np.asarray(scene["selected_index"])
    if selected.size != 1 or not 0 <= int(selected.reshape(-1)[0]) < 256:
        raise ProbeDataError(f"{token}: invalid selected index {selected}")


def _validate_effect_scene(
    token: str, scene: Mapping[str, npt.NDArray[Any]], candidates: int
) -> None:
    required_shapes = {
        "ego_effect": (candidates, 8, 16),
        "map_effect": (candidates, 8, 8),
        "actor_effect": (candidates, 8, 16, len(ACTOR_EFFECT_NAMES)),
        "actor_mask": (candidates, 8, 16),
        "interaction_mask": (candidates, 8, 16),
        "shared_logged_future": (8, 16, 8),
        "shared_actor_mask": (8, 16),
    }
    for key, shape in required_shapes.items():
        if key not in scene or np.asarray(scene[key]).shape != shape:
            actual = None if key not in scene else np.asarray(scene[key]).shape
            raise ProbeDataError(f"{token}: {key} expected {shape}, got {actual}")
    for key, value in scene.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ProbeDataError(f"{token}: effect {key} contains NaN/Inf")


def _interaction_mask_for_clearance(
    effect: ReplayEffectTensors,
    thresholds: InteractionThresholds,
) -> npt.NDArray[np.bool_]:
    name_to_index = {name: index for index, name in enumerate(ACTOR_EFFECT_NAMES)}
    actor = effect.actor_effect
    mask = effect.actor_mask
    interaction = (
        (actor[..., name_to_index["oriented_box_clearance"]] < thresholds.clearance_m)
        | (
            actor[..., name_to_index["swept_box_distance"]]
            < thresholds.conflict_zone_clearance_m
        )
        | (
            (actor[..., name_to_index["time_to_closest_approach"]] < thresholds.tca_seconds)
            & (
                actor[..., name_to_index["distance_at_closest_approach"]]
                < thresholds.tca_distance_m
            )
        )
    )
    return np.asarray(interaction & mask, dtype=bool)


def _shared_logged_future(
    context: Any, effect: ReplayEffectTensors, actor_slots: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    output = np.zeros((8, actor_slots, 8), dtype=np.float32)
    mask = np.zeros((8, actor_slots), dtype=bool)
    actors = context.logged_actors
    for slot, actor_index in enumerate(effect.selected_actor_indices):
        valid = actors.valid[:, actor_index]
        mask[:, slot] = valid
        output[:, slot, 0:2] = actors.positions[:, actor_index]
        output[:, slot, 2] = np.sin(actors.headings[:, actor_index])
        output[:, slot, 3] = np.cos(actors.headings[:, actor_index])
        output[:, slot, 4:6] = actors.velocities[:, actor_index]
        output[:, slot, 6:8] = actors.sizes[:, actor_index]
        output[~valid, slot] = 0.0
    return output, mask


def cache_replay_effects(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing effect cache: {args.output}")
    frozen_reader = FeatureShardReader(args.frozen_cache)
    frozen_identity = frozen_reader.manifest["identity"]
    if int(frozen_identity["candidate_count"]) != 256 or int(frozen_identity["horizon"]) != 8:
        raise ProbeDataError("effect Gate requires exactly 256 candidates and eight steps")

    sys.path.insert(0, str(args.wote_root))
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    all_tokens = [
        str(record["scene_token"])
        for shard in frozen_reader.manifest["shards"]
        for record in json.loads(
            (args.frozen_cache / shard["sidecar"]).read_text(encoding="utf-8")
        )["records"]
    ]
    if len(all_tokens) != len(set(all_tokens)):
        raise ProbeDataError("frozen cache contains duplicate scene tokens")
    scene_loader = SceneLoader(
        data_path=args.data_root / "navsim_logs/trainval",
        sensor_blobs_path=args.data_root / "sensor_blobs/trainval",
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            tokens=all_tokens,
        ),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if set(scene_loader.tokens) != set(all_tokens):
        raise ProbeDataError("scene loader tokens do not exactly match frozen cache")
    metric_loader = MetricCacheLoader(args.metric_cache)
    missing_metric = sorted(set(all_tokens) - set(metric_loader.tokens))
    if missing_metric:
        raise ProbeDataError(
            f"metric cache missing {len(missing_metric)} scenes; first={missing_metric[:5]}"
        )
    builder = ReplayGroundedEffectBuilder(
        EffectBuilderConfig(
            actor_slots=args.actor_slots,
            interval_seconds=args.interval_seconds,
            ego_length_m=args.ego_length_m,
            ego_width_m=args.ego_width_m,
            interaction=InteractionThresholds(
                clearance_m=args.clearance_m,
                tca_seconds=args.tca_seconds,
                tca_distance_m=args.tca_distance_m,
                conflict_zone_clearance_m=args.conflict_zone_clearance_m,
            ),
        )
    )
    identity = CacheIdentity(
        run_id=f"{frozen_identity['run_id']}-effects",
        split=str(frozen_identity["split"]),
        checkpoint_sha256=str(frozen_identity["checkpoint_sha256"]),
        wote_commit_sha=str(frozen_identity["wote_commit_sha"]),
        feature_schema_version="replay_effect.v1",
    )
    writer = FeatureShardWriter(args.output, identity)
    for shard_index, (sidecar, arrays) in enumerate(frozen_reader.iter_shards()):
        shard_values: dict[str, list[npt.NDArray[Any]]] = {}
        records: list[SceneCacheRecord] = []
        for scene_index, record in enumerate(sidecar["records"]):
            token = str(record["scene_token"])
            scene = scene_loader.get_scene_from_token(token)
            context = context_from_navsim_scene(scene, metric_loader.get_from_token(token))
            trajectories = np.asarray(arrays["trajectory"][scene_index], dtype=np.float32)
            effect = builder.build(trajectories, context)
            shared, shared_mask = _shared_logged_future(context, effect, args.actor_slots)
            values = effect.as_tensor_dict()
            values["shared_logged_future"] = shared
            values["shared_actor_mask"] = shared_mask
            for sensitivity in args.sensitivity_clearance_m:
                sensitivity_thresholds = InteractionThresholds(
                    clearance_m=float(sensitivity),
                    tca_seconds=args.tca_seconds,
                    tca_distance_m=args.tca_distance_m,
                    conflict_zone_clearance_m=args.conflict_zone_clearance_m,
                )
                values[f"interaction_mask_clearance_{sensitivity:g}m"] = (
                    _interaction_mask_for_clearance(effect, sensitivity_thresholds)
                )
            for key, value in values.items():
                shard_values.setdefault(key, []).append(np.asarray(value))
            records.append(
                SceneCacheRecord(
                    scene_token=token,
                    candidate_indices=tuple(record["candidate_indices"]),
                    trajectory_hash=str(record["trajectory_hash"]),
                    label_hash=str(record["label_hash"]),
                )
            )
        writer.write_shard(
            shard_index,
            {key: np.stack(value, axis=0) for key, value in shard_values.items()},
            records,
        )
    writer.finalize()


def _masked_actor_summary(
    values: npt.NDArray[Any], mask: npt.NDArray[Any]
) -> npt.NDArray[np.float32]:
    actor = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if actor.ndim != 4 or valid.shape != actor.shape[:-1]:
        raise ProbeDataError(f"actor summary shape mismatch: {actor.shape}/{valid.shape}")
    counts = valid.sum(axis=2, keepdims=True)
    denominator = np.maximum(counts, 1)
    mean = (actor * valid[..., None]).sum(axis=2) / denominator
    minimum = np.where(valid[..., None], actor, np.inf).min(axis=2)
    maximum = np.where(valid[..., None], actor, -np.inf).max(axis=2)
    empty = counts[..., 0] == 0
    minimum[empty] = 0.0
    maximum[empty] = 0.0
    return np.concatenate([mean, minimum, maximum], axis=-1).reshape(len(actor), -1)


def summarize_replay_effect(effects: Mapping[str, npt.NDArray[Any]]) -> npt.NDArray[np.float32]:
    ego = np.asarray(effects["ego_effect"], dtype=np.float32)
    map_effect = np.asarray(effects["map_effect"], dtype=np.float32)
    actor_mask = np.asarray(effects["actor_mask"], dtype=bool)
    interaction = np.asarray(effects["interaction_mask"], dtype=bool)
    actor = _masked_actor_summary(effects["actor_effect"], actor_mask)
    valid_counts = np.maximum(actor_mask.sum(axis=2), 1)
    valid_fraction = actor_mask.mean(axis=2)
    interaction_fraction = (interaction & actor_mask).sum(axis=2) / valid_counts
    summary = np.concatenate(
        [
            ego.reshape(len(ego), -1),
            map_effect.reshape(len(map_effect), -1),
            actor,
            valid_fraction,
            interaction_fraction,
        ],
        axis=-1,
    ).astype(np.float32)
    if summary.shape[1] > AUXILIARY_RAW_DIM or not np.isfinite(summary).all():
        raise ProbeDataError(f"replay effect summary has invalid shape/data {summary.shape}")
    return summary


def summarize_shared_logged_future(
    values: npt.NDArray[Any], mask: npt.NDArray[Any]
) -> npt.NDArray[np.float32]:
    actor = np.asarray(values, dtype=np.float32)[None]
    valid = np.asarray(mask, dtype=bool)[None]
    summary = _masked_actor_summary(actor, valid)[0]
    extras = np.concatenate([valid.mean(axis=2).reshape(-1), valid.sum(axis=2).reshape(-1)])
    output = np.concatenate([summary, extras]).astype(np.float32)
    if len(output) > AUXILIARY_RAW_DIM or not np.isfinite(output).all():
        raise ProbeDataError(f"shared future summary is invalid: {output.shape}")
    return output


def _pad_numpy(value: npt.NDArray[Any], width: int, name: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] > width:
        raise ProbeDataError(f"{name} has {array.shape[-1]} features; limit is {width}")
    if not np.isfinite(array).all():
        raise ProbeDataError(f"{name} contains NaN/Inf")
    output = np.zeros(array.shape[:-1] + (width,), dtype=np.float32)
    output[..., : array.shape[-1]] = array
    return output


def raw_scene_inputs(
    scene: ProbeScene,
    model_type: str,
    candidate_indices: npt.NDArray[np.int64] | None = None,
    effect_permutation: npt.NDArray[np.int64] | None = None,
    effect_override: Mapping[str, npt.NDArray[Any]] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], int]:
    if model_type not in ALL_SCORER_TYPES:
        raise ValueError(f"unsupported scorer type: {model_type}")
    frozen = scene.frozen
    trajectories = np.asarray(frozen["trajectory"], dtype=np.float32)
    candidates = len(trajectories)
    indices = (
        np.arange(candidates, dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64)
    )
    if indices.ndim != 1 or len(indices) == 0 or np.any((indices < 0) | (indices >= candidates)):
        raise ProbeDataError(f"{scene.token}: invalid candidate subset")
    trajectory = trajectories[indices].reshape(len(indices), -1)
    ego = np.asarray(frozen["ego_status_feature"], dtype=np.float32).reshape(-1)
    ego_repeated = np.broadcast_to(ego[None], (len(indices), len(ego)))
    base = _pad_numpy(np.concatenate([trajectory, ego_repeated], axis=-1), BASE_RAW_DIM, "base")

    include_current = model_type != "trajectory_only"
    current_pool = np.asarray(frozen["current_bev_pool"], dtype=np.float32).reshape(-1)
    if len(current_pool) != CURRENT_RAW_DIM:
        raise ProbeDataError(
            f"{scene.token}: current BEV pool expected 256, got {len(current_pool)}"
        )
    current = np.broadcast_to(current_pool[None], (len(indices), CURRENT_RAW_DIM)).copy()
    if not include_current:
        current.fill(0.0)

    auxiliary = np.zeros((len(indices), AUXILIARY_RAW_DIM), dtype=np.float32)
    effects = effect_override if effect_override is not None else scene.effects
    if model_type in {"oracle_replay_effect", "predicted_replay_effect"}:
        if effects is None:
            raise ProbeDataError(f"{scene.token}: {model_type} requires replay effects")
        effect_summary = summarize_replay_effect(effects)
        if effect_permutation is not None:
            permutation = np.asarray(effect_permutation, dtype=np.int64)
            if sorted(permutation.tolist()) != list(range(candidates)):
                raise ProbeDataError(f"{scene.token}: effect permutation is not bijective")
            effect_summary = effect_summary[permutation]
        auxiliary[:, : effect_summary.shape[1]] = effect_summary[indices]
    elif model_type == "shared_logged_future":
        if effects is None:
            raise ProbeDataError(f"{scene.token}: shared scorer requires logged future")
        shared = summarize_shared_logged_future(
            effects["shared_logged_future"], effects["shared_actor_mask"]
        )
        auxiliary[:, : len(shared)] = shared[None]
    elif model_type == "wote_full_future":
        reward = np.asarray(frozen["reward_feature"], dtype=np.float32).reshape(candidates, -1)
        auxiliary[:, : reward.shape[1]] = reward[indices]
    elif model_type == "wote_environment_only":
        environment = np.asarray(
            frozen["environment_only_future"], dtype=np.float32
        ).reshape(candidates, -1)
        if environment.shape[1] > AUXILIARY_RAW_DIM:
            raise ProbeDataError("environment-only future exceeds auxiliary schema")
        auxiliary[:, : environment.shape[1]] = environment[indices]

    labels = np.asarray(frozen["factor_labels"], dtype=np.float32)[indices]
    selected = int(np.asarray(frozen["selected_index"]).reshape(-1)[0])
    return base, current, auxiliary, labels, selected


def _candidate_subset(token: str, candidates: int, count: int, seed: int, epoch: int) -> npt.NDArray[np.int64]:
    if count >= candidates:
        return np.arange(candidates, dtype=np.int64)
    digest = hashlib.sha256(f"{token}:{seed}:{epoch}".encode("utf-8")).digest()
    scene_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(scene_seed)
    return np.sort(rng.choice(candidates, size=count, replace=False)).astype(np.int64)


def iter_raw_batches(
    frozen_root: Path,
    effect_root: Path | None,
    model_type: str,
    batch_scenes: int,
    candidate_count: int,
    seed: int,
    epoch: int,
    full_candidates: bool,
) -> Iterator[RawProbeBatch]:
    pending: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene in iter_probe_scenes(frozen_root, effect_root):
        subset = None
        if not full_candidates:
            subset = _candidate_subset(scene.token, 256, candidate_count, seed, epoch)
        raw = raw_scene_inputs(scene, model_type, subset)
        pending.append((scene.token, raw))
        if len(pending) < batch_scenes:
            continue
        yield _stack_raw_batch(pending)
        pending.clear()
    if pending:
        yield _stack_raw_batch(pending)


def _stack_raw_batch(
    scenes: Sequence[tuple[str, tuple[npt.NDArray[Any], ...]]]
) -> RawProbeBatch:
    candidate_counts = {item[1][0].shape[0] for item in scenes}
    if len(candidate_counts) != 1:
        raise ProbeDataError(f"batch candidate counts differ: {candidate_counts}")
    return RawProbeBatch(
        tokens=tuple(item[0] for item in scenes),
        base=torch.from_numpy(np.stack([item[1][0] for item in scenes])),
        current=torch.from_numpy(np.stack([item[1][1] for item in scenes])),
        auxiliary=torch.from_numpy(np.stack([item[1][2] for item in scenes])),
        factor_labels=torch.from_numpy(np.stack([item[1][3] for item in scenes])),
        selected_indices=np.asarray([item[1][4] for item in scenes], dtype=np.int64),
    )


def _pdms_tensor(factors: Tensor) -> Tensor:
    nc, dac, ep, ttc, comfort = factors.unbind(dim=-1)
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


@torch.inference_mode()
def validation_objective(
    composer: MatchedInputComposer,
    probe: MatchedCapacityFactorProbe,
    frozen_root: Path,
    effect_root: Path | None,
    model_type: str,
    batch_scenes: int,
    device: torch.device,
) -> tuple[float, float]:
    composer.eval()
    probe.eval()
    selected_sum = 0.0
    regret_sum = 0.0
    scenes = 0
    for batch in iter_raw_batches(
        frozen_root,
        effect_root,
        model_type,
        batch_scenes,
        256,
        seed=0,
        epoch=0,
        full_candidates=True,
    ):
        common = composer(
            batch.base.to(device),
            batch.current.to(device),
            batch.auxiliary.to(device),
        )
        prediction = probe(common)["score"]
        target_score = _pdms_tensor(batch.factor_labels.to(device))
        selected = prediction.argmax(dim=1)
        rows = torch.arange(len(selected), device=device)
        selected_true = target_score[rows, selected]
        oracle = target_score.max(dim=1).values
        selected_sum += float(selected_true.sum().cpu())
        regret_sum += float((oracle - selected_true).sum().cpu())
        scenes += len(selected)
    if scenes == 0:
        raise ProbeDataError("validation cache is empty")
    return regret_sum / scenes, selected_sum / scenes


def _train_trial(
    *,
    train_cache: Path,
    val_cache: Path,
    train_effects: Path | None,
    val_effects: Path | None,
    model_type: str,
    seed: int,
    learning_rate: float,
    pairwise_weight: float,
    hidden_dim: int,
    max_epochs: int,
    patience: int,
    batch_scenes: int,
    train_candidates: int,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    composer = MatchedInputComposer().to(device)
    probe = MatchedCapacityFactorProbe(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    pair_generator = torch.Generator(device=device)
    pair_generator.manual_seed(seed + 91_337)
    best_regret = float("inf")
    best_selected = -float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    steps = 0
    stale = 0
    peak_memory = 0
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(max_epochs):
        composer.train()
        probe.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in iter_raw_batches(
            train_cache,
            train_effects,
            model_type,
            batch_scenes,
            train_candidates,
            seed,
            epoch,
            full_candidates=False,
        ):
            base = batch.base.to(device)
            current = batch.current.to(device)
            auxiliary = batch.auxiliary.to(device)
            targets = batch.factor_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_values = probe(composer(base, current, auxiliary))
            factor_loss = factorized_probe_loss(output_values["logits"], targets)
            ranking_loss = pairwise_ranking_loss(
                output_values["score"], _pdms_tensor(targets), generator=pair_generator
            )
            loss = factor_loss + pairwise_weight * ranking_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{model_type} seed {seed}: non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1
            steps += 1
        if epoch_steps == 0:
            raise ProbeDataError("training cache is empty")
        val_regret, val_selected = validation_objective(
            composer,
            probe,
            val_cache,
            val_effects,
            model_type,
            batch_scenes,
            device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": epoch_loss / epoch_steps,
                "validation_regret": val_regret,
                "validation_selected_pdms": val_selected,
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
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("early stopping did not capture a valid model")
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    audit = audit_parameters(composer, probe)
    payload = {
        "schema_version": "matched_factor_probe.v1",
        "model_type": model_type,
        "seed": seed,
        "learning_rate": learning_rate,
        "pairwise_weight": pairwise_weight,
        "hidden_dim": hidden_dim,
        "best_epoch": best_epoch,
        "training_steps": steps,
        "validation_regret": best_regret,
        "validation_selected_pdms": best_selected,
        "parameter_audit": asdict(audit),
        "peak_gpu_memory_bytes": peak_memory,
        "probe_state_dict": best_state,
        "composer_state_dict": {
            key: value.detach().cpu() for key, value in composer.state_dict().items()
        },
        "history": history,
    }
    _atomic_torch_save(output, payload)
    return {
        key: payload[key]
        for key in (
            "model_type",
            "seed",
            "learning_rate",
            "pairwise_weight",
            "hidden_dim",
            "best_epoch",
            "training_steps",
            "validation_regret",
            "validation_selected_pdms",
            "parameter_audit",
            "peak_gpu_memory_bytes",
        )
    } | {"checkpoint": str(output)}


def train_suite(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing training output: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_config = config["probe"]
    seeds = [int(value) for value in config["run"]["seeds"]]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    pairwise_weights = [
        float(value)
        for value in probe_config.get(
            "pairwise_weights", [probe_config.get("pairwise_weight", 0.5)]
        )
    ]
    if not seeds or not learning_rates or not pairwise_weights:
        raise ValueError("seed/lr/pairwise grids must be non-empty")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    trials: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for model_type, seed in itertools.product(args.models, seeds):
        model_trials: list[dict[str, Any]] = []
        for learning_rate, pairwise_weight in itertools.product(
            learning_rates, pairwise_weights
        ):
            trial_path = (
                args.output
                / "search"
                / model_type
                / f"seed-{seed}-lr-{learning_rate:g}-pair-{pairwise_weight:g}.pt"
            )
            result = _train_trial(
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                train_effects=args.train_effects,
                val_effects=args.val_effects,
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
                output=trial_path,
            )
            model_trials.append(result)
            trials.append(result)
        best = min(
            model_trials,
            key=lambda row: (
                float(row["validation_regret"]),
                -float(row["validation_selected_pdms"]),
                float(row["learning_rate"]),
                float(row["pairwise_weight"]),
            ),
        )
        selected.append(best)

    parameter_counts = {int(row["parameter_audit"]["trainable_parameters"]) for row in trials}
    if len(parameter_counts) != 1:
        raise RuntimeError(f"matched probe parameter counts differ: {parameter_counts}")
    manifest = {
        "schema_version": "matched_probe_training.v1",
        "config": str(args.config),
        "models": list(args.models),
        "seeds": seeds,
        "optimizer": "AdamW",
        "learning_rate_grid": learning_rates,
        "pairwise_weight_grid": pairwise_weights,
        "early_stopping": {
            "metric": "validation_top1_regret",
            "patience": int(probe_config["patience"]),
            "max_epochs": int(probe_config["max_epochs"]),
        },
        "trainable_parameter_count": next(iter(parameter_counts)),
        "selected_trials": selected,
        "all_trials": trials,
    }
    atomic_write_json(args.output / "training_manifest.json", manifest)


def _add_effect_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frozen-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wote-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--ego-length-m", type=float, default=4.87)
    parser.add_argument("--ego-width-m", type=float, default=2.27)
    parser.add_argument("--clearance-m", type=float, default=6.0)
    parser.add_argument("--tca-seconds", type=float, default=3.0)
    parser.add_argument("--tca-distance-m", type=float, default=10.0)
    parser.add_argument("--conflict-zone-clearance-m", type=float, default=1.0)
    parser.add_argument(
        "--sensitivity-clearance-m", type=float, nargs="+", default=(4.0, 6.0, 8.0)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    effects = subparsers.add_parser("cache-effects")
    _add_effect_cache_arguments(effects)
    train = subparsers.add_parser("train-suite")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--val-cache", type=Path, required=True)
    train.add_argument("--train-effects", type=Path, required=True)
    train.add_argument("--val-effects", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--models", nargs="+", choices=ALL_SCORER_TYPES, default=G2_MODEL_TYPES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cache-effects":
        cache_replay_effects(args)
    elif args.command == "train-suite":
        train_suite(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
