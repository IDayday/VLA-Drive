"""Fixed, non-learned 32x64 auxiliary token contract for Gate2O v2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from .replay_effect_builder import (
    PRIMITIVE_ACTOR_INDICES,
    PRIMITIVE_MAP_INDICES,
)


AUXILIARY_TOKEN_COUNT = 32
AUXILIARY_TOKEN_WIDTH = 64
TIME_STEPS = 8
ACTOR_SLOTS = 16
INTERVENTION_SEED = 20260827

MODEL_VARIANTS = (
    "trajectory_only",
    "direct_current",
    "ego_kinematic_effect",
    "static_primitive_effect",
    "shared_logged_future",
    "dynamic_replay_effect",
    "full_primitive_action_effect",
    "full_primitive_no_interaction_mask",
    "interaction_mask_only",
    "full_engineered_action_effect",
    "wote_full_future",
    "wote_environment_only",
)

STRUCTURED_GROUP_SLICES = {
    "ego": slice(0, 8),
    "map": slice(8, 16),
    "actor": slice(16, 24),
    "mask": slice(24, 32),
}


class EffectTokenError(RuntimeError):
    """An effect tensor violates the pre-registered packing contract."""


def _as_finite(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise EffectTokenError(f"{name} contains NaN/Inf")
    return array


def _primitive_effects(
    effects: Mapping[str, npt.ArrayLike],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    ego = _as_finite(
        effects.get("primitive_ego_effect", effects.get("ego_effect")), "ego effect"
    )
    if "primitive_map_effect" in effects:
        map_effect = _as_finite(effects["primitive_map_effect"], "map effect")
    else:
        map_effect = _as_finite(effects["map_effect"], "map effect")[
            ..., PRIMITIVE_MAP_INDICES
        ]
    if "primitive_actor_effect" in effects:
        actor = _as_finite(effects["primitive_actor_effect"], "actor effect")
    else:
        actor = _as_finite(effects["actor_effect"], "actor effect")[
            ..., PRIMITIVE_ACTOR_INDICES
        ]
    validity = _as_finite(
        effects.get("primitive_actor_mask", effects.get("actor_mask")),
        "actor validity",
    )
    interaction = _as_finite(
        effects.get("primitive_interaction_mask", effects.get("interaction_mask")),
        "interaction mask",
    )
    candidates = ego.shape[0]
    expected = {
        "ego": (candidates, 8, 16),
        "map": (candidates, 8, 7),
        "actor": (candidates, 8, 16, 10),
        "validity": (candidates, 8, 16),
        "interaction": (candidates, 8, 16),
    }
    values = {
        "ego": ego,
        "map": map_effect,
        "actor": actor,
        "validity": validity,
        "interaction": interaction,
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise EffectTokenError(f"{name} expected {shape}, got {values[name].shape}")
    if np.any(interaction > validity + 1.0e-6):
        raise EffectTokenError("interaction mask marks an invalid actor")
    return ego, map_effect, actor, validity, interaction


def actor_summary(
    actor: npt.ArrayLike,
    validity: npt.ArrayLike,
    interaction: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float32]:
    """Mean/min/max plus label-free validity and interaction fractions."""

    values = _as_finite(actor, "actor summary input")
    mask = _as_finite(validity, "actor summary validity") > 0.5
    if values.ndim != 4 or mask.shape != values.shape[:-1]:
        raise EffectTokenError(
            f"actor summary expects [K,8,A,D]/[K,8,A], got {values.shape}/{mask.shape}"
        )
    counts = mask.sum(axis=2)
    denominator = np.maximum(counts, 1)[..., None]
    mean = (values * mask[..., None]).sum(axis=2) / denominator
    minimum = np.where(mask[..., None], values, np.inf).min(axis=2)
    maximum = np.where(mask[..., None], values, -np.inf).max(axis=2)
    empty = counts == 0
    minimum[empty] = 0.0
    maximum[empty] = 0.0
    valid_fraction = mask.mean(axis=2, keepdims=True)
    if interaction is None:
        interaction_fraction = np.zeros_like(valid_fraction)
    else:
        interaction_mask = _as_finite(interaction, "interaction summary") > 0.5
        if interaction_mask.shape != mask.shape:
            raise EffectTokenError("interaction/validity mask shapes differ")
        interaction_fraction = (
            (interaction_mask & mask).sum(axis=2)[..., None] / denominator
        )
    output = np.concatenate(
        [mean, minimum, maximum, valid_fraction, interaction_fraction], axis=-1
    ).astype(np.float32)
    return output


def _fixed_group_time_embedding(group_index: int) -> npt.NDArray[np.float32]:
    """Sixteen deterministic, non-learned dimensions for one 8-token group."""

    time = np.arange(8, dtype=np.float32)[:, None]
    frequency = np.exp(-np.arange(4, dtype=np.float32)[None] * np.log(10_000.0) / 3.0)
    time_embedding = np.concatenate([np.sin(time * frequency), np.cos(time * frequency)], axis=1)
    group = np.zeros((8, 8), dtype=np.float32)
    group[:, group_index % 4] = 1.0
    group[:, 4 + group_index % 4] = -1.0
    return np.concatenate([group, time_embedding], axis=1)


def _put_group(
    output: npt.NDArray[np.float32],
    group: str,
    values: npt.ArrayLike,
) -> None:
    array = _as_finite(values, f"{group} token values")
    candidates = output.shape[0]
    if array.ndim != 3 or array.shape[:2] != (candidates, 8):
        raise EffectTokenError(
            f"{group} tokens expected [{candidates},8,D], got {array.shape}"
        )
    if array.shape[-1] > 48:
        raise EffectTokenError(f"{group} raw width {array.shape[-1]} exceeds 48")
    target = output[:, STRUCTURED_GROUP_SLICES[group]]
    target[..., : array.shape[-1]] = array
    group_index = tuple(STRUCTURED_GROUP_SLICES).index(group)
    target[..., 48:64] = _fixed_group_time_embedding(group_index)[None]


def _shared_actor_summary(effects: Mapping[str, npt.ArrayLike], candidates: int) -> tuple[np.ndarray, np.ndarray]:
    if "shared_logged_future" not in effects or "shared_actor_mask" not in effects:
        raise EffectTokenError("shared_logged_future requires value and validity mask")
    shared = _as_finite(effects["shared_logged_future"], "shared logged future")
    mask = _as_finite(effects["shared_actor_mask"], "shared actor mask")
    if shared.shape != (8, 16, 8) or mask.shape != (8, 16):
        raise EffectTokenError(
            f"shared future expected [8,16,8]/[8,16], got {shared.shape}/{mask.shape}"
        )
    repeated = np.broadcast_to(shared[None], (candidates,) + shared.shape)
    repeated_mask = np.broadcast_to(mask[None], (candidates,) + mask.shape)
    summary = actor_summary(repeated, repeated_mask)
    return summary, repeated_mask.astype(np.float32)


def _mask_summary(
    validity: npt.NDArray[np.float32], interaction: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    valid_count = validity.sum(axis=2, keepdims=True)
    interaction_count = (validity * interaction).sum(axis=2, keepdims=True)
    return np.concatenate(
        [
            valid_count / ACTOR_SLOTS,
            interaction_count / ACTOR_SLOTS,
            interaction_count / np.maximum(valid_count, 1.0),
            (valid_count > 0).astype(np.float32),
            (interaction_count >= 2).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _engineered_summary(
    effects: Mapping[str, npt.ArrayLike], validity: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if "map_engineered_effect" in effects:
        map_engineered = _as_finite(
            effects["map_engineered_effect"], "map engineered effect"
        )
    else:
        map_engineered = _as_finite(effects["map_effect"], "map effect")[..., (5,)]
    if "actor_engineered_effect" in effects:
        actor_engineered = _as_finite(
            effects["actor_engineered_effect"], "actor engineered effect"
        )
    else:
        actor_engineered = _as_finite(effects["actor_effect"], "actor effect")[
            ..., (9, 10, 11)
        ]
    return map_engineered, actor_summary(actor_engineered, validity)[..., :9]


def _pack_wote_feature(
    frozen: Mapping[str, npt.ArrayLike], *, environment_only: bool
) -> npt.NDArray[np.float32]:
    trajectory = _as_finite(frozen["trajectory"], "trajectory")
    candidates = trajectory.shape[0]
    if environment_only:
        feature = _as_finite(
            frozen["environment_only_future"], "WoTE environment-only future"
        ).reshape(candidates, -1)
    else:
        reward = _as_finite(frozen["reward_feature"], "WoTE reward feature").reshape(candidates, -1)
        future_ego = _as_finite(
            frozen["future_ego_features_by_step"], "WoTE future ego feature"
        ).reshape(candidates, -1)
        future_bev = _as_finite(
            frozen["future_bev_tokens_by_step"], "WoTE future BEV feature"
        )
        if future_bev.ndim != 4 or future_bev.shape[0] != candidates:
            raise EffectTokenError(f"invalid future BEV shape: {future_bev.shape}")
        future_bev_pool = future_bev.mean(axis=2).reshape(candidates, -1)
        feature = np.concatenate([reward, future_ego, future_bev_pool], axis=-1)
    capacity = AUXILIARY_TOKEN_COUNT * AUXILIARY_TOKEN_WIDTH
    if feature.shape[1] > capacity:
        raise EffectTokenError(
            f"WoTE feature width {feature.shape[1]} exceeds fixed capacity {capacity}"
        )
    output = np.zeros((candidates, capacity), dtype=np.float32)
    output[:, : feature.shape[1]] = feature
    return output.reshape(candidates, AUXILIARY_TOKEN_COUNT, AUXILIARY_TOKEN_WIDTH)


@dataclass(frozen=True)
class PackedVariant:
    auxiliary_tokens: npt.NDArray[np.float32]
    use_current_bev: bool


class EffectTokenPacker:
    """Pack every A-L variant into one identical auxiliary tensor shape."""

    def pack(
        self,
        model_type: str,
        effects: Mapping[str, npt.ArrayLike] | None,
        frozen: Mapping[str, npt.ArrayLike],
    ) -> PackedVariant:
        if model_type not in MODEL_VARIANTS:
            raise EffectTokenError(f"unknown model type: {model_type}")
        trajectory = _as_finite(frozen["trajectory"], "trajectory")
        if trajectory.shape != (256, 8, 3):
            raise EffectTokenError(f"trajectory expected [256,8,3], got {trajectory.shape}")
        candidates = len(trajectory)
        use_current = model_type != "trajectory_only"
        if model_type in {"trajectory_only", "direct_current"}:
            return PackedVariant(
                np.zeros((candidates, 32, 64), dtype=np.float32), use_current
            )
        if model_type == "wote_full_future":
            return PackedVariant(_pack_wote_feature(frozen, environment_only=False), True)
        if model_type == "wote_environment_only":
            return PackedVariant(_pack_wote_feature(frozen, environment_only=True), True)
        if effects is None:
            raise EffectTokenError(f"{model_type} requires replay effect tensors")
        if model_type == "shared_logged_future":
            output = np.zeros((candidates, 32, 64), dtype=np.float32)
            shared_actor, shared_mask = _shared_actor_summary(effects, candidates)
            _put_group(output, "actor", shared_actor)
            _put_group(
                output,
                "mask",
                _mask_summary(shared_mask, np.zeros_like(shared_mask)),
            )
            return PackedVariant(output, True)

        ego, map_effect, actor, validity, interaction = _primitive_effects(effects)
        output = np.zeros((candidates, 32, 64), dtype=np.float32)
        include_ego = model_type in {
            "ego_kinematic_effect",
            "static_primitive_effect",
            "full_primitive_action_effect",
            "full_primitive_no_interaction_mask",
            "full_engineered_action_effect",
        }
        include_map = model_type in {
            "static_primitive_effect",
            "full_primitive_action_effect",
            "full_primitive_no_interaction_mask",
            "full_engineered_action_effect",
        }
        include_actor = model_type in {
            "dynamic_replay_effect",
            "full_primitive_action_effect",
            "full_primitive_no_interaction_mask",
            "full_engineered_action_effect",
        }
        include_masks = model_type in {
            "dynamic_replay_effect",
            "full_primitive_action_effect",
            "full_primitive_no_interaction_mask",
            "interaction_mask_only",
            "full_engineered_action_effect",
        }
        effective_interaction = (
            np.zeros_like(interaction)
            if model_type == "full_primitive_no_interaction_mask"
            else interaction
        )
        if include_ego:
            _put_group(output, "ego", ego)
        if include_map:
            map_values = map_effect
            if model_type == "full_engineered_action_effect":
                map_engineered, _ = _engineered_summary(effects, validity)
                map_values = np.concatenate([map_values, map_engineered], axis=-1)
            _put_group(output, "map", map_values)
        if include_actor:
            actor_values = actor_summary(actor, validity, effective_interaction)
            if model_type == "full_engineered_action_effect":
                _, actor_engineered = _engineered_summary(effects, validity)
                actor_values = np.concatenate([actor_values, actor_engineered], axis=-1)
            _put_group(output, "actor", actor_values)
        if include_masks:
            _put_group(output, "mask", _mask_summary(validity, effective_interaction))
        return PackedVariant(output, True)


def deterministic_effect_permutation(
    scene_token: str,
    candidates: int = 256,
    seed: int = INTERVENTION_SEED,
) -> npt.NDArray[np.int64]:
    if candidates <= 1:
        raise ValueError("effect permutation requires at least two candidates")
    digest = hashlib.sha256(f"{scene_token}:{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    permutation = rng.permutation(candidates).astype(np.int64)
    if np.array_equal(permutation, np.arange(candidates)):
        permutation = np.roll(permutation, 1)
    return permutation


def intervene_effects(
    effects: Mapping[str, npt.ArrayLike],
    intervention: str,
    scene_token: str,
    seed: int = INTERVENTION_SEED,
) -> dict[str, npt.NDArray[Any]]:
    """Apply a registered candidate-specificity intervention to effects only."""

    if intervention not in {
        "none",
        "full_effect_swap",
        "actor_only_swap",
        "static_only_swap",
        "scene_mean_effect",
    }:
        raise EffectTokenError(f"unknown intervention: {intervention}")
    output = {name: np.asarray(value).copy() for name, value in effects.items()}
    if intervention == "none":
        return output
    permutation = deterministic_effect_permutation(scene_token, seed=seed)
    static_keys = {
        "ego_effect",
        "map_effect",
        "primitive_ego_effect",
        "primitive_map_effect",
        "map_engineered_effect",
    }
    actor_keys = {
        "actor_effect",
        "actor_mask",
        "interaction_mask",
        "primitive_actor_effect",
        "primitive_actor_mask",
        "primitive_interaction_mask",
        "actor_engineered_effect",
    }
    if intervention == "full_effect_swap":
        selected_keys = static_keys | actor_keys
    elif intervention == "actor_only_swap":
        selected_keys = actor_keys
    elif intervention == "static_only_swap":
        selected_keys = static_keys
    else:
        selected_keys = static_keys | actor_keys
    for name in selected_keys & output.keys():
        value = output[name]
        if value.shape[0] != 256:
            raise EffectTokenError(f"candidate-specific {name} lacks 256 axis")
        if intervention == "scene_mean_effect":
            mean = value.astype(np.float32).mean(axis=0, keepdims=True)
            output[name] = np.broadcast_to(mean, value.shape).copy()
        else:
            output[name] = value[permutation].copy()
    return output
