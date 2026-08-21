"""Strict base-checkpoint loading with explicitly scoped DDP-DRS keys."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


PLANNER_PREFIX = "multi_trajectory_planner."


def load_base_checkpoint_strict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> None:
    """Strictly load either a full DDP-DRS state or an original base state.

    For an original Qwen+DiT checkpoint, every pre-existing key must match
    exactly.  The only locally allowed missing keys are printed and must be
    under the new planner prefix; no unexpected key is accepted.
    """

    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    if checkpoint_keys == model_keys:
        model.load_state_dict(dict(state_dict), strict=True)
        print("DDP-DRS full checkpoint missing keys: []")
        print("DDP-DRS full checkpoint unexpected keys: []")
        return

    planner_keys = {key for key in model_keys if key.startswith(PLANNER_PREFIX)}
    original_model_keys = model_keys - planner_keys
    missing_original = sorted(original_model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - original_model_keys)
    allowed_missing = sorted(model_keys - checkpoint_keys)
    print(f"original Qwen+DiT checkpoint missing base keys: {missing_original}")
    print(f"original Qwen+DiT checkpoint unexpected keys: {unexpected}")
    print(f"DDP-DRS locally allowed planner keys: {allowed_missing}")
    if missing_original or unexpected or set(allowed_missing) != planner_keys:
        raise RuntimeError(
            "original Qwen+DiT checkpoint does not strictly match base model keys"
        )

    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    actual_missing = sorted(incompatible.missing_keys)
    actual_unexpected = sorted(incompatible.unexpected_keys)
    if actual_missing != allowed_missing or actual_unexpected:
        raise RuntimeError(
            "localized DDP-DRS checkpoint compatibility result changed unexpectedly"
        )
