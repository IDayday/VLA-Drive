from types import SimpleNamespace

import pytest
import torch

from navsim.planning.training.input_only_cache import (
    reject_dynamic_feature_cache,
    validate_input_only_cache_policy,
)


@pytest.mark.parametrize(
    "key",
    [
        "last_hidden_state",
        "patch_features",
        "semantic_tokens",
        "planning_registers",
        "future_registers",
        "ema_registers",
    ],
)
def test_every_dynamic_feature_cache_key_is_rejected(key):
    with pytest.raises(RuntimeError, match="change the scientific method"):
        reject_dynamic_feature_cache(
            {key: torch.zeros(1)}, enabled=True, source="unit-test"
        )


def test_guard_can_be_disabled_for_legacy_static_feature_workflows():
    reject_dynamic_feature_cache(
        {"last_hidden_state": torch.zeros(1)}, enabled=False, source="legacy"
    )


def test_formal_cache_policy_rejects_enabled_model_output_cache():
    policy = SimpleNamespace(
        mode="input_only",
        cache_vlm_hidden_state=False,
        cache_patch_features=False,
        cache_semantic_tokens=False,
        cache_planning_registers=True,
        cache_future_ema_registers=False,
    )
    with pytest.raises(ValueError, match="forbidden dynamic representations"):
        validate_input_only_cache_policy(policy)
