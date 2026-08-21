"""Unit tests for the action-effect sample isolation contract."""

from __future__ import annotations

import pytest

from research.action_effect.data_contract import (
    assert_no_future_leakage,
    collate_isolated_samples,
    isolate_sample,
)


def test_future_fields_are_target_only() -> None:
    sample = isolate_sample(
        {"current_images": [1], "candidate_trajectory": [2], "navigation": 3},
        {"future_actor_states": [4], "future_bev": [5]},
    )
    batch = collate_isolated_samples([sample])
    assert "future_actor_states" not in batch["input"][0]
    assert batch["target"][0]["future_actor_states"] == [4]


@pytest.mark.parametrize(
    "bad_input",
    [
        {"future_images": object()},
        {"nested": {"future_actor_states": object()}},
        {"metric_cache": object()},
        {"reactive_model": {"collision": True}},
    ],
)
def test_privileged_input_is_rejected(bad_input: dict[str, object]) -> None:
    with pytest.raises(AssertionError, match="privileged future"):
        assert_no_future_leakage(bad_input)


def test_collate_rejects_unpartitioned_fields() -> None:
    with pytest.raises(AssertionError, match="exactly input/target"):
        collate_isolated_samples([{"input": {}, "target": {}, "future_bev": []}])
