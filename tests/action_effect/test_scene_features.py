from __future__ import annotations

import numpy as np
import pytest

from research.action_effect.scene_features import sanitize_current_observation


def _sample() -> dict:
    return {
        "image": [object(), object(), object()],
        "state": np.zeros((1, 4), dtype=np.float32),
        "lang": "keep straight",
        "token": "scene",
        "action": np.ones((8, 4), dtype=np.float32),
        "future_actor_states": np.ones((2, 3), dtype=np.float32),
    }


def test_scene_feature_input_discards_dataset_future_labels() -> None:
    safe = sanitize_current_observation(_sample())
    assert set(safe) == {"image", "state", "lang", "token"}
    assert "action" not in safe
    assert "future_actor_states" not in safe


def test_scene_feature_input_requires_runtime_observation() -> None:
    sample = _sample()
    del sample["state"]
    with pytest.raises(KeyError, match="state"):
        sanitize_current_observation(sample)
