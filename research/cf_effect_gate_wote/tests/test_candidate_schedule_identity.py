from __future__ import annotations

from pathlib import Path

import numpy as np

from research.cf_effect_gate_wote.src.effect_tokenizer import MODEL_VARIANTS
from research.cf_effect_gate_wote.src.oracle_effect_data import (
    build_fixed_split,
    deterministic_candidate_schedule,
    deterministic_pair_schedule,
    read_tokens,
)


def test_candidate_and_pair_schedules_are_model_independent() -> None:
    schedules = [
        deterministic_candidate_schedule("scene-a", 2, 7, selected_index=93)
        for _ in MODEL_VARIANTS
    ]
    for value in schedules[1:]:
        np.testing.assert_array_equal(value, schedules[0])
    assert len(schedules[0]) == 64
    assert 93 in schedules[0]
    pairs = [
        deterministic_pair_schedule("scene-a", 2, 7, 64)
        for _ in MODEL_VARIANTS
    ]
    for value in pairs[1:]:
        np.testing.assert_array_equal(value, pairs[0])
    assert not np.any(pairs[0][:, 0] == pairs[0][:, 1])


def test_registered_split_is_disjoint_and_excludes_g1_set() -> None:
    split_dir = Path(__file__).resolve().parents[1] / "configs" / "splits"
    split = build_fixed_split(split_dir)
    assert (len(split.train), len(split.val), len(split.test)) == (1024, 256, 512)
    assert not set(split.train) & set(split.val)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)
    assert not set(split.test) & set(read_tokens(split_dir / "relabel_headroom_tokens.txt"))
    assert split.test == read_tokens(split_dir / "test_tokens.txt")[200:712]

