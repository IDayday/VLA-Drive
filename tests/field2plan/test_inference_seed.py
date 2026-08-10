import random

import numpy as np
import torch

from infer import seed_inference


def _draw(seed: int, rank: int):
    used_seed = seed_inference(seed, rank)
    return (
        used_seed,
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )


def test_inference_seed_is_repeatable_per_rank():
    first = _draw(20260808, 1)
    second = _draw(20260808, 1)
    assert first == second


def test_inference_seed_is_rank_specific():
    assert _draw(20260808, 0) != _draw(20260808, 1)


def test_inference_seed_rejects_negative_values():
    try:
        seed_inference(-1, 0)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative inference seed must fail")
