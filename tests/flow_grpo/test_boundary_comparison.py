from scripts.flow_grpo.compare_boundaries import compare_values
import torch


def test_boundary_comparison_covers_moments_rng_and_integer_precision():
    before = {
        "master": torch.tensor([1.0]),
        "exp_avg": torch.tensor([0.1]),
        "rng": torch.tensor([2**62], dtype=torch.int64),
        "cursor": 1,
        "pending": [],
    }
    assert all(x["allclose"] for x in compare_values(before, before).values())
    after = {
        **before,
        "master": torch.tensor([1.1]),
        "exp_avg": torch.tensor([0.2]),
        "rng": torch.tensor([2**62 + 1], dtype=torch.int64),
        "pending": {},
    }
    rows = compare_values(before, after)
    assert {k for k, v in rows.items() if not v["allclose"]} == {
        "/master",
        "/exp_avg",
        "/rng",
        "/pending/@type",
    }
