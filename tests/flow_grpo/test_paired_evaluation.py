import json
import numpy as np
import pandas as pd
import pytest
import torch
from starVLA.rl.flow_grpo.evaluation import (
    scene_noise_seed,
    paired_scores,
    validate_evaluation_tokens,
)
from starVLA.rl.flow_grpo.math import clipped_surrogate


def test_noise_is_stable_across_order_and_shards():
    tokens = ["scene-a", "scene-b", "scene-c", "scene-d"]

    def sample(token):
        return torch.randn(
            8, 4, generator=torch.Generator().manual_seed(scene_noise_seed(42, token))
        )

    serial = {t: sample(t) for t in tokens}
    reordered = {
        t: sample(t) for shard in [tokens[1::2], tokens[::2]] for t in reversed(shard)
    }
    assert all(torch.equal(serial[t], reordered[t]) for t in tokens)
    assert scene_noise_seed(42, "a") != scene_noise_seed(43, "a")


def test_paired_log_bootstrap_and_tail_counts():
    base = pd.DataFrame(
        {
            "token": list("abcd"),
            "log_name": ["log1", "log1", "log2", "log2"],
            "score": [0.0, 0.5, 0.95, 0.9],
        }
    )
    after = base.copy()
    after["score"] = [0.5, 0.0, 0.85, 1.0]
    rows, report = paired_scores(base, after)
    assert (
        report["zero_to_nonzero"]
        == report["nonzero_to_zero"]
        == report["high_score_degraded"]
        == 1
    )
    assert report["mean_paired_delta"] == pytest.approx(0.0)
    assert report["logs"] == 2 and report["bootstrap_unit"] == "log"
    with pytest.raises(ValueError, match="identical"):
        paired_scores(base, after.iloc[:3])


def test_eval_navtest_is_explicit_and_reward_stays_forbidden(tmp_path):
    from starVLA.rl.flow_grpo.reward import RewardService

    p = tmp_path / "test.json"
    p.write_text(json.dumps(["test_a", "test_b"]))
    cfg = {"paths": {"test_list": str(p)}}
    assert validate_evaluation_tokens(cfg, "navtest", ["test_b", "test_a"])
    assert not validate_evaluation_tokens(cfg, "navtest", ["test_a"])
    with pytest.raises(ValueError):
        validate_evaluation_tokens(cfg, "navtest", ["train_a"])
    with pytest.raises(ValueError, match="forbidden"):
        RewardService("unused", "unused", "navtest", ["test_a"])


@pytest.mark.parametrize(
    "adv,ratio,expected_gradient",
    [
        (1.0, 0.97, -0.97),
        (1.0, 1.0, -1.0),
        (1.0, 1.03, 0.0),
        (-1.0, 0.97, 0.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.03, 1.03),
    ],
)
def test_ppo_clip_sign_boundaries_and_old_immutability(adv, ratio, expected_gradient):
    old = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    before = old.detach().clone()
    current = torch.tensor(
        [2.0 + np.log(ratio)], dtype=torch.float64, requires_grad=True
    )
    loss, result = clipped_surrogate(current, old, torch.tensor([adv]), clip=0.02)
    loss.sum().backward()
    assert current.grad.item() == pytest.approx(expected_gradient)
    assert old.grad is None and torch.equal(old.detach(), before)
    assert result.item() == pytest.approx(ratio)
