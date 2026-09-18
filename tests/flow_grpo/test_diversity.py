import numpy as np
import pytest
from starVLA.rl.flow_grpo.diversity import trajectory_diversity


def test_identical_trajectories_and_nonbinary_constant_reward():
    poses = np.zeros((16, 8, 3))
    r = trajectory_diversity(poses, np.full(16, 5/7), np.zeros((16, 11, 8, 4)))
    assert r["all_equal_rewards"] and r["distinct_rewards"] == 1
    assert r["pair_ade_m"]["1.0"] == r["xy_covariance_effective_rank"] == 0
    assert r["normalized_chain_rms_std"] == [0.]*11


def test_physical_units_pair_counts_and_circular_heading():
    poses = np.zeros((2, 8, 3))
    poses[1, :, 0] = 2.
    poses[0, :, 2] = np.pi-1e-3
    poses[1, :, 2] = -np.pi+1e-3
    r = trajectory_diversity(poses, [.7, .8])
    assert r["pairs"] == 1 and r["pair_ade_m"]["0.5"] == 2
    assert r["endpoint_std_xy_m"] == [1., 0.]
    assert r["heading_circular_std_mean_rad"] == pytest.approx(.001, rel=1e-5)
    assert r["reward_span"] == pytest.approx(.1)
    assert r["xy_covariance_effective_rank"] == pytest.approx(1.)
    with pytest.raises(ValueError):
        trajectory_diversity(poses, [0., np.nan])


def test_production_metrics_expose_signal_and_physical_spread():
    from types import SimpleNamespace
    import torch
    from starVLA.rl.flow_grpo.metrics import Metrics
    from starVLA.rl.flow_grpo.advantages import assign_behavior_advantages

    rollout = SimpleNamespace(rewards=torch.tensor([[.9, 1.]]), advantages=None,
        transition_mask=torch.ones(1, 2, 10, dtype=torch.bool), score_records=[],
        physical_trajectories=np.zeros((1, 2, 8, 3)))
    rollout.physical_trajectories[0, 1, :, 0] = 1.
    assign_behavior_advantages([rollout], {"advantage_normalization": "global_batch",
        "advantage_epsilon": 1e-6, "advantage_clip": 5.}, "cpu")
    result = {k: torch.tensor(0.) for k in ["loss", "grpo", "reference", "sft"]}
    result.update(ratio=torch.ones(1, 2, 10), logratio=torch.zeros(1, 2, 10), components={})
    metrics = Metrics()
    metrics.update_scene(result, rollout, .02)
    report = metrics.result()
    assert report["candidate_pair_ade_scene_median_m"] == 1.
    assert report["behavior_global_candidate_count"] == 2
    assert report["advantage_nonzero_group_fraction"] == 1.
