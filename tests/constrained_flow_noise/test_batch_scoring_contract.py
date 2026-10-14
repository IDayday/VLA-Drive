"""Proposal-pool leakage and official threshold/mask boundary regressions."""
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.constrained_flow_noise.batched_canonical_score import independent_pair_aggregation
from starVLA.rl.safe_flow_improvement import v1_reward


@pytest.fixture(scope='module')
def official():
    repo = Path(__file__).resolve().parents[2]
    v1_reward.initialize(dict(repo=str(repo), maps_root='/mnt/navsim/maps',
                              protocol_sha256=v1_reward.protocol_identity(repo)['sha256']))
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return PDMScorer, WeightedMetricIndex.PROGRESS, TrajectorySampling(num_poses=40, interval_length=.1)


@pytest.mark.parametrize('progress,nc,dac', [
    ([10., 6., 12.], [1., 1., 1.], [1., 1., 1.]),
    ([100., 6., 12.], [0., 1., 1.], [1., 1., 1.]),
    ([10., 6., 12.], [.5, .5, 1.], [1., 1., 0.]),
    ([0., 0., 0.], [0., 0., 1.], [1., 1., 1.]),
    ([5., 5., 5. + 1e-12, 5. - 1e-12], [1.] * 4, [1.] * 4),
])
def test_every_candidate_matches_real_official_pair(official, progress, nc, dac):
    cls, index, sampling = official
    scorer = cls(sampling)
    scorer._multi_metrics = np.array([nc, dac], dtype=np.float64)
    scorer._weighted_metrics = np.ones((4, len(progress)), dtype=np.float64)
    scorer._weighted_metrics[1, -1] = 0
    scorer._progress_raw = np.array(progress, dtype=np.float64)
    before = copy.deepcopy(scorer)
    actual = independent_pair_aggregation(scorer, cls._aggregate_scores, index)
    for i in range(1, len(progress)):
        pair = copy.deepcopy(before)
        pair._multi_metrics = pair._multi_metrics[:, [0, i]]
        pair._weighted_metrics = pair._weighted_metrics[:, [0, i]]
        pair._progress_raw = pair._progress_raw[[0, i]]
        expected = cls._aggregate_scores(pair)
        assert actual[i] == expected[1]
        assert scorer._weighted_metrics[index, i] == pair._weighted_metrics[index, 1]
    np.testing.assert_array_equal(scorer._multi_metrics, before._multi_metrics)
    np.testing.assert_array_equal(scorer._progress_raw, before._progress_raw)


def test_naive_pool_negative_control_exposes_ep_leakage(official):
    cls, index, sampling = official
    scorer = cls(sampling)
    scorer._multi_metrics = np.ones((2, 3), dtype=np.float64)
    scorer._weighted_metrics = np.ones((4, 3), dtype=np.float64)
    scorer._progress_raw = np.array([10., 6., 12.])
    naive = copy.deepcopy(scorer)
    wrong = cls._aggregate_scores(naive)
    correct = independent_pair_aggregation(scorer, cls._aggregate_scores, index)
    assert naive._weighted_metrics[index, 1] == .5
    assert scorer._weighted_metrics[index, 1] == .6
    assert correct[1] > wrong[1]


def test_randomized_reference_pair_and_permutation(official):
    cls, index, sampling = official
    rng = np.random.RandomState(42)
    for _ in range(50):
        scorer = cls(sampling)
        scorer._multi_metrics = rng.choice([0., .5, 1.], (2, 17))
        scorer._weighted_metrics = rng.choice([0., .5, 1.], (4, 17))
        scorer._progress_raw = rng.uniform(0, 25, 17)
        baseline = copy.deepcopy(scorer)
        values = independent_pair_aggregation(scorer, cls._aggregate_scores, index)
        perm = np.r_[0, rng.permutation(np.arange(1, 17))]
        baseline._multi_metrics = baseline._multi_metrics[:, perm]
        baseline._weighted_metrics = baseline._weighted_metrics[:, perm]
        baseline._progress_raw = baseline._progress_raw[perm]
        reordered = independent_pair_aggregation(baseline, cls._aggregate_scores, index)
        np.testing.assert_array_equal(reordered, values[perm])
