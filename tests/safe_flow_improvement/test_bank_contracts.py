from dataclasses import replace
import numpy as np
import pytest

from starVLA.rl.safe_flow_improvement.candidate_bank import Candidate, deduplicate
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, V1_COMPONENTS, WindowIdentity, require_same_window


def candidate(x, reference=False, **kwargs):
    poses = np.zeros((8, 3))
    poses[:, 0] = x
    return Candidate(token='abc', log_name='log1', source='reference_ode' if reference else 'ode',
                     behavior_sha256='sft', normalized_action=np.zeros((8, 4)), requested_trajectory=poses,
                     simulated_states=np.zeros((41, 11)), score=.8,
                     metrics=dict.fromkeys(V1_COMPONENTS, 1.), protocol_sha256='canonical',
                     noise_keys={'root': x}, is_reference=reference, **kwargs)


def test_duplicate_reference_and_sample_masses_merge_without_changing_prior():
    bank = [candidate(1, True), candidate(1, True), candidate(2, True), candidate(3, True),
            candidate(1), candidate(4), candidate(4), candidate(5)]
    unique, q, empirical, groups = deduplicate(bank)
    assert len(unique) == 5
    assert q == pytest.approx([.5, .25, .25, 0, 0])
    assert empirical == pytest.approx([3/8, 1/8, 1/8, 2/8, 1/8])
    prior = .5*q + .5*empirical
    assert prior.sum() == pytest.approx(1.)
    assert sum(map(len, groups)) == 8


def test_identical_request_with_different_feedback_is_an_error():
    bank = [candidate(1, True) for _ in range(4)] + [replace(candidate(1), score=.9)]
    with pytest.raises(ValueError, match='different canonical outcomes'):
        deduplicate(bank)


@pytest.mark.parametrize('source', ['latent_search', 'behavior_search', 'ode'])
def test_external_and_ode_candidates_cannot_claim_sde_likelihood(source):
    with pytest.raises(ValueError, match='old-policy logprob'):
        replace(candidate(1), source=source, old_logprob=0.).validate()


def test_nft_rejects_search_reference_and_wrong_behavior():
    candidate(1).require_nft_behavior('sft')
    for row in [candidate(1, True), replace(candidate(1), source='behavior_search'),
                replace(candidate(1), behavior_sha256='changed')]:
        with pytest.raises(ValueError):
            row.require_nft_behavior('sft')


def test_failures_or_missing_metrics_cannot_silently_enter_bank():
    for row in [replace(candidate(1), status='worker_error'), replace(candidate(1), metrics={}),
                replace(candidate(1), simulated_states=np.zeros((8, 3))),
                replace(candidate(1), protocol='navsim_v2_one_stage_single_scene')]:
        with pytest.raises(ValueError):
            row.validate()


def test_log_isolation_and_update_calibration_boundaries():
    with pytest.raises(ValueError, match='log overlap'):
        PopulationContract(frozenset(['a']), frozenset(['b']), frozenset(['c']),
                           {'a': 'log', 'b': 'log', 'c': 'test'})
    pop = PopulationContract(frozenset(['a']), frozenset(['b']), frozenset(['c']),
                             {'a': 'train', 'b': 'dev', 'c': 'test'})
    pop.require_training(['a'], 'eta calibration')
    for token in ['b', 'c', 'missing']:
        with pytest.raises(ValueError):
            pop.require_training([token], 'eta calibration')


def test_resume_rejects_changed_behavior_bank_and_objective():
    identity = WindowIdentity('P', 'P1', 42, 0, 'sft', 'behavior', 'split', 'bank', 'protocol', 'objective', 'numeric')
    require_same_window(identity, identity)
    for field in ['behavior_sha256', 'bank_sha256', 'objective_sha256', 'split_sha256']:
        with pytest.raises(ValueError):
            require_same_window(identity, replace(identity, **{field: 'changed'}))
