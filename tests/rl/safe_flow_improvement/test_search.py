import numpy as np

from starVLA.rl.safe_flow_improvement.candidate_search import BoundedCEM, CEMSpec, safety_order, smooth_behavior_request


def metrics(score, nc=1., dac=1., ttc=1.):
    return dict(score=score, no_at_fault_collisions=nc, drivable_area_compliance=dac, time_to_collision_within_bound=ttc)


def test_safe_search_beats_unsafe_high_progress():
    assert safety_order(metrics(.7)) < safety_order(metrics(.99, nc=.5))
    assert safety_order(metrics(.7)) < safety_order(metrics(.99, ttc=0.))


def test_elites_reused_without_extra_draws_and_latent_prior_bounded():
    search = BoundedCEM('latent_search', np.random.default_rng(42))
    first, record = search.draw(15)
    assert first.shape == (15, 32) and record['retained_elites_not_rescored'] == 0
    for _ in range(4):
        values, record = search.draw(15)
        search.update(values, [metrics(float(i)/15) for i in range(15)])
        assert len(search.elites) == 4
        assert np.linalg.norm(search.mean) <= 2.+1e-12
        assert np.all(search.std >= .5) and np.all(search.std <= 1)
        assert all(x['chi2_prior_radius_cdf'] <= .995+1e-12 for x in record['draws'])


def test_behavior_braking_is_smooth_with_fixed_initial_velocity():
    base = np.stack([np.arange(1, 9)*5., np.zeros(8), np.zeros(8)], axis=1)
    request, feedback = smooth_behavior_request(base, [10., 0.], [-.5, 1., 0., 0.])
    assert feedback['status'] == 'PHYSICAL_PROPOSAL_VALID'
    assert np.allclose(request[:2], base[:2])
    assert request[-1, 0] < base[-1, 0]
    assert np.all(np.diff(request[:, 0]) > 0)
    assert feedback['initial_velocity_xy'] == [10., 0.]


def test_invalid_physical_proposal_is_not_zero_reward():
    base = np.stack([np.arange(1, 9)*5., np.zeros(8), np.zeros(8)], axis=1)
    base[4, 1] = 20.
    _, feedback = smooth_behavior_request(base, [10., 0.], [0., 1., 0., 0.])
    assert feedback['status'] == 'INVALID_PHYSICAL_PROPOSAL'
    assert 'score' not in feedback
