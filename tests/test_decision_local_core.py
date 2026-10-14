import numpy as np
import pytest
import torch

from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig, trajectory_queries, time_warp, smooth_offset
from starVLA.model.modules.decision_local.consequence_contract import ModelInputs, PredictionQueries, OracleRelations
from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss
from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
from starVLA.model.modules.structured_world.queries import body_points


def fixtures(T=8):
    torch.manual_seed(42)
    inputs = ModelInputs(torch.randn(2, 8, 16), torch.randn(2, 12, 16),
        torch.randn(2, T, 4), torch.randn(2, T, 3), torch.randn(2, 4),
        torch.tensor([[5.176, 2.297, 1.461]]).expand(2, 3))
    queries = PredictionQueries(torch.randn(2, 7, T, 3), torch.arange(1, T+1)*.5,
                                torch.ones(2, 7, T, dtype=torch.bool), torch.ones(2, 7, dtype=torch.bool))
    return inputs, queries


@pytest.mark.parametrize('T', [6, 8])
def test_complete_queries_and_time_domain(T):
    t = np.arange(1, T+1)*.5
    q0 = np.column_stack((5*t, np.zeros(T), np.zeros(T)))
    out = trajectory_queries(q0, [5., 0.])
    np.testing.assert_array_equal(out['poses'][0], q0)
    np.testing.assert_array_equal(out['times'], t)
    for i, offset in enumerate((-.5, .5, -1., 1.), 1):
        np.testing.assert_allclose(out['poses'][i, :, 1], offset*smooth_offset(t/t[-1]), atol=1e-6)
        assert out['kinematics'][i]['initial_position_error_m'] == 0.
    for amplitude in (-.5, .5):
        dense = np.linspace(0, t[-1], 10001)
        phi = time_warp(dense, t[-1], amplitude)
        assert phi[0] == 0 and phi[-1] == t[-1]
        assert np.diff(phi).min() > 0
        assert 0 <= phi.min() <= phi.max() <= t[-1]
    assert out['environment_time_is_unwarped']


def test_lateral_is_path_normal_not_global_y_and_stationary_heading():
    t = np.arange(1, 9)*.5
    yaw = np.pi/3
    q0 = np.column_stack((5*t*np.cos(yaw), 5*t*np.sin(yaw), np.full(8, yaw)))
    out = trajectory_queries(q0, [2.5, 4.330127])
    displacement = out['poses'][4, -1, :2]-out['poses'][0, -1, :2]
    assert displacement[0] < -.7 and displacement[1] > .4
    assert abs(np.dot(displacement, [np.cos(yaw), np.sin(yaw)])) < 1e-5
    stop = trajectory_queries(np.zeros((8, 3)), [0., 0.])
    assert np.isfinite(stop['poses']).all()
    np.testing.assert_array_equal(stop['poses'][5:], np.zeros((2, 8, 3)))
    assert stop['kinematic_valid'][0] and stop['kinematic_valid'][5:].all()
    assert not stop['kinematic_valid'][1:5].any()  # no instantaneous sideways heading jump
    with pytest.raises(ValueError):
        time_warp(t, 3., 1.)


def test_full_body_rear_axle_and_rotation():
    body = dict(length=5.176, width=2.297, rear_axle_to_center=1.461)
    p = body_points(torch.tensor([[0., 0., 0.], [0., 0., torch.pi/2]]), body)
    torch.testing.assert_close(p[0, :, 0].max(), torch.tensor(5.176/2+1.461))
    torch.testing.assert_close(p[0, :, 0].min(), torch.tensor(-5.176/2+1.461))
    torch.testing.assert_close(p[1, :, 1].max(), p[0, :, 0].max())


@pytest.mark.parametrize('T', [6, 8])
def test_no_candidate_id_and_matched_abs_diff_initialization(T):
    inputs, queries = fixtures(T)
    torch.manual_seed(1); absolute = ConsequenceQueryHead(16, dim=32)
    torch.manual_seed(1); difference = ConsequenceQueryHead(16, dim=32)
    assert list(absolute.state_dict()) == list(difference.state_dict())
    for a, b in zip(absolute.parameters(), difference.parameters()):
        assert torch.equal(a, b)
    queries.poses[:, 1].copy_(queries.poses[:, 0])
    pred = absolute(inputs, queries)
    torch.testing.assert_close(pred[:, 1], pred[:, 0], rtol=0, atol=0)
    other = difference(inputs, queries)
    torch.testing.assert_close(pred, other, rtol=0, atol=0)
    loss_a = consequence_loss(pred, torch.zeros_like(pred), torch.ones_like(pred, dtype=torch.bool))
    loss_b = consequence_loss(other, torch.zeros_like(pred), torch.ones_like(pred, dtype=torch.bool), difference_weight=1.)
    torch.testing.assert_close(loss_a['absolute'], loss_b['absolute'], rtol=0, atol=0)


def test_masked_nan_and_pair_normalization():
    raw = torch.randn(2, 7, 8, 5, requires_grad=True)
    truth = torch.randn_like(raw); truth[..., 2] = 0.; truth[..., 4] = 1.
    valid = torch.ones_like(raw, dtype=torch.bool)
    valid[0] = False; truth[0] = float('nan')
    values = consequence_loss(raw, truth, valid, difference_weight=1.)
    values['loss'].backward()
    assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[0], torch.zeros_like(raw.grad[0]))
    # All six identical variant errors have the same mean as a single pair.
    raw = torch.zeros(1, 7, 8, 5); raw[:, 1:, :, 0] = 1.
    valid = torch.ones_like(raw, dtype=torch.bool)
    a = consequence_loss(raw, torch.zeros_like(raw), valid)['difference']
    valid[:, 2:] = False
    b = consequence_loss(raw, torch.zeros_like(raw), valid)['difference']
    torch.testing.assert_close(a, b)


def test_pair_requires_both_endpoints_and_swap_preserves_signed_error():
    # A missing anchor cannot supervise a valid-looking variant by itself.
    raw = torch.zeros(1, 2, 3, 5)
    truth = torch.zeros_like(raw)
    raw[:, 1, :, 0] = 2.
    truth[:, 1, :, 0] = .5
    valid = torch.ones_like(raw, dtype=torch.bool)
    original = consequence_loss(raw, truth, valid)['difference']
    assert original > 0
    swapped = consequence_loss(raw.flip(1), truth.flip(1), valid.flip(1))['difference']
    torch.testing.assert_close(original, swapped, rtol=0, atol=0)
    torch.testing.assert_close(raw[:, 1]-raw[:, 0], -(raw.flip(1)[:, 1]-raw.flip(1)[:, 0]))
    valid[:, 0] = False
    truth[:, 0] = float('nan')
    missing = consequence_loss(raw, truth, valid)
    assert missing['difference'] == 0
    assert all(count.sum() == 0 for key, count in missing['counts'].items() if key.startswith('pair_'))
    same = raw[:, :1].expand(-1, 2, -1, -1)
    identical = consequence_loss(same, same, torch.ones_like(same, dtype=torch.bool))
    assert identical['difference'] == 0


def test_refiner_zero_initial_and_oracle_entry_separation():
    inputs, queries = fixtures()
    model = ResidualRefiner(16, dim=32)
    torch.testing.assert_close(model.camera_only(inputs, queries), inputs.q0_encoded, rtol=0, atol=0)
    oracle = OracleRelations(torch.randn(2, 7, 8, 5), torch.ones(2, 7, 8, 5, dtype=torch.bool), 'FUTURE')
    with pytest.raises(TypeError):
        model.camera_only(inputs, queries, oracle)
    with pytest.raises(ValueError):
        ModelInputs.from_mapping({**vars(inputs), 'future_map': torch.randn(1)})
    output = model.oracle_probe(inputs, queries, oracle)
    output.square().mean().backward()
    assert model.output.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_query_only_independent_training_architecture_does_not_read_visual():
    inputs, queries = fixtures()
    model = ConsequenceQueryHead(16, dim=32, query_only=True)
    a = model(inputs, queries)
    changed = ModelInputs(torch.full_like(inputs.H_A, float('nan')), torch.full_like(inputs.W, float('nan')),
                           inputs.q0_encoded, inputs.q0, inputs.state, inputs.body)
    torch.testing.assert_close(a, model(changed, queries), rtol=0, atol=0)


def test_camera_relation_support_is_current_kinematics_only():
    inputs,queries=fixtures()
    model=ResidualRefiner(16,dim=32)
    # Make the output sensitive after its required zero-initialized start.
    torch.nn.init.normal_(model.output.weight,std=.02)
    queries.kinematic_valid[:,2]=False
    queries.kinematic_valid[:,0]=False  # q0 is retained even if it is poor.
    relations=torch.randn(2,7,8,5)
    original=model.camera_only(inputs,queries,relations)
    changed=relations.clone();changed[:,2]=10000.
    torch.testing.assert_close(original,model.camera_only(inputs,queries,changed),rtol=0,atol=0)
    changed=relations.clone();changed[:,0]+=10.
    assert not torch.equal(original,model.camera_only(inputs,queries,changed))
