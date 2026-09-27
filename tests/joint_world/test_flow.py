import io
import torch
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow, actor_mask, training_loss_sums


def setup():
    torch.manual_seed(7)
    model = JointTrajectoryFlow(12, dim=16, heads=4, layers=2, steps=3)
    noise = torch.randn(2, 4, 3, 2)
    current = dict(actor_features=torch.randn(2, 4, 12), context=torch.randn(2, 5, 12),
                   current_xy=torch.randn(2, 4, 2), existence=torch.rand(2, 4))
    return model, noise, current


def test_actor_masks_and_no_future_deployment_signature():
    hidden = actor_mask(256, 4, 'cpu', torch.Generator().manual_seed(1))
    assert hidden.any(-1).all() and hidden.all(-1).sum() > 80
    import inspect
    assert not {'target', 'known_xy', 'valid', 'known_mask'} & set(inspect.signature(JointTrajectoryFlow.sample).parameters)


def test_exact_single_actor_mask_and_same_rng_draw_schedule():
    first=torch.Generator().manual_seed(23);second=torch.Generator().manual_seed(23)
    one=actor_mask(256,65,'cpu',first,all_hidden_probability=0.,partial_mode='single_actor')
    actor_mask(256,65,'cpu',second,all_hidden_probability=0.,partial_mode='bernoulli')
    assert (one.sum(-1)==1).all()
    assert torch.equal(first.get_state(),second.get_state())
    mixed=actor_mask(256,65,'cpu',all_hidden_probability=.5,partial_mode='single_actor')
    assert ((mixed.sum(-1)==1)|(mixed.sum(-1)==65)).all()


def test_masked_clean_context_cannot_leak_even_nan():
    m, x, c = setup()
    time = torch.tensor([.2, .7]); mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    a = m(x, time, **c, known_xy=torch.randn_like(x), known_mask=mask)[0]
    b = m(x, time, **c, known_xy=torch.full_like(x, float('nan')), known_mask=mask)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_non_ego_actor_permutation_equivariance():
    m, x, c = setup(); perm = torch.tensor([0, 3, 1, 2]); time = torch.tensor([.2, .7])
    y, _ = m(x, time, **c)
    p = {k: (v if k == 'context' else v[:, perm]) for k, v in c.items()}
    z, _ = m(x[:, perm], time, **p)
    torch.testing.assert_close(y[:, perm], z, rtol=2e-5, atol=2e-6)


def test_empty_missing_and_all_hidden_gradients():
    m, x, c = setup(); target = torch.randn_like(x); valid = torch.ones(2, 4, 3, dtype=torch.bool)
    valid[:, 1:] = False; target[:, 1:] = float('nan')
    hidden = torch.ones(2, 4, dtype=torch.bool)
    sums, counts = training_loss_sums(m, target, valid, hidden, x, torch.tensor([.2, .7]), **c)
    assert counts['agents'] == 0 and counts['ego'] == 12
    loss = sum(sums[k] / max(counts[k], 1) for k in sums)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.blocks[0].actor.in_proj_weight.grad.norm() > 0
    assert m.condition[1].weight.grad.norm() > 0


def test_joint_sample_and_save_restore():
    m, x, c = setup(); y, h = m.sample(x, **c, sampling_steps=2)
    stream = io.BytesIO(); torch.save(m.state_dict(), stream); stream.seek(0)
    n, _, _ = setup(); n.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    z, g = n.sample(x, **c, sampling_steps=2)
    torch.testing.assert_close(y, z, rtol=0, atol=0)
    torch.testing.assert_close(h, g, rtol=0, atol=0)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_residual_roundtrip_stationary_and_no_centre_gradient():
    model = JointTrajectoryFlow(12, dim=16, heads=4, layers=1, steps=3,
                                trajectory_mode='current_residual', agent_scale_m=5.)
    centres = torch.tensor([[[0., 0.], [24., 7.]]], requires_grad=True)
    zero = torch.zeros(1, 2, 3, 2, requires_grad=True)
    static = model.decode_trajectories(zero, centres)
    torch.testing.assert_close(static, centres.detach()[:, :, None].expand_as(static))
    future = torch.randn_like(static) * 10
    torch.testing.assert_close(model.decode_trajectories(model.encode_trajectories(future, centres), centres), future)
    static.sum().backward()
    assert centres.grad is None
    torch.testing.assert_close(zero.grad[:, 0], torch.full_like(zero.grad[:, 0], 20.))
    torch.testing.assert_close(zero.grad[:, 1], torch.full_like(zero.grad[:, 1], 5.))


def test_residual_unknown_future_values_do_not_enter_context():
    model = JointTrajectoryFlow(12, dim=16, heads=4, layers=1, steps=3,
                                trajectory_mode='current_residual', agent_scale_m=5.)
    _, x, current = setup()
    mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    a = model(x, torch.ones(2), **current, known_xy=torch.randn_like(x), known_mask=mask)[0]
    b = model(x, torch.ones(2), **current, known_xy=torch.full_like(x, float('nan')), known_mask=mask)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
