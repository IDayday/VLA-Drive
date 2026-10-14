from dataclasses import asdict
import pytest
import torch
from iqe.contracts import TrajectoryContract
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.models import StudyScorer
from iqe.conditional_readout.model import ConditionalReadoutScorer


def contract():
    return TrajectoryContract(8, .5, 'normalized_xy_sincos', 'ego_relative', 'rear_axle',
                              'radian', (10., .3, 0., 0.), (8., 2., 1., 1.), 'test')


def inputs(k=3):
    return dict(world=torch.randn(2, 5, 8), action=torch.randn(2, 2, 8),
                current_ego=torch.randn(2, 4), world_valid=torch.ones(2, 5, dtype=torch.bool),
                action_valid=torch.ones(2, 2, dtype=torch.bool), visual=torch.randn(2, 3, 7, 8),
                visual_valid=torch.ones(2, 3, 7, dtype=torch.bool), visual_xy=torch.randn(2, 3, 7, 2)), torch.randn(2, k, 8, 3)


def model(conditioned=True):
    with initialization_rng(41):
        return ConditionalReadoutScorer(8, 8, contract(), visual_dim=8, views=3, conditioned=conditioned)


def test_same_random_parameters_and_exact_original_global_control():
    with initialization_rng(41):
        old = StudyScorer(8, 8, contract(), spatial=True, visual_dim=8, views=3).eval()
    control, conditional = model(False).eval(), model(True).eval()
    assert module_hash(old) == module_hash(control) == module_hash(conditional)
    x, t = inputs()
    with torch.no_grad():
        assert torch.equal(old(x, t)['values'], control(x, t)['values'])
        assert not torch.equal(control(x, t)['values'], conditional(x, t)['values'])


def test_candidate_permutation_append_duplicate_and_padding():
    m = model().eval(); x, t = inputs()
    with torch.no_grad():
        expected = m(x, t)['values']
        assert torch.allclose(m(x, t[:, [2, 0, 1]])['values'], expected[:, [2, 0, 1]], atol=2e-6, rtol=0)
        assert torch.allclose(m(x, torch.cat((t, t[:, :1]), 1))['values'][:, :3], expected, atol=2e-6, rtol=0)
        duplicate = m(x, t[:, :1].expand(-1, 2, -1, -1))['values']
        assert torch.equal(duplicate[:, 0], duplicate[:, 1])
        x['visual_valid'][:, :, -1] = False
        out = m(x, t)['values']
        x['visual'][:, :, -1] = 10000
        assert torch.allclose(m(x, t)['values'], out, atol=2e-6, rtol=0)
        invalid = t.clone(); invalid[:, 1] = float('nan')
        pred = m(x, invalid)
        assert (~pred['valid'][:, 1]).all()
        assert torch.allclose(pred['values'][:, [0, 2]], out[:, [0, 2]], atol=2e-6, rtol=0)


def test_real_gradient_paths_detach_inputs_and_one_shared_projection():
    m = model(); x, t = inputs(k=4)
    for value in x.values():
        if value.dtype.is_floating_point: value.requires_grad_(True)
    t.requires_grad_(True)
    count = {'semantic': 0, 'visual': 0}
    def hit(name):
        def f(*_): count[name] += 1
        return f
    hs = [m.world_projection.register_forward_hook(hit('semantic')),
          m.spatial_reader.projection.register_forward_hook(hit('visual'))]
    try:
        p = m(x, t)
        (p['values'].square().sum()+p['component_logits'].square().mean()).backward()
    finally:
        for h in hs: h.remove()
    assert count == {'semantic': 1, 'visual': 1}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.spatial_reader.queries.grad.norm() > 0
    assert m.trajectory_encoder[0].weight.grad.norm() > 0
    assert t.grad is None and all(v.grad is None for v in x.values())
    with pytest.raises(ValueError, match='whitelist'):
        m(x | {'GT': torch.ones(2, 8, 3)}, t)
