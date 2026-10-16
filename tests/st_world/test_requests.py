import pytest
import torch
from starVLA.model.modules.st_world.requests import (
    SampledRequests, sample_requests, gather_current, gather_future)


def streams(seed=12):
    return dict(patch_generator=torch.Generator().manual_seed(seed),
                horizon_generator=torch.Generator().manual_seed(seed + 100))


def test_sampling_reproducible_independent_and_recoverable():
    torch.manual_seed(777)
    global_before = torch.get_rng_state().clone()
    generators = streams()
    first = sample_requests(5, 3, (18, 24), 96, future=True, **generators)
    states = {k: g.get_state() for k, g in generators.items()}
    second = sample_requests(5, 3, (18, 24), 96, future=True, **generators)
    for key, state in states.items():
        generators[key].set_state(state)
    restored = sample_requests(5, 3, (18, 24), 96, future=True, **generators)
    assert torch.equal(second.patch_indices, restored.patch_indices)
    assert torch.equal(second.future_pair, restored.future_pair)
    assert not torch.equal(first.patch_indices, second.patch_indices)
    for selected in first.patch_indices.flatten(0, 1):
        assert len(torch.unique(selected)) == 96
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.equal(torch.cat([first.slice(0, 2).patch_indices, first.slice(2, 5).patch_indices]), first.patch_indices)


def test_full_grid_is_in_order_and_does_not_advance_patch_rng():
    generators = streams()
    before = generators['patch_generator'].get_state()
    request = sample_requests(2, 3, (6, 8), 48, **generators)
    assert torch.equal(request.patch_indices, torch.arange(48).expand(2, 3, -1))
    assert torch.equal(before, generators['patch_generator'].get_state())
    assert request.future_pair is None


def test_rectangular_coordinates_times_and_view_order():
    request = SampledRequests(torch.tensor([[[0, 5], [1, 4], [2, 3]]]), (2, 3), torch.tensor([[0, 1]]))
    views, xy, times = request.query_layout(current=True, future=True)
    assert views.tolist() == [[0, 0, 1, 1, 2, 2] * 3]
    torch.testing.assert_close(xy[0, :2], torch.tensor([[-2/3, -.5], [2/3, .5]]))
    assert times.tolist() == [[0.] * 6 + [.5] * 6 + [1.] * 6]
    assert torch.equal(xy[:, :6], xy[:, 6:12]) and torch.equal(xy[:, :6], xy[:, 12:])


def test_future_pair_gathers_same_patches_and_keeps_masks_label_side():
    request = SampledRequests(torch.tensor([[[0, 5], [1, 4], [2, 3]], [[3, 2], [4, 1], [5, 0]]]),
                              (2, 3), torch.tensor([[0, 1], [2, 3]]))
    features = torch.arange(2 * 4 * 3 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 3, 2)
    features.requires_grad_(True)
    valid = torch.ones(features.shape[:-1], dtype=torch.bool)
    valid[1, 2, 1, 1, 1] = False
    selected, masks = gather_future(features, valid, request)
    for b in range(2):
        for t in range(2):
            for v in range(3):
                for k, patch in enumerate(request.patch_indices[b, v]):
                    y, x = divmod(patch.item(), 3)
                    source = (b, request.future_pair[b, t], v, y, x)
                    assert torch.equal(selected[b, t, v, k], features[source])
                    assert masks[b, t, v, k] == valid[source]
    assert not selected.requires_grad
    before = request.query_layout(current=True, future=True)
    gather_future(features * 17, ~valid, request)
    assert all(torch.equal(a, b) for a, b in zip(before, request.query_layout(current=True, future=True)))


def test_current_gather_and_tail_batch_slices_do_not_resample():
    request = sample_requests(5, 3, (3, 4), 4, **streams())
    features = torch.arange(5 * 3 * 3 * 4 * 2).float().reshape(5, 3, 3, 4, 2)
    valid = torch.ones(features.shape[:-1], dtype=torch.bool)
    full, mask = gather_current(features, valid, request)
    tail, tail_mask = gather_current(features[4:], valid[4:], request.slice(4, 5))
    assert torch.equal(full[4:], tail) and torch.equal(mask[4:], tail_mask)


@pytest.mark.parametrize('kwargs', [
    {'grid_hw': None}, {'grid_hw': (2,)}, {'grid_hw': (True, 6)},
    {'positions_per_view': 433}, {'batch_size': 0}, {'future': 'false'},
])
def test_invalid_sampling_rejected(kwargs):
    arguments = dict(batch_size=2, views=3, grid_hw=(18, 24), positions_per_view=96)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        sample_requests(**arguments, **streams())


def test_separate_generators_and_adjacent_times_required():
    g = torch.Generator()
    with pytest.raises(ValueError, match='separate'):
        sample_requests(2, 3, (6, 8), 48, future=True, patch_generator=g, horizon_generator=g)
    with pytest.raises(ValueError, match='adjacent'):
        SampledRequests(torch.zeros(1, 3, 2, dtype=torch.long), (2, 3), torch.tensor([[0, 2]])).validate()
    with pytest.raises(ValueError, match='no sampled time'):
        SampledRequests(torch.zeros(1, 3, 1, dtype=torch.long), (2, 3)).query_layout(future=True)


def test_wrong_target_layout_and_address_fail():
    request = sample_requests(1, 3, (2, 3), 2, **streams())
    with pytest.raises(ValueError, match='different target layout'):
        gather_current(torch.zeros(1, 3, 3, 2, 4), torch.ones(1, 3, 3, 2, dtype=torch.bool), request)
    with pytest.raises(ValueError, match='does not exist'):
        SampledRequests(torch.full((1, 3, 1), 6, dtype=torch.long), (2, 3)).validate()
