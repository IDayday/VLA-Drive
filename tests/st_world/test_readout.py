import pytest
import torch
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.st_world import SparseFeatureReadout, sample_requests


def make_head():
    return SparseFeatureReadout(8, 5, dim=16, heads=4, layers=2)


@pytest.mark.parametrize('world_count', [7, 144, 288])
def test_dense_registered_control_strict_loads_and_matches_old_readout(world_count):
    torch.manual_seed(3)
    legacy = DINOFeatureHead(8, 5, dim=16, heads=4, layers=2).eval()
    sparse = make_head().eval()
    sparse.load_state_dict(legacy.state_dict(), strict=True)
    world = torch.randn(2, world_count, 8)
    request = sample_requests(2, 3, (2, 3), 6, patch_generator=torch.Generator())
    ids, xy, time = request.query_layout()
    actual = sparse(world, ids, xy, time).reshape(2, 3, 2, 3, 5)
    expected = legacy(world, torch.zeros(2), (2, 3)).permute(0, 1, 3, 4, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert set(sparse.state_dict()) == set(legacy.state_dict())


def test_future_time_encoding_keeps_legacy_fp32_then_autocast_boundary():
    torch.manual_seed(17)
    legacy = DINOFeatureHead(8, 5, dim=16, heads=4, layers=2).eval()
    sparse = make_head().eval()
    sparse.load_state_dict(legacy.state_dict(), strict=True)
    world = torch.randn(3, 9, 8)
    request = sample_requests(3, 3, (2, 3), 6, patch_generator=torch.Generator())
    ids, xy, _ = request.query_layout()
    horizons = torch.tensor([1., 2., 4.])
    times = horizons[:, None].expand_as(ids)
    captures = []
    h0 = legacy.time.register_forward_pre_hook(lambda m, args: captures.append(args[0].detach().clone()))
    h1 = sparse.time.register_forward_pre_hook(lambda m, args: captures.append(args[0].detach().clone()))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        legacy(world, horizons, (2, 3))
        sparse(world, ids, xy, times)
    h0.remove()
    h1.remove()
    assert captures[0].dtype == captures[1].dtype == torch.bfloat16
    assert torch.equal(captures[0][:, None].expand(-1, ids.shape[1], -1), captures[1])


def test_query_reordering_duplicates_and_independent_of_world_count():
    torch.manual_seed(1)
    head, world = make_head().eval(), torch.randn(2, 11, 8)
    request = sample_requests(2, 3, (18, 24), 4, future=True,
        patch_generator=torch.Generator(), horizon_generator=torch.Generator().manual_seed(7))
    ids, xy, times = request.query_layout(future=True)
    prediction = head(world, ids, xy, times)
    permutation = torch.randperm(ids.shape[1])
    torch.testing.assert_close(head(world, ids[:, permutation], xy[:, permutation], times[:, permutation]),
                               prediction[:, permutation], rtol=1e-6, atol=1e-7)
    duplicated = head(world, ids[:, :1].expand(-1, 2), xy[:, :1].expand(-1, 2, -1), times[:, :1].expand(-1, 2))
    torch.testing.assert_close(duplicated[:, 0], duplicated[:, 1], rtol=0, atol=0)


def test_full_memory_projection_once_and_gradients_through_all_readout_components():
    torch.manual_seed(5)
    head, world = make_head(), torch.randn(2, 9, 8, requires_grad=True)
    request = sample_requests(2, 3, (2, 3), 6, future=True,
        patch_generator=torch.Generator(), horizon_generator=torch.Generator().manual_seed(5))
    calls = []
    handle = head.project.register_forward_hook(lambda m, a, o: calls.append(a[0].shape))
    prediction = head(world, *request.query_layout(future=True))
    prediction.square().sum().backward()
    handle.remove()
    assert calls == [world.shape]
    assert world.grad is not None and world.grad.isfinite().all() and world.grad.norm() > 0
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        assert parameter.grad.norm() > 0, name


def test_masked_memory_nan_is_padding_not_a_label_or_model_input():
    torch.manual_seed(9)
    head = make_head().eval()
    world = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    world[:, 2:] = float('nan')
    request = sample_requests(1, 3, (2, 2), 4, patch_generator=torch.Generator())
    args = request.query_layout()
    actual = head(world, *args, memory_valid=mask)
    expected = head(world[:, :2], *args)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match='at least one'):
        head(world, *args, memory_valid=torch.zeros_like(mask))
    with pytest.raises(ValueError, match='Nonfinite W'):
        head(world, *args)


@pytest.mark.parametrize('kind', ['time', 'view', 'xy', 'world', 'device_mask'])
def test_invalid_model_queries_fail_closed(kind):
    head, world = make_head(), torch.randn(1, 3, 8)
    request = sample_requests(1, 3, (2, 2), 4, patch_generator=torch.Generator())
    ids, xy, times = request.query_layout()
    extra = {}
    if kind == 'time': times[0, 0] = 1.5
    if kind == 'view': ids[0, 0] = 3
    if kind == 'xy': xy[0, 0, 0] = float('nan')
    if kind == 'world': world[0, 0, 0] = float('inf')
    if kind == 'device_mask': extra['memory_valid'] = torch.ones(1, 3)
    with pytest.raises(ValueError):
        head(world, ids, xy, times, **extra)
