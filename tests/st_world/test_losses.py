from datetime import timedelta
from itertools import combinations
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from starVLA.model.modules.st_world import (
    masked_mse, optimizer_batch_counts, temporal_losses, weighted_temporal_loss)
from starVLA.model.modules.st_world.config import SpaceTimeConfig


def test_temporal_terms_use_separate_valid_counts_and_raw_delta_amplitude():
    p = torch.tensor([[[[[1., 2.], [3., 4.]]], [[[1.01, 2.02], [3.2, 4.4]]]]], requires_grad=True)
    target = torch.zeros_like(p, requires_grad=True)
    valid = torch.tensor([[[[True, True]], [[True, False]]]])
    counts = optimizer_batch_counts(2, future_valid=valid, delta=True)
    losses, terms = temporal_losses(counts, future_prediction=p, future_target=target, future_valid=valid, delta=True)
    assert {k: v.item() for k, v in counts.items()} == {'delta': 2, 'future_a': 4, 'future_b': 2}
    torch.testing.assert_close(losses['future'], .5 * (p[:, 0].square().mean() + p[:, 1][valid[:, 1]].square().mean()))
    expected_delta = torch.tensor((.01**2 + .02**2) / 2)
    torch.testing.assert_close(losses['delta'], expected_delta, rtol=1e-5, atol=1e-9)
    (losses['future'] + losses['delta']).backward()
    assert target.grad is None
    assert p.grad is not None and p.grad.isfinite().all()
    assert not terms['delta'].numerator.requires_grad


def dataset(all_invalid=False):
    inputs = torch.arange(4 * 3 * 1 * 2 * 3).float().reshape(4, 3, 1, 2, 3) / 23.
    targets = torch.cos(inputs[..., :2] * 3)
    current = torch.tensor([[[True, True]], [[True, False]], [[False, False]], [[False, False]]])
    future = torch.tensor([[[[True, True]], [[True, False]]], [[[True, False]], [[True, True]]],
                           [[[False, False]], [[False, False]]], [[[False, False]], [[False, False]]]])
    if all_invalid:
        current.zero_()
        future.zero_()
        targets.fill_(float('nan'))
    return inputs, targets, current, future


def linear_model():
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.1, .2, -.3], [-.2, .4, .1]]))
        model.bias.copy_(torch.tensor([.05, -.1]))
    return model


def objective(prediction, target, current_valid, future_valid, counts):
    losses, _ = temporal_losses(counts, current_prediction=prediction[:, 0], current_target=target[:, 0],
        current_valid=current_valid, future_prediction=prediction[:, 1:], future_target=target[:, 1:],
        future_valid=future_valid, delta=True)
    return losses['current'] + .2 * losses['future'] + .05 * losses['delta']


@pytest.mark.parametrize('split', [(1, 3), (2, 2), (1, 1, 1, 1)])
def test_microbatch_sum_matches_global_batch_loss_and_gradient(split):
    inputs, targets, current, future = dataset()
    counts = optimizer_batch_counts(2, current_valid=current, future_valid=future, delta=True)
    full, micro = linear_model(), linear_model()
    expected = objective(full(inputs), targets, current, future, counts)
    expected.backward()
    offset, actual = 0, torch.tensor(0.)
    for size in split:
        sl = slice(offset, offset + size)
        loss = objective(micro(inputs[sl]), targets[sl], current[sl], future[sl], counts)
        loss.backward()
        actual += loss.detach()
        offset += size
    torch.testing.assert_close(actual, expected.detach())
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-6, atol=1e-7)


def test_invalid_padding_is_differentiable_zero_and_valid_nan_is_error():
    prediction = torch.full((2, 3, 4), float('nan'), requires_grad=True)
    target = torch.full_like(prediction, float('nan'))
    valid = torch.zeros(2, 3, dtype=torch.bool)
    term = masked_mse(prediction, target, valid, 0)
    assert term.loss.item() == 0 and term.loss.requires_grad and term.loss.dtype == torch.float32
    term.loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    valid[0, 0] = True
    with pytest.raises(ValueError, match='Nonfinite'):
        masked_mse(prediction, target, valid, 4)


def test_all_invalid_temporal_graph_has_zero_gradient():
    inputs, targets, current, future = dataset(all_invalid=True)
    model = linear_model()
    counts = optimizer_batch_counts(2, current_valid=current, future_valid=future, delta=True)
    loss = objective(model(inputs), targets, current, future, counts)
    loss.backward()
    assert loss.item() == 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.equal(parameter.grad, torch.zeros_like(parameter))


@pytest.mark.parametrize('bad_count', [-1, 1.5, 3, float('nan'), [4]])
def test_wrong_global_denominator_rejected(bad_count):
    p = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match='denominator'):
        masked_mse(p, torch.zeros_like(p), torch.ones(1, 2, dtype=torch.bool), bad_count)


def test_counts_must_match_active_objectives_and_complete_requests():
    p = torch.ones(1, 1, 2, 2)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    for counts in ({}, {'current': 4, 'delta': 4}):
        with pytest.raises(ValueError, match='inventory'):
            temporal_losses(counts, current_prediction=p, current_target=p, current_valid=valid)
    with pytest.raises(ValueError, match='together'):
        temporal_losses({'current': 4}, current_prediction=p)
    with pytest.raises(ValueError, match='both future'):
        optimizer_batch_counts(2, current_valid=valid, delta=True)
    with pytest.raises(ValueError, match='same original'):
        optimizer_batch_counts(2, current_valid=valid, future_valid=torch.ones(2, 2, 1, 2, dtype=torch.bool))


def test_uniform_sparse_mse_expectation_matches_full_grid():
    p = torch.tensor([.1, .3, .5, 1., 2., 4.]).reshape(1, 6, 1)
    target = p * .4
    full = masked_mse(p, target, torch.ones(1, 6, dtype=torch.bool), 6).loss
    samples = []
    for indices in combinations(range(6), 2):
        samples.append(masked_mse(p[:, indices], target[:, indices], torch.ones(1, 2, dtype=torch.bool), 2).loss)
    torch.testing.assert_close(torch.stack(samples).mean(), full)


def test_registered_weights_apply_once_without_repeat_or_accumulation_scale():
    teacher = dict(image_hw=[288, 384], patch_hw=[16, 16], native_grid_hw=[18, 24],
        output_grid_hw=[18, 24], channels=256, storage_dtype='float32', compute_dtype='float32',
        view_names=['CAM_F0', 'CAM_L0', 'CAM_R0'], teacher_identity='a'*64, transform_identity='b'*64,
        pooling={'kind': 'none', 'kernel_hw': [1, 1], 'stride_hw': [1, 1]})
    config = SpaceTimeConfig.from_dict(dict(cell='T4', spatial_parent='S-P2', teacher=teacher,
        world_queries_per_view=48, sampled_positions=96, history=True, future=True, delta=True,
        future_weight=.2, delta_weight=.05))
    terms = {'current': torch.tensor(2.), 'future': torch.tensor(3.), 'delta': torch.tensor(4.)}
    torch.testing.assert_close(weighted_temporal_loss(terms, config), torch.tensor(2.8))
    with pytest.raises(ValueError, match='differ'):
        weighted_temporal_loss({'current': torch.tensor(2.)}, config)


def _distributed_worker(rank, rendezvous, result_dir, case):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        if case == 'invalid_rank':
            mask = torch.ones(1, 1, 2, dtype=torch.bool if rank == 0 else torch.float32)
            caught = None
            try:
                optimizer_batch_counts(2, current_valid=mask)
            except ValueError as exc:
                caught = str(exc)
            Path(result_dir, f'{rank}.json').write_text(json.dumps({'error': caught}))
            return
        inputs, targets, current, future = dataset(all_invalid=case == 'all_invalid')
        sl = slice(rank * 2, (rank + 1) * 2)
        inputs, targets, current, future = inputs[sl], targets[sl], current[sl], future[sl]
        counts = optimizer_batch_counts(2, current_valid=current, future_valid=future, delta=True)
        model = DistributedDataParallel(linear_model())
        value = torch.tensor(0.)
        for offset in range(2):
            loss = objective(model(inputs[offset:offset+1]), targets[offset:offset+1],
                             current[offset:offset+1], future[offset:offset+1], counts)
            loss.backward()
            value += loss.detach()
        dist.all_reduce(value)
        value /= 2
        torch.save({'gradient': [p.grad for p in model.module.parameters()], 'loss': value,
                    'counts': counts}, Path(result_dir, f'{rank}.pt'))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('case', ['rank_without_labels', 'all_invalid', 'invalid_rank'])
def test_two_rank_gloo_matches_global_gradient_and_fails_together(tmp_path, case):
    rendezvous = str(tmp_path / 'rendezvous')
    mp.spawn(_distributed_worker, args=(rendezvous, str(tmp_path), case), nprocs=2, join=True)
    if case == 'invalid_rank':
        for rank in range(2):
            result = json.loads((tmp_path / f'{rank}.json').read_text())
            assert 'Distributed count validation failed' in result['error']
        return
    inputs, targets, current, future = dataset(all_invalid=case == 'all_invalid')
    reference = linear_model()
    counts = optimizer_batch_counts(2, current_valid=current, future_valid=future, delta=True, distributed=False)
    expected = objective(reference(inputs), targets, current, future, counts)
    expected.backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f'{rank}.pt', weights_only=True)
        torch.testing.assert_close(actual['loss'], expected.detach())
        for name in counts:
            assert actual['counts'][name] == counts[name]
        for a, b in zip(actual['gradient'], reference.parameters()):
            torch.testing.assert_close(a, b.grad, rtol=2e-6, atol=1e-7)
