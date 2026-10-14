from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from iqe.specialist_study.objectives import terms, denominators


def fixture():
    score = torch.tensor([[.95, .1, .7], [.2, .8, float('nan')],
                          [.6, float('nan'), float('nan')], [float('nan')] * 3])
    valid = torch.isfinite(score)
    component = torch.full((4, 3, 6), .7)
    cv = valid[..., None].expand(-1, -1, 6).clone()
    cv[2] = False
    return {'scores': score, 'valid': valid, 'components': component, 'component_valid': cv,
            'natural': torch.tensor([True, False, True, False]),
            'comparison': torch.tensor([False, True, False, False])}


class SmallHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor([.12, -.09, .06]))
        self.component = torch.nn.Parameter(torch.linspace(-.1, .1, 18).reshape(3, 6))

    def forward(self, x):
        return {'values': torch.sigmoid(x[:, None] * self.value),
                'component_logits': x[:, None, None] * self.component}


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    model = SmallHead()
    wrapped = DistributedDataParallel(model)
    data = {k: v[rank::2] for k, v in fixture().items()}
    denominator = denominators(data)
    for value in denominator.values():
        dist.all_reduce(value)
    result = terms(wrapped(torch.arange(1., 5.)[rank::2]), data)
    weights = {'value': 1., 'components': .2, 'ranking': .2}
    loss = sum(v.numerator * 2 / denominator[k].clamp_min(1) * weights[k] for k, v in result.items())
    loss.backward()
    torch.save({k: v.grad for k, v in model.named_parameters()}, Path(output) / f'rank{rank}.pt')
    dist.destroy_process_group()


def test_variable_k_two_flow_global_denominators_gloo(tmp_path):
    reference = SmallHead()
    data = fixture()
    result = terms(reference(torch.arange(1., 5.)), data)
    weights = {'value': 1., 'components': .2, 'ranking': .2}
    sum(v.numerator / v.denominator.clamp_min(1) * weights[k] for k, v in result.items()).backward()
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        grads = torch.load(tmp_path / f'rank{rank}.pt', weights_only=True)
        for name, parameter in reference.named_parameters():
            assert torch.allclose(grads[name], parameter.grad, atol=2e-7, rtol=1e-6)
