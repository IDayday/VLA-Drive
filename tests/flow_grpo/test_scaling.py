import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from starVLA.rl.flow_grpo.distributed import globally_weighted_sum


def objective(p, scene_values, g, k):
    # Policy terms mean over G,K; SFT replay enters once per scene.
    policy = ((p * scene_values[:, None, None]).expand(-1, g, k) ** 2).mean((1, 2))
    sft = (p - scene_values).square()
    return policy + 0.1 * sft


@pytest.mark.parametrize("g,k", [(2, 3), (8, 10)])
def test_group_step_microbatch_and_accumulation_scaling(g, k):
    x = torch.tensor([0.2, 1.0, 2.0, 4.0], dtype=torch.float64)
    p = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    objective(p, x, g, k).mean().backward()
    reference = p.grad.clone()
    for micro in [1, 2, 4]:
        q = p.detach().clone().requires_grad_()
        for chunk in x.split(micro):
            (objective(q, chunk, g, k).sum() / len(x)).backward()
        torch.testing.assert_close(q.grad, reference, rtol=1e-12, atol=1e-12)


def _rank_worker(rank, world, path, result):
    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=world
    )
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.7)
    model = DistributedDataParallel(model)
    # Different valid counts, rank0 all-equal reward => zero policy term, but SFT
    # still contributes. No dummy zero loss to unused parameter branches.
    xs = [
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64),
    ]
    x = xs[rank]
    prediction = model(x)
    local = (
        prediction.square() * (0 if rank == 0 else 1) + 0.1 * (prediction - x).square()
    ).sum()
    loss = globally_weighted_sum(local, 4.0)
    loss.backward()
    if rank == 0:
        torch.save(model.module.weight.grad, result)
    dist.destroy_process_group()


def test_real_torch_ddp_unequal_counts_and_equal_group(tmp_path):
    rendezvous = str(tmp_path / "init")
    out = str(tmp_path / "result.pt")
    mp.spawn(_rank_worker, args=(2, rendezvous, out), nprocs=2, join=True)
    p = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    loss = (
        (p * x).square() * torch.tensor([0, 1, 1, 1]) + 0.1 * (p * x - x).square()
    ).mean()
    loss.backward()
    actual = torch.load(out, weights_only=True)
    torch.testing.assert_close(actual.squeeze(), p.grad, rtol=1e-12, atol=1e-12)
