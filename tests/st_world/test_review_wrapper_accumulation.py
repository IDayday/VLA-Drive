"""Independent full-wrapper reduction and coordinated bad-cache checks."""
import copy
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.model.framework.st_world_model import SpaceTimeStudent
from starVLA.model.modules.st_world.losses import optimizer_batch_counts
from starVLA.model.modules.st_world.requests import sample_requests, gather_current
from tools.st_world_policy.train_world import collective_call

from test_student_wrapper import Parent, inputs, options


def test_full_wrapper_ego_and_aux_microbatch_gradient_agree():
    torch.manual_seed(123)
    whole = SpaceTimeStudent(Parent(), options(), initialization_seed_value=42).train()
    accumulated = copy.deepcopy(whole)
    observations = inputs(3)
    request = sample_requests(3, 3, (18, 24), 96, patch_generator=torch.Generator().manual_seed(7))
    target = {"ego": torch.randn(3, 8, 4), "current_dense_features": torch.randn(3, 3, 18, 24, 2),
              "current_valid": torch.ones(3, 3, 18, 24, dtype=torch.bool)}
    target["current_valid"][1].zero_()
    target["current_valid"][2, :, :9].zero_()
    _, valid = gather_current(target["current_dense_features"], target["current_valid"], request)
    counts = optimizer_batch_counts(2, current_valid=valid)
    counts["ego_scenes"] = 3

    def execute(model, start, end):
        return model(observations[start:end], {k: v[start:end] for k, v in target.items()},
            requests=request.slice(start, end), global_counts=counts,
            noise_generator=torch.Generator().manual_seed(31), time_generator=torch.Generator().manual_seed(32))["loss"]

    expected = execute(whole, 0, 3)
    expected.backward()
    actual = 0.
    for start, end in ((0, 1), (1, 3)):
        loss = execute(accumulated, start, end)
        loss.backward()
        actual += float(loss.detach())
    torch.testing.assert_close(torch.tensor(actual), expected.detach(), rtol=2e-6, atol=1e-7)
    compared = 0
    for (name_a, a), (name_b, b) in zip(whole.named_parameters(), accumulated.named_parameters()):
        assert name_a == name_b
        if not a.requires_grad:
            assert a.grad is None and b.grad is None
        elif a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=3e-7, msg=name_a)
            compared += 1
    assert compared > 10


def _bad_cache_rank(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        def cache_read():
            if rank == 1:
                raise FileNotFoundError("review fixture missing rank1 teacher record")
            return torch.ones(1)
        try:
            collective_call(cache_read, "current data and teacher labels")
            status = {"failed": False}
        except RuntimeError as error:
            status = {"failed": True, "error": str(error)}
        Path(output, str(rank) + ".json").write_text(json.dumps(status))
    finally:
        dist.destroy_process_group()


def test_single_rank_bad_cache_exits_both_ranks_before_backward(tmp_path):
    mp.spawn(_bad_cache_rank, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / (str(rank) + ".json")).read_text()) for rank in range(2)]
    assert all(result["failed"] for result in results)
    assert results[0]["error"] == results[1]["error"]
    assert "missing rank1 teacher record" in results[0]["error"]
