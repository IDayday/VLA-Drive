import pytest
import torch

from starVLA.training.hierarchical_schedule import build_hierarchical_schedule
from starVLA.training.train_starvla import _combine_hierarchical_losses


def test_single_task_curriculum_boundaries():
    start = build_hierarchical_schedule(0, 100)
    assert not start.dynamic_enabled and start.dynamic_topm == 0
    ramp = build_hierarchical_schedule(15, 100)
    assert ramp.dynamic_enabled
    assert ramp.num_dynamic_candidates == 64
    assert ramp.dynamic_topm == 48
    assert ramp.lambda_drivor == pytest.approx(0.5)
    final = build_hierarchical_schedule(20, 100)
    assert final.dynamic_enabled and final.dynamic_topm == 32
    assert final.lambda_drivor == 1.0


def test_zero_weight_loss_is_omitted_from_backward_graph():
    schedule = build_hierarchical_schedule(10, 100)
    assert schedule.dynamic_enabled and schedule.lambda_drivor == 0.0
    parameters = {
        name: torch.nn.Parameter(torch.tensor(float(index + 1)))
        for index, name in enumerate(
            ("flow", "drivor", "suprim_coarse", "suprim_fine")
        )
    }
    losses = {name: parameter.square() for name, parameter in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=0.1, weight_decay=0.1)
    inactive_before = parameters["drivor"].detach().clone()

    _combine_hierarchical_losses(losses, schedule).backward()

    assert parameters["drivor"].grad is None
    assert all(
        parameter.grad is not None
        for name, parameter in parameters.items()
        if name != "drivor"
    )
    optimizer.step()
    torch.testing.assert_close(parameters["drivor"], inactive_before)
