import torch

from starVLA.model.modules.field2plan.trajectory_refiner import TrajectoryRefiner


def test_zero_initialized_refiner_is_bitwise_draft_identity() -> None:
    refiner = TrajectoryRefiner(context_dim=16, hidden_dim=32, num_layers=2)
    draft = torch.randn(2, 3, 8, 4)
    context = torch.randn(2, 3, 8, 16)

    output = refiner(draft, context)

    assert torch.equal(output.final_action, draft)
    assert torch.count_nonzero(output.delta_physical) == 0
    assert output.delta_norm.item() == 0.0


def test_zero_initialized_output_projection_can_receive_gradient() -> None:
    refiner = TrajectoryRefiner(context_dim=8, hidden_dim=16, num_layers=1)
    draft = torch.zeros(1, 1, 8, 4)
    draft[..., 3] = 1.0
    context = torch.randn(1, 1, 8, 8)
    output = refiner(draft, context)
    loss = output.final_action[..., 0].sum()
    loss.backward()
    assert refiner.output_projection.weight.grad is not None
    assert torch.count_nonzero(refiner.output_projection.weight.grad) > 0
