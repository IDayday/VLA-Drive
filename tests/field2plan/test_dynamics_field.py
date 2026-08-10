import inspect

import torch

from starVLA.model.modules.field2plan.dynamics_field_writer import (
    ActionFreeDynamicsFieldWriter,
)
from starVLA.model.modules.field2plan.dynamics_supervision import (
    DynamicsSupervisionHead,
    DynamicsTargets,
    dynamics_supervision_losses,
)


def test_action_free_writer_interface_and_output_gradient() -> None:
    signature = inspect.signature(ActionFreeDynamicsFieldWriter.forward)
    forbidden = {"action", "draft", "future_action", "trajectory"}
    assert forbidden.isdisjoint(signature.parameters)

    writer = ActionFreeDynamicsFieldWriter(
        input_channels=8,
        output_channels=12,
        horizon=8,
        history_length=4,
        hidden_channels=16,
    )
    current = torch.randn(2, 8, 6, 7, requires_grad=True)
    history_current_from_ego = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 4, 1, 1)
    history_current_from_ego[:, :, 0, 3] = torch.arange(4)

    output = writer(current, history_current_from_ego)

    assert output.field.shape == (2, 8, 12, 6, 7)
    assert output.log_variance.shape == (2, 8, 1, 6, 7)
    output.field.sum().backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_action_free_writer_keeps_history_order_and_spatial_context() -> None:
    torch.manual_seed(7)
    writer = ActionFreeDynamicsFieldWriter(
        input_channels=4,
        output_channels=6,
        horizon=3,
        history_length=4,
        hidden_channels=8,
    ).eval()
    current = torch.randn(1, 4, 5, 5)
    ordered = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1)
    ordered[0, :, 0, 3] = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    permuted = ordered[:, torch.tensor([2, 1, 0, 3])]

    ordered_output = writer(current, ordered).field
    permuted_output = writer(current, permuted).field

    assert not torch.allclose(ordered_output, permuted_output)
    assert writer.spatial_context_mode == "avg_pool3x3_linear_fusion"
    assert not any(isinstance(module, torch.nn.Conv2d) for module in writer.modules())

    empty = torch.zeros_like(current)
    neighbor_impulse = empty.clone()
    neighbor_impulse[0, 0, 2, 1] = 1.0
    empty_center = writer(empty, ordered).field[..., 2, 2]
    impulse_center = writer(neighbor_impulse, ordered).field[..., 2, 2]
    assert not torch.allclose(empty_center, impulse_center)


def test_dynamics_losses_are_confidence_masked_and_probe_shuffled_teacher() -> None:
    head = DynamicsSupervisionHead(12, 6)
    student = torch.randn(2, 8, 12, 4, 5, requires_grad=True)
    prediction = head(student)
    target = torch.randn(2, 8, 6, 4, 5)
    weights = torch.ones(2, 8, 4, 5)
    targets = DynamicsTargets(target, weights).validate()

    losses, metrics = dynamics_supervision_losses(
        prediction,
        targets,
        log_variance=torch.zeros(2, 8, 1, 4, 5),
        temporal_contrast_margin=0.05,
    )

    assert set(losses) == {
        "dynamics_cosine",
        "dynamics_smooth_l1",
        "dynamics_temporal_contrast",
        "dynamics_uncertainty",
    }
    assert metrics["dynamics_cosine_similarity"].isfinite()
    assert metrics["dynamics_shuffled_cosine_similarity"].isfinite()
    assert (
        metrics["dynamics_cosine_similarity"]
        - metrics["dynamics_shuffled_cosine_similarity"]
    ).isfinite()
    sum(losses.values()).backward()
    assert student.grad is not None
