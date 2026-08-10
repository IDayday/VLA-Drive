import torch

from starVLA.model.modules.grounded_world.consequence import (
    CONSEQUENCE_NAMES,
    ConsequenceTargets,
    PlanningConsequenceHead,
    consequence_losses,
)


def test_training_only_consequence_head_predicts_physical_components() -> None:
    head = PlanningConsequenceHead(context_dim=32, hidden_dim=48)
    context = torch.randn(2, 5, 8, 32, requires_grad=True)
    valid = torch.ones(2, 5, 8, dtype=torch.bool)
    prediction = head(context, valid).validate()
    assert prediction.values.shape == (2, 5, len(CONSEQUENCE_NAMES))
    assert prediction.log_variance.shape == prediction.values.shape

    targets = ConsequenceTargets(
        values=torch.rand_like(prediction.values),
        valid_mask=torch.ones_like(prediction.values, dtype=torch.bool),
    ).validate()
    scales = torch.tensor([10.0, 4.0, 1.0, 5.0, 20.0, 1.0])
    losses, metrics = consequence_losses(prediction, targets, scales=scales)
    assert set(losses) == {f"consequence_{name}" for name in CONSEQUENCE_NAMES}
    assert "consequence_valid_ratio" in metrics
    for name in CONSEQUENCE_NAMES:
        assert f"consequence_{name}_valid_ratio" in metrics
        assert f"consequence_{name}_mae" in metrics
        assert f"consequence_{name}_prediction_std" in metrics
        assert f"consequence_{name}_target_std" in metrics
    assert "consequence_collision_accuracy" in metrics
    sum(losses.values()).backward()
    assert context.grad is not None


def test_consequence_scales_normalize_regression_without_changing_units() -> None:
    values = torch.zeros(1, 2, 6, requires_grad=True)
    prediction = type("Prediction", (), {})()
    prediction.values = values
    prediction.log_variance = torch.zeros_like(values)
    prediction.validate = lambda: prediction
    targets = ConsequenceTargets(
        values=torch.tensor(
            [[[10.0, 4.0, 0.0, 5.0, 20.0, 1.0]] * 2]
        ),
        valid_mask=torch.ones(1, 2, 6, dtype=torch.bool),
    )
    losses, metrics = consequence_losses(
        prediction,
        targets,
        scales=torch.tensor([10.0, 4.0, 1.0, 5.0, 20.0, 1.0]),
    )
    for name in ("clearance", "ttc", "lane_distance", "progress", "comfort"):
        assert torch.allclose(losses[f"consequence_{name}"], torch.tensor(0.5))
    assert torch.allclose(metrics["consequence_progress_mae"], torch.tensor(20.0))


def test_consequence_head_has_no_candidate_selection_api() -> None:
    head = PlanningConsequenceHead(context_dim=8, hidden_dim=16)
    assert not hasattr(head, "select")
    assert not hasattr(head, "rank")
    assert not hasattr(head, "predict_epdms")
