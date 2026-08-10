import pytest
import torch

from starVLA.model.modules.grounded_world.supervision import (
    FeatureAlignmentTarget,
    FutureTargetContract,
    alignment_losses,
    future_prediction_losses,
    global_alignment_losses,
)
from starVLA.model.modules.grounded_world.prior_adapter import ExternalPriorAdapter


def test_external_prior_and_future_target_contracts_cannot_be_conflated() -> None:
    contract = FutureTargetContract(
        source="student_ema",
        target_id="stage1-world-ema-v1",
        shared_across_teacher_controls=True,
    ).validate()
    assert contract.source == "student_ema"

    for forbidden in ("vjepa2_1", "drive_jepa", "external_teacher"):
        with pytest.raises(ValueError, match="student_ema"):
            FutureTargetContract(
                source=forbidden,
                target_id="bad",
                shared_across_teacher_controls=True,
            ).validate()
    with pytest.raises(ValueError, match="shared"):
        FutureTargetContract(
            source="student_ema",
            target_id="bad",
            shared_across_teacher_controls=False,
        ).validate()


def test_prior_and_future_losses_expose_noninvasive_probe_metrics() -> None:
    current_prediction = torch.randn(3, 8, 6, 6, requires_grad=True)
    current_target = FeatureAlignmentTarget(
        features=torch.randn(3, 8, 6, 6),
        weights=torch.ones(3, 6, 6),
        target_id="driving-jepa-history-v1",
    ).validate()
    prior_losses, prior_metrics = alignment_losses(
        current_prediction,
        current_target,
        prefix="prior",
    )
    assert set(prior_losses) == {"prior_cosine", "prior_smooth_l1"}
    assert "prior_scene_shuffle_margin" in prior_metrics

    future_prediction = torch.randn(3, 4, 8, 6, 6, requires_grad=True)
    future_target = FeatureAlignmentTarget(
        features=torch.randn(3, 4, 8, 6, 6),
        weights=torch.ones(3, 4, 6, 6),
        target_id="stage1-world-ema-v1",
    ).validate()
    future_losses, future_metrics = future_prediction_losses(
        future_prediction,
        future_target,
        contract=FutureTargetContract(
            source="student_ema",
            target_id="stage1-world-ema-v1",
            shared_across_teacher_controls=True,
        ),
    )
    assert set(future_losses) == {
        "future_cosine",
        "future_smooth_l1",
        "future_temporal_contrast",
    }
    assert "future_temporal_margin" in future_metrics
    total = sum(prior_losses.values()) + sum(future_losses.values())
    total.backward()
    assert current_prediction.grad is not None
    assert future_prediction.grad is not None


def test_external_prior_is_global_when_teacher_has_no_bev_layout() -> None:
    adapter = ExternalPriorAdapter(teacher_channels=6, output_channels=8)
    features = torch.randn(2, 4, 3, 6, 5, 7)
    confidence = torch.ones(2, 4, 3, 5, 7)
    target, weights = adapter(features, confidence)
    assert target.shape == (2, 8)
    assert weights.shape == (2,)

    prediction = torch.randn(2, 8, requires_grad=True)
    losses, metrics = global_alignment_losses(
        prediction,
        target=target,
        weights=weights,
        target_id="driving-jepa-current-history",
        prefix="prior",
    )
    sum(losses.values()).backward()
    assert prediction.grad is not None
    assert "prior_scene_shuffle_margin" in metrics
