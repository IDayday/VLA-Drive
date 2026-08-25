import pytest
import torch

from starVLA.model.modules.register_planner.losses import RegisterTrajectoryLoss


def test_final_only_wta_loss():
    gt = torch.zeros(1, 8, 3)
    first = torch.full((1, 2, 8, 3), 100.0, requires_grad=True)
    final = torch.stack((torch.ones(8, 3), torch.full((8, 3), 2.0)))[None]
    final.requires_grad_()
    output = RegisterTrajectoryLoss()([first, final], gt)
    assert output.loss.item() == pytest.approx(3.0)
    output.loss.backward()
    assert first.grad is None
    assert final.grad is not None


def test_winner_index_shape():
    output = RegisterTrajectoryLoss()([torch.randn(3, 64, 8, 3)], torch.randn(3, 8, 3))
    assert output.winner_index.shape == (3,)


def test_min_over_64_matches_manual_computation():
    proposals, gt = torch.randn(2, 64, 8, 3), torch.randn(2, 8, 3)
    expected = torch.linalg.norm(
        proposals - gt[:, None], ord=1, dim=-1
    ).mean(dim=-1).min(dim=1).values.mean()
    actual = RegisterTrajectoryLoss()([proposals], gt).loss
    torch.testing.assert_close(actual, expected)


def test_all_layers_uses_configured_weights_without_renormalizing():
    gt = torch.zeros(1, 8, 3)
    stages = [
        torch.ones(1, 2, 8, 3),
        torch.full((1, 2, 8, 3), 2.0),
    ]
    loss = RegisterTrajectoryLoss(
        stage_loss_mode="all_layers", stage_loss_weights=[0.25, 1.0]
    )(stages, gt).loss
    assert loss.item() == pytest.approx(0.25 * 3.0 + 6.0)


def test_diversity_weight_zero_by_default():
    assert RegisterTrajectoryLoss().diversity_weight == 0.0
    with pytest.raises(ValueError, match="diversity"):
        RegisterTrajectoryLoss(diversity_weight=0.1)
