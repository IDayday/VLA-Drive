from __future__ import annotations

import torch

from research.action_effect.losses import ConsequencePredictionLoss
from research.action_effect.world_probe import ActionEffectWorldProbe, count_parameters


def _model(mode: str = "scene_action") -> ActionEffectWorldProbe:
    return ActionEffectWorldProbe(
        scene_input_dim=12,
        consequence_dim=9,
        latent_dim=16,
        trajectory_input_dim=4,
        trajectory_token_dim=8,
        action_hidden_dim=10,
        dropout=0.0,
        input_mode=mode,
    )


def test_world_probe_contract_and_shapes() -> None:
    model = _model()
    output = model(torch.randn(3, 8, 12), torch.randn(3, 8, 4), torch.randn(3, 8, 10))
    assert output["effect_latent"].shape == (3, 16)
    assert output["consequence_prediction"].shape == (3, 9)
    assert output["structured_future_prediction"] is None


def test_same_parameter_no_action_really_ignores_action() -> None:
    model = _model("zero_action").eval()
    scene = torch.randn(2, 8, 12)
    left = model(scene, torch.randn(2, 8, 4))["consequence_prediction"]
    right = model(scene, torch.randn(2, 8, 4))["consequence_prediction"]
    torch.testing.assert_close(left, right)
    assert count_parameters(model) == count_parameters(_model())


def test_consequence_loss_splits_hard_and_soft() -> None:
    loss = ConsequencePredictionLoss(hard_dim=3)
    output = loss(torch.zeros(4, 5), torch.zeros(4, 5), soft_mask=torch.tensor([True, False]))
    assert set(output) == {"total", "hard", "soft"}
    assert output["total"].ndim == 0


def test_structured_future_decoder_shape() -> None:
    model = ActionEffectWorldProbe(
        scene_input_dim=12,
        consequence_dim=9,
        latent_dim=16,
        trajectory_input_dim=4,
        trajectory_token_dim=8,
        dropout=0.0,
        structured_future_shape=(3, 7, 16, 16),
    )
    output = model(torch.randn(2, 8, 12), torch.randn(2, 8, 4))
    assert output["structured_future_prediction"].shape == (2, 3, 7, 16, 16)
