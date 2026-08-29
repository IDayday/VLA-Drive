from __future__ import annotations

import torch
import pytest

from research.cf_effect_gate_wote.src.models.structured_six_factor_probe import (
    StructuredSixFactorProbe,
    pdms_from_six_factors,
    six_factor_probe_loss,
)


def test_structured_probe_outputs_b_k_six() -> None:
    model = StructuredSixFactorProbe().eval()
    with torch.inference_mode():
        output = model(
            torch.zeros(2, 3, 8, 3),
            torch.zeros(2, 8),
            torch.zeros(2, 64, 256),
            torch.zeros(2, 3, 32, 64),
        )
    assert output["logits"].shape == (2, 3, 6)
    assert output["factors"].shape == (2, 3, 6)
    assert output["score"].shape == (2, 3)


def test_score_includes_ddc_and_rejects_five_factors() -> None:
    factors = torch.ones(1, 1, 6)
    factors[..., 2] = 0.5
    torch.testing.assert_close(pdms_from_six_factors(factors), torch.tensor([[0.5]]))
    with pytest.raises(ValueError, match="dimension 6"):
        pdms_from_six_factors(torch.ones(1, 1, 5))


def test_ddc_half_is_a_valid_soft_bce_target() -> None:
    logits = torch.zeros(1, 4, 6, requires_grad=True)
    labels = torch.ones(1, 4, 6)
    labels[..., 2] = 0.5
    scores = pdms_from_six_factors(labels)
    prediction = pdms_from_six_factors(logits.sigmoid())
    pairs = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]])
    loss = six_factor_probe_loss(logits, labels, prediction, scores, pairs, 0.5)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

