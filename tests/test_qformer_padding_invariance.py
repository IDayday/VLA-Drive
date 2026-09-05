import pytest
import torch
from navsim.agents.EpisodeDrive.layers.q_former.q_former import VisionOnlyQFormer


def test_all_valid_legacy_parity_padding_values_and_mask_polarity():
    torch.manual_seed(8)
    model = VisionOnlyQFormer(vision_dim=32, hidden_dim=16, num_heads=4).eval()
    q = torch.randn(1, 16, 16) * 0.02
    tokens = torch.randn(2, 11, 32)
    expected = model(q, tokens)
    assert torch.equal(expected, model(q, tokens, torch.ones(2, 11, dtype=torch.bool)))
    padded = torch.cat([torch.randn(2, 7, 32) * 100, tokens], dim=1)
    valid = torch.cat([torch.zeros(2, 7), torch.ones(2, 11)], dim=1).bool()
    actual = model(q, padded, valid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    padded[:, :7] *= -500
    torch.testing.assert_close(model(q, padded, valid), actual, atol=0, rtol=0)
    with pytest.raises(ValueError, match='at least one'):
        model(q, tokens, torch.zeros(2, 11))
