import torch
from torch import nn

from starVLA.model.modules.field2plan.visual_feature_tap import VisualFeatureTap


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, tokens: torch.Tensor, *, grid_thw: torch.Tensor):
        self.calls += 1
        return tokens + 1.0, [tokens + 2.0]

    def vit_tokens_to_featmap(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_views=2, view_order=(0, 1)
    ):
        chunks = torch.split(hidden_states, [4, 4], dim=0)
        features = torch.stack([chunk.T.reshape(3, 2, 2) for chunk in chunks])
        return torch.cat([features[None, 0], features[None, 1]], dim=-1), features[None]


def test_explicit_feature_tap_does_not_rerun_visual_encoder() -> None:
    visual = FakeVisual()
    tap = VisualFeatureTap(enabled=True, mode="explicit", num_views=2, view_order=(0, 1))
    tokens = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    output = visual(tokens, grid_thw=grid)
    tapped = tap.from_visual_output(visual, output, grid)

    assert visual.calls == 1
    assert tapped.features.shape == (1, 2, 3, 2, 2)
    assert tapped.grid_thw.shape == (2, 3)


def test_hook_fallback_is_scoped_and_clears_tensor_references() -> None:
    visual = FakeVisual()
    tap = VisualFeatureTap(enabled=True, mode="hook", num_views=2, view_order=(0, 1))
    tokens = torch.zeros(8, 3)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])

    with tap.capture(visual):
        visual(tokens, grid_thw=grid)
        tapped = tap.consume(visual)

    assert visual.calls == 1
    assert tapped.features.shape == (1, 2, 3, 2, 2)
    assert tap.has_pending_capture is False


def test_disabled_tap_registers_no_hook() -> None:
    visual = FakeVisual()
    tap = VisualFeatureTap(enabled=False, mode="hook", num_views=2)
    before = len(visual._forward_hooks)
    with tap.capture(visual):
        visual(torch.zeros(8, 3), grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]))
    assert len(visual._forward_hooks) == before
