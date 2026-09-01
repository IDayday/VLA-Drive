from __future__ import annotations

import pytest
import torch

from local_stage2.export_multiview_m0_observation_replay import (
    pool_m0_visual_tokens,
)


def test_m0_multiview_pooling_uses_fixed_camera_blocks() -> None:
    # Two scenes, four cameras. Counts vary so a naive concatenation would
    # shift the later camera semantics between scenes.
    counts = torch.tensor([[1, 2, 1, 1], [2, 1, 2, 1]], dtype=torch.int16)
    total = int(counts.sum())
    raw = torch.stack(
        [torch.full((256, 3), float(index + 1)) for index in range(total)]
    )
    tokens, mask = pool_m0_visual_tokens(
        raw,
        counts,
        pool_grid=(2, 2),
        max_crops_per_camera=3,
    )
    assert tokens.shape == (2, 4 * 3 * 4, 3)
    assert mask.shape == tokens.shape[:2]
    assert mask.sum(dim=1).tolist() == [20, 24]

    camera_block = 12
    # Scene 0 camera L0 owns two crops (8 tokens) in its own fixed block.
    assert mask[0, camera_block : camera_block + 8].all()
    assert not mask[0, camera_block + 8 : 2 * camera_block].any()
    # Scene 1 camera L0 still starts at exactly the same block boundary.
    assert mask[1, camera_block : camera_block + 4].all()
    assert not mask[1, camera_block + 4 : 2 * camera_block].any()


def test_m0_multiview_pooling_is_deterministic_and_finite() -> None:
    generator = torch.Generator().manual_seed(19)
    counts = torch.ones(2, 4, dtype=torch.int16)
    raw = torch.randn(8, 256, 7, generator=generator)
    first = pool_m0_visual_tokens(raw, counts, (1, 2), 2)
    second = pool_m0_visual_tokens(raw, counts, (1, 2), 2)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert torch.equal(first[1], second[1])
    assert torch.isfinite(first[0]).all()


@pytest.mark.parametrize(
    "raw,counts,max_crops,error",
    [
        (torch.zeros(1, 255, 4), torch.ones(1, 4), 2, "raw_visual_tokens"),
        (torch.zeros(4, 256, 4), torch.ones(1, 3), 2, "crop_counts"),
        (
            torch.zeros(4, 256, 4),
            torch.ones(1, 4, dtype=torch.bool),
            2,
            "not bool",
        ),
        (torch.zeros(4, 256, 4), torch.tensor([[1, 1, 1, 3]]), 2, "capacity"),
        (torch.zeros(5, 256, 4), torch.ones(1, 4), 2, "do not match"),
    ],
)
def test_m0_multiview_pooling_rejects_invalid_layout(
    raw: torch.Tensor,
    counts: torch.Tensor,
    max_crops: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        pool_m0_visual_tokens(raw, counts, (2, 2), max_crops)
