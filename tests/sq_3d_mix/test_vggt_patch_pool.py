import pytest
import torch

from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_per_view,
)


def _payload(view_values, grids=((2, 3), (2, 3), (2, 3)), dim=4):
    views = []
    for value, (height, width) in zip(view_values, grids):
        if torch.is_tensor(value):
            view = value.reshape(height * width, 1).expand(-1, dim).clone()
        else:
            view = torch.full((height * width, dim), float(value))
        views.append(view)
    features = torch.cat(views, dim=0)
    return {
        "features": features,
        "valid_mask": torch.ones(features.shape[0], dtype=torch.bool),
        "patch_grid_hw": torch.tensor(grids, dtype=torch.int16),
        "view_ids": torch.full((features.shape[0],), -1),
    }


def test_vggt_pool_output_shape():
    pooled = pool_dense_vggt_per_view([_payload((1, 2, 3))])

    assert pooled.shape == (1, 180, 4)


def test_vggt_pool_detaches_offline_cache_features():
    payload = _payload((1, 2, 3))
    payload["features"].requires_grad_(True)

    pooled = pool_dense_vggt_per_view([payload])

    assert not pooled.requires_grad


def test_vggt_pool_preserves_view_order():
    pooled = pool_dense_vggt_per_view([_payload((10, 20, 30))])

    torch.testing.assert_close(pooled[0, :60], torch.full((60, 4), 10.0))
    torch.testing.assert_close(pooled[0, 60:120], torch.full((60, 4), 20.0))
    torch.testing.assert_close(pooled[0, 120:], torch.full((60, 4), 30.0))


def test_vggt_pool_preserves_row_major_contract():
    front = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    left = front + 10
    right = front + 20
    pooled = pool_dense_vggt_per_view(
        [_payload((front, left, right), dim=1)],
        output_hw=(2, 3),
    )

    expected = torch.cat([front.flatten(), left.flatten(), right.flatten()])
    torch.testing.assert_close(pooled[0, :, 0], expected)


def test_vggt_pool_rejects_invalid_mask():
    payload = _payload((1, 2, 3))
    payload["valid_mask"][4] = False

    with pytest.raises(ValueError, match="invalid or padded"):
        pool_dense_vggt_per_view([payload])


def test_vggt_pool_rejects_grid_count_mismatch():
    payload = _payload((1, 2, 3))
    payload["patch_grid_hw"][0, 0] = 3

    with pytest.raises(ValueError, match="patch grids sum"):
        pool_dense_vggt_per_view([payload])
