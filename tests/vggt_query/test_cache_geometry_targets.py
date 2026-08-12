import torch
from PIL import Image

from tools.precompute_vggt_query_cache import build_physical_geometry_targets


def test_dense_vggt_heads_build_relative_geometry_targets(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"view_{index}.jpg"
        Image.new("RGB", (160, 90)).save(path)
        paths.append(path)
    depth = torch.full((1, 3, 518, 518, 1), 2.0)
    points = torch.zeros(1, 3, 518, 518, 3)
    points[..., 0] = 1.0
    points[..., 2] = 2.0
    confidence = torch.ones(1, 3, 518, 518)

    target, weight, valid = build_physical_geometry_targets(
        depth,
        confidence,
        points,
        confidence,
        path_batches=[paths],
        output_size=(2, 3),
    )

    assert target.shape == (1, 18, 3)
    assert weight.shape == valid.shape == (1, 18)
    assert valid.all()
    torch.testing.assert_close(target[..., 0], torch.full((1, 18), 0.5))
    torch.testing.assert_close(target[..., 1:], torch.zeros(1, 18, 2))
    torch.testing.assert_close(weight, torch.ones_like(weight))
