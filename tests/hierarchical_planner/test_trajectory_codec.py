import torch

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec


def test_flow_navsim_round_trip_and_leading_dimensions():
    codec = TrajectoryCodec()
    action = torch.randn(2, 5, 8, 4)
    heading = torch.randn(2, 5, 8)
    action[..., 2] = heading.sin()
    action[..., 3] = heading.cos()
    restored = codec.navsim_to_flow(codec.flow_to_navsim(action))
    torch.testing.assert_close(restored, action, rtol=1e-5, atol=1e-6)


def test_8_to_40_round_trip_and_heading_wrap():
    codec = TrajectoryCodec()
    trajectory = torch.randn(2, 3, 8, 3)
    trajectory[..., 2] = torch.linspace(3.12, -3.12, 8)
    dense = codec.upsample_8_to_40(trajectory)
    assert dense.shape == (2, 3, 40, 3)
    torch.testing.assert_close(codec.downsample_40_to_8(dense), trajectory)
    assert torch.isfinite(dense).all()


def test_8_to_40_preserves_bfloat16_dtype_and_anchors():
    codec = TrajectoryCodec()
    trajectory = torch.randn(2, 8, 3, dtype=torch.bfloat16)

    dense = codec.upsample_8_to_40(trajectory)

    assert dense.dtype is torch.bfloat16
    torch.testing.assert_close(codec.downsample_40_to_8(dense), trajectory)
