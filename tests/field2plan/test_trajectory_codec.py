import numpy as np
import torch

from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec


def test_round_trip_supports_batch_and_candidate_dims() -> None:
    codec = TrajectoryCodec()
    trajectory = torch.randn(2, 3, 8, 3, dtype=torch.float64)
    trajectory[..., 0] = trajectory[..., 0].abs() * 12.0
    trajectory[..., 2] = trajectory[..., 2] * 8.0

    decoded = codec.decode_action(codec.encode_trajectory(trajectory))

    torch.testing.assert_close(decoded[..., :2], trajectory[..., :2])
    torch.testing.assert_close(
        torch.sin(decoded[..., 2]), torch.sin(trajectory[..., 2])
    )
    torch.testing.assert_close(
        torch.cos(decoded[..., 2]), torch.cos(trajectory[..., 2])
    )


def test_numpy_encoding_matches_legacy_dataloader_formula() -> None:
    codec = TrajectoryCodec()
    physical = np.tile(
        np.array([[[0.0, -1.0, -3.0], [10.0, 2.0, 3.4]]], dtype=np.float32),
        (1, 4, 1),
    )
    encoded = codec.encode_trajectory(physical)
    expected = np.stack(
        [
            (physical[..., 0] - 10.172484) / 8.805105,
            (physical[..., 1] - 0.360762) / 2.277741,
            np.sin((physical[..., 2] + np.pi) % (2 * np.pi) - np.pi),
            np.cos((physical[..., 2] + np.pi) % (2 * np.pi) - np.pi),
        ],
        axis=-1,
    ).astype(np.float32)
    np.testing.assert_allclose(encoded, expected, rtol=0.0, atol=1e-7)


def test_heading_wrap_normalization_and_zero_delta_are_stable() -> None:
    codec = TrajectoryCodec()
    action = torch.tensor(
        [[[0.1, -0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    ).repeat(1, 4, 1)
    pair = codec.normalize_heading_pair(action[..., 2:])
    torch.testing.assert_close(torch.linalg.vector_norm(pair[0, 0]), torch.tensor(1.0))
    torch.testing.assert_close(pair[0, 1], torch.tensor([0.0, 1.0]))

    zero_delta = torch.zeros(*action.shape[:-1], 3)
    composed = codec.compose_delta(action, zero_delta)
    assert torch.equal(composed, action)


def test_tube_points_shape_and_orientation() -> None:
    codec = TrajectoryCodec()
    trajectory = torch.tensor([[[[2.0, 3.0, 0.0]]]])  # [B,M,H,3]
    points = codec.tube_points(
        trajectory,
        lateral_offsets_m=(-1.0, 1.0),
        longitudinal_offsets_m=(0.0, 2.0),
    )
    assert points.shape == (1, 1, 1, 4, 2)
    expected = torch.tensor([[2.0, 2.0], [4.0, 2.0], [2.0, 4.0], [4.0, 4.0]])
    torch.testing.assert_close(points[0, 0, 0], expected)
