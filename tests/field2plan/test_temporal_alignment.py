import numpy as np
import pytest
import torch

from starVLA.model.modules.field2plan.temporal_alignment import (
    build_temporal_alignment,
    interpolate_temporal_features,
    se2_poses_to_transforms,
)


def test_se2_transforms_and_current_frame_round_trip() -> None:
    poses = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 1.0, torch.pi / 2.0],
        ],
        dtype=torch.float32,
    )
    global_from_ego = se2_poses_to_transforms(poses)
    alignment = build_temporal_alignment(
        poses,
        current_index=3,
        history_indices=(0, 1, 2, 3),
        future_indices=(4,),
        frame_interval_s=0.5,
    ).validate()

    assert global_from_ego.shape == (5, 4, 4)
    assert alignment.current_from_ego.shape == (5, 4, 4)
    torch.testing.assert_close(
        alignment.current_from_ego @ alignment.ego_from_current,
        torch.eye(4).expand(5, -1, -1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        alignment.current_from_ego[3], torch.eye(4), atol=1e-6, rtol=0
    )
    assert alignment.history_indices == (0, 1, 2, 3)
    assert alignment.future_indices == (4,)
    torch.testing.assert_close(alignment.frame_times_s, torch.arange(5) * 0.5)


def test_temporal_alignment_supports_batch_and_rejects_future_as_history() -> None:
    poses = torch.zeros(2, 12, 3)
    poses[:, :, 0] = torch.arange(12)
    alignment = build_temporal_alignment(
        poses,
        current_index=3,
        history_indices=(0, 1, 2, 3),
        future_indices=tuple(range(4, 12)),
        frame_interval_s=0.5,
    ).validate()
    assert alignment.global_from_ego.shape == (2, 12, 4, 4)
    assert alignment.valid_mask.shape == (2, 12)

    with pytest.raises(ValueError, match="history indices"):
        build_temporal_alignment(
            poses,
            current_index=3,
            history_indices=(0, 1, 4),
            future_indices=tuple(range(4, 12)),
            frame_interval_s=0.5,
        )


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_temporal_feature_interpolation_has_expected_values(mode: str) -> None:
    features = torch.tensor([0.0, 2.0, 4.0]).reshape(1, 3, 1, 1, 1)
    source_times = torch.tensor([0.0, 1.0, 2.0])
    query_times = torch.tensor([0.5, 1.5])

    output = interpolate_temporal_features(
        features, source_times, query_times, mode=mode
    )

    assert output.shape == (1, 2, 1, 1, 1)
    expected = [0.0, 2.0] if mode == "nearest" else [1.0, 3.0]
    np.testing.assert_allclose(output.flatten().numpy(), expected)

