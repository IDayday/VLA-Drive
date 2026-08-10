from pathlib import Path

import numpy as np
import pytest

from tools.field2plan.visualize_temporal_alignment import (
    render_temporal_alignment,
)


def _transforms_from_x(x_values: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], len(x_values), axis=0)
    transforms[:, 0, 3] = x_values
    return transforms


def test_render_temporal_alignment_reports_current_frame_coordinates(
    tmp_path: Path,
) -> None:
    frame_indices = np.arange(12, dtype=np.int64)
    current_index = 3
    current_x = frame_indices.astype(np.float32) - float(current_index)
    arrays = {
        "current_frame_index": np.asarray(current_index, dtype=np.int64),
        "history_frame_indices": np.arange(4, dtype=np.int64),
        "future_frame_indices": np.arange(4, 12, dtype=np.int64),
        "frame_times_s": (frame_indices - current_index).astype(np.float32) * 0.5,
        "current_from_ego": _transforms_from_x(current_x),
        "ego_from_current": _transforms_from_x(-current_x),
        "valid_mask": np.ones(12, dtype=np.bool_),
        "teacher_frame_indices": np.arange(4, 12, dtype=np.int64),
        "teacher_frame_times_s": np.arange(1, 9, dtype=np.float32) * 0.5,
    }
    output = tmp_path / "temporal.png"

    summary = render_temporal_alignment(arrays, output)

    assert output.is_file()
    assert output.stat().st_size > 1000
    assert summary["current_origin_xy_m"] == pytest.approx([0.0, 0.0])
    assert np.asarray(summary["history_origins_xy_m"])[:, 0].tolist() == pytest.approx(
        [-3.0, -2.0, -1.0, 0.0]
    )
    assert np.asarray(summary["future_origins_xy_m"])[:, 0].tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    assert summary["teacher_alignment_max_abs_time_error_s"] == pytest.approx(0.0)


def test_render_temporal_alignment_rejects_teacher_frame_mismatch(
    tmp_path: Path,
) -> None:
    arrays = {
        "current_frame_index": np.asarray(1, dtype=np.int64),
        "history_frame_indices": np.asarray([0, 1], dtype=np.int64),
        "future_frame_indices": np.asarray([2, 3], dtype=np.int64),
        "frame_times_s": np.asarray([-0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "current_from_ego": _transforms_from_x(
            np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        ),
        "ego_from_current": _transforms_from_x(
            np.asarray([1.0, 0.0, -1.0, -2.0], dtype=np.float32)
        ),
        "valid_mask": np.ones(4, dtype=np.bool_),
        "teacher_frame_indices": np.asarray([2, 4], dtype=np.int64),
        "teacher_frame_times_s": np.asarray([0.5, 1.0], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="teacher frame indices"):
        render_temporal_alignment(arrays, tmp_path / "bad.png")
