from pathlib import Path

import numpy as np
import pytest
import torch

from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec
from tools.field2plan.visualize_fields import render_field_diagnostics


def test_render_field_and_trajectory_bundle(tmp_path: Path) -> None:
    physical = torch.zeros(8, 3)
    physical[:, 0] = torch.linspace(0.0, 28.0, 8)
    physical[:, 1] = torch.linspace(-1.0, 2.0, 8)
    action = TrajectoryCodec().encode_trajectory(physical).numpy()
    tube = np.zeros((8, 6, 3), dtype=np.float32)
    tube[..., 0] = physical[:, None, 0].numpy()
    tube[..., 1] = physical[:, None, 1].numpy()
    arrays = {
        "geometry_field": np.ones((4, 8, 8), dtype=np.float32),
        "field_valid_mask": np.eye(8, dtype=np.float32),
        "geometry_target_depth": np.full((3, 3, 8, 8), 12.0, np.float32),
        "geometry_pred_depth": np.full((3, 3, 8, 8), 11.0, np.float32),
        "draft_action": action,
        "final_action": action,
        "tube_points": tube,
    }
    output = tmp_path / "fields.png"
    panels = render_field_diagnostics(arrays, output)
    assert panels == (
        "geometry_field",
        "field_valid_mask",
        "geometry_target_depth",
        "geometry_pred_depth",
        "trajectory_readout",
    )
    assert output.is_file()
    assert output.stat().st_size > 1000


def test_render_rejects_unrecognized_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no recognized"):
        render_field_diagnostics({"loss": np.array(1.0)}, tmp_path / "bad.png")
