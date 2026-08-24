import inspect
import types

import numpy as np
import torch
from torch import nn

from starVLA.gp_sq3dmix_v2 import (
    spatial_derangement_indices,
    spatial_shuffle_pooled_geometry,
)
from starVLA.model.framework.QwenOFT_GPSQ3DMix import QwenOFT_GPSQ3DMix
from tools.build_gp_sq3dmix_hard_negative_map import build_map


def _pooled(batch=2, dim=3):
    features = torch.arange(batch * 180 * dim).reshape(batch, 180, dim).float()
    return {
        "features": features,
        "view_ids": torch.arange(3).repeat_interleave(60).repeat(batch, 1),
        "uv_coords": torch.randn(batch, 180, 2),
        "ray_features": torch.randn(batch, 180, 6),
    }


def test_spatial_shuffle_is_deterministic_derangement_and_topology_independent():
    expected = spatial_derangement_indices(20260824, "token-a", 1)
    for _world_size in (1, 2, 16):
        actual = spatial_derangement_indices(20260824, "token-a", 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.any(expected == torch.arange(60))


def test_spatial_shuffle_preserves_per_view_distribution_and_metadata():
    pooled = _pooled()
    shuffled, fixed = spatial_shuffle_pooled_geometry(
        pooled, ["token-a", "token-b"], 20260824
    )
    assert fixed.item() == 0
    for key in ("view_ids", "uv_coords", "ray_features"):
        assert shuffled[key] is pooled[key]
    for batch in range(2):
        for view in range(3):
            sl = slice(view * 60, (view + 1) * 60)
            expected = pooled["features"][batch, sl].sort(dim=0).values
            actual = shuffled["features"][batch, sl].sort(dim=0).values
            torch.testing.assert_close(actual, expected)


def _synthetic_hard_map_inputs(sample_count=300, target_count=16):
    generator = torch.Generator().manual_seed(7)
    actions = torch.randn(sample_count, 8, 4, generator=generator)
    descriptors = torch.nn.functional.normalize(
        torch.randn(sample_count, 128, generator=generator), dim=-1
    )
    tokens = [f"source-{index:04d}" for index in range(sample_count)]
    source = {
        "tokens": tokens,
        "actions": actions,
        "commands": ["keep straight"] * sample_count,
        "log_ids": [f"log-{index:04d}" for index in range(sample_count)],
        "episode_ids": [f"log-{index:04d}" for index in range(sample_count)],
    }
    target_tokens = tokens[:target_count]
    target_records = [
        {
            "action": actions[index].numpy(),
            "command": "keep straight",
            "log_id": source["log_ids"][index],
        }
        for index in range(target_count)
    ]
    return target_tokens, target_records, descriptors[:target_count], source, descriptors


def test_fixed_hard_donors_obey_identity_log_command_and_batch_independence():
    values = _synthetic_hard_map_inputs()
    first, statistics = build_map(
        target_tokens=values[0],
        target_records=values[1],
        target_descriptors=values[2],
        source=values[3],
        source_descriptors=values[4],
        action_mean=np.zeros((1, 1, 4)),
        action_std=np.ones((1, 1, 4)),
    )
    second, _ = build_map(
        target_tokens=values[0],
        target_records=values[1],
        target_descriptors=values[2],
        source=values[3],
        source_descriptors=values[4],
        action_mean=np.zeros((1, 1, 4)),
        action_std=np.ones((1, 1, 4)),
    )
    assert first == second
    assert statistics["fallback_rate"] <= 0.01
    for row in first:
        assert row["target_token"] != row["donor_token"]
        assert row["same_log"] is False
        assert row["command"] == "keep straight"
        assert 9 <= row["action_neighbor_rank"] <= 128


def test_hard_shuffled_uses_complete_donor_payload_and_metadata():
    model = QwenOFT_GPSQ3DMix.__new__(QwenOFT_GPSQ3DMix)
    nn.Module.__init__(model)
    model.spatial_shuffle_seed = 20260824
    model.register_buffer("pooled_feature_slot_mean", torch.zeros(180, 3))
    real = _pooled(batch=1)
    donor = {key: value.clone() + (10 if value.is_floating_point() else 1) for key, value in real.items()}
    extension = {
        "pooled_geometry": real,
        "pooled_hard_geometry": donor,
        "tokens": ["target"],
        "examples": [
            {
                "gp_hard_negative_metadata": {
                    "action_distance": 1.0,
                    "geometry_cosine_distance": 0.7,
                    "fallback_level": 0,
                }
            }
        ],
    }
    selected, diagnostics = model._apply_intervention(extension, "hard_shuffled")
    for key in donor:
        torch.testing.assert_close(selected[key], donor[key])
    assert diagnostics["gp_sq3dmix/hard_donor_fallback_level"] == 0


def test_stage_a_v2_has_no_batch_local_nearest_shuffle():
    assert not hasattr(QwenOFT_GPSQ3DMix, "_nearest_target_shuffle")
    source = inspect.getsource(QwenOFT_GPSQ3DMix._condition_action_queries)
    assert "nearest" not in source


def test_gated_scene_shuffle_diagnostic_uses_requested_derangement():
    model = QwenOFT_GPSQ3DMix.__new__(QwenOFT_GPSQ3DMix)
    nn.Module.__init__(model)
    model.gp_mode = "gated_residual"
    observed = {}

    def fake_enhance(self, action_queries, extension, mode, *, scene_override=None):
        observed["mode"] = mode
        observed["scene"] = scene_override.clone()
        return action_queries + scene_override[..., :1], {}, {"retention": {}}

    model._enhance = types.MethodType(fake_enhance, model)
    queries = torch.zeros(2, 8, 4)
    scene = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 0.0]]])
    permutation = torch.tensor([1, 0])
    enhanced, _, _ = model.evaluate_scene_shuffled_queries(
        queries, {"scene_summary": scene}, permutation
    )
    assert observed["mode"] == "real"
    torch.testing.assert_close(observed["scene"], scene.index_select(0, permutation))
    assert torch.all(enhanced[0] == 2.0)
    assert torch.all(enhanced[1] == 1.0)
