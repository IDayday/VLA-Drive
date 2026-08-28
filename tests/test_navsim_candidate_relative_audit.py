"""Unit and deployment-backed regression tests for the NAVSIM audit tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.navsim_candidate_relative_audit.build_soft_contrastive_labels import (
    consequence_distance_matrix,
    prefix_distance_matrix,
    softmax_negative_distance,
)
from tools.navsim_candidate_relative_audit.candidate_generator import (
    fallback_candidates,
    quintic_smoothstep,
)
from tools.navsim_candidate_relative_audit.common import (
    resolve_horizon_index,
    se2_global_to_local,
    se2_local_to_global,
    wrap_heading,
)


REPORT_ROOT = (
    Path(__file__).resolve().parents[1] / "reports" / "navsim_candidate_relative_audit"
)


def curved_anchor() -> np.ndarray:
    x = np.arange(1, 9, dtype=np.float64) * 1.5
    y = 0.08 * x**2
    heading = np.arctan2(np.diff(np.r_[0.0, y]), np.diff(np.r_[0.0, x]))
    return np.stack([x, y, heading], axis=1)


def test_horizon_to_timestamp_mapping_uses_nearest_actual_time() -> None:
    timestamps = np.asarray([1_000_000, 1_498_000, 2_001_000, 3_002_000, 5_003_000])
    assert resolve_horizon_index(timestamps, 0.5) == 1
    assert resolve_horizon_index(timestamps, 1.0) == 2
    assert resolve_horizon_index(timestamps, 2.0) == 3
    assert resolve_horizon_index(timestamps, 4.0) == 4


def test_heading_wrap() -> None:
    values = wrap_heading(np.asarray([-3 * np.pi, -np.pi, 0, np.pi, 3 * np.pi]))
    assert np.all(values >= -np.pi)
    assert np.all(values < np.pi)
    assert wrap_heading(2 * np.pi + 0.2) == pytest.approx(0.2)


def test_local_global_local_round_trip() -> None:
    local = np.asarray([[0.0, 0.0, 0.0], [4.0, -2.0, 2.8], [-3.0, 5.0, -2.9]])
    origin = np.asarray([331_000.2, 4_691_000.7, 2.4])
    recovered = se2_global_to_local(se2_local_to_global(local, origin), origin)
    assert np.max(np.abs(recovered[:, :2] - local[:, :2])) < 1e-9
    assert np.max(np.abs(wrap_heading(recovered[:, 2] - local[:, 2]))) < 1e-12


def test_actor_box_coordinate_transform_preserves_dimensions() -> None:
    # A box uses a pose plus dimensions. SE(2) changes only the pose.
    box = np.asarray([6.0, -1.5, 2.9, 4.8, 2.1])
    candidate_pose = np.asarray([3.0, 2.0, -2.7])
    global_pose = se2_local_to_global(box[:3], [10.0, 20.0, 1.2])
    candidate_local = se2_global_to_local(global_pose, candidate_pose)
    recovered_global = se2_local_to_global(candidate_local, candidate_pose)
    assert np.allclose(recovered_global[:2], global_pose[:2])
    assert wrap_heading(recovered_global[2] - global_pose[2]) == pytest.approx(0.0)
    assert np.array_equal(box[3:], np.asarray([4.8, 2.1]))


def test_gt_candidate_is_unchanged_and_candidates_share_implicit_start() -> None:
    anchor = curved_anchor()
    candidates = fallback_candidates(anchor)
    assert candidates[0][0] == "gt"
    assert np.array_equal(candidates[0][2], anchor)
    for _, _, trajectory in candidates:
        first = trajectory[0]
        expected_heading = np.arctan2(first[1], first[0])
        assert wrap_heading(first[2] - expected_heading) == pytest.approx(0.0)
    manifest_path = REPORT_ROOT / "candidate_manifest.parquet"
    if manifest_path.is_file():
        manifest = pd.read_parquet(manifest_path)
        start_columns = [
            "implicit_start_x_m",
            "implicit_start_y_m",
            "implicit_start_heading_rad",
        ]
        assert np.array_equal(
            manifest[start_columns].to_numpy(),
            np.zeros((len(manifest), len(start_columns))),
        )


def test_smooth_perturbation_continuity_and_deduplication() -> None:
    u = np.linspace(0, 1, 1001)
    ramp = quintic_smoothstep(u)
    derivative = np.gradient(ramp, u)
    assert ramp[0] == pytest.approx(0.0)
    assert ramp[-1] == pytest.approx(1.0)
    assert derivative[0] < 1e-4
    assert derivative[-1] < 1e-4
    assert np.all(np.diff(ramp) >= -1e-12)
    candidates = [item[2] for item in fallback_candidates(curved_anchor())]
    assert len({np.round(item, 5).tobytes() for item in candidates}) == len(candidates)


def test_target_mask_and_hash_sentinel_if_deployment_artifact_exists() -> None:
    targets = sorted((REPORT_ROOT / "targets").glob("*.npz"))
    if not targets:
        pytest.skip("deployment target artifact not generated")
    with np.load(targets[0]) as payload:
        mask = np.asarray(payload["candidate_relative_actor_mask"], dtype=bool)
        hashes = np.asarray(payload["candidate_relative_actor_track_hash"])
        tensor = np.asarray(payload["candidate_relative_actor_tensor"])
    assert mask.shape == hashes.shape == tensor.shape[:-1]
    assert np.all(hashes[~mask] == 0)
    assert np.isfinite(tensor[mask]).all()


def test_target_schema_has_field_level_provenance_if_generated() -> None:
    path = REPORT_ROOT / "target_schema.json"
    if not path.is_file():
        pytest.skip("deployment target schema not generated")
    schema = json.loads(path.read_text())
    required = {
        "name",
        "unit",
        "coordinate_frame",
        "target_horizons_s",
        "source_frequency_hz",
        "valid_mask",
        "candidate_dependent",
        "logged_future_dependent",
        "reactive_only",
        "inference_availability",
    }
    for array in schema["arrays"].values():
        metadata = array["field_metadata"]
        assert array["fields"] == [field["name"] for field in metadata]
        assert all(required <= field.keys() for field in metadata)


def test_consequence_pairwise_distance_is_symmetric() -> None:
    environment = np.asarray([[0.0, 1.0], [1.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    actor = np.zeros((3, 2, 10), dtype=np.float32)
    actor[:, 0, 1] = [0.0, 1.0, 2.0]
    mask = np.asarray([[True, False], [True, False], [True, False]])
    hashes = np.asarray([[1, 0], [1, 0], [1, 0]], dtype=np.uint64)
    distance = consequence_distance_matrix(environment, actor, mask, hashes, np.ones(2))
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance[np.triu_indices(3, 1)] > 0)


def test_soft_label_rows_sum_to_one() -> None:
    distance = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5], [2.0, 0.5, 0.0]])
    probabilities = softmax_negative_distance(distance, sigma=0.7, axis=1)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_same_prefix_different_tail_behavior() -> None:
    base = np.zeros((8, 3), dtype=np.float64)
    base[:, 0] = np.arange(1, 9)
    diverged = base.copy()
    diverged[2:, 1] = np.linspace(0.0, 4.0, 6)
    diverged[2:, 2] = np.arctan2(np.gradient(diverged[2:, 1]), 1.0)
    values = np.stack([base, diverged])
    scales = {"s_p_m": 1.0, "s_heading_rad": 0.2, "s_v_mps": 1.0}
    short = prefix_distance_matrix(values, 1.0, scales)
    long = prefix_distance_matrix(values, 4.0, scales)
    assert short[0, 1] == pytest.approx(0.0)
    assert long[0, 1] > 0.5
    short_q = softmax_negative_distance(short, sigma=1.0, axis=1)
    long_q = softmax_negative_distance(long, sigma=1.0, axis=1)
    assert short_q[0, 1] > long_q[0, 1]


def test_batch_and_single_official_scoring_consistency_if_audited() -> None:
    path = REPORT_ROOT / "candidate_scoring_summary.json"
    if not path.is_file():
        pytest.skip("official scoring audit artifact not generated")
    result = json.loads(path.read_text())
    assert result["criteria"]["candidate_order_preserved"]
    assert result["criteria"]["state_horizon_alignment"]
    assert result["repeat_verification"]["deterministic"]
    assert result["repeat_verification"]["repeat_metric_max_abs_error"] < 1e-12


def test_fixed_seed_candidate_result_is_deterministic() -> None:
    first = fallback_candidates(curved_anchor())
    second = fallback_candidates(curved_anchor())
    assert [item[0] for item in first] == [item[0] for item in second]
    for left, right in zip(first, second):
        assert np.array_equal(left[2], right[2])
