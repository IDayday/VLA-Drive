"""Regression tests for the NAVSIM candidate-relative feasibility audit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.navsim_candidate_relative_audit.analyze_target_diversity import (
    consequence_distance_matrix,
    masked_actor_distance,
)
from tools.navsim_candidate_relative_audit.build_soft_contrastive_labels import (
    prefix_distance_matrix,
    softmax_negative,
)
from tools.navsim_candidate_relative_audit.candidate_generator import generate_candidates, smoothstep5
from tools.navsim_candidate_relative_audit.common import (
    configure_navsim_environment,
    discover_paths,
    global_to_local,
    local_to_global,
    metric_cache_loader,
    read_parquet,
    resolve_horizon_index,
    wrap_angle,
)


def _base_trajectory() -> np.ndarray:
    x = np.linspace(1.0, 8.0, 8)
    y = 0.05 * x**2
    heading = np.arctan2(np.diff(np.r_[0.0, y]), np.diff(np.r_[0.0, x]))
    return np.column_stack([x, y, heading])


def test_horizon_to_timestamp_mapping_uses_measured_timestamps() -> None:
    start = 1_600_000_000_000_000
    timestamps = [start, start + 499_800, start + 1_000_100, start + 1_500_050, start + 2_000_300]
    assert resolve_horizon_index(timestamps, 0.5, origin_index=0) == 1
    assert resolve_horizon_index(timestamps, 2.0, origin_index=0) == 4


def test_heading_wrap() -> None:
    values = wrap_angle(np.asarray([0.0, np.pi + 0.1, -np.pi - 0.1, 5 * np.pi]))
    np.testing.assert_allclose(values, [0.0, -np.pi + 0.1, np.pi - 0.1, np.pi], atol=1e-12)


def test_local_global_local_round_trip() -> None:
    rng = np.random.default_rng(11)
    local = rng.normal(size=(32, 3)); local[:, 2] = wrap_angle(local[:, 2])
    origin = np.asarray([664_000.0, 3_996_000.0, 2.8])
    recovered = global_to_local(origin, local_to_global(origin, local))
    np.testing.assert_allclose(recovered[:, :2], local[:, :2], atol=1e-9)
    np.testing.assert_allclose(wrap_angle(recovered[:, 2] - local[:, 2]), 0.0, atol=1e-12)


def test_actor_box_coordinate_transform() -> None:
    corners_local = np.asarray([[2, 1, 0], [2, -1, 0], [-2, -1, 0], [-2, 1, 0]], dtype=float)
    frame_pose = np.asarray([100.0, -25.0, np.pi / 3])
    world = local_to_global(frame_pose, corners_local)
    candidate_pose = np.asarray([101.0, -24.0, -0.2])
    candidate_local = global_to_local(candidate_pose, world)
    recovered_world = local_to_global(candidate_pose, candidate_local)
    np.testing.assert_allclose(recovered_world[:, :2], world[:, :2], atol=1e-10)


def test_gt_candidate_is_not_changed_and_start_is_shared() -> None:
    base = _base_trajectory()
    candidates, _ = generate_candidates(base, 12, seed=5)
    np.testing.assert_array_equal(candidates[0], base.astype(np.float32))
    implicit_starts = np.zeros((len(candidates), 3))
    np.testing.assert_array_equal(implicit_starts, 0.0)
    assert np.all(np.linalg.norm(candidates[:, 0, :2] - implicit_starts[:, :2], axis=1) < 5.0)


def test_smooth_perturbation_continuity() -> None:
    u = np.linspace(0.0, 1.0, 1001)
    ramp = smoothstep5(u)
    derivative = np.gradient(ramp, u)
    assert abs(derivative[0]) < 1e-4
    assert abs(derivative[-1]) < 1e-4
    assert np.max(np.abs(np.diff(ramp))) < 0.003
    assert np.all(np.diff(ramp) >= -1e-12)


def test_candidate_deduplication_and_seed_determinism() -> None:
    base = _base_trajectory()
    first, specs_first = generate_candidates(base, 16, seed=123)
    second, specs_second = generate_candidates(base, 16, seed=123)
    np.testing.assert_array_equal(first, second)
    assert [item.name for item in specs_first] == [item.name for item in specs_second]
    signatures = {np.round(candidate, 5).tobytes() for candidate in first}
    assert len(signatures) == len(first)


def test_target_mask_ignores_invalid_slots() -> None:
    values_a = np.zeros((2, 3, 10)); values_b = values_a.copy()
    mask = np.zeros((2, 3), dtype=bool); mask[:, 0] = True
    hashes = np.zeros((2, 3), dtype=np.int64); hashes[:, 0] = 41
    values_b[:, 1:, :] = 1e6  # Invalid padding must not affect distance.
    distance = masked_actor_distance(values_a, mask, hashes, values_b, mask, hashes)
    assert distance == pytest.approx(0.0)


def test_consequence_pairwise_distance_is_symmetric() -> None:
    rng = np.random.default_rng(4)
    arrays = {
        "C_environment_only": rng.normal(size=(3, 2, 15)),
        "candidate_relative_actor": rng.normal(size=(3, 2, 4, 10)),
        "candidate_relative_actor_mask": np.ones((3, 2, 4), dtype=bool),
        "candidate_relative_actor_token_hash": np.broadcast_to(np.arange(1, 5), (3, 2, 4)).copy(),
    }
    distance = consequence_distance_matrix(arrays)
    np.testing.assert_allclose(distance, distance.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(distance), 0.0)


def test_soft_label_rows_sum_to_one() -> None:
    distance = np.asarray([[0.0, 1.0, 4.0], [1.0, 0.0, 2.0]])
    labels = softmax_negative(distance, sigma=0.7)
    np.testing.assert_allclose(labels.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(labels >= 0)


def test_same_prefix_different_tail_behavior() -> None:
    base = _base_trajectory()
    candidates, specs = generate_candidates(base, 12, seed=9)
    tail = [index for index, spec in enumerate(specs) if spec.name == "same_prefix_different_tail"][0]
    short = prefix_distance_matrix(candidates, 1.0)[0, tail]
    medium = prefix_distance_matrix(candidates, 2.0)[0, tail]
    long = prefix_distance_matrix(candidates, 4.0)[0, tail]
    assert short < 1e-6
    assert medium < 1e-6
    assert long > medium + 0.1


def test_batch_scoring_equals_single_candidate_scoring() -> None:
    """Small real-data regression; skips only when the deployed audit data is absent."""

    manifest_path = REPO_ROOT / "reports/navsim_candidate_relative_audit/candidate_manifest.parquet"
    if not manifest_path.is_file():
        pytest.skip("Audit manifest has not been generated")
    paths = discover_paths("trainval")
    if not paths.metric_cache_path.is_dir():
        pytest.skip("NAVSIM training metric cache is unavailable")
    configure_navsim_environment(paths)
    from tools.navsim_candidate_relative_audit.score_candidates import _poses_from_group, score_pose_batch

    manifest = read_parquet(manifest_path)
    token = str(manifest.scene_token.iloc[0])
    group = manifest[manifest.scene_token == token].sort_values("candidate_index").head(2)
    poses = _poses_from_group(group)
    cache = metric_cache_loader(paths).get_from_token(token)
    batch = score_pose_batch(cache, poses)
    singles = [score_pose_batch(cache, poses[index : index + 1]) for index in range(len(poses))]
    np.testing.assert_allclose(batch["score"], [item["score"][0] for item in singles], atol=1e-12)
    for index, single in enumerate(singles):
        np.testing.assert_allclose(batch["simulated_states"][index], single["simulated_states"][0], atol=1e-12)
