"""Fast regression tests for the Gate C data and oracle controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared_future_candidate_consequence.build_controlled_candidates import (
    generate_randomized_candidates,
)
from tools.shared_future_candidate_consequence.build_balanced_split import balanced_select
from tools.shared_future_candidate_consequence.build_gate_c_targets import target_schema_v3
from tools.shared_future_candidate_consequence.build_model_candidate_bank import (
    select_diverse_candidates,
    trajectory_distance,
)
from tools.shared_future_candidate_consequence.aggregate_all_log_pipeline import (
    _candidate_audit,
    _metric_audit,
)
from tools.shared_future_candidate_consequence.common import (
    assert_feature_names_safe,
    assert_inference_batch_safe,
    validate_training_split,
)
from tools.shared_future_candidate_consequence.export_episode_drive_candidates import (
    _feature_payload,
)
from tools.shared_future_candidate_consequence.run_oracle_decomposition import (
    OracleDataset,
    _calibration_curve,
    _completed_prefix_length,
    _risk_from_actor,
    feature_group,
)
from tools.shared_future_candidate_consequence.run_all_log_pipeline import (
    _cache_covers_selection,
)


def _trajectory() -> np.ndarray:
    x = np.linspace(1.0, 10.0, 8)
    y = 0.15 * np.sin(np.linspace(0.0, np.pi, 8))
    heading = np.arctan2(np.diff(np.r_[0.0, y]), np.diff(np.r_[0.0, x]))
    return np.column_stack([x, y, heading])


def _oracle_fixture() -> OracleDataset:
    rng = np.random.default_rng(19)
    scenes, candidates = 4, 5
    dynamic = rng.normal(size=(scenes, candidates, 1448)).astype(np.float32)
    return OracleDataset(
        scene_tokens=np.asarray([f"scene_{index}" for index in range(scenes)]),
        log_names=np.asarray(["log_a", "log_b", "log_c", "log_d"]),
        folds=np.arange(scenes),
        candidate_indices=np.broadcast_to(np.arange(candidates), (scenes, candidates)).copy(),
        candidate_families=np.full((scenes, candidates), "speed_change"),
        trajectory=rng.normal(size=(scenes, candidates, 24)).astype(np.float32),
        current=rng.normal(size=(scenes, 6)).astype(np.float32),
        k_exact=rng.normal(size=(scenes, candidates, 48)).astype(np.float32),
        static=rng.normal(size=(scenes, candidates, 48)).astype(np.float32),
        d_state=dynamic,
        d_risk=rng.normal(size=(scenes, candidates, 40)).astype(np.float32),
        d_signal=rng.normal(size=(scenes, candidates, 24)).astype(np.float32),
        recomputed_risk=rng.normal(size=(scenes, candidates, 40)).astype(np.float32),
        score=rng.random(size=(scenes, candidates)).astype(np.float32),
        factors=rng.random(size=(scenes, candidates, 6)).astype(np.float32),
        family_names=("speed_change",),
    )


def test_target_schema_separates_static_dynamic_and_risk() -> None:
    schema = target_schema_v3(16)
    groups = schema["groups"]
    assert set(groups) == {
        "K_exact", "S_static", "D_state", "D_risk", "D_signal",
        "shared_actor_future", "current_actor_state",
    }
    assert groups["K_exact"]["candidate_direct"] is True
    assert groups["K_exact"]["depends_on_logged_future"] is False
    assert groups["K_exact"]["inference_available"] is True
    assert groups["S_static"]["depends_on_static_map"] is True
    assert groups["S_static"]["inference_available"] is False
    assert groups["D_state"]["depends_on_logged_future"] is True
    assert groups["D_state"]["official_metric_proxy"] is False
    assert groups["D_risk"]["official_metric_proxy"] is True
    assert groups["shared_actor_future"]["coordinate_frame"] == "current ego frame"
    assert groups["current_actor_state"]["depends_on_logged_future"] is False
    assert groups["current_actor_state"]["inference_available"] is False


def test_official_scores_and_future_fields_are_rejected_as_inputs() -> None:
    for key in ("official_score", "aggregate_score", "future_image", "future_annotations"):
        with pytest.raises(AssertionError):
            assert_inference_batch_safe({key: np.zeros(1)})
    with pytest.raises(AssertionError):
        assert_feature_names_safe(["trajectory", "official_collision"])
    assert_inference_batch_safe(
        {"current_images": np.zeros(1), "ego_status": np.zeros(1), "proposals": np.zeros(1)}
    )


def test_only_legal_training_splits_are_accepted() -> None:
    assert validate_training_split("train") == "train"
    assert validate_training_split("trainval") == "trainval"
    for split in ("navtest", "navhard", "test", "private_test"):
        with pytest.raises(ValueError):
            validate_training_split(split)


def test_randomized_candidates_are_deterministic_unique_and_index_randomized() -> None:
    base = _trajectory()
    first, first_specs = generate_randomized_candidates(base, 16, 11)
    second, second_specs = generate_randomized_candidates(base, 16, 11)
    np.testing.assert_array_equal(first, second)
    assert first_specs == second_specs
    assert len({np.round(value, 5).tobytes() for value in first}) == 16
    gt_index = [index for index, spec in enumerate(first_specs) if spec.family == "gt"]
    assert len(gt_index) == 1
    np.testing.assert_array_equal(first[gt_index[0]], base.astype(np.float32))
    observed_gt_indices = {
        next(index for index, spec in enumerate(generate_randomized_candidates(base, 16, seed)[1]) if spec.family == "gt")
        for seed in range(8)
    }
    assert len(observed_gt_indices) > 1


def test_episode_drive_current_feature_payload_contains_no_future_inputs() -> None:
    torch = pytest.importorskip("torch")
    cached = {
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.zeros(4),
        "status_feature": torch.zeros(8),
        "image_path_tensor": torch.zeros(32, dtype=torch.int64),
    }
    payload = _feature_payload(cached)
    assert set(payload) == set(cached)
    assert_inference_batch_safe(payload)


def test_log_level_folds_have_no_train_validation_overlap() -> None:
    fold_dir = Path("reports/shared_future_candidate_consequence_gate_c/folds")
    if not fold_dir.is_dir():
        pytest.skip("Balanced split has not been generated")
    import json

    fold_paths = sorted(fold_dir.glob("fold_*.json"))
    assert len(fold_paths) == 5
    validation_sets = []
    for path in fold_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        training = set(payload["train_logs"])
        validation = set(payload["validation_logs"])
        assert not training & validation
        assert not payload["log_overlap"]
        validation_sets.append(validation)
    assert not any(
        validation_sets[left] & validation_sets[right]
        for left in range(5)
        for right in range(left + 1, 5)
    )


def test_full_entrypoint_preserves_complete_scene_inventory() -> None:
    script = (
        REPO_ROOT / "tools/shared_future_candidate_consequence/run_gate_c.sh"
    ).read_text(encoding="utf-8")
    assert 'cp "${OUTPUT_DIR}/balanced_scene_manifest.parquet"' not in script
    builder = (
        REPO_ROOT / "tools/shared_future_candidate_consequence/build_balanced_split.py"
    ).read_text(encoding="utf-8")
    assert 'report_dir / "all_scene_inventory.parquet"' in builder
    report = REPO_ROOT / "reports/shared_future_candidate_consequence_gate_c"
    inventory_path = report / "all_scene_inventory.parquet"
    manifest_path = report / "balanced_scene_manifest.parquet"
    if inventory_path.is_file() and manifest_path.is_file():
        inventory = pd.read_parquet(inventory_path, columns=["scene_token", "log_name"])
        manifest = pd.read_parquet(manifest_path, columns=["scene_token", "log_name"])
        assert len(inventory) >= len(manifest)
        assert set(manifest.scene_token).issubset(set(inventory.scene_token))
        assert set(manifest.log_name) == set(inventory.log_name)


def test_oracle_dynamic_shuffle_changes_only_dynamic_suffix() -> None:
    dataset = _oracle_fixture()
    base = feature_group(dataset, "O3", 7)
    full = feature_group(dataset, "O8", 7)
    shuffled = feature_group(dataset, "O10", 7)
    assert full.shape == shuffled.shape
    np.testing.assert_array_equal(full[..., : base.shape[-1]], shuffled[..., : base.shape[-1]])
    assert not np.array_equal(full[..., base.shape[-1] :], shuffled[..., base.shape[-1] :])
    # The within-scene control preserves every dynamic row as a multiset.
    for scene in range(dataset.scenes):
        expected = np.sort(full[scene, :, base.shape[-1] :], axis=0)
        actual = np.sort(shuffled[scene, :, base.shape[-1] :], axis=0)
        np.testing.assert_allclose(actual, expected)


def test_recomputed_ttc_is_finite_and_responds_to_closing_actor() -> None:
    summary = np.full((1, 2, 8, 5), 20.0, dtype=np.float32)
    actor = np.zeros((1, 2, 8, 1, 10), dtype=np.float32)
    mask = np.ones((1, 2, 8, 1), dtype=bool)
    actor[..., 1] = 20.0  # relative x
    actor[..., 2] = 0.0   # relative y
    actor[:, 0, ..., 3] = -5.0
    actor[:, 1, ..., 3] = 1.0
    actor[..., 6] = 4.0
    actor[..., 7] = 2.0
    risk = _risk_from_actor(summary, actor, mask)
    assert np.isfinite(risk).all()
    assert np.all(risk[0, 0, :, 1] < 10.0)
    assert np.all(risk[0, 1, :, 1] == 10.0)


def test_factor_calibration_bins_preserve_all_examples() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=bool)
    probability = np.asarray([0.0, 0.2, 0.8, 1.0])
    curve = _calibration_curve(truth, probability, bins=5)
    assert len(curve) == 5
    assert sum(row["count"] for row in curve) == len(truth)
    assert curve[0]["observed_rate"] == 0.0
    assert curve[-1]["observed_rate"] == 1.0


def test_oracle_store_allows_only_trailing_incomplete_scenes() -> None:
    assert _completed_prefix_length(np.asarray([True, True, True])) == 3
    assert _completed_prefix_length(np.asarray([True, True, False])) == 2
    with pytest.raises(RuntimeError, match="incomplete hole"):
        _completed_prefix_length(np.asarray([True, False, True]))


def test_model_candidate_distance_and_selection_are_deterministic() -> None:
    rng = np.random.default_rng(31)
    proposals = np.cumsum(rng.normal(0, 0.2, size=(64, 8, 3)), axis=1).astype(np.float32)
    baseline = rng.normal(size=64)
    official = rng.random(64)
    distance = trajectory_distance(proposals)
    np.testing.assert_allclose(distance, distance.T, atol=1e-7)
    np.testing.assert_array_equal(np.diag(distance), 0.0)
    first, reasons, _ = select_diverse_candidates(proposals, baseline, official, 16, 41)
    second, _, _ = select_diverse_candidates(proposals, baseline, official, 16, 41)
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 16
    assert set(first).issubset(reasons)


def test_generated_fold_manifest_is_log_disjoint() -> None:
    path = Path("reports/shared_future_candidate_consequence_gate_c/balanced_scene_manifest.parquet")
    if not path.is_file():
        pytest.skip("Balanced all-log manifest has not been generated")
    frame = pd.read_parquet(path, columns=["log_name", "fold"])
    assert frame.groupby("log_name").fold.nunique().max() == 1
    assert set(frame.fold.unique()) == set(range(5))


def test_all_log_selection_caps_each_log_without_dropping_small_logs() -> None:
    rows = []
    for log_name, count in (("large", 80), ("medium", 50), ("small", 7)):
        for index in range(count):
            rows.append(
                {
                    "log_name": log_name,
                    "scene_token": f"{log_name}_{index}",
                    "cache_exists": True,
                }
            )
    metadata = pd.DataFrame(rows)
    target_count = sum(min(count, 50) for count in (80, 50, 7))
    selected = balanced_select(metadata, target_count, 3, 50, 13)
    assert set(selected.log_name) == {"large", "medium", "small"}
    assert selected.groupby("log_name").size().to_dict() == {
        "large": 50, "medium": 50, "small": 7,
    }


def test_all_log_aggregate_filters_cache_superset(tmp_path: Path) -> None:
    candidates = []
    metrics = []
    for token in ("keep", "extra"):
        for index in range(3):
            candidates.append(
                {
                    "scene_token": token,
                    "candidate_index": index,
                    "candidate_family": "gt" if index == 1 else "speed_change",
                    "is_gt": index == 1,
                }
            )
            metrics.append(
                {
                    "scene_token": token,
                    "candidate_index": index,
                    "scoring_success": True,
                    "aggregate_score": 0.5 + 0.1 * index,
                }
            )
    candidate_path = tmp_path / "candidate.parquet"
    metric_path = tmp_path / "metric.parquet"
    pd.DataFrame(candidates).to_parquet(candidate_path)
    pd.DataFrame(metrics).to_parquet(metric_path)
    candidate_audit = _candidate_audit(candidate_path, {"keep"}, 3)
    metric_audit = _metric_audit(metric_path, {"keep"}, 3)
    assert candidate_audit["candidate_valid"]
    assert candidate_audit["candidate_rows"] == 3
    assert metric_audit["metric_valid"]
    assert metric_audit["metric_rows"] == 3


def test_formal_log_reuses_complete_superset_cache(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.parquet"
    metric_path = tmp_path / "metrics.parquet"
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    rows = [
        {"scene_token": token, "candidate_index": index}
        for token in ("formal", "superset_only")
        for index in range(3)
    ]
    pd.DataFrame(rows).to_parquet(candidate_path)
    pd.DataFrame([{**row, "scoring_success": True} for row in rows]).to_parquet(metric_path)
    np.savez_compressed(target_dir / "formal.npz", valid=np.ones(1, dtype=bool))
    assert _cache_covers_selection(
        candidate_path, metric_path, target_dir, ["formal"], num_candidates=3
    )
    assert not _cache_covers_selection(
        candidate_path, metric_path, target_dir, ["missing"], num_candidates=3
    )
