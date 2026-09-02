from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_candidate_bank_is_current_only_and_all_64(tmp_path) -> None:
    evaluation = _load(
        REPO_ROOT / "navsim/planning/script/run_pdm_score_multi_gpu.py"
    )
    component_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    )
    predictions = {}
    for scene_index, token in enumerate(("token-a", "token-b")):
        predictions[token] = {
            "proposals": np.full((64, 8, 3), scene_index, dtype=np.float32),
            "predicted_log_pdm": np.linspace(-2, 0, 64, dtype=np.float32),
            "predicted_pdms": np.linspace(0, 1, 64, dtype=np.float32),
            "selected_index": 63,
            "component_probabilities": {
                name: np.full(64, 0.5, dtype=np.float32)
                for name in component_names
            },
            "planning_registers": np.ones((16, 256), dtype=np.float32),
            "tile_gate": 0.0,
            "semantic_gate": 0.2,
            "inference_latency_seconds": 0.01,
        }
    output = tmp_path / "candidate_bank.npz"
    manifest = evaluation.save_formal_candidate_bank(
        predictions, ["token-a", "token-b"], output
    )
    assert manifest["inference_uses_future_inputs"] is False
    assert manifest["official_candidate_scores_in_bank"] is False
    with np.load(output, allow_pickle=False) as bank:
        assert bank["proposals"].shape == (2, 64, 8, 3)
        assert bank["planning_registers"].shape == (2, 16, 256)
        assert not any("future" in key for key in bank.files)


def test_formal_candidate_metrics_use_official_all_64_scores(tmp_path) -> None:
    metrics = _load(
        REPO_ROOT / "local_planreg_wm_v1/collect_candidate_metrics.py"
    )
    scores = np.zeros((2, 64, 7), dtype=np.float32)
    scores[..., :6] = 0.5
    scores[0, :, 6] = np.linspace(0.0, 1.0, 64)
    scores[1, :, 6] = np.linspace(1.0, 0.0, 64)
    selected = np.asarray([63, 63], dtype=np.int64)
    proposals = np.zeros((2, 64, 8, 3), dtype=np.float32)
    proposals[:, :, :, 0] = np.arange(64, dtype=np.float32)[None, :, None]
    path = tmp_path / "scored.npz"
    np.savez_compressed(
        path,
        pdm_scores=scores,
        selected_indices=selected,
        proposals=proposals,
        planning_registers=np.ones((2, 16, 256), dtype=np.float32),
        tile_gate=np.zeros(2, dtype=np.float32),
        semantic_gate=np.full(2, 0.2, dtype=np.float32),
        inference_latency_seconds=np.asarray([0.01, 0.02]),
    )
    report = metrics.collect(path)
    assert report["candidate_count"] == 64
    assert report["oracle_at_64_pdms"] == 1.0
    assert np.isclose(report["selected_pdms"], 0.5)
    assert np.isclose(report["scorer_regret"], 0.5)
    assert report["candidate_duplicate_rate"] == 0.0
    assert report["effective_clusters_rms_0p5"] == 64.0
