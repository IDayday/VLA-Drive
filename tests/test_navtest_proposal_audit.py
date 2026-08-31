import pickle

import numpy as np

from local_stage2.run_navtest_proposal_audit import _candidate_geometry
from local_stage2.score_cached_navtest_proposals import (
    _completed_log,
    _load_predictions,
    _persist_log_rows,
    _read_persisted_rows,
)


def test_candidate_geometry_detects_unique_proposals() -> None:
    proposals = np.zeros((3, 8, 3), dtype=np.float32)
    proposals[1, :, 0] = np.linspace(0.0, 2.0, 8)
    proposals[2, :, 1] = np.linspace(0.0, 3.0, 8)
    metrics = _candidate_geometry(proposals)
    assert metrics["unique_candidate_count"] == 3.0
    assert metrics["mean_pairwise_endpoint_distance_m"] > 0.0
    assert metrics["mean_pairwise_ade_m"] > 0.0


def test_per_log_candidate_scores_are_atomic_and_resumable(tmp_path) -> None:
    log_name = "2021.01.01.00.00.00_veh-1_00000_00100"
    scores = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    predictions = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    row = {
        "token": "token-a",
        "log_name": log_name,
        "valid": True,
        "candidate_scores": scores,
        "predicted_scores": predictions,
    }
    manifest = _persist_log_rows(tmp_path, log_name, [row])
    assert manifest["valid_scene_count"] == 1
    assert _completed_log(tmp_path, log_name)

    rows, arrays = _read_persisted_rows(tmp_path)
    assert rows[0]["token"] == "token-a"
    np.testing.assert_array_equal(arrays["token-a"][0], scores)
    np.testing.assert_array_equal(arrays["token-a"][1], predictions)


def test_prediction_cache_requires_proposals_and_scorer_outputs(tmp_path) -> None:
    valid_path = tmp_path / "valid.pkl"
    with valid_path.open("wb") as file:
        pickle.dump(
            {
                "token-a": {
                    "proposals": np.zeros((64, 8, 3), dtype=np.float32),
                    "predicted_scores": np.zeros(64, dtype=np.float32),
                }
            },
            file,
        )
    loaded = _load_predictions(valid_path)
    assert list(loaded) == ["token-a"]

    invalid_path = tmp_path / "invalid.pkl"
    with invalid_path.open("wb") as file:
        pickle.dump({"token-a": {"proposals": np.zeros((64, 8, 3))}}, file)
    try:
        _load_predictions(invalid_path)
    except ValueError as error:
        assert "Malformed prediction" in str(error)
    else:
        raise AssertionError("Malformed candidate cache was accepted")
