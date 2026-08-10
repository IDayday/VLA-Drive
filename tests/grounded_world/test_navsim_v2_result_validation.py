from pathlib import Path

import pandas as pd
import pytest

from tools.grounded_world.validate_navsim_v2_results import validate_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_navtest_requires_all_rows_valid_and_finite_score(tmp_path: Path) -> None:
    csv_path = tmp_path / "navtest.csv"
    _write_csv(
        csv_path,
        [
            {"token": "a", "valid": True, "score": 0.4},
            {"token": "b", "valid": True, "score": 0.6},
            {"token": "average_all_frames", "valid": True, "score": 0.5},
        ],
    )
    summary = validate_results(csv_path, suite="navtest", expected_scenarios=2)
    assert summary["valid_scenarios"] == 2
    assert summary["mean_score"] == pytest.approx(0.5)


def test_validate_navhard_requires_stage_summaries(tmp_path: Path) -> None:
    csv_path = tmp_path / "navhard.csv"
    _write_csv(
        csv_path,
        [
            {"token": "a", "valid": True, "score": 0.5},
            {"token": "b", "valid": True, "score": 0.25},
            {"token": "extended_pdm_score_stage_one", "valid": True, "score": 0.6},
            {"token": "extended_pdm_score_stage_two", "valid": True, "score": 0.4},
            {"token": "extended_pdm_score_combined", "valid": True, "score": 0.24},
        ],
    )
    summary = validate_results(csv_path, suite="navhard_two_stage", expected_scenarios=2)
    assert summary["stage_one_score"] == pytest.approx(0.6)
    assert summary["stage_two_score"] == pytest.approx(0.4)
    assert summary["combined_score"] == pytest.approx(0.24)


def test_validate_results_rejects_invalid_scenarios(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    _write_csv(csv_path, [{"token": "a", "valid": False, "score": 0.0}])
    with pytest.raises(ValueError, match="invalid scenarios"):
        validate_results(csv_path, suite="navtest", expected_scenarios=1)
