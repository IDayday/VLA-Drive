import json
from pathlib import Path

import pandas as pd
import pytest

from tools.grounded_world.aggregate_navsim_results import aggregate_matrix


def _result(root: Path, name: str, scores: dict[str, float]) -> Path:
    csv_path = root / f"{name}.csv"
    rows = [
        {"token": token, "valid": True, "score": score}
        for token, score in scores.items()
    ]
    rows.append(
        {
            "token": "average_all_frames",
            "valid": True,
            "score": sum(scores.values()) / len(scores),
        }
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = root / f"{name}.json"
    summary.write_text(
        json.dumps(
            {
                "suite": "navtest",
                "csv": str(csv_path),
                "expected_scenarios": len(scores),
                "valid_scenarios": len(scores),
                "mean_score": sum(scores.values()) / len(scores),
                "official_summary_score": sum(scores.values()) / len(scores),
            }
        )
    )
    return summary


def test_aggregate_uses_actual_results_and_paired_tokens(tmp_path: Path) -> None:
    base = _result(tmp_path, "base", {"a": 0.2, "b": 0.4, "c": 0.6})
    full = _result(tmp_path, "full", {"a": 0.3, "b": 0.5, "c": 0.7})
    matrix = {
        "runs": [
            {"arm": "b0", "seed": 42, "step": 30000, "suite": "navtest", "summary": str(base)},
            {"arm": "b5", "seed": 42, "step": 30000, "suite": "navtest", "summary": str(full)},
            {"arm": "b5", "seed": 43, "step": 30000, "suite": "navtest", "summary": str(tmp_path / "missing.json")},
        ]
    }
    report = aggregate_matrix(matrix, reference_arm="b0", bootstrap_samples=200, seed=9)
    statuses = {(row["arm"], row["seed"]): row["status"] for row in report["runs"]}
    assert statuses[("b5", 43)] == "MISSING"
    paired = report["paired"]
    assert len(paired) == 1
    assert paired[0]["mean_difference"] == pytest.approx(0.1)
    assert paired[0]["paired_tokens"] == 3
