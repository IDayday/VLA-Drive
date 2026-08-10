import json
from pathlib import Path

from tools.grounded_world.summarize_training_signals import (
    load_metric_records,
    summarize_training_signals,
)


def test_training_signal_report_uses_only_actual_jsonl_records(tmp_path: Path) -> None:
    path = tmp_path / "training_metrics.jsonl"
    records = [
        {
            "step": step,
            "prior_scene_shuffle_margin": 0.01 + step / 100000.0,
            "future_temporal_margin": 0.02 + step / 100000.0,
            "world_tube_valid_ratio": 0.8,
            "world_delta_norm": step / 10000.0,
            "consequence_progress_prediction_std": 0.4,
            "consequence_progress_target_std": 2.0,
        }
        for step in (100, 200, 300, 400)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    loaded = load_metric_records(path)
    report = summarize_training_signals(loaded, stage="planning", window=2)
    assert report["record_count"] == 4
    assert report["last_step"] == 400
    assert report["checks"]["prior_alignment"]["status"] == "PASS"
    assert report["checks"]["future_temporal_identity"]["status"] == "PASS"
    assert report["checks"]["trajectory_tube_coverage"]["status"] == "PASS"
    assert report["checks"]["refiner_nonzero_update"]["status"] == "PASS"
    assert report["checks"]["consequence_progress_noncollapse"]["status"] == "PASS"


def test_training_signal_report_marks_absent_signals_missing() -> None:
    report = summarize_training_signals([{"step": 10, "plan_loss": 1.0}], "prior", 1)
    assert report["checks"]["prior_alignment"]["status"] == "MISSING"
    assert report["checks"]["geometry_coverage"]["status"] == "MISSING"


def test_predictive_and_planning_audits_prefer_retention_margin() -> None:
    report = summarize_training_signals(
        [{"step": 10, "retention_scene_shuffle_margin": 0.25}],
        "predictive",
        1,
    )
    assert report["checks"]["prior_alignment"]["status"] == "PASS"
    assert (
        report["checks"]["prior_alignment"]["metric"]
        == "retention_scene_shuffle_margin"
    )
