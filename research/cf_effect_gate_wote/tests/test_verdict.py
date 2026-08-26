from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.cf_effect_gate_wote.src.verdict import (
    _candidate_table,
    _effect_table,
    _inverse_table,
    determine_verdict,
)


def _payload(key: str, passed: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {key: passed}
    if key == "gate_g1_pass":
        payload.update(oracle_gap_raw=0.03, oracle_gap_points=3.0)
    if key == "gate_g2_pass":
        payload.update(mean_direct_selected_pdms_gain_raw=0.006)
    return payload


def test_verdict_matrix_and_not_run_are_distinct(tmp_path: Path) -> None:
    g0 = _payload("gate_g0_pass")
    g1_pass = _payload("gate_g1_pass")
    g2_pass = _payload("gate_g2_pass")
    g3_pass = _payload("gate_g3_pass")
    g4_pass = _payload("gate_g4_pass")
    missing_metrics = tmp_path / "missing.csv"
    assert determine_verdict(None, None, None, None, None, missing_metrics)[
        "final_verdict"
    ] == "NOT_RUN"
    assert determine_verdict(
        g0, _payload("gate_g1_pass", False), None, None, None, missing_metrics
    )["final_verdict"] == "STOP_DIRECTION"
    assert determine_verdict(
        g0, g1_pass, g2_pass, _payload("gate_g3_pass", False), None, missing_metrics
    )["final_verdict"] == "EFFECT_TARGET_VALID_BUT_PREDICTION_BOTTLENECK"
    assert determine_verdict(
        g0, g1_pass, g2_pass, g3_pass, _payload("gate_g4_pass", False), missing_metrics
    )["final_verdict"] == "EFFECT_MODEL_ONLY"
    assert determine_verdict(
        g0, g1_pass, g2_pass, g3_pass, g4_pass, missing_metrics
    )["final_verdict"] == "EFFECT_PLUS_INVERSE"


def test_g0_alignment_failure_records_exact_blocker(tmp_path: Path) -> None:
    g0 = {
        "gate_g0_pass": False,
        "scene_count": 200,
        "official_debug_equivalence": True,
        "cache_reproducible": True,
        "alignment_mismatched_candidate_fraction": 0.005,
        "alignment_maximum_absolute_error": 0.622247964,
        "alignment_tolerance": 1e-6,
        "alignment_published_generator_default_cache_conflict": True,
    }
    verdict = determine_verdict(g0, None, None, None, None, tmp_path / "missing.csv")
    assert verdict["final_verdict"] == "STOP_DIRECTION"
    assert verdict["gate_g1_candidate_headroom"] == "NOT_RUN"
    assert "0.005000" in verdict["blocking_evidence"][0]
    assert "80 proposal poses" in verdict["blocking_evidence"][1]
    assert len(verdict["positive_evidence"]) == 2


def test_not_run_tables_retain_required_control_names() -> None:
    effect_rows = _effect_table(pd.DataFrame())
    inverse_rows = _inverse_table(pd.DataFrame())
    assert len(effect_rows) == 8
    assert all("NOT_RUN" in row for row in effect_rows)
    assert "Trajectory-only" in effect_rows[0]
    assert "Effect swap" in effect_rows[-1]
    assert len(inverse_rows) == 3
    assert "environment_only" in inverse_rows[1]
    assert _candidate_table(None).startswith("| WoTE fixed base anchors")


def test_simulator_supervision_special_case(tmp_path: Path) -> None:
    metrics = tmp_path / "probe.csv"
    pd.DataFrame(
        [
            {"model": "direct_current", "top1_regret": 0.10, "selected_pdms": 0.50},
            {"model": "wote_full_future", "top1_regret": 0.07, "selected_pdms": 0.51},
        ]
    ).to_csv(metrics, index=False)
    verdict = determine_verdict(
        _payload("gate_g0_pass"),
        _payload("gate_g1_pass"),
        _payload("gate_g2_pass", False),
        None,
        None,
        metrics,
    )
    assert verdict["final_verdict"] == "SIMULATOR_SUPERVISION_DEPENDENT"
