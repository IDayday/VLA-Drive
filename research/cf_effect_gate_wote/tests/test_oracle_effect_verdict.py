from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.cf_effect_gate_wote.src.models.structured_six_factor_probe import (
    load_v2_checkpoint,
)
from research.cf_effect_gate_wote.src.oracle_effect_report import (
    EXPECTED_OLD_REPORT_HASHES,
    automatic_verdict,
    audit_legacy_reports,
    direction_non_compliance,
    hard_false_safe,
)


class Comparison:
    def __init__(self, delta: float = 0.01, lower: float = 0.005) -> None:
        self.score_delta = delta
        self.score_ci_lower = lower


def comparisons() -> dict[str, Comparison]:
    return {
        "wote_full_vs_direct": Comparison(),
        "wote_env_vs_direct": Comparison(),
    }


def statuses() -> dict[str, object]:
    return {
        "direct_baseline_quality": True,
        "full_vs_direct": True,
        "full_vs_static": True,
        "full_vs_shared_future": True,
        "candidate_specificity": True,
        "primitive_requirement": True,
        "engineered_pass": True,
        "mask_proxy": False,
        "interaction_conditional": False,
        "interaction_subset_status": "FAIL",
    }


def test_verdict_priority_and_positive_terminal() -> None:
    result = automatic_verdict(
        data_contract_pass=True,
        probe_contract_pass=True,
        statuses=statuses(),
        comparisons=comparisons(),  # type: ignore[arg-type]
    )
    assert result["final_verdict"] == "ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE"
    underfit = statuses()
    underfit["direct_baseline_quality"] = False
    result = automatic_verdict(
        data_contract_pass=True,
        probe_contract_pass=True,
        statuses=underfit,
        comparisons=comparisons(),  # type: ignore[arg-type]
    )
    assert result["final_verdict"] == "DIRECT_BASELINE_UNDERFIT"


def test_hard_false_safe_and_direction_include_ddc() -> None:
    factors = np.ones((3, 6), dtype=np.float32)
    factors[0, 2] = 0.0
    factors[1, 2] = 0.5
    np.testing.assert_array_equal(hard_false_safe(factors), [True, False, False])
    np.testing.assert_array_equal(direction_non_compliance(factors), [True, True, False])


def test_legacy_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": "matched_factor_probe.v1"}, path)
    with pytest.raises(ValueError, match="refuses checkpoint schema"):
        load_v2_checkpoint(path)


def test_old_report_hashes_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[3]
    audit = audit_legacy_reports(root)
    assert audit["status"] == "UNCHANGED"
    assert audit["actual"] == EXPECTED_OLD_REPORT_HASHES
