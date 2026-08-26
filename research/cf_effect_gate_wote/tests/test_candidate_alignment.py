from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.candidate_alignment import (
    CandidateAlignmentError,
    CandidateLabels,
    CandidateScoreTable,
    audit_alignment,
    build_fixed_splits,
    candidate_oracle,
    load_anchors,
    sha1_sorted_tokens,
    source_alignment_audit,
)


def _score_payload(tokens: list[str], factors: np.ndarray) -> dict[str, object]:
    keys = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
    )
    table = {key: factors[:, index] for index, key in enumerate(keys)}
    table["score"] = factors.mean(axis=1)
    return {
        token: {"trajectory_scores": [table]}
        for token in tokens
    }


def test_sha1_split_is_deterministic_and_disjoint() -> None:
    tokens = [f"token-{index:05d}" for index in range(20_000)]
    first = build_fixed_splits(tokens)
    second = build_fixed_splits(reversed(tokens))
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 8192,
        "val": 1024,
        "test": 2048,
    }
    assert first["train"] + first["val"] + first["test"] == sha1_sorted_tokens(tokens)[
        : 8192 + 1024 + 2048
    ]
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))


def test_split_falls_back_to_70_10_20() -> None:
    split = build_fixed_splits([f"small-{index}" for index in range(100)])
    assert {name: len(values) for name, values in split.items()} == {
        "train": 70,
        "val": 10,
        "test": 20,
    }


def test_anchor_shape_and_duplicates_are_rejected(tmp_path: Path) -> None:
    anchors = np.zeros((256, 8, 3), dtype=np.float32)
    path = tmp_path / "anchors.npy"
    np.save(path, anchors)
    with pytest.raises(CandidateAlignmentError, match="duplicate"):
        load_anchors(path)


def test_source_audit_identifies_base_anchor_offset_risk(tmp_path: Path) -> None:
    root = tmp_path / "WoTE"
    directory = root / "navsim/agents/WoTE"
    directory.mkdir(parents=True)
    (directory / "WoTE_targets.py").write_text(
        "sim_reward_dict_single = self.sim_reward_dict[token]['trajectory_scores'][0]\n"
        "combined = np.vstack([sim_reward_dict_single[key] for key in self.sim_keys])\n",
        encoding="utf-8",
    )
    (directory / "WoTE_model.py").write_text(
        'result["trajectory_anchors"] = self.trajectory_anchors\n'
        "trajectory_anchors = self.trajectory_anchors + offset\n",
        encoding="utf-8",
    )
    (directory / "WoTE_loss.py").write_text(
        'trajectory_anchors = predictions["trajectory_anchors"]\n', encoding="utf-8"
    )
    audit = source_alignment_audit(root)
    assert audit["pass"] is True
    assert audit["score_alignment_domain"] == "base_anchors"
    assert audit["offset_label_mismatch_risk"] is True


def test_dynamic_alignment_detects_candidate_reindexing(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    anchors = rng.normal(size=(256, 8, 3)).astype(np.float32)
    tokens = [f"scene-{index}" for index in range(20)]
    factors = rng.uniform(size=(256, 5)).astype(np.float32)
    score_path = tmp_path / "scores.npy"
    np.save(score_path, _score_payload(tokens, factors), allow_pickle=True)
    table = CandidateScoreTable(score_path)

    exact = CandidateLabels(factors=factors, score=factors.mean(axis=1))
    _, summary = audit_alignment(anchors, table, tokens, lambda _token, _anchors: exact)
    assert summary["pass"] is True
    assert summary["maximum_absolute_error"] == 0.0

    swapped = CandidateLabels(
        factors=factors[::-1].copy(), score=factors.mean(axis=1)[::-1].copy()
    )
    _, swapped_summary = audit_alignment(
        anchors, table, tokens, lambda _token, _anchors: swapped
    )
    assert swapped_summary["pass"] is False
    assert swapped_summary["mismatched_candidate_fraction"] > 0.9


def test_candidate_oracle_uses_fixed_selected_indices(tmp_path: Path) -> None:
    tokens = [f"scene-{index}" for index in range(20)]
    factors = np.ones((256, 5), dtype=np.float32)
    factors[:, 2] = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    score_path = tmp_path / "scores.npy"
    np.save(score_path, _score_payload(tokens, factors), allow_pickle=True)
    table = CandidateScoreTable(score_path)
    selected = {token: 0 for token in tokens}
    frame, summary = candidate_oracle(table, selected, tokens)
    assert len(frame) == 20
    assert (frame["selected_index"] == 0).all()
    assert (frame["oracle_index"] == 255).all()
    assert summary["oracle_gap_raw"] > 0.19
    assert summary["improvement_gt_0_05_fraction"] == 1.0
