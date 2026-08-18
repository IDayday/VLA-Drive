from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "evaluate_vlm_retention.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_vlm_retention", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def test_extract_choice_requires_an_unambiguous_allowed_letter() -> None:
    assert TOOLS.extract_choice("A", "ABCD") == "A"
    assert TOOLS.extract_choice("The answer is (b).", "ABCD") == "B"
    assert TOOLS.extract_choice("Answer: D: beneath the book", "ABCD") == "D"
    assert TOOLS.extract_choice("I choose A, not B", "ABCD") is None
    assert TOOLS.extract_choice("J", "ABCD") is None


def test_stratified_sample_is_balanced_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "item_id": list(range(12)),
            "category": ["a"] * 6 + ["b"] * 6,
        }
    )
    first = TOOLS.stratified_sample(
        frame, group_column="category", per_group=3, seed=17, id_column="item_id"
    )
    second = TOOLS.stratified_sample(
        frame, group_column="category", per_group=3, seed=17, id_column="item_id"
    )
    assert first["item_id"].tolist() == second["item_id"].tolist()
    assert first["category"].value_counts().to_dict() == {"a": 3, "b": 3}


def test_paired_statistics_use_item_intersection_and_report_direction() -> None:
    baseline = {
        "a": True,
        "b": True,
        "c": False,
        "d": False,
    }
    treatment = {
        "a": True,
        "b": False,
        "c": True,
        "d": True,
        "extra": True,
    }
    stats = TOOLS.paired_statistics(
        baseline, treatment, seed=7, bootstrap_samples=2000
    )
    assert stats["sample_count"] == 4
    assert stats["baseline_accuracy"] == 0.5
    assert stats["treatment_accuracy"] == 0.75
    assert stats["delta_accuracy"] == 0.25
    assert stats["baseline_only_correct"] == 1
    assert stats["treatment_only_correct"] == 2
    assert np.isfinite(stats["mcnemar_exact_p"])


def test_checkpoint_key_mapping_and_component_groups() -> None:
    prefix = "qwen_vl_interface.model."
    visual = prefix + "model.visual.blocks.3.attn.qkv.weight"
    language = prefix + "model.language_model.layers.7.mlp.down_proj.weight"
    assert TOOLS.standard_qwen_key(visual) == "model.visual.blocks.3.attn.qkv.weight"
    assert TOOLS.parameter_groups(visual) == (
        "qwen",
        "visual",
        "visual.blocks.3",
    )
    assert TOOLS.parameter_groups(language) == (
        "qwen",
        "language",
        "language.layers.7",
    )


def test_paired_continuous_statistics_report_mean_delta() -> None:
    baseline = {"a": 0.2, "b": 0.4, "c": 0.8}
    treatment = {"a": 0.3, "b": 0.2, "c": 1.0, "extra": 0.0}
    stats = TOOLS.paired_continuous_statistics(
        baseline, treatment, seed=3, bootstrap_samples=1000
    )
    assert stats["sample_count"] == 3
    assert np.isclose(stats["baseline_mean"], 1.4 / 3)
    assert np.isclose(stats["treatment_mean"], 1.5 / 3)
    assert np.isclose(stats["delta_mean"], 0.1 / 3)
    assert stats["improved_count"] == 2
    assert stats["regressed_count"] == 1
