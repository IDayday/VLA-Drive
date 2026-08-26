from pathlib import Path

import pytest

from research.cf_effect_gate_wote.src.cache_metric_subset import _tokens


def test_metric_subset_combines_disjoint_files_in_order(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("scene-a\nscene-b\n", encoding="utf-8")
    val.write_text("scene-c\n", encoding="utf-8")

    assert _tokens([train, val], None) == ["scene-a", "scene-b", "scene-c"]
    assert _tokens([train, val], 2) == ["scene-a", "scene-b"]


def test_metric_subset_rejects_cross_file_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("scene-a\nscene-b\n", encoding="utf-8")
    val.write_text("scene-b\nscene-c\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutually disjoint"):
        _tokens([train, val], None)
