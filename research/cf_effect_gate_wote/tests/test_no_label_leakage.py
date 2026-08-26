from __future__ import annotations

import ast
from pathlib import Path

from research.cf_effect_gate_wote.src import replay_effect_builder


def test_replay_builder_never_reads_planning_metric_labels() -> None:
    source_path = Path(replay_effect_builder.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_exact = {
        "score",
        "scores",
        "factor",
        "factors",
        "pdms",
        "epdms",
        "selected_index",
        "candidate_index_label",
    }
    names = {
        node.id.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    strings = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (forbidden_exact & names)
    assert not (forbidden_exact & strings)
