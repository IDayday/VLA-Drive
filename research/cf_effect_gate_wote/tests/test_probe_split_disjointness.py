from __future__ import annotations

from pathlib import Path

from research.cf_effect_gate_wote.src.oracle_effect_verdict import build_probe_split


def test_fixed_probe_split_is_disjoint_and_uses_required_slices() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "splits"
    split = build_probe_split(root, root / "relabel_headroom_tokens.txt")
    assert (len(split.train), len(split.val), len(split.test)) == (1024, 256, 512)
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))
    assert not (set(split.test) & set((root / "relabel_headroom_tokens.txt").read_text().split()))
    original_test = (root / "test_tokens.txt").read_text().split()
    assert list(split.test) == original_test[200:712]

