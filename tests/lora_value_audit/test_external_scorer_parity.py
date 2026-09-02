import numpy as np

from tools.lora_value_audit.evaluate_union_scorer import _settings


def test_duplicate_control_preserves_true_oracle_and_budget() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(64, 8, 3)).astype(np.float32)
    external = rng.normal(size=(64, 8, 3)).astype(np.float32)
    base_true = np.linspace(0, 1, 64)
    external_true = np.linspace(0.2, 0.9, 64)
    predicted = np.linspace(1, 0, 64)
    settings = _settings(base, base_true, predicted, external, external_true, True)
    for count in (8, 16):
        proposals, true_scores, _, additive = settings[f"duplicate{count}"]
        assert proposals.shape[0] == 64 + count
        assert additive
        assert np.max(true_scores) == np.max(base_true)


def test_fixed_budget_counts_are_exact() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(size=(64, 8, 3)).astype(np.float32)
    external = rng.normal(size=(64, 8, 3)).astype(np.float32)
    settings = _settings(base, rng.random(64), rng.random(64), external, rng.random(64), False)
    for name in ("fixed_top_base56_ideal8", "fixed_diverse_base56_ideal8", "fixed_top_base48_ideal16", "fixed_diverse_base48_ideal16", "fixed_top_base32_ideal32", "fixed_diverse_base32_ideal32"):
        assert settings[name][0].shape == (64, 8, 3)
        assert settings[name][1].shape == (64,)
