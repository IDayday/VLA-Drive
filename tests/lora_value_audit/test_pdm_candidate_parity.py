import numpy as np

from tools.lora_value_audit.exact_candidate_scorer import _candidate_indices


def test_parity_sample_includes_selected_oracle_and_unique_random() -> None:
    indices = _candidate_indices(3, 9, 64, 4, np.random.default_rng(2))
    assert indices[:2] == [3, 9]
    assert len(indices) == len(set(indices)) == 4


def test_parity_sample_handles_selected_equal_oracle() -> None:
    indices = _candidate_indices(7, 7, 64, 4, np.random.default_rng(2))
    assert indices[0] == 7
    assert len(indices) == len(set(indices)) == 4
