import pytest
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import merge_action_context


def test_none_context_is_an_exact_noop():
    action_queries = torch.randn(2, 8, 32)
    merged = merge_action_context(action_queries, None)
    assert merged.data_ptr() == action_queries.data_ptr()


def test_context_is_appended_on_sequence_axis():
    action_queries = torch.randn(2, 8, 32)
    context = torch.randn(2, 27, 32)
    merged = merge_action_context(action_queries, context)
    assert merged.shape == (2, 35, 32)
    torch.testing.assert_close(merged[:, :8], action_queries)
    torch.testing.assert_close(merged[:, 8:], context)


def test_context_contract_fails_before_action_head():
    with pytest.raises(AssertionError, match="batch"):
        merge_action_context(torch.randn(2, 8, 32), torch.randn(1, 27, 32))
