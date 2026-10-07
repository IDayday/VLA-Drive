import pytest
from tools.planning_interface_transfer.accelerate_navtest_tail import partition_missing


def test_delegation_covers_only_missing_tail_and_never_frontier():
    index=[{'token':str(i)} for i in range(12146)]
    existing={str(i) for i in range(10001)}
    guards,parts=partition_missing(index,existing,4,32,6)
    tail=[i for p in parts for i in p]
    assert len(tail)==len(set(tail))
    assert set(tail).isdisjoint(guards)
    assert set(tail)|set(guards)==set(range(10001,12146))
    assert all(str(i) not in existing for i in tail)
    for rank in range(4):
        missing=list(range(rank,12146,4));missing=[i for i in missing if i>=10001]
        assert set(missing[:32])<=set(guards)


def test_guarded_short_tail_is_left_to_original_writer():
    index=[{'token':str(i)} for i in range(24)]
    guards,parts=partition_missing(index,set(),4,32,6)
    assert guards==list(range(24)) and parts==[[]]*6


def test_layout_bound_bank_cannot_change_world_size():
    with pytest.raises(ValueError):partition_missing([],set(),8,32,4)


def test_helpers_do_not_use_cache_tokens_as_indices():
    index=[{'token':f'random-{100-i}'} for i in range(64)]
    guards,parts=partition_missing(index,{'random-100','random-99'},4,2,3)
    assert set(guards)|set(i for p in parts for i in p)==set(range(2,64))
