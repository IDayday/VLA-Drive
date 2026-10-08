import pytest
import ast
import shlex
import signal
from enum import IntEnum
from tools.planning_interface_transfer import accelerate_navtest_tail
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


class LegacySignal(IntEnum):
    SIGCONT = int(signal.SIGCONT)

    def __str__(self):
        # Python 3.10's signal enum spelling is not defined in remote code.
        return 'Signals.SIGCONT'


@pytest.mark.parametrize('signum', [signal.SIGSTOP, signal.SIGCONT, LegacySignal.SIGCONT])
def test_remote_owned_signal_uses_portable_integer(monkeypatch, signum):
    calls = []
    monkeypatch.setattr(accelerate_navtest_tail.subprocess, 'run',
                        lambda command, **kwargs: calls.append((command, kwargs)))
    accelerate_navtest_tail.signal_owned('registered-host',
                                        [{'pid': 123, 'bank': '/registered/bank'}], signum)
    command, kwargs = calls[0]
    assert command[:2] == ['ssh', 'registered-host']
    assert kwargs == {'check': True, 'timeout': 20}
    code = shlex.split(command[2])[2]
    tree = ast.parse(code)
    kill = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'kill')
    assert isinstance(kill.args[1], ast.Constant)
    assert kill.args[1].value == int(signum)
    assert 'tools.foresight.export_predictions' in code
    assert "r['bank']" in code and 'Owned native PID mismatch' in code
