import json
import threading
import pytest
from scripts.analysis.accelerated_epoch_run import await_baseline


def test_existing_completed_baseline_is_reused_after_training_migration(tmp_path):
    result={'status':'COMPLETE','seeds':{'42':'fixture'}}
    (tmp_path/'progress.json').write_text(json.dumps({'status':'FAIL','failures':{'training':'deliberate migration'},'evaluations':{'sft':result}}))
    assert await_baseline(tmp_path,threading.Event(),timeout=.1,poll=.01)==result


@pytest.mark.parametrize('state',[{'baseline_error':'scorer crashed'},{'evaluations':{'sft':{'status':'RUNNING'}}}])
def test_partial_or_failed_baseline_never_accepted(tmp_path,state):
    (tmp_path/'progress.json').write_text(json.dumps(state))
    with pytest.raises((ValueError,RuntimeError)):await_baseline(tmp_path,threading.Event(),timeout=.1,poll=.01)


def test_wait_is_bounded_and_cancellable(tmp_path):
    with pytest.raises(TimeoutError):await_baseline(tmp_path,threading.Event(),timeout=.01,poll=.001)
    event=threading.Event();event.set()
    with pytest.raises(RuntimeError,match='cancelled'):await_baseline(tmp_path,event)
