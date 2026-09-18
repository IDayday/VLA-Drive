"""Real epoch controller with subprocess boundaries injected, no model claims."""
import json
import threading
import pytest
from scripts.analysis.full_epoch_run import orchestrate


def test_baseline_overlaps_training_and_final_uses_completed_checkpoint(tmp_path):
    started=threading.Event();scoring=threading.Event();calls=[]
    def train(cancel):
        started.set();assert scoring.wait(2)
        assert not cancel.is_set()
        return {'status':'PASS','exported':'fixed_last','update':12912}
    def evaluate(label,checkpoint,cancel):
        calls.append((label,checkpoint))
        if label=='sft':
            assert started.wait(2);scoring.set()
        return {'status':'COMPLETE'}
    result=orchestrate({'control_dir':str(tmp_path),'baseline_checkpoint':'original_F'},train,evaluate,poll_seconds=.01)
    assert result['status']=='COMPLETE' and calls==[('sft','original_F'),('last','fixed_last')]


def test_scoring_failure_does_not_cancel_training_or_hide_final_attempt(tmp_path):
    calls=[];scored=threading.Event()
    def train(cancel):
        assert scored.wait(2);assert not cancel.is_set()
        calls.append('trained')
        return {'status':'PASS','exported':'fixed_last'}
    def evaluate(label,checkpoint,cancel):
        calls.append(label)
        if label=='sft':
            scored.set();raise IOError('score output failed')
        return {'status':'COMPLETE'}
    with pytest.raises(RuntimeError,match='sft_evaluation'):
        orchestrate({'control_dir':str(tmp_path),'baseline_checkpoint':'original'},train,evaluate,poll_seconds=.01)
    assert 'trained' in calls and calls[-1]=='last'
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['training']['status']=='PASS'


def test_training_failure_does_not_evaluate_partial_checkpoint(tmp_path):
    calls=[]
    def train(cancel):raise RuntimeError('numerical failure')
    def evaluate(label,checkpoint,cancel):calls.append(label);return {'status':'COMPLETE'}
    with pytest.raises(RuntimeError,match='numerical failure'):
        orchestrate({'control_dir':str(tmp_path),'baseline_checkpoint':'original'},train,evaluate,poll_seconds=.01)
    assert calls==['sft']
