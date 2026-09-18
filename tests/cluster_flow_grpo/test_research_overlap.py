"""Real orchestration function, injected executors; not a GPU/evaluation test."""
import json
from pathlib import Path
import threading
from scripts.analysis.credit_research_run import orchestrate


def test_evaluation_does_not_block_training_to_next_boundary(tmp_path):
    reached_final=threading.Event();evaluation_started=threading.Event()
    runroot=tmp_path/'run'
    spec={'control_dir':str(tmp_path/'control'),'evaluate_updates':[32,64],
          'arms':{'uniform':{'train_spec':'fixture','run':str(runroot)}}}
    def publish(update):
        p=runroot/f'checkpoints/update_{update:06d}';p.mkdir(parents=True)
        (p/'trainer_state.json').write_text(json.dumps({'update':update,'world_size':8}))
        (p/'COMPLETE').write_text('complete')
    def train(path,cancel):
        publish(32)
        assert evaluation_started.wait(3)
        # Evaluation is deliberately waiting for training to reach64.
        publish(64);reached_final.set()
        return {'status':'PASS','exit_codes':[0]}
    order=[]
    def evaluate(label,arm,update,cancel):
        order.append(update)
        if update==32:
            evaluation_started.set();assert reached_final.wait(3)
        return {'status':'COMPLETE'}
    result=orchestrate(spec,train,evaluate,pause=lambda _:threading.Event().wait(.01))
    assert result['status']=='COMPLETE' and order==[32,64]


def test_failed_training_never_publishes_complete(tmp_path):
    import pytest
    spec={'control_dir':str(tmp_path/'control'),'evaluate_updates':[32,64],
          'arms':{'uniform':{'train_spec':'fixture','run':str(tmp_path/'run')}}}
    with pytest.raises(RuntimeError,match='training failed'):
        orchestrate(spec,lambda *a:{'status':'FAIL'},lambda *a:None,pause=lambda _:None)
    assert json.loads((tmp_path/'control/result.json').read_text())['status']=='FAIL'
