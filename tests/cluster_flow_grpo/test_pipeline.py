"""Real asynchronous controller tests with CPU-only executor fixtures."""
import json
from pathlib import Path
import threading
import pytest

from scripts.cluster_flow_grpo.pipeline import continuous_pipeline
from scripts.cluster_flow_grpo.paired import validate_plan


def publish(run, update):
    root=run/"checkpoints"/f"update_{update:06d}"
    root.mkdir(parents=True)
    (root/"trainer_state.json").write_text(json.dumps({"update":update}))
    (root/"COMPLETE").write_text("CPU control fixture only")
    return root


def validator(path):
    return json.loads((path/"trainer_state.json").read_text())


def test_slow_evaluation_does_not_block_optimizer_progress(tmp_path):
    entered=threading.Event();advanced=threading.Event();cancel=threading.Event();seen=[]
    def train(resume, target):
        assert resume is None and target==400
        publish(tmp_path,100)
        assert entered.wait(3)
        publish(tmp_path,200);publish(tmp_path,400);advanced.set()
    def evaluate(exported, update):
        if update==100:
            entered.set()
            assert advanced.wait(3), "training waited for evaluation"
        seen.append(update);return {"complete_split":True,"epdms":.5}
    continuous_pipeline(tmp_path,[100,200,400],validate=validator,train=train,
        export=lambda cp,u:cp,evaluate=evaluate,on_result=lambda *a:None,
        baseline=lambda:None,cancelled=cancel,poll_seconds=.001)
    assert seen==[100,200,400] and not cancel.is_set()


def test_resume_uses300_and_never_retrains_saved_evaluation_boundaries(tmp_path):
    for u in [100,200,300]:publish(tmp_path,u)
    calls=[]
    def train(resume,target):
        calls.append((resume.name,target));publish(tmp_path,400)
    continuous_pipeline(tmp_path,[100,200,400],validate=validator,train=train,
        export=lambda cp,u:cp,evaluate=lambda *a:{},on_result=lambda *a:None,
        baseline=lambda:None,cancelled=threading.Event(),poll_seconds=.001)
    assert calls==[("update_000300",400)]


def test_evaluation_failure_cancels_only_this_producer(tmp_path):
    cancel=threading.Event();terminated=threading.Event()
    def train(*args):
        publish(tmp_path,100)
        assert cancel.wait(3);terminated.set()
        raise RuntimeError("cancelled child")
    def fail(*args):raise ValueError("evaluation fixture failure")
    with pytest.raises(ValueError,match="evaluation fixture"):
        continuous_pipeline(tmp_path,[100,200],validate=validator,train=train,
            export=lambda cp,u:cp,evaluate=fail,on_result=lambda *a:None,
            baseline=lambda:None,cancelled=cancel,poll_seconds=.001)
    assert terminated.is_set()


def test_adoption_finishes_existing_segment_before_continuous_resume(tmp_path):
    calls=[]
    def adopt():
        for u in [100,200,300,400]:publish(tmp_path,u)
        calls.append("adopted400")
    def train(resume,target):
        calls.append((resume.name,target));publish(tmp_path,600)
    continuous_pipeline(tmp_path,[100,200,400,600],validate=validator,train=train,
        export=lambda cp,u:cp,evaluate=lambda *a:{},on_result=lambda *a:None,
        baseline=lambda:None,cancelled=threading.Event(),adopt=adopt,poll_seconds=.001)
    assert calls==["adopted400",("update_000400",600)]


def test_complete_checkpoint_mutation_is_rejected(tmp_path):
    published=publish(tmp_path,100)
    def baseline():
        # Mutation after the consumer's native validation, before the producer
        # validates final completion, is exercised via the evaluate callback.
        pass
    finish=threading.Event()
    def train(resume,target):
        assert finish.wait(3);publish(tmp_path,200)
    def evaluate(cp,u):
        if u==100:
            (published/"trainer_state.json").write_text('{"update": 999}')
            finish.set()
        return {}
    with pytest.raises(ValueError,match="checkpoint changed"):
        continuous_pipeline(tmp_path,[100,200],validate=validator,train=train,
            export=lambda cp,u:cp,evaluate=evaluate,on_result=lambda *a:None,
            baseline=baseline,cancelled=threading.Event(),poll_seconds=.001)


def test_async_gpu_accounting_includes_eval_and_reserved_devices():
    plan={"groups":{v:{"nodes":[{"host":h,"devices":[0,1]}]}
                    for v,h in [("frozen_visual","F"),("unfrozen_visual","U")]},
          "resource_limits":{"F":{"max_gpus":3,"reserved_devices":[3]}},
          "async_evaluation":{"slots":[{"host":"F","gpu":2}]}}
    assert validate_plan(plan)==2
    plan["async_evaluation"]["slots"][0]["gpu"]=0
    with pytest.raises(ValueError,match="disjoint"):validate_plan(plan)
    plan["async_evaluation"]["slots"][0]["gpu"]=3
    with pytest.raises(ValueError,match="resource limit"):validate_plan(plan)
