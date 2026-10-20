"""Independent Route A objective/sampling tests; synthetic cached contexts only."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tools.action_effect import train_refiner as entry
from tools.action_effect.common import write_json
from tools.action_effect.training import run_training, private_state
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel
from test_model_effect import cfg, spec
from test_model_refiner_entry import fixture_dataset, refiner, recipe


torch.set_num_threads(2)


def test_actual_train_entry_virtual_references_are_exactly_balanced_and_uniform(tmp_path,monkeypatch):
    dataset,_=fixture_dataset(tmp_path)
    template_repair=deepcopy(dataset.fit[1]);template_keep=deepcopy(dataset.fit[0])
    dataset.fit=[]
    for kind,count,template in (("repair",3,template_repair),("keep",5,template_keep)):
        for i in range(count):
            row=deepcopy(template);row["scene_id"]=kind+str(i);row["source_log"]=kind+"log"+str(i)
            dataset.fit.append(row)
    monkeypatch.setattr(entry,"prepare_dataset",lambda *_args,**_kwargs:dataset)
    monkeypatch.setattr(entry,"build_model",lambda *_args,**_kwargs:refiner())
    seen={}
    def capture(model,**kwargs):
        seen.update(kwargs);return {"status":"TEST_ONLY_NO_TRAINING"}
    monkeypatch.setattr(entry,"run_training",capture)
    result=entry.train(recipe(),dataset_path=dataset.path,context_manifest=dataset.reader.manifest_path,
        output=tmp_path/"run",arm="A0",device="cpu",seed=7,max_updates=4)
    counts=Counter(r["scene_id"] for r in dataset.fit)
    assert result["status"]=="TEST_ONLY_NO_TRAINING" and seen["dataset_size"]==30
    assert all(counts["repair"+str(i)]==5 for i in range(3))
    assert all(counts["keep"+str(i)]==3 for i in range(5))
    assert sum(counts[k] for k in counts if k.startswith("repair"))==15
    assert sum(counts[k] for k in counts if k.startswith("keep"))==15
    assert seen["identity"]["sampling"]["new_unique_scenes"]==0


@pytest.mark.parametrize("arm",["A0","A1","ScoreRefiner"])
def test_global_objective_microbatch_gradient_and_teacher_freezing(tmp_path,arm):
    dataset,_=fixture_dataset(tmp_path)
    # Different true label counts and one fully empty label scene must not
    # become locally normalized or change another task's global denominator.
    for name in ("base_labels","selected_labels"):
        dataset.fit[1][name]["valid"]=[i==0 for i in range(len(spec().names))]
        dataset.fit[2][name]["valid"]=[False]*len(spec().names)
    teacher=None if arm=="A0" else ActionEffectModel(cfg("absolute_score" if arm=="ScoreRefiner" else "delta_feedback"),spec())
    full=refiner(teacher); micro=deepcopy(full)
    with torch.no_grad():
        full.refiner.coefficient_head.bias.fill_(.05)
        micro.refiner.coefficient_head.bias.fill_(.05)
    frozen_before=None if teacher is None else {n:p.detach().clone() for n,p in teacher.named_parameters()}
    ids=[0,1,2,1,0]
    a=entry.make_step(dataset,arm=arm,microbatch_scenes=5,device="cpu")(full,ids,0)
    b=entry.make_step(dataset,arm=arm,microbatch_scenes=1,device="cpu")(micro,ids,0)
    for key in a: assert a[key]==pytest.approx(b[key],rel=1e-5,abs=1e-6)
    assert a["preserve_denominator"]==3 and a["supervision_denominator"]==5
    if teacher is not None:
        assert a["effect_consistency_denominator"]==4 and a["predicted_safety_denominator"]==2
        full.train();assert not teacher.training
        for name,param in teacher.named_parameters():
            assert not param.requires_grad and param.grad is None and torch.equal(param,frozen_before[name])
    for (name,p),(other,q) in zip(full.named_parameters(),micro.named_parameters()):
        assert name==other
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
        else:assert p.grad is q.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in full.refiner.parameters())


def test_all_arms_same_private_parameter_identity_and_own_teacher_mode(tmp_path,monkeypatch):
    dataset,_=fixture_dataset(tmp_path)
    from tools.action_effect import train_effect_model
    loaded=[]
    def own_teacher(path,**kwargs):
        mode=kwargs["allowed_modes"][0];loaded.append((path,mode,kwargs["expected_base_binding_hash"]))
        teacher=ActionEffectModel(cfg(mode),spec())
        teacher.provenance={"test_only":True,"external_weights_loaded":False,"checkpoint":str(path)}
        return teacher
    monkeypatch.setattr(train_effect_model,"load_effect_teacher",own_teacher)
    models=[entry.build_model(recipe(),dataset,arm=arm,seed=7,device="cpu",
        effect_checkpoint=None if arm=="A0" else "synthetic_own_"+arm) for arm in entry.ARMS]
    states=[private_state(m) for m in models]
    assert all(set(s)==set(states[0]) for s in states)
    assert all(torch.equal(s[n],states[0][n]) for s in states for n in states[0])
    assert all(not any(p.requires_grad for p in m.effect_teacher.parameters()) for m in models[1:])
    assert [x[1] for x in loaded]==["delta_feedback","absolute_score"]


def test_a1_actual_shared_loop_resume_preserves_parameters_teacher_and_next_samples(tmp_path):
    dataset,_=fixture_dataset(tmp_path); initial=refiner(ActionEffectModel(cfg(),spec()))
    whole=deepcopy(initial);split=deepcopy(initial);seen_full=[];seen_split=[]
    def fit(root,model,seen,**kwargs):
        fn=entry.make_step(dataset,arm="A1",microbatch_scenes=1,device="cpu")
        def step(m,ids,n):seen.append(list(ids));return fn(m,ids,n)
        return run_training(model,output=root,identity={"source_commit":"independent_synthetic_route_a"},
            dataset_size=len(dataset.fit),step_fn=step,cfg=recipe(),device="cpu",seed=11,max_updates=4,**kwargs)
    fit(tmp_path/"whole",whole,seen_full)
    fit(tmp_path/"split",split,seen_split,stop_after=2)
    resumed=deepcopy(initial);torch.randn(19)
    fit(tmp_path/"split",resumed,seen_split,resume=True)
    assert seen_full==seen_split
    for key,value in whole.state_dict().items():assert torch.equal(value,resumed.state_dict()[key])
    terminal_seen=[];out=fit(tmp_path/"split",resumed,terminal_seen,resume=True)
    assert out["segment_updates"]==0 and terminal_seen==[]
