from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from iqe.io import atomic_json,atomic_torch,digest,file_hash
from iqe.evaluation.retention import module_hash,module_hash_parameters
from iqe.decision_selection import evaluation as ev
from iqe.decision_selection.data import canonical_bank


def test_final_test_rejected_before_opening_any_artifact():
    with pytest.raises(ValueError,match='forbidden'):canonical_bank(None,'final_test')
    with pytest.raises(ValueError,match='forbidden'):canonical_bank(None,'navtest')


def setup_endpoint(tmp_path,monkeypatch):
    kind='M_set';root=tmp_path/'refit'/kind;root.mkdir(parents=True)
    model=nn.Linear(2,2);model.register_buffer('time',torch.arange(3.))
    initial=module_hash(model)
    config=SimpleNamespace(root=tmp_path)
    plan=dict(binding_hash='base',split_hash='split',models={kind:dict(updates=8,initialization_hash=initial)})
    plan['hash']=digest(plan);atomic_json(tmp_path/'REFIT_PLAN.json',plan)
    atomic_json(root/'RUN_IDENTITY.json',dict(dependencies=dict(refit_plan=plan['hash'],phase='refit')))
    atomic_json(root/'INITIALIZATION.json',dict(initial_state_hash=initial,external_scorer_weights_loaded=False,own_pretrained_scorer_weights_loaded=False))
    with torch.no_grad():model.weight.add_(1.)
    state=model.state_dict();atomic_torch(root/'best.pt',state);atomic_torch(root/'step_000008.pt',dict(module=state))
    atomic_json(str(root/'step_000008.pt')+'.COMPLETE.json',dict(checksum=file_hash(root/'step_000008.pt')))
    result=dict(status='COMPLETE',optimizer_steps=8,termination=dict(terminal=True,termination_reason='MAX_STEPS'),
        best_selection_key=None,checkpoint=str(root/'step_000008.pt'),best_checkpoint_hash=file_hash(root/'best.pt'),parameter_hash_after=module_hash_parameters(model))
    atomic_json(root/'result.json',result)
    monkeypatch.setattr(ev,'verify_binding',lambda _:dict(hash='base'))
    monkeypatch.setattr(ev,'read_split',lambda _:dict(hash='split'))
    monkeypatch.setattr(ev,'create',lambda *_:(deepcopy(model),dict(initial_state_hash=initial)))
    return config,root,result


def test_refit_endpoint_requires_exact_last_state_and_scratch_identity(tmp_path,monkeypatch):
    c,root,result=setup_endpoint(tmp_path,monkeypatch)
    model,identity=ev.final_endpoint(c,'M_set')
    assert identity['updates']==8
    # A model with parameters identical but different normalizer/time buffers is not compatible.
    state=torch.load(root/'best.pt',weights_only=True);state['time']+=1
    atomic_torch(root/'best.pt',state);result['best_checkpoint_hash']=file_hash(root/'best.pt');atomic_json(root/'result.json',result)
    with pytest.raises(ValueError,match='buffers'):ev.final_endpoint(c,'M_set')


@pytest.mark.parametrize('change',[dict(optimizer_steps=7),dict(best_selection_key=[.99]),dict(status='STOPPED_AT_BOUNDARY')])
def test_external_eval_cannot_select_incomplete_or_validation_refit(tmp_path,monkeypatch,change):
    c,root,result=setup_endpoint(tmp_path,monkeypatch);atomic_json(root/'result.json',result|change)
    with pytest.raises(ValueError):ev.final_endpoint(c,'M_set')


def test_wrong_checkpoint_hash_is_rejected(tmp_path,monkeypatch):
    c,root,result=setup_endpoint(tmp_path,monkeypatch)
    with open(root/'step_000008.pt','ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='incomplete'):ev.final_endpoint(c,'M_set')


@pytest.mark.parametrize('field',['evaluation_code','deployment_code'])
def test_external_lock_cannot_silently_reuse_changed_execution_code(tmp_path,monkeypatch,field):
    path=Path(ev.__file__)
    record=dict(binding_hash='fixed',models={},evaluation_code=file_hash(path),
                deployment_code=file_hash(path.with_name('deployment.py')))
    record[field]='wrong-source-hash';record['hash']=digest(record)
    atomic_json(tmp_path/'FINAL_EVALUATION_LOCK.json',record)
    monkeypatch.setattr(ev,'verify_binding',lambda _:dict(hash='fixed'))
    with pytest.raises(ValueError,match='code changed'):ev._final_lock(SimpleNamespace(root=tmp_path))
