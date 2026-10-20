"""Independent offline audit negative tests; no GPU training is performed."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from tools.action_effect.audit_effect_resume import check_subset,compare_states,exact_hash,read_state
from tools.action_effect.common import artifact,digest,write_json


def load_candidate():
    root=Path(os.environ.get('ACTION_EFFECT_QUALIFICATION_REVIEW_SOURCE',Path(__file__).resolve().parents[2]))
    path=root/'tools/action_effect/qualify_effect_resume.py'
    if not path.is_file():pytest.skip('Qualification source is not in this source snapshot')
    spec=importlib.util.spec_from_file_location('tools.action_effect._independent_qualification_candidate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def state(step=9):
    g=torch.Generator().manual_seed(4)
    return {'schema':'action_effect_private_checkpoint_v1','model':{'p':torch.zeros(3)},
        'optimizer':{'state':{0:{'exp_avg':torch.zeros(3),'exp_avg_sq':torch.ones(3),'step':torch.tensor(float(step))}},
                     'param_groups':[{'params':[0]}]},'scheduler':{'last_epoch':step},
        'sampler':g.get_state(),'stateful':{'pair_rng':g.get_state()},'rng':{'cpu':g.get_state(),'cuda':[]},
        'consumed_scenes':step*2,'optimizer_step':step,'run_identity':'test','termination_reason':'RUNNING_CHECKPOINT',
        'terminal':False,'history':[{'optimizer_step':i,'losses':{'value':.2},'gradient_norm':.1,'lr':.0001,
            'consumed_scenes':i*2,'seconds':.01} for i in range(1,step+1)]}


def test_exact_typed_hash_rejects_dtype_signed_zero_nonfinite_and_changed_moment():
    assert exact_hash(torch.tensor([0.]))!=exact_hash(torch.tensor([-0.]))
    assert exact_hash(torch.ones(1))!=exact_hash(torch.ones(1,dtype=torch.float64))
    with pytest.raises(ValueError):exact_hash(np.array([float('nan')]))
    a=state();b=deepcopy(a);assert compare_states(a,b,9)['exact']
    b['optimizer']['state'][0]['exp_avg'][0]=.1
    assert not compare_states(a,b,9)['exact']
    b=deepcopy(a);b['history'][8]['seconds']=999.
    assert compare_states(a,b,9)['exact'] # Timing is deliberately not model state.
    b['history'][8]['losses']['value']+=.1
    assert not compare_states(a,b,9)['exact']


def test_expected_provenance_is_independent_of_later_review_source_revision():
    expected={'source_commit':'actually-executed-old-source','seed':42,'recipe':{'max_updates':1000}}
    check_subset({**expected,'review_tool_source_commit':'newer-review-only-source'},expected)
    for field,value in [('source_commit','wrong'),('seed',43),('recipe',{'max_updates':16})]:
        with pytest.raises(ValueError,match=field):check_subset({**expected,field:value},expected)


def test_checkpoint_name_to_Adam_mapping_and_consumed_boundary(tmp_path):
    identity={'private_names':['p'],'recipe':{'batch_scenes':2}};write_json(tmp_path/'IDENTITY.json',identity)
    s=state();s['run_identity']=digest(identity)
    def save():
        p=tmp_path/'step_000009.pt';torch.save(s,p)
        receipt={'checkpoint':artifact(p),'run_identity':digest(identity),'optimizer_step':9}
        (p.with_suffix('.COMPLETE.json')).write_text(__import__('json').dumps(receipt));return p
    read_state(save(),digest(identity),9)
    s['optimizer']['state'][0]['exp_avg']=torch.zeros(4)
    with pytest.raises(ValueError,match='Adam moment'):read_state(save(),digest(identity),9)
    s=state();s['run_identity']=digest(identity);s['consumed_scenes']+=1
    with pytest.raises(ValueError,match='Consumed data cursor'):read_state(save(),digest(identity),9)


def test_candidate_refuses_foreign_completed_receipt_namespace(tmp_path):
    q=load_candidate();old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    for root,label in ((old,'old'),(new,'new')):
        write_json(root/'IDENTITY.json',{'scope':label});write_json(root/'QUALIFICATION_CONFIG.json',{'scope':label})
    cp=old/'fixture.pt';cp.write_bytes(b'unit-only, not actual training')
    result={'status':'PASS','exact_resume_this_runtime':True,'checkpoints':{'x':artifact(cp)},
        'identity':artifact(old/'IDENTITY.json'),'qualification_config':artifact(old/'QUALIFICATION_CONFIG.json'),
        'commands':[],'new_updates_this_invocation':32}
    write_json(old/'RESULT.json',result)
    complete={'status':'PASS','identity':artifact(old/'IDENTITY.json'),'result':artifact(old/'RESULT.json')}
    write_json(old/'COMPLETE.json',complete);write_json(new/'COMPLETE.json',complete)
    assert q.reuse_complete(old)['new_updates_this_invocation']==0
    with pytest.raises(ValueError,match='output directory'):q.reuse_complete(new)


@pytest.mark.parametrize('field,value',[('source_commit','foreign'),('arm','absolute_score'),('seed',43),
    ('data_artifact',{'sha256':'other'}),('context_artifact',{'sha256':'other'}),('max_updates',16),
    ('recipe',{'max_updates':1000,'warmup_updates':1})])
def test_candidate_refuses_completed_training_from_another_experiment(tmp_path,field,value):
    q=load_candidate();expected={'source_commit':'actual','arm':'delta_feedback','seed':42,
        'data_artifact':{'sha256':'data'},'context_artifact':{'sha256':'ctx'},'max_updates':1000,
        'recipe':{'max_updates':1000,'warmup_updates':50}}
    write_json(tmp_path/'IDENTITY.json',{**expected,field:value})
    with pytest.raises(ValueError,match='provenance mismatch'):
        q.validate_training_identity(tmp_path,expected)
