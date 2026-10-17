"""CPU plan/accounting fixtures, not a real teacher qualification."""
from copy import deepcopy
import json

import pytest
import torch

from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheReader
from tools.st_world_policy import qualify_fixed_batch_teacher as q
from test_qualify_ondemand_teacher import assets


def test_plan_is_metadata_only_with_padding_cost_and_fixed_backend(tmp_path,monkeypatch):
    args = assets(tmp_path)
    def forbidden(*a,**k): raise AssertionError('Plan must not discover CUDA, load weights or read tensors')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'device_count',forbidden)
    monkeypatch.setattr(TeacherCacheReader,'read_image',forbidden)
    p=q.make_plan(*args)
    assert p['backend']['deterministic_algorithms'] is True
    assert p['batch_policy']['compute_batch_images']==8
    assert p['budget']['max_image_forwards']==p['budget']['max_batch_calls']*8
    assert p['budget']['ram_peak_bytes']==8*q.shared.GiB
    assert p['budget']['tensor_disk_bytes']==0
    assert 'legacy_reference_compatibility' in p['gate']
    assert p['gate']['numeric_tolerance'] is None
    q.validate_plan(p)


@pytest.mark.parametrize('section,key,value',[
    ('backend','deterministic_algorithms',False),('budget','max_seconds',901),
    ('budget','max_image_forwards',99999),('batch_policy','compute_batch_images',4)])
def test_changed_plan_rejected_before_cuda_or_output(tmp_path,monkeypatch,section,key,value):
    p=q.make_plan(*assets(tmp_path)); p[section][key]=value
    p['identity']=identity_hash({k:v for k,v in p.items() if k!='identity'})
    path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('No CUDA')))
    with pytest.raises(ValueError):q.run(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_guard_budgets_all_compute_slots_even_for_one_real_image(monkeypatch):
    class E:
        execution_identity='a'*64;contract_identity='b'*64
        def compute_batch(self,records,**kwargs):return len(records)
    b=q.shared.Budget({'max_seconds':900,'ram_peak_bytes':100*q.shared.GiB,
                       'max_image_forwards':8,'max_batch_calls':1})
    monkeypatch.setattr(q.shared,'rng_digest',lambda device:{'fixture':True})
    e=q.GuardedFixedExecutor(E(),b,'cpu')
    assert e.compute_batch([{'fixture':True}])==1
    assert b.images==8 and b.calls==1 and e.rng_checks==1
    with pytest.raises(RuntimeError,match='budget'):e.compute_batch([{}])


def test_fixed_qualification_cli_has_no_launch_or_relaxation_flag():
    p=q.parser()
    assert p.parse_args(['run','--plan','a','--output','b']).command=='run'
    with pytest.raises(SystemExit):p.parse_args(['run','--plan','a','--output','b','--tolerance','.1'])
