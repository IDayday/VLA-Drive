"""CPU metadata/protocol tests only, not real-teacher numerical evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheReader,TeacherCacheWriter
from tools.st_world_policy import qualify_ondemand_teacher as q
from test_ondemand_teacher import fixture,sign,z_for_key


def assets(tmp_path):
    c,m=fixture(tmp_path,missing_source=True)
    for name,manifest in [('reference',m),('current',deepcopy(m))]:
        if name=='current':manifest['times_s']=[0.];manifest['scenes'][0]['frames']=manifest['scenes'][0]['frames'][:1];sign(manifest)
        w=TeacherCacheWriter(tmp_path/name,c,manifest,images_per_shard=2,
            execution={'batch_size':8,'teacher':{'actual':'fixture'},'transform':{'actual':'fixture'}})
        for shard in range(w.shard_count):
            z=torch.stack([z_for_key(r['key']) for r in w.rows(shard)])
            w.commit_shard(shard,z,torch.ones(z.shape[:-1],dtype=torch.bool))
        w.finalize()
    requests=tmp_path/'requests.json';requests.write_text(json.dumps(m))
    config=tmp_path/'resolved_build.json';config.write_text(json.dumps({'command':'build','variant':'SP2',
        'storage_dtype':'float32','batch_size':8,'output':str(tmp_path/'reference'),'teacher_root':'/not-loaded/fixture'}))
    return requests,tmp_path/'reference',tmp_path/'current',config


def test_plan_no_cuda_weights_or_feature_reads(tmp_path,monkeypatch):
    args=assets(tmp_path)
    def forbidden(*a,**k):raise AssertionError('plan must be CPU metadata only')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'device_count',forbidden)
    monkeypatch.setattr(TeacherCacheReader,'read_image',forbidden)
    p=q.make_plan(*args)
    assert p['scenes']==1 and p['current_unique_images']==1 and p['future_unique_images']==3
    assert p['true_absent_views']==3 and p['budget']['tensor_disk_bytes']==0
    assert p['budget']['max_image_forwards']==4+3+1+6
    assert p['gate']['numeric_tolerance'] is None
    assert p['teacher_root']=='/not-loaded/fixture'


@pytest.mark.parametrize('field,value', [('command','verify'),('batch_size',1),('storage_dtype','float16'),('teacher_root',None)])
def test_plan_refuses_wrong_actual_build_provenance(tmp_path,field,value):
    args=assets(tmp_path);d=json.loads(args[3].read_text());d[field]=value;args[3].write_text(json.dumps(d))
    with pytest.raises(ValueError,match='resolved_config'):q.make_plan(*args)


def test_plan_refuses_unbudgeted_large_identity_before_reader(tmp_path,monkeypatch):
    args=assets(tmp_path)
    p=args[1]/'identity.json'
    with p.open('wb') as f:f.truncate(32*q.MiB+1)
    with pytest.raises(ValueError,match='Metadata exceeds'):q.make_plan(*args)


def test_plan_changed_input_and_budget_rejected_before_cuda(tmp_path,monkeypatch):
    args=assets(tmp_path);p=q.make_plan(*args);plan=tmp_path/'plan.json';plan.write_text(json.dumps(p))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('No GPU access')))
    p['budget']['max_image_forwards']+=100;p['identity']=identity_hash({k:v for k,v in p.items() if k!='identity'})
    plan.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='protocol or budget'):q.run(plan,tmp_path/'output')
    assert not (tmp_path/'output').exists()
    p=q.make_plan(*args);plan.write_text(json.dumps(p));args[3].write_text('{}')
    with pytest.raises(ValueError,match='Pinned qualification input'):q.run(plan,tmp_path/'output')


def test_full_tensor_difference_and_no_tolerance_passthrough():
    reference=torch.tensor([0.,1.,-2.],dtype=torch.float32)
    same=q.difference(reference,reference.clone())
    assert same['bitwise_equal'] and same['max_abs']==0
    changed=reference.clone();changed[1]=torch.nextafter(changed[1],torch.tensor(float('inf')))
    diff=q.difference(changed,reference)
    assert not diff['bitwise_equal'] and diff['max_abs']>0 and diff['elements']==3
    total=q.summarize([same,diff])
    assert not total['bitwise_equal'] and total['elements']==6 and total['comparisons']==2
    with pytest.raises(ValueError):q.difference(torch.tensor([float('nan')]),torch.zeros(1))


def test_metadata_atomic_staging_cap(tmp_path):
    path=tmp_path/'result.json';q.bounded_json(path,{'a':1},100)
    before=path.read_bytes()
    with pytest.raises(RuntimeError,match='staging'):q.bounded_json(path,{'long':'x'*90},100)
    assert path.read_bytes()==before


def test_runtime_scope_and_forward_budget():
    with pytest.raises(ValueError,match='900'):q.make_plan(None,None,None,None,max_seconds=901)
    b=q.Budget({'max_seconds':900,'ram_peak_bytes':100*q.GiB,'max_image_forwards':3,'max_batch_calls':2})
    b.admit(2)
    with pytest.raises(RuntimeError,match='forward budget'):b.admit(2)
    assert b.images==2 and b.calls==1


def test_cli_help_is_real_and_has_only_plan_and_run():
    p=q.parser()
    args=p.parse_args(['run','--plan','x','--output','y'])
    assert args.command=='run'
    with pytest.raises(SystemExit):p.parse_args(['run','--plan','x','--output','y','--skip-review'])
