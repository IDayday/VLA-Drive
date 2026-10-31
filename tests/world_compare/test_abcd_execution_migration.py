"""Execution-only exact native restore, never a fresh optimizer/RNG phase."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from test_recovery import bundle, identity, state, advance
from starVLA.rl.qwen_world_abcd_research import checkpoint as cp
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from starVLA.rl.simwam_recipe.lineage import SCIENTIFIC_KEYS, verify_migration
from tools.qwen_world_abcd_research.train import seal_run_identity


def migration_identity():
    old=identity()
    for key in SCIENTIFIC_KEYS:
        old.setdefault(key, {'fixture':key})
    old['source_files']={'train.py':'old_execution_source'}
    old['IL_parent']={'export_parent':{'source_migration':{'differences':{'adapter':('old','new')}}}}
    return seal_run_identity(old)


def make_receipt(tmp_path, old, new, parent):
    evidence=tmp_path/'CPU_PASS.json';evidence.write_text(json.dumps({'status':'PASS','fixture':True}))
    value=dict(schema=cp.MIGRATION_SCHEMA,old_identity=old,new_identity=new,
        parent_files_sha256=file_sha(Path(parent)/'files.json'),
        evidence=[dict(path=str(evidence),sha256=file_sha(evidence))])
    value['sha256']=fingerprint(value)
    path=tmp_path/'migration.json';path.write_text(json.dumps(value))
    return path,value


def test_json_native_sealing_preserves_hash_and_requires_real_source_migration():
    old=migration_identity()
    assert old==json.loads(json.dumps(old))
    raw=copy.deepcopy(old)
    raw['IL_parent']['export_parent']['source_migration']['differences']['adapter']=('old','new')
    assert raw!=old and fingerprint(raw)==fingerprint(old)
    assert seal_run_identity(raw)==old
    raw['source_files']['train.py']='new_execution_source'
    new=seal_run_identity(raw)
    assert new['sha256']!=old['sha256']
    assert new['scientific_identity']==old['scientific_identity']
    assert new['execution_identity']!=old['execution_identity']


def test_native_midinner_migration_preserves_Adam_all_RNG_cursor_and_successor(tmp_path,monkeypatch):
    random.seed(41);np.random.seed(42);torch.manual_seed(43)
    old=migration_identity();a,o,s,g=bundle()
    t,r=advance(a,o,s,g,state(),None,old,2)
    monkeypatch.setattr(cp,'EXECUTION_IDENTITY',old['execution_identity'])
    monkeypatch.setattr(cp,'MIGRATION_RECEIPT',None)
    parent=cp.save(tmp_path,a,o,s,g,t,r,old)
    # The original run/scientific identity remains the checkpoint identity.
    new=copy.deepcopy(old);new['source_files']['train.py']='new_execution_source'
    new=seal_run_identity(new)
    path,receipt=make_receipt(tmp_path,old,new,parent)
    assert verify_migration(path,old,new,parent)==receipt
    expected_t,expected_r=advance(a,o,s,g,copy.deepcopy(t),copy.deepcopy(r),old,8)
    expected_weights=a.lora_A.detach().clone();expected_adam=tensor_digest(o.state_dict())
    expected_scheduler=copy.deepcopy(s.state_dict());expected_g=g.get_state().clone()
    # Perturb all global RNGs and adapter state before exact native restore.
    a2,o2,s2,g2=bundle();random.seed(99);np.random.seed(99);torch.manual_seed(99);g2.manual_seed(99)
    monkeypatch.setattr(cp,'EXECUTION_IDENTITY',new['execution_identity'])
    monkeypatch.setattr(cp,'MIGRATION_RECEIPT',receipt)
    t2,r2=cp.restore(parent,a2,o2,s2,g2,old,migration=receipt)
    assert t2['next_inner']==2 and t2['scene_cursor']==t['scene_cursor']
    actual_rng=(random.random(),np.random.rand(),torch.rand(3))
    own=torch.load(Path(parent)/'rank0.pt',weights_only=False)['rng']
    random.setstate(own['python']);np.random.set_state(own['numpy']);torch.set_rng_state(own['torch'])
    expected_rng=(random.random(),np.random.rand(),torch.rand(3))
    assert actual_rng[:2]==expected_rng[:2] and torch.equal(actual_rng[2],expected_rng[2])
    t2,r2=advance(a2,o2,s2,g2,t2,r2,old,4)
    child=cp.save(tmp_path,a2,o2,s2,g2,t2,r2,old)
    transaction=json.loads((Path(child)/'transaction.json').read_text())
    assert transaction['execution_identity']==new['execution_identity']
    assert transaction['migration_receipt_sha256']==receipt['sha256']
    assert json.loads((Path(child)/'identity.json').read_text())==old
    assert verify_migration(path,old,new,child)==receipt
    a3,o3,s3,g3=bundle()
    t3,r3=cp.restore(child,a3,o3,s3,g3,old,migration=receipt)
    actual_t,actual_r=advance(a3,o3,s3,g3,t3,r3,old,8)
    assert actual_t==expected_t and actual_r is expected_r is None
    assert torch.equal(a3.lora_A,expected_weights)
    assert tensor_digest(o3.state_dict())==expected_adam and s3.state_dict()==expected_scheduler
    assert torch.equal(g3.get_state(),expected_g)
    # Exact receipt cannot authorize a changed recipe/reference or new source.
    bad=copy.deepcopy(receipt);bad['new_identity']['IL_parent']['export_parent']['other_parent']='forbidden'
    payload=bad['new_identity'];payload.pop('sha256');payload['sha256']=fingerprint(payload)
    bad.pop('sha256');bad['sha256']=fingerprint(bad)
    with torch.no_grad():a3.lora_A.fill_(99)
    with pytest.raises(RuntimeError,match='nonexecution'):
        cp.restore(parent,a3,o3,s3,g3,old,migration=bad)
    assert torch.equal(a3.lora_A,torch.full((3,),99.))
    bad=copy.deepcopy(receipt);bad['evidence']=[];bad.pop('sha256');bad['sha256']=fingerprint(bad)
    with pytest.raises(RuntimeError,match='nonempty'):
        cp.restore(parent,a3,o3,s3,g3,old,migration=bad)
