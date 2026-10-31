import copy
import json
from pathlib import Path
import pytest
from starVLA.rl.qwen_world_abcd_research.identity import canonical_json_identity, validate_execution_receipt
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha


def test_parent_tuple_migration_keeps_exact_persisted_identity():
    value={'IL_parent':{'export_parent':{'source_migration':{
        'differences':{'adapter.py':('old_sha','new_sha')}}}}}
    persisted=json.loads(json.dumps(value))
    assert persisted!=value  # The real old resume guard rejected this pair.
    assert fingerprint(persisted)==fingerprint(value)
    assert canonical_json_identity(value)==persisted
    assert canonical_json_identity(persisted)==persisted


@pytest.mark.parametrize('field,value',[('learning_rate',1e-4),('clip',.02),('parent','different')])
def test_real_identity_change_remains_visible(field,value):
    old={'learning_rate':5e-5,'clip':.005,'parent':'fixed'}
    new=copy.deepcopy(old);new[field]=value
    assert canonical_json_identity(old)!=canonical_json_identity(new)
    assert fingerprint(old)!=fingerprint(new)


def test_nonfinite_identity_is_rejected():
    with pytest.raises(ValueError):canonical_json_identity({'bad':float('nan')})


def receipt_bundle(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();(parent/'files.json').write_text('{}')
    (parent/'transaction.json').write_text('{}')
    evidence=tmp_path/'test_evidence.json';evidence.write_text('{"status":"PASS"}')
    old=dict(recipe={'lr':5e-5},parent='same',source_files={'a':'old'},execution_identity={'sha256':'old'},sha256='old_identity')
    new=copy.deepcopy(old);new.update(source_files={'a':'new'},execution_identity={'sha256':'new'},sha256='new_identity')
    receipt=dict(old_identity=old,new_identity=new,parent_files_sha256=file_sha(parent/'files.json'),
                 evidence=[dict(path=str(evidence),sha256=file_sha(evidence))])
    receipt['sha256']=fingerprint(receipt)
    return parent,evidence,old,new,receipt


def test_explicit_same_science_parent_and_descendant_receipt(tmp_path):
    parent,proof,old,new,receipt=receipt_bundle(tmp_path)
    validate_execution_receipt(receipt,old,new['execution_identity'],parent)
    (parent/'files.json').write_text('{"next":"inventory"}')
    with pytest.raises(ValueError,match='neither audited'):validate_execution_receipt(receipt,old,new['execution_identity'],parent)
    (parent/'transaction.json').write_text(json.dumps(dict(migration_receipt_sha256=receipt['sha256'],execution_identity=new['execution_identity'])))
    validate_execution_receipt(receipt,old,new['execution_identity'],parent)


@pytest.mark.parametrize('bad',['science','evidence','receipt','execution'])
def test_execution_receipt_rejects_real_change(tmp_path,bad):
    parent,proof,old,new,receipt=receipt_bundle(tmp_path)
    execution=new['execution_identity']
    if bad=='science':
        receipt['new_identity']['recipe']['lr']=1e-4
        receipt.pop('sha256');receipt['sha256']=fingerprint(receipt)
    elif bad=='evidence':proof.write_text('{"status":"FAIL"}')
    elif bad=='receipt':receipt['parent_files_sha256']='wrong'
    else:execution={'sha256':'other'}
    with pytest.raises(ValueError):validate_execution_receipt(receipt,old,execution,parent)


@pytest.mark.parametrize('inner',[0,1,2,3])
def test_migration_restores_identical_Adam_RNG_and_pending_chain(tmp_path,monkeypatch,inner):
    import random
    import numpy as np
    import torch
    from starVLA.rl.qwen_world_abcd_research import checkpoint as cp
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    old=dict(world=1,sha256='old_identity',recipe={'lr':.01},execution_identity={'sha256':'old'},source_files={'a':'old'})
    new=copy.deepcopy(old);new.update(sha256='new_identity',execution_identity={'sha256':'new'},source_files={'a':'new'})
    class Hooks:
        def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None):return dict(state)
        def validate_ranks(self,summaries):return summaries
    hooks=Hooks();actor=torch.nn.Linear(2,1);actor.device=torch.device('cpu')
    opt=torch.optim.AdamW(actor.parameters(),lr=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    actor(torch.ones(1,2)).sum().backward();opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
    generator=torch.Generator().manual_seed(42)
    state=dict(phase='READY'if inner==0 else'INNER_COMMITTED',updates=4+inner,next_inner=inner)
    chain=None if inner==0 else dict(chain=torch.randn(8,11,8,3),old_logp=torch.randn(8,3),advantages=torch.randn(8))
    monkeypatch.setattr(cp,'EXECUTION_IDENTITY',old['execution_identity'])
    path=Path(cp.save(tmp_path,actor,opt,scheduler,generator,state,chain,old,runtime=hooks))
    proof=tmp_path/'evidence.json';proof.write_text('{"status":"PASS"}')
    receipt=dict(old_identity=old,new_identity=new,parent_files_sha256=file_sha(path/'files.json'),
                 evidence=[dict(path=str(proof),sha256=file_sha(proof))])
    receipt['sha256']=fingerprint(receipt)
    outputs=[]
    for migration in(None,receipt):
        with torch.no_grad():
            for p in actor.parameters():p.add_(10)
        monkeypatch.setattr(cp,'EXECUTION_IDENTITY',new['execution_identity'])
        restored,pending=cp.restore(path,actor,opt,scheduler,generator,old,migration=migration,runtime=hooks)
        outputs.append((restored,pending,copy.deepcopy(opt.state_dict()),copy.deepcopy(scheduler.state_dict()),
            {n:p.detach().clone()for n,p in actor.named_parameters()},torch.rand(5),torch.rand(5,generator=generator),random.random(),np.random.rand()))
    def same(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    same(*outputs)
