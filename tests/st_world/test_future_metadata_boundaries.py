"""Synthetic adversarial metadata tests; no real data or teacher execution."""
import json
import os
from pathlib import Path

import pytest

from tests.st_world.test_prepare_future_metadata import setup
from tools.st_world_policy import prepare_future_metadata as f
from starVLA.dataloader import st_world_future_provenance as provenance
from starVLA.dataloader.st_world_targets import atomic_json


@pytest.mark.parametrize('kind',['shard_symlink','root_symlink','ancestor_symlink','fifo','file_hardlink'])
def test_unsafe_metadata_roots_rejected_before_write(tmp_path,kind):
    args,run,p,_=setup(tmp_path); output=Path(args.output)
    outside=tmp_path/'outside'; outside.mkdir()
    if kind=='shard_symlink': (output/'log_shards').symlink_to(outside,target_is_directory=True)
    elif kind=='root_symlink':
        alias=tmp_path/'alias'; alias.symlink_to(output,target_is_directory=True)
        run.plan=str(alias/'PLAN.json')
    elif kind=='ancestor_symlink':
        alias=tmp_path/'alias'; alias.symlink_to(tmp_path,target_is_directory=True)
        run.plan=str(alias/'future/PLAN.json')
    elif kind=='fifo': os.mkfifo(output/'unexpected.pipe')
    else: (outside/'plan_alias').hardlink_to(output/'PLAN.json')
    before=sorted(x.name for x in outside.iterdir())
    with pytest.raises(ValueError): f.build(run)
    assert sorted(x.name for x in outside.iterdir())==before
    assert not (output/'COMPLETE.json').exists()


def test_late_unsafe_parent_rejected_by_writer(tmp_path):
    args,_,p,_=setup(tmp_path); output=Path(args.output)
    budget=f.PreparationBudget(output,p['limits'])
    outside=tmp_path/'outside'; outside.mkdir()
    (output/'log_shards').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError): f.write_json(output/'log_shards/a.json',{'a':1},budget)
    assert list(outside.iterdir())==[]


@pytest.mark.parametrize('kind',['hardlink','symlink','missing','unexpected'])
def test_completed_shard_physical_coverage_is_checked(tmp_path,kind):
    args,run,_,_=setup(tmp_path); f.build(run); out=Path(args.output)
    shard=next((out/'log_shards').glob('*.json'))
    if kind=='hardlink': (tmp_path/'alias').hardlink_to(shard)
    elif kind=='symlink':
        original=tmp_path/'outside.json'; shard.rename(original); shard.symlink_to(original)
    elif kind=='missing': shard.unlink()
    else: (out/'log_shards/extra.json').write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)): f.build(run)


@pytest.mark.parametrize('field,value',[
    ('status','PARTIAL'),('schema','wrong'),('plan_identity','0'*64),('scenes',999),('logs',999),
    ('scenes',True),('target_times_s',[0.,.5,1.,2.]),('original_selection',{}),
    ('teacher_extractions',1),('tensor_disk_bytes',1),('training_admission',True),
    ('request_identity','invalid'),('files',{}),('log_costs',[])])
def test_complete_control_tampering_rejected(tmp_path,field,value):
    args,run,_,_=setup(tmp_path); f.build(run); path=Path(args.output)/'COMPLETE.json'
    done=json.loads(path.read_text()); done[field]=value; atomic_json(path,done)
    with pytest.raises(ValueError): f.build(run)


@pytest.mark.parametrize('kind',['order','path','hash','schema'])
def test_shard_receipt_checked_even_when_outer_hash_is_updated(tmp_path,kind):
    args,run,_,_=setup(tmp_path); f.build(run); out=Path(args.output)
    path=out/'SHARD_RECEIPTS.json'; receipt=json.loads(path.read_text())
    if kind=='order': receipt['shards'].reverse()
    elif kind=='path': receipt['shards'][0]['path']=str(tmp_path/'outside.json')
    elif kind=='hash': receipt['shards'][0]['sha256']='0'*64
    else: receipt['schema']='foreign'
    atomic_json(path,receipt)
    done=json.loads((out/'COMPLETE.json').read_text()); done['files'][path.name]=f.file_sha256(path)
    atomic_json(out/'COMPLETE.json',done)
    with pytest.raises(ValueError): f.build(run)


@pytest.mark.parametrize('kind',['growth','replace'])
def test_ledger_single_bounded_snapshot_rejects_read_mutation(tmp_path,monkeypatch,kind):
    args,run,_,_=setup(tmp_path); f.build(run)
    manifest=json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    path=Path(manifest['source_logs_ledger']['path']); original=Path.open; reads=[]
    class MutatingRead:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): return self
        def __exit__(self,*_): self.stream.close()
        def read(self,n):
            assert n==provenance.MAX_LEDGER_BYTES+1; reads.append(n); data=self.stream.read(n)
            if kind=='growth':
                with original(path,'ab') as stream: stream.write(b' '*provenance.MAX_LEDGER_BYTES)
            else:
                other=path.with_suffix('.replacement')
                with original(other,'wb') as stream: stream.write(data)
                other.replace(path)
            return data
    def patched(p,mode='r',*args,**kwargs):
        stream=original(p,mode,*args,**kwargs)
        return MutatingRead(stream) if p==path and mode=='rb' else stream
    monkeypatch.setattr(Path,'open',patched)
    with pytest.raises(ValueError,match='changed'): provenance.validate_ledger(manifest)
    assert reads==[provenance.MAX_LEDGER_BYTES+1]


def test_ledger_uses_one_bounded_read_and_no_read_text(tmp_path,monkeypatch):
    args,run,_,_=setup(tmp_path); f.build(run)
    manifest=json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    monkeypatch.setattr(Path,'read_text',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('unbounded text read')))
    assert provenance.validate_ledger(manifest)['sha256']==manifest['source_logs_ledger']['sha256']
