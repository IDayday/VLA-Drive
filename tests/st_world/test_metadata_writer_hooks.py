"""Actual registered CLI parsers + tiny synthetic metadata; never real data."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from tools.st_world_policy import metadata_storage as storage
from tools.st_world_policy import prepare_future_metadata as future
from tools.st_world_policy import prepare_current_teacher as current
from starVLA.dataloader import st_world_metadata_index as index
from tests.st_world.test_metadata_storage_guard import fixture, write, update_registration, ledger
from tests.st_world.test_prepare_future_metadata import setup
from tests.st_world.test_ondemand_teacher import fixture as request_fixture


def registered_cli(tmp_path,kind):
    f=fixture(tmp_path,nested=kind=='future_preparation')
    f.reg.update(cwd=str(Path.cwd()),callback_protocol='st_metadata_storage_check_v1',
        file_patterns=[r'METADATA_ADMISSION\.json',r'BUILD\.lock',r'COMPLETE\.json(?:\.[0-9a-f]+\.tmp)?',
            r'metadata\.(?:sqlite(?:-journal)?|[0-9a-f]+\.partial\.sqlite(?:-journal)?|orphan\.[0-9a-f]+\.sqlite)'])
    data=tmp_path/'data'; data.mkdir()
    if kind=='future_preparation':
        a,_,_,_=setup(data); a.output=str(f.root); a.max_metadata_bytes=3*storage.MIB
        plan=future.plan(a)
        registration=future.signed({'schema':future.REG_SCHEMA,'plan_identity':plan['identity'],
            'plan_sha256':future.file_sha256(f.root/'PLAN.json'),'phase':'data_preparation',
            'approved_scope':plan['scope'],'gpu_authorized':False,
            'teacher_extraction_authorized':False,'training_authorized':False})
        prepare_reg=tmp_path/'prepare_registration.json'; prepare_sha=write(prepare_reg,registration)
        module='tools.st_world_policy.prepare_future_metadata'
        argv=['build','--plan',str(f.root/'PLAN.json'),'--registration',str(prepare_reg),
              '--registration-sha256',prepare_sha]
        f.reg.update(directory_patterns=[r'log_shards'],preexisting_pins=[
            {'relative_path':name,'sha256':future.file_sha256(f.root/name)} for name in ('PLAN.json','PLAN.lock')])
        f.reg['file_patterns'] += [r'PLAN\.(?:json|lock)',
            r'(?:TEACHER_REQUESTS|RAW_LOG_PROVENANCE|SHARD_RECEIPTS|COMPLETE)\.json(?:\.[0-9a-f]+\.partial)?',
            r'log_shards/[0-9a-f]{64}\.json(?:\.[0-9a-f]+\.partial)?']
    else:
        contract,manifest=request_fixture(data)
        request=data/'requests.json'; write(request,manifest)
        receipt=data/'contract.json'
        receipt_sha=write(receipt,{'schema':'st_teacher_cache_v1','contract':contract.to_dict()})
        module='tools.st_world_policy.index_teacher_metadata'
        argv=['build','--requests',str(request),'--expected-request-identity',manifest['identity'],
              '--contract-receipt',str(receipt),'--contract-receipt-sha256',receipt_sha,
              '--output',str(f.root),'--max-scenes','1','--max-disk-bytes',str(3*storage.MIB)]
    module_value=importlib.import_module(module)
    f.reg['command']=[sys.executable,'-m',module,*argv]
    f.reg['source_pins']=[{'path':str(Path(importlib.import_module(name).__file__).resolve()),
        'sha256':future.file_sha256(importlib.import_module(name).__file__)}
        for name in (module,*module_value.METADATA_STORAGE_CALLBACK_DEPENDENCIES)]
    update_registration(f); storage.migrate_policy(**f.kwargs)
    return f


def run_cli(f,monkeypatch,capsys):
    args=['metadata_storage','run']
    for key,value in f.guard.items(): args += ['--'+key.replace('_','-'),str(value)]
    monkeypatch.setattr(sys,'argv',args)
    storage.main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize('kind',['current_index','future_preparation'])
def test_real_registered_cli_callbacks_and_same_ledger_resume(tmp_path,monkeypatch,capsys,kind):
    f=registered_cli(tmp_path,kind); before=ledger(f)
    result=run_cli(f,monkeypatch,capsys)
    assert result['status']=='GUARDED_METADATA_RETURN'
    assert result['writer_callbacks']>0 and result['writer_commit_callbacks']>=3
    assert result['artifact']['tensor_disk_bytes']==0
    if kind=='future_preparation': assert result['artifact']['status']=='COMPLETE'
    complete=(f.root/'COMPLETE.json').read_bytes()
    after=ledger(f)
    assert len(after['claims'])==len(before['claims'])+1
    again=run_cli(f,monkeypatch,capsys)
    assert again['status']=='GUARDED_METADATA_RETURN'
    assert (f.root/'COMPLETE.json').read_bytes()==complete
    resumed=ledger(f)['claims']
    assert set(resumed)==set(after['claims'])
    for root,claim in after['claims'].items():
        assert {k:v for k,v in resumed[root].items() if k!='observed_bytes'}=={
            k:v for k,v in claim.items() if k!='observed_bytes'}
        assert resumed[root]['observed_bytes']>=claim['observed_bytes']


@pytest.mark.parametrize('kind',['current_index','future_preparation'])
def test_guard_rejection_precedes_complete_publication(tmp_path,monkeypatch,capsys,kind):
    f=registered_cli(tmp_path,kind); original=storage.MetadataWriteContext.check; observed=[]
    def stop(self,*,before_commit=False,path=None):
        if path is not None: observed.append(str(path))
        if before_commit and path is not None and Path(path).name=='COMPLETE.json':
            self.failed='fixture boundary rejection'
            raise storage.b.StorageBudgetError('fixture boundary rejection')
        return original(self,before_commit=before_commit,path=path)
    monkeypatch.setattr(storage.MetadataWriteContext,'check',stop)
    with pytest.raises(SystemExit) as exc: run_cli(f,monkeypatch,capsys)
    assert exc.value.code==2 and not (f.root/'COMPLETE.json').exists()
    assert any('.partial' in path for path in observed)
    assert str(f.root) in ledger(f)['claims']  # Reservation survives failure.


@pytest.mark.parametrize('kind',['current_index','future_preparation'])
def test_actual_creation_path_is_checked_before_any_new_file(tmp_path,monkeypatch,capsys,kind):
    f=registered_cli(tmp_path,kind); original=storage.MetadataWriteContext.check; targets=[]
    def stop(self,*,before_commit=False,path=None):
        if before_commit and path is not None and (
                (kind=='current_index' and str(path).endswith('.partial.sqlite')) or
                (kind=='future_preparation' and 'log_shards' in Path(path).parts)):
            targets.append(Path(path)); assert not Path(path).exists()
            self.failed='fixture creation rejected'
            raise storage.b.StorageBudgetError('fixture creation rejected')
        return original(self,before_commit=before_commit,path=path)
    monkeypatch.setattr(storage.MetadataWriteContext,'check',stop)
    with pytest.raises(SystemExit) as exc: run_cli(f,monkeypatch,capsys)
    assert exc.value.code==2 and targets and all(not p.exists() for p in targets)
    assert not (f.root/'COMPLETE.json').exists()


def test_source_population_optional_callback_preserves_exact_output(tmp_path):
    a,_,_,_=setup(tmp_path)
    reference=current.source_population(a.student_root,a.partition,a.split)
    calls=[]
    actual=current.source_population(a.student_root,a.partition,a.split,resource_check=lambda:calls.append(True))
    assert actual==reference and len(calls)>12
    def stop(): raise RuntimeError('source metadata budget')
    with pytest.raises(RuntimeError,match='budget'):
        current.source_population(a.student_root,a.partition,a.split,resource_check=stop)


def test_long_chunk_helpers_call_hook_and_publications_never_throttle(tmp_path,monkeypatch):
    clock=[10.]; calls=[]
    monkeypatch.setattr(index,'_LAST_STORAGE_CHECK',0.)
    monkeypatch.setattr(index.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(index,'check_active_metadata_storage',lambda **kw:calls.append(dict(kw)))
    path=tmp_path/'source'; path.write_bytes(b'x'*(3*index.MiB+3))
    original=Path.open
    class SlowRead:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): return self
        def __exit__(self,*_): self.stream.close()
        def read(self,n):
            assert n==index.MiB; clock[0]+=1.1; return self.stream.read(n)
    monkeypatch.setattr(Path,'open',lambda p,*a,**k:SlowRead(original(p,*a,**k)))
    assert index.file_sha256(path)==hashlib.sha256(b'x'*(3*index.MiB+3)).hexdigest()
    assert len(calls)>=5
    before=len(calls)
    index._storage_check(before_commit=True,path=tmp_path/'one')
    index._storage_check(before_commit=True,path=tmp_path/'two')
    assert len(calls)==before+2 and all(x['before_commit'] for x in calls[-2:])
