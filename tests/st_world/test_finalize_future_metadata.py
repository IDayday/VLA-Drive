"""Tiny synthetic shards only; no real dataset, teacher, GPU or raw recompute."""
from copy import deepcopy
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import zlib

import pytest

from tools.st_world_policy import finalize_future_metadata as final
from tools.st_world_policy import prepare_future_metadata as f
from tools.st_world_policy import metadata_storage as storage
from tests.st_world.test_prepare_future_metadata import setup
from tests.st_world.test_metadata_storage_guard import fixture, write, update_registration


def make_case(tmp_path, monkeypatch, output=None, full=False):
    data = tmp_path/'data'; data.mkdir()
    a, old_args, plan, original = setup(data)
    if full:
        a.scope='full_population'; a.output=str(data/'future_full'); plan=f.plan(a)
        original=f.signed({'schema':f.REG_SCHEMA,'plan_identity':plan['identity'],
            'plan_sha256':f.file_sha256(Path(a.output)/'PLAN.json'),'phase':'data_preparation',
            'approved_scope':plan['scope'],'gpu_authorized':False,'teacher_extraction_authorized':False,'training_authorized':False})
        path=tmp_path/'ORIGINAL_FULL_REGISTRATION.json'; write(path,original)
        old_args=SimpleNamespace(plan=str(Path(a.output)/'PLAN.json'),registration=str(path),registration_sha256=f.file_sha256(path))
    # Simulate the real failure window: all shards and ledger committed, no requests/COMPLETE.
    with monkeypatch.context() as m:
        def fail(*_): raise RuntimeError('synthetic original finalization failure')
        m.setattr(f, 'write_row_document', fail)
        with pytest.raises(RuntimeError, match='synthetic'): f.build(old_args)
    source = Path(a.output)
    terminal = tmp_path/'CONTROL_BUILD_RESULT.json'
    output = output or tmp_path/'finalized'; output.mkdir(parents=True,exist_ok=True)
    pin = lambda path: {'path':str(path), 'sha256':f.file_sha256(path)}
    # Synthetic version of the actual FAILED_NO_AUTO_RETRY supervisor receipt chain.
    controller=tmp_path/'old_controller.py'; controller.write_text('# synthetic controller, never executed\n')
    storage_path=tmp_path/'OLD_STORAGE_REGISTRATION.json'
    write(storage_path,{'schema':'st_metadata_build_registration_v1','writer_kind':'future_preparation',
        'output_root':str(source),'cwd':str(Path.cwd()),'command':[sys.executable,'-m',
        'tools.st_world_policy.prepare_future_metadata','build','--plan',str(source/'PLAN.json'),
        '--registration',old_args.registration,'--registration-sha256',old_args.registration_sha256]})
    prep_path=tmp_path/'OLD_PREPARATION.json'
    write(prep_path,{'schema':'st_full_future_metadata_controls_v1','source_inputs':{
        'roles':{'train':{'plan':pin(source/'PLAN.json')}}},'registrations':{'train':pin(storage_path)}})
    admission_path=tmp_path/'OLD_ADMISSION.json'
    write(admission_path,{'schema':'st_full_future_build_resource_admission_v1','status':'PASS_BOUNDED_TRAIN_BUILD_ONLY',
        'role':'train','controller':pin(controller),'preparation':pin(prep_path),'registration':pin(storage_path),
        'limits':{'automatic_retry':False},'attempt':1,'source_runtime':{'path':str(Path.cwd())},
        'exact_population':{'root':str(source),'scenes':plan['selected_scene_count'],'logs':len(plan['selected_logs'])}})
    write(terminal,{'schema':'st_full_future_build_invocation_v1','status':'FAILED_NO_AUTO_RETRY',
        'automatic_retry':False,'training_authorized':False,'scientific_result':False,'new_teacher_tensor_bytes':0,
        'role':'train','resources':{'cleanup':{'live_owned_remaining':0},'returncode':-15},
        'admission':pin(admission_path),'preparation':pin(prep_path),'controller':pin(controller),
        'limits':{'automatic_retry':False},'attempt':1})
    limits = dict(plan['limits'], max_rss_bytes=2*final.GiB, max_source_read_bytes=8*final.MiB,
                  max_metadata_bytes=3*final.MiB, max_seconds=60.)
    reg = f.signed({'schema':final.REG_SCHEMA, 'phase':'data_preparation', 'output_root':str(output),
        'source_root':str(source), 'source_code_root':str(Path.cwd()),
        'source_plan':pin(source/'PLAN.json'), 'source_registration':pin(Path(old_args.registration)),
        'source_terminal':pin(terminal), 'source_ledger':pin(source/'RAW_LOG_PROVENANCE.json'),
        'source_shard_receipts':None, 'source_shards':final.shard_census(plan, source)[0], 'limits':limits,
        'implementation':[pin(Path(importlib.import_module(name).__file__).resolve()) for name in
            ('tools.st_world_policy.finalize_future_metadata',*final.METADATA_STORAGE_CALLBACK_DEPENDENCIES)],
        'max_cache_bytes':final.MiB, 'max_decoded_shard_bytes':4*final.MiB, 'max_row_bytes':64*1024,
        'gpu_authorized':False, 'teacher_extraction_authorized':False, 'training_authorized':False})
    reg_path = tmp_path/'FINALIZATION_REGISTRATION.json'; write(reg_path, reg)
    args = SimpleNamespace(command='finalize-existing', output=str(output), registration=str(reg_path),
                           registration_sha256=f.file_sha256(reg_path))
    return SimpleNamespace(args=args, reg=reg, source=source, output=output, plan=plan, original=original,
                           terminal=terminal, old_args=old_args)


def repin(case):
    case.reg.pop('identity'); case.reg = f.signed(case.reg)
    Path(case.args.registration).write_text(json.dumps(case.reg, sort_keys=True, indent=2)+'\n')
    case.args.registration_sha256=f.file_sha256(case.args.registration)


def test_finalize_reads_each_shard_once_and_preserves_rows_bytes_and_failed_attempt(tmp_path, monkeypatch):
    case=make_case(tmp_path, monkeypatch)
    old={str(p):p.read_bytes() for p in case.source.rglob('*') if p.is_file()}
    old_terminal=case.terminal.read_bytes()
    original_read=final._read; counts=Counter()
    def counted(path,*args,**kw):
        if Path(path).parent.name=='log_shards': counts[str(path)]+=1
        return original_read(path,*args,**kw)
    monkeypatch.setattr(final,'_read',counted)
    original_open=Path.open
    def metadata_only(path,*args,**kw):
        assert path.suffix not in ('.pkl','.png','.pt','.safetensors')
        return original_open(path,*args,**kw)
    monkeypatch.setattr(Path,'open',metadata_only)
    done=final.run(case.args)
    assert len(counts)==2 and set(counts.values())=={1}
    assert done['schema']==final.COMPLETE_SCHEMA and done['raw_log_deserialized'] is False
    assert done['source_shards_reused'] is True and not (case.source/'COMPLETE.json').exists()
    assert not (case.output/'COMPLETE.json').exists()
    assert case.terminal.read_bytes()==old_terminal
    assert {str(p):p.read_bytes() for p in case.source.rglob('*') if p.is_file()}==old
    manifest=json.loads((case.output/'TEACHER_REQUESTS.json').read_text())
    rows=[]
    for log in case.plan['selected_logs']:
        rows.extend(json.loads(f.shard_path(case.source,log).read_text())['scenes'])
    by_token={r['token']:r for r in rows}
    expected_rows=[by_token[r['token']] for r in case.plan['selected_rows']]
    expected=f.signed({**{k:v for k,v in manifest.items() if k not in ('scenes','identity')}, 'scenes':expected_rows})
    assert (case.output/'TEACHER_REQUESTS.json').read_bytes()==(json.dumps(expected,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    assert done['request_identity']==f.identity_hash({k:v for k,v in expected.items() if k!='identity'})
    final.validate_completion(done,case.output/final.COMPLETE_NAME, request_sha256=f.file_sha256(case.output/'TEACHER_REQUESTS.json'),
        request_identity=manifest['identity'],manifest_header={k:v for k,v in manifest.items() if k!='scenes'},scene_count=2)


def test_interleaved_scene_cache_three_passes_exact_and_read_free(tmp_path,monkeypatch):
    limits={'max_seconds':30,'max_rss_bytes':2*final.GiB,'max_metadata_bytes':final.MiB,
            'max_source_read_bytes':final.MiB,'read_mib_per_second':64.}
    budget=f.PreparationBudget(tmp_path,limits)
    rows=[{'token':str(i),'log':'AB'[i%2]} for i in range(64)]
    values=[dict(row,nested={'unicode':'世界','mask':[None,0.,1.],'ordinal':i}) for i,row in enumerate(rows)]
    cache=final.CompressedScenes(rows,budget,final.MiB,64*1024)
    for log in ('A','B'):
        for scene in values:
            if scene['log']==log: cache.add(scene)
    def forbidden(*a,**kw): raise AssertionError('Replay must never read files')
    monkeypatch.setattr(Path,'open',forbidden)
    for _ in range(3): assert list(cache())==values
    assert cache.bytes<final.MiB and budget.bytes==0


@pytest.mark.parametrize('kind',['truncated','trailing','bomb','checksum'])
def test_bounded_decompression_rejects_corruption_and_expansion(tmp_path,kind):
    budget=f.PreparationBudget(tmp_path,{'max_seconds':30,'max_rss_bytes':2*final.GiB,'max_metadata_bytes':final.MiB})
    cache=final.CompressedScenes([{'token':'x'}],budget,final.MiB,128)
    cache.add({'token':'x','value':'ok'})
    compressed,length,digest=cache.values['x']
    if kind=='truncated': compressed=compressed[:-2]
    if kind=='trailing': compressed+=b'trailing'
    if kind=='bomb': compressed=zlib.compress(b'x'*1000000); length=1000000
    if kind=='checksum': digest=b'0'*32
    cache.values['x']=(compressed,length,digest)
    with pytest.raises(ValueError,match='length/checksum'): list(cache())


@pytest.mark.parametrize('point',['TEACHER_REQUESTS.json','FINALIZATION_COMPLETE.json'])
def test_publication_failure_preserves_partials_and_resumes_identically(tmp_path,monkeypatch,point):
    case=make_case(tmp_path,monkeypatch)
    original=f._storage_check
    def fail(*,before_commit=False,path=None,**kw):
        if before_commit and path is not None and Path(path).name==point: raise RuntimeError('publication failure')
        return original(before_commit=before_commit,path=path,**kw)
    with monkeypatch.context() as m:
        m.setattr(f,'_storage_check',fail)
        with pytest.raises(RuntimeError,match='publication'): final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()
    partials={p.name:p.read_bytes() for p in case.output.glob('*.partial')}
    assert partials
    done=final.run(case.args)
    complete=(case.output/final.COMPLETE_NAME).read_bytes()
    again=final.run(case.args)
    assert again['reused'] is True and again['request_identity']==done['request_identity']
    assert (case.output/final.COMPLETE_NAME).read_bytes()==complete
    assert {p.name:p.read_bytes() for p in case.output.glob('*.partial')}==partials


@pytest.mark.parametrize('kind',['missing','changed','budget','cache','decode','terminal','source_code'])
def test_fail_closed_before_complete(tmp_path,monkeypatch,kind):
    case=make_case(tmp_path,monkeypatch)
    shard=f.shard_path(case.source,case.plan['selected_logs'][0])
    if kind=='missing': shard.unlink()
    if kind=='changed': shard.write_bytes(shard.read_bytes()+b' ')
    if kind=='budget': case.reg['limits']['max_source_read_bytes']=1; repin(case)
    if kind=='cache': case.reg['max_cache_bytes']=32; repin(case)
    if kind=='decode': case.reg['max_decoded_shard_bytes']=1; repin(case)
    if kind=='terminal': case.terminal.write_text('{"status":"COMPLETE"}')
    if kind=='source_code': case.reg['source_code_root']=str(tmp_path); repin(case)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)): final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()


def test_pinned_shard_receipt_and_corrupt_semantics_rejected(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch)
    path=f.shard_path(case.source,case.plan['selected_logs'][0])
    data=json.loads(path.read_text()); data['scenes'][0]['log']='foreign'
    data.pop('identity'); write(path,f.signed(data))
    case.reg['source_shards']=final.shard_census(case.plan,case.source)[0]; repin(case)
    with pytest.raises(ValueError,match='selection'): final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()


@pytest.mark.parametrize('kind',['cleanup','controller','admission'])
def test_real_supervisor_schema_requires_cleanup_and_pinned_lineage(tmp_path,monkeypatch,kind):
    case=make_case(tmp_path,monkeypatch)
    terminal=json.loads(case.terminal.read_text())
    if kind=='cleanup':
        terminal['resources']['cleanup']['live_owned_remaining']=1
        write(case.terminal,terminal);case.reg['source_terminal']['sha256']=f.file_sha256(case.terminal);repin(case)
    elif kind=='controller': Path(terminal['controller']['path']).write_text('# changed\n')
    else: Path(terminal['admission']['path']).write_text('{}')
    with pytest.raises(ValueError): final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()


def test_actual_storage_dispatcher_new_claim_hooks_and_receipt(tmp_path,monkeypatch):
    guard=fixture(tmp_path)
    case=make_case(tmp_path,monkeypatch,output=guard.root)
    guard.reg.update(writer_kind='future_finalization',cwd=str(Path.cwd()), callback_protocol=final.METADATA_STORAGE_CALLBACK_PROTOCOL,
        file_patterns=[r'METADATA_ADMISSION\.json',r'FINALIZE\.lock',
            r'(?:TEACHER_REQUESTS|RAW_LOG_PROVENANCE|SHARD_RECEIPTS|FINALIZATION_COMPLETE)\.json(?:\.[0-9a-f]+\.partial)?'],
        directory_patterns=[],preexisting_pins=[])
    guard.reg['command']=[sys.executable,'-m','tools.st_world_policy.finalize_future_metadata','finalize-existing',
        '--registration',case.args.registration,'--registration-sha256',case.args.registration_sha256,'--output',str(guard.root)]
    names=('tools.st_world_policy.finalize_future_metadata',*final.METADATA_STORAGE_CALLBACK_DEPENDENCIES)
    guard.reg['source_pins']=[{'path':str(Path(importlib.import_module(name).__file__).resolve()),
        'sha256':f.file_sha256(importlib.import_module(name).__file__)} for name in names]
    update_registration(guard); storage.migrate_policy(**guard.kwargs)
    done=storage.run_registered_metadata(**guard.guard)
    assert done['status']=='GUARDED_METADATA_RETURN' and done['artifact']['schema']==final.COMPLETE_SCHEMA
    assert done['writer_callbacks']>0 and done['writer_commit_callbacks']>=8


def test_downstream_rejects_lineage_and_file_tamper(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch); done=final.run(case.args)
    changed=deepcopy(done); changed['source_shards_reused']=False; changed.pop('identity'); changed=f.signed(changed)
    with pytest.raises(ValueError,match='provenance'): final.validate_completion(changed,case.output/final.COMPLETE_NAME)
    (case.output/'SHARD_RECEIPTS.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='checksum'): final.validate_completion(done,case.output/final.COMPLETE_NAME)


@pytest.mark.parametrize('mode',['finalized','legacy'])
def test_population_consumer_keeps_current_order_checks_and_strict_new_receipt(tmp_path,monkeypatch,mode):
    from tools.st_world_policy import teacher_provider as tp
    case=make_case(tmp_path,monkeypatch,full=True); done=final.run(case.args)
    manifest=json.loads((case.output/'TEACHER_REQUESTS.json').read_text())
    selected=Path(case.plan['original_selection']['path']); current_root=selected.parent
    recipe=tmp_path/'RECIPE.json'; write(recipe,{'data':{'root':case.plan['source']['student_root']}})
    receipt_path=case.output/final.COMPLETE_NAME
    if mode=='legacy':
        # Original consumer contract remains accepted with its existing filename/schema.
        receipt_path=case.output/'COMPLETE.json'
        write(receipt_path,{**done,'schema':'st_future_metadata_complete_v1'})
    d={'requests':tp.pin(case.output/'TEACHER_REQUESTS.json'), 'profile_admission':{
        'population':{'current_registration':tp.pin(current_root/'REGISTRATION.json'),
            'current_complete':tp.pin(current_root/'COMPLETE.json'),'future_complete':tp.pin(receipt_path)},
        'data_manifest':tp.pin(selected),'recipe':tp.pin(recipe),'max_scenes':2}}
    request=SimpleNamespace(receipt={'scene_count':2,'manifest_header':{k:v for k,v in manifest.items() if k!='scenes'},
        'request_identity':manifest['identity'],'dependencies':{'source_logs_ledger':{
            'path':manifest['source_logs_ledger']['path'],'sha256':done['files']['RAW_LOG_PROVENANCE.json']}}},
        manifest=manifest,check=lambda:None)
    current_rows=[{**s,'frames':s['frames'][:1]} for s in manifest['scenes']]
    source=SimpleNamespace(receipt={'scene_count':2},manifest={'times_s':[0.],
        'selection_identity':case.plan['selection_identity'],'scenes':current_rows},check=lambda:None)
    original=final.validate_completion; calls=[]
    def checked(*a,**kw): calls.append(True); return original(*a,**kw)
    monkeypatch.setattr(final,'validate_completion',checked)
    tp._validate_full_population(d,request,source)
    assert bool(calls)==(mode=='finalized')
    source.manifest['scenes']=list(reversed(current_rows))
    with pytest.raises(ValueError,match='order'): tp._validate_full_population(d,request,source)
    if mode=='finalized':
        case.terminal.write_bytes(b'{}')
        with pytest.raises(ValueError,match='checksum'): tp._validate_full_population(d,request,source)
