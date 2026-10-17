"""Full-index admission tests use synthetic tiny records, never real qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from starVLA.dataloader.st_world_targets import atomic_json, TeacherCacheWriter
from starVLA.dataloader.st_world_metadata_index import build_index, Rows
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_future_provenance import LogCoverage, LEDGER_SCHEMA, REF_SCHEMA, FIELDS
from tools.st_world_policy import teacher_provider as tp, qualify_indexed_provider as q
from test_provider_profile_upgrade import f, sign
from test_indexed_composition import exercise, descriptor


@pytest.fixture(scope='module')
def full(f, tmp_path_factory):
    root=tmp_path_factory.mktemp('full-profile'); n=258
    # Repeated synthetic observations give >256 distinct scene IDs without large tensors.
    m=deepcopy(f.a.current.manifest); proto=m['scenes'][0]
    m['scenes']=[dict(deepcopy(proto),token='full-'+str(i)) for i in range(n)]
    m['selection_identity']=identity_hash([dict(token=x['token'],log=x['log'],current_record_sha256='a'*64,ego_record_sha256='b'*64) for x in m['scenes']]);sign(m)
    current=TeacherCacheWriter(root/'current',f.a.fit.contract,m,execution=f.a.current.identity['execution'])
    current.commit_shard(0,torch.zeros(1,18,24,1024),torch.ones(1,18,24,dtype=torch.bool));current.finalize()
    source_index=root/'current_index'
    build_index(source_index,requests=current.root/'requests.json',expected_request_identity=m['identity'],
        contract=current.contract,cache_root=current.root,expected_cache_identity=current.identity['identity'],max_scenes=n)
    p=q.make_plan(Path(f.p['request_index']['complete']['path']).parent,source_index,f.p['teacher_config']['path'],f.a.report)
    r=exercise(p);parent,parentpath=descriptor(p,r,root)
    original=Path(tp.__file__).resolve().parents[2]
    p['source_files']=[tp.pin(f.archive/Path(v['path']).relative_to(original)) for v in p['source_files']]
    p['runner_source_sha256']=tp.pin(f.archive/'tools/st_world_policy/qualify_indexed_provider.py')['sha256'];sign(p)
    atomic_json(parent['qualification_plan']['path'],p)
    r['plan_identity']=p['identity'];sign(r);atomic_json(parent['qualification']['path'],r)
    parent.update(qualification=tp.pin(parent['qualification']['path']),qualification_plan=tp.pin(parent['qualification_plan']['path']))
    sign(parent);atomic_json(parentpath,parent)
    migrationpath=root/'MIGRATION.json';q.prepare_control_migration(parentpath,f.archive,migrationpath)
    old=deepcopy(f.old);old['data']['teacher_provider']['descriptor']=tp.pin(parentpath)
    oldpath=root/'ORIGINAL.json';atomic_json(oldpath,old)
    recipe=deepcopy(f.recipe);recipe['execution'].update(sample_limit=n,stop_after=64,world_size=2,global_batch=2,micro_batch=1)
    from tools.st_world_policy.official_runtime import REFERENCE_RECIPE
    effective={**{k:recipe['execution'][k] for k in ('world_size','micro_batch','global_batch','max_updates')},
               **recipe['optimizer'],'gradient_checkpointing':False}
    recipe['runtime']['official_execution']['declared_recipe_overrides']=sorted(k for k,v in REFERENCE_RECIPE.items() if effective[k]!=v)
    recipepath=root/'RECIPE.json';atomic_json(recipepath,recipe)
    selected=dict(schema='st_fixed_scene_list_v1',split='train',source_index_hash=m['source_index_sha256'],
        source_student_identity=m['source_student_identity'],ego_identity_sha256='e'*64,
        scenes=[dict(token=x['token'],log=x['log'],current_record_sha256='a'*64,ego_record_sha256='b'*64) for x in m['scenes']])
    selectedpath=root/'population'/'SELECTED_SCENES.json';atomic_json(selectedpath,selected)
    binding=dict(student_root=recipe['data']['root'],source_identity=m['source_student_identity'],
        source_index_identity=m['source_index_sha256'],partition_sha256=m['partition_sha256'],
        ego_identity_sha256='e'*64,split='train',scenes=n,logs=1,
        ordered_scene_ids_hash=identity_hash([x['token'] for x in m['scenes']]))
    reg=sign(dict(schema='st_current_population_registration_v1',scope='all_original_current_observations',
        source=binding,max_scenes=n,target_times_s=[0.]))
    regpath=selectedpath.with_name('REGISTRATION.json');atomic_json(regpath,reg)
    selected.update(registration_identity=reg['identity'],selection_rule='entire_original_index_in_original_order');atomic_json(selectedpath,selected)
    current_complete=dict(status='COMPLETE',scope='current_only_metadata',registration_identity=reg['identity'],
        scenes=n,logs=1,selected_sha256=tp.pin(selectedpath)['sha256'])
    ccpath=selectedpath.with_name('COMPLETE.json');atomic_json(ccpath,current_complete)
    fm=deepcopy(f.manifest); proto=fm['scenes'][0]
    fm['scenes']=[dict(deepcopy(proto),token=x['token'],log=x['log']) for x in m['scenes']]
    for target,source in zip(fm['scenes'],m['scenes']):
        target['timestamp_us']=source['timestamp_us'];target['frames'][0]=deepcopy(source['frames'][0])
    fm.pop('source_logs',None)
    fm.update(source='original_raw_log_future_frames_v1',registration_identity='d'*64,selection_identity=identity_hash(selected['scenes']))
    coverage=LogCoverage()
    for row in fm['scenes']:coverage.add(row['token'],row['log'])
    ledger=sign(dict(schema=LEDGER_SCHEMA,split='train',**{k:fm[k] for k in FIELDS},
        logs=[dict(row,path=str(root/(row['log']+'.pkl')),bytes=1,mtime_ns=0,sha256='c'*64) for row in coverage.records()]))
    future_root=root/'future';ledgerpath=future_root/'RAW_LOG_PROVENANCE.json';atomic_json(ledgerpath,ledger)
    fm['source_logs_ledger']=dict(schema=REF_SCHEMA,ledger_schema=LEDGER_SCHEMA,**tp.pin(ledgerpath),identity=ledger['identity']);sign(fm)
    requestpath=future_root/'TEACHER_REQUESTS.json';atomic_json(requestpath,fm)
    future= dict(schema='st_future_metadata_complete_v1',status='COMPLETE',scope='full_population',
        scenes=n,logs=1,target_times_s=[0.,.5,1.,2.,4.],request_identity=fm['identity'],
        files={'TEACHER_REQUESTS.json':tp.pin(requestpath)['sha256'],'RAW_LOG_PROVENANCE.json':tp.pin(ledgerpath)['sha256']},
        original_selection={**tp.pin(selectedpath), 'current_registration_identity':reg['identity'],
            'current_registration_sha256':tp.pin(regpath)['sha256'],'current_complete_sha256':tp.pin(ccpath)['sha256']},
        registration_identity=fm['registration_identity'],plan_identity='f'*64,tensor_disk_bytes=0,training_admission=False)
    fcpath=future_root/'COMPLETE.json';atomic_json(fcpath,future)
    index=root/'future_index';build_index(index,requests=requestpath,expected_request_identity=fm['identity'],contract=current.contract,max_scenes=n)
    admission=dict(schema=tp.FULL_PROFILE_SCHEMA,scope='PROFILE',parent_descriptor=tp.pin(parentpath),
        source_migration=tp.pin(migrationpath),data_manifest=tp.pin(selectedpath),recipe=tp.pin(recipepath),
        reference_recipe=tp.pin(oldpath),max_scenes=n,max_updates=256,new_population_tensor_qualified=False,formal_approved=False,
        population=dict(current_registration=tp.pin(regpath),current_complete=tp.pin(ccpath),future_complete=tp.pin(fcpath)),
        consumption=tp.full_profile_consumption(recipe))
    path=root/'PROVIDER.json';d=tp.prepare_indexed(index,[source_index],p['teacher_config']['path'],parent['qualification']['path'],path,
        limits=p['limits'],profile_admission=admission)
    return SimpleNamespace(**locals())


def test_full_population_actual_indices_and_cumulative_consumption(full,monkeypatch):
    original=Path.read_text
    forbidden={full.requestpath, full.selectedpath, full.current.root/'requests.json', full.current.root/'identity.json'}
    def guarded(path,*a,**k):
        assert path not in forbidden, 'Full request/selection parsed eagerly'
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',guarded)
    def no_slices(*a,**k):raise AssertionError('No ordinal/slice population materialization')
    monkeypatch.setattr(Rows,'__getitem__',no_slices)
    d,_,provider=tp.load_descriptor(tp.pin(full.path))
    try:
        assert len(provider.manifest['scenes'])==258
        assert d['profile_admission']['consumption']==dict(tp.full_profile_consumption(full.recipe))
        assert d['profile_admission']['consumption']['cumulative_scene_exposures']==128
        assert provider.memory_report()['metadata_population_materialized'] is False
    finally:tp.close_provider_metadata(provider)
    tp.validate_training_manifest({'descriptor':tp.pin(full.path)},tp.pin(full.selectedpath))
    with pytest.raises(ValueError,match='dataset path/SHA'):
        tp.validate_training_manifest({'descriptor':tp.pin(full.path)},dict(tp.pin(full.selectedpath),sha256='0'*64))


def test_full_configuration_uses_existing_pca_and_trainer_contract(full,f,tmp_path):
    from tools.st_world_policy.training_config import validate_config
    from tools.st_world_policy.check_formal_approval import validate_execution
    cfgpath=tp.configure_smoke(full.recipepath,full.path,f.a.fit.root,tmp_path/'configured',fit_cache_index=f.a.fit_index)
    cfg=json.loads(Path(cfgpath).read_text());validate_config(cfg);validate_execution(cfg['execution'])
    assert cfg['execution']['sample_limit']==258 and cfg['execution']['stop_after']==64
    assert cfg['execution']['max_updates']==20000 and cfg['optimizer']['warmup']==1000
    assert cfg['space_time']==full.old['space_time'] and cfg['runtime']['loader_workers']==0


@pytest.mark.parametrize('kind',['scope','count','role','time','source','current_complete','future_complete','wrong_request','original_selection','consumption','257','global32','recipe','source_cache','raw_sha','teacher','missing_complete'])
def test_full_population_negative_boundaries(full,tmp_path,kind,monkeypatch):
    d=deepcopy(full.d); a=d['profile_admission']
    if kind in ('consumption','257'):
        a['consumption']['cumulative_optimizer_updates'] = 257 if kind=='257' else 65
    elif kind=='scope':a['scope']='FORMAL'
    elif kind=='count':a['max_scenes']-=1
    elif kind=='source_cache':d['sources'][0]['cache_identity']='0'*64
    elif kind=='raw_sha':a['data_manifest']['sha256']='0'*64
    elif kind=='teacher':d['execution']['tf32']=True
    elif kind=='missing_complete':a['population']['future_complete']['path']=str(tmp_path/'absent.json')
    elif kind in ('recipe','global32'):
        cfg=deepcopy(full.recipe)
        if kind=='recipe':cfg['optimizer']['lr']*=2
        else:cfg['execution']['global_batch']=32
        path=tmp_path/'recipe.json';atomic_json(path,cfg);a['recipe']=tp.pin(path)
    else:
        key='current_registration' if kind=='source' else ('current_complete' if kind=='current_complete' else 'future_complete')
        body=tp.pinned(a['population'][key])
        if kind=='source':body['source']['source_identity']='0'*64;sign(body)
        elif kind in ('current_complete','future_complete'):body['status']='PARTIAL'
        elif kind=='role':body['scope']='qualification'
        elif kind=='time':body['target_times_s']=[0.]
        elif kind=='wrong_request':body['request_identity']='0'*64
        elif kind=='original_selection':body['original_selection']['sha256']='0'*64
        # Keep the actual canonical path binding, so each distinct semantic guard
        # is exercised instead of failing only the prerequisite path check.
        original_pinned=tp.pinned; target=deepcopy(a['population'][key])
        monkeypatch.setattr(tp,'pinned',lambda pin: deepcopy(body) if pin==target else original_pinned(pin))
    sign(d)
    with pytest.raises((ValueError,FileNotFoundError)):tp.validate_descriptor(d)


@pytest.mark.parametrize('mutation',['reverse','omit','duplicate','current_pin','ego_pin','source'])
def test_stream_population_rejects_order_coverage_and_pins(full,mutation,monkeypatch):
    # Retain all real index/receipt checks; mutate only the decoded selected row.
    from starVLA.dataloader import st_world_metadata_index as mi
    original=mi._atomic_rows
    def changed(path,arrays,consume,**kw):
        def row(key,pos,value,encoded):
            value=deepcopy(value)
            if pos==0:
                if mutation in ('reverse','duplicate'):value['token']='full-1'
                elif mutation=='omit':return
                elif mutation=='current_pin':value.pop('current_record_sha256')
                elif mutation=='ego_pin':value.pop('ego_record_sha256')
            consume(key,pos,value,encoded)
        h,c,digest=original(path,arrays,row,**kw)
        if mutation=='source':h['source_index_hash']='0'*64
        return h,c,digest
    monkeypatch.setattr(mi,'_atomic_rows',changed)
    with pytest.raises(ValueError):tp.validate_descriptor(full.d)


def test_old_profile_does_not_inherit_full_permission(f,full):
    with pytest.raises(ValueError):tp.validate_profile_recipe(f.admission,full.recipe)
    d=deepcopy(full.d);d['profile_admission'].pop('population');d['profile_admission'].pop('consumption')
    d['profile_admission'].update(schema=tp.PROFILE_SCHEMA,max_scenes=256);sign(d)
    with pytest.raises(ValueError):tp.validate_descriptor(d)


def test_consumption_resume_is_total_not_a_fresh_increment(full):
    c=tp.full_profile_consumption(full.recipe)
    assert c['initial_completed_updates']==0 and c['resume_budget']=='same_run_cumulative_not_per_invocation'
    assert c['per_rank_scene_exposures']==[64,64] and c['gradient_accumulation_steps']==1
    cfg=deepcopy(full.recipe);cfg['execution']['world_size']=1
    assert tp.full_profile_consumption(cfg)['per_rank_scene_exposures']==[128]
    cfg['execution']['stop_after']=256
    assert tp.full_profile_consumption(cfg)['cumulative_scene_exposures']==512
    cfg['execution']['stop_after']=257
    with pytest.raises(ValueError):tp.full_profile_consumption(cfg)


def test_cli_incomplete_full_optin_rejected(monkeypatch):
    monkeypatch.setattr('sys.argv',['teacher_provider','--request-index','none','--source-index','none',
        '--teacher-config','none','--qualification-result','none','--output','none','--profile-full-population'])
    with pytest.raises(SystemExit) as e:tp.main()
    assert e.value.code==2


@pytest.mark.parametrize('field',['initial_completed_updates','initial_epoch','initial_offset','gradient_accumulation_steps'])
def test_counter_type_cannot_alias_integer_with_boolean(full,field):
    a=deepcopy(full.admission);a['consumption'][field]=bool(a['consumption'][field])
    with pytest.raises(ValueError,match='consumption'):tp.validate_profile_recipe(a,full.recipe)


def test_explicit_reference_recipe_change_is_rejected(full,tmp_path):
    cfg=deepcopy(full.recipe);cfg['optimizer']['warmup']=200;path=tmp_path/'recipe.json';atomic_json(path,cfg)
    admission=deepcopy(full.admission);admission['recipe']=tp.pin(path)
    with pytest.raises(ValueError,match='recipe'):tp.validate_profile_recipe(admission,cfg)


def test_actual_full_cli_with_synthetic_existing_indices(full,f,tmp_path,monkeypatch,capsys):
    budget=tmp_path/'CONSUMPTION.json';atomic_json(budget,full.admission['consumption'])
    out=tmp_path/'PROVIDER.json';configured=tmp_path/'configured'
    args=['teacher_provider','--request-index',str(full.index),'--source-index',str(full.source_index),
        '--teacher-config',full.p['teacher_config']['path'],'--qualification-result',full.parent['qualification']['path'],
        '--output',str(out),'--max-cached-bytes',str(full.p['limits']['max_cached_bytes']),
        '--profile-parent-descriptor',str(full.parentpath),'--profile-source-migration',str(full.migrationpath),
        '--profile-data-manifest',str(full.selectedpath),'--profile-reference-config',str(full.oldpath),
        '--profile-full-population','--profile-current-registration',str(full.regpath),
        '--profile-current-complete',str(full.ccpath),'--profile-future-complete',str(full.fcpath),
        '--profile-consumption-budget',str(budget),'--training-config',str(full.recipepath),
        '--configured-output',str(configured),'--pca-fit-cache-root',str(f.a.fit.root),
        '--pca-fit-cache-index',str(f.a.fit_index)]
    monkeypatch.setattr('sys.argv',args);tp.main()
    printed=[json.loads(v) for v in capsys.readouterr().out.splitlines()]
    assert printed[0]['tensor_disk_bytes']==0 and printed[1]['started'] is False
    d=json.loads(out.read_text());assert d['profile_admission']['schema']==tp.FULL_PROFILE_SCHEMA
    assert d['profile_admission']['consumption']['cumulative_scene_exposures']==128
