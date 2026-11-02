"""Only new S6 family and exact priority-return race, CPU only."""
import copy,json,inspect
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import launch as L,evaluate as E
from starVLA.rl.qwen_world_research_data.manifest import fingerprint


def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def test_S6_pure_independent_registration_early_cadence_and_exact_export_protocol(tmp_path,monkeypatch):
    original=L.read(L.ROOT/'RUNS.json');before=(L.ROOT/'RUNS.json').read_bytes()
    # Registration is tested against a private pre-S6 fixture, regardless of
    # whether the real campaign has subsequently admitted that exact family.
    original['runs']=[r for r in original['runs']if r['id']in set(L.BASE_ORDER)|{L.FAST}]
    base=next(r for r in original['runs']if r['id']=='S1_NOMINAL_RESCUE')
    cfg=yaml.safe_load(Path(base['config']).read_text());p=cfg['research']['performance']
    cfg['research'].update(arm=L.QUALITY,verified_quality={'schema':'CPU interface fixture, not qualification'})
    p.update(algorithm='verified_quality',scenes_per_collection=64,evaluation_epochs=L.QUALITY_EPOCHS,
             stages=[dict(name='main',start_epoch=0.,B=64,action_lr=5e-5,memory_lr=5e-6,rescue=True,history=False)])
    cfg.update(max_steps=129120,gradient_accumulation_steps=8)
    config=tmp_path/'S6.yaml';config.write_text(yaml.safe_dump(cfg));source=tmp_path/'frozenfixture';source.mkdir()
    actual=L.validate_run_recipe
    def recipe(run):
        if run['id']!=L.QUALITY:return actual(run)
        if run['trainer_module']!='tools.qwen_world_verified_quality.train':raise ValueError('Exact S6 trainer')
        return cfg
    monkeypatch.setattr(L,'validate_run_recipe',recipe)
    real_check=L.subprocess.check_output
    def check(command,**kw):
        if command[:3]==['git','-C',str(source)]:return ''if 'status'in command else'fixturecommit\n'
        return real_check(command,**kw)
    monkeypatch.setattr(L.subprocess,'check_output',check)
    result=L.quality_registration(original,config,source,worker_source=tmp_path/'ops5')
    assert (L.ROOT/'RUNS.json').read_bytes()==before and len(result['runs'])==len(original['runs'])+1
    run=next(r for r in result['runs']if r['id']==L.QUALITY)
    assert [r['id']for r in result['runs']][:3]==['S1_NOMINAL_RESCUE','S5_PERFORMANCE_MAX',L.QUALITY]
    assert run['native_gates'][:4]==[404,808,1616,3228]
    assert [L.point(run['exposure_mapping'],e)['updates']for e in(1,6,20)]==[6456,38736,129120]
    assert not any(k.startswith('execution_')and k!='execution_output_root'for k in run)
    assert run['root']!=base['root']and run['parent_checkpoint']==base['parent_checkpoint']
    assert all(x['host']!='training-vlawm-tsinghua-zt5'for x in run['deployment_candidates'])
    bad=copy.deepcopy(result);next(r for r in bad['runs']if r['id']==L.QUALITY)['navtest_epochs'].remove(.0625)
    with pytest.raises(ValueError,match='exact registered early'):L.validate_registry(bad)
    cp=dict(path=str(tmp_path/'cp'),files_sha256='files',updates=404)
    receipt=dict(schema='world_verified_quality_export_native_validation_v1',status='PASS',checkpoint=cp['path'],files_sha256='files',counts=dict(updates=404),quality_protocol_sha256=fingerprint(cfg['research']['verified_quality']))
    assert E.validate_native_export_receipt(run,cp,receipt)==receipt
    with pytest.raises(ValueError,match='S6 export'):E.validate_native_export_receipt(run,cp,dict(receipt,quality_protocol_sha256='0'*64))
    from tools.qwen_world_safety_exploration import job_worker
    text=inspect.getsource(job_worker.task_ranks)
    assert "'tools.qwen_world_verified_quality.train'"in text and 'd44_native_rl'not in text
    from tools.qwen_world_performance import deadline_guard as G
    row=dict(args=['model','-m','tools.qwen_world_verified_quality.train','--output',run['root']])
    assert G.classify_direct(row,{L.TASK},'host')['role']=='native'
    unknown=dict(args=['model','-m','tools.unrelated.train','--output',run['root']])
    assert G.classify_direct(unknown,{L.TASK},'host')is None


def test_priority_exact_EXIT0_unharvested_stage_blocks_lower_but_science_failure_does_not(tmp_path,monkeypatch):
    monkeypatch.setattr(L,'ROOT',tmp_path);run=dict(id='S1_NOMINAL_RESCUE')
    job=tmp_path/'jobs'/run['id']/'old_attempt';command=['model','-m','exact.train']
    state=dict(status='TRAINING',stage_target_updates=808,stages={'808':dict(job=str(job),host='permanent',command=command)})
    assert not L.priority_needs_admission(run,state)
    dump(job/'RESULT.json',dict(returncode=0,host='permanent',command=command))
    assert L.priority_needs_admission(run,state)
    done=copy.deepcopy(state);done['stages']['808']['complete']=True
    assert not L.priority_needs_admission(run,done)
    assert not L.priority_needs_admission(run,dict(state,status='FAILED_REVIEW_REQUIRED'))
    dump(job/'RESULT.json',dict(returncode=1,host='permanent',command=command))
    assert not L.priority_needs_admission(run,state)
    dump(job/'RESULT.json',dict(returncode=0,host='different',command=command))
    assert not L.priority_needs_admission(run,state)
