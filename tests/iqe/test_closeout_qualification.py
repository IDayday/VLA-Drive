from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from iqe.io import atomic_json, atomic_torch, read_json, file_hash, digest, BlockedError
from iqe.base_qualification import evaluate_base, qualify_base, require_qualification, binding
from iqe.data.manifests import load_scenes, save_scenes
from test_base_qualification import prepared
from test_two_round_pipeline import pipeline_fixture
from conftest import TensorAdapter


def evaluation_fixture(tmp_path, monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);ck,manifest=prepared(p,tmp_path)
    adapter=TensorAdapter().eval().requires_grad_(False)
    calls={'inference':0,'scoring':[]}
    encode=adapter.encode_scene
    def encode_scene(batch):
        calls['inference']+=1
        return encode(batch[0])
    adapter.encode_scene=encode_scene
    adapter.predict_s0=lambda f:adapter.action(f)
    monkeypatch.setattr('iqe.base_qualification.S0Adapter.load_and_validate_s0',lambda *a:adapter)
    monkeypatch.setattr('iqe.base_qualification.load_observation',lambda *a:torch.ones(1,3,8))
    return p,ck,manifest,adapter,calls


def quality_fixture(tmp_path, monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    rules={'status':'PREREGISTERED','registered_unix':1.,'validation_manifest_hash':file_hash(manifest),
        'score_protocol_hash':p.backend().protocol_hash,'minimum_mean_PDMS_01':.8,'maximum_zero_rate':0.,
        'maximum_NC_violation_rate':0.,'maximum_DAC_violation_rate':0.,'minimum_scenes':4,'minimum_logs':4}
    path=tmp_path/'rules.json';atomic_json(path,rules);p.config['base_evaluation']['rules']=str(path)
    def report(checkpoint, value, name):
        scenes=load_scenes(manifest)
        rows=[{'scene_id':s.scene_id,'score':{'score_valid':True,'total_score_01':value,
            'component_valid_masks':{'no_at_fault_collisions':True,'drivable_area_compliance':True},
            'named_components':{'no_at_fault_collisions':1.,'drivable_area_compliance':1.}}} for s in scenes]
        out=tmp_path/(name+'.json')
        atomic_json(out,{'status':'COMPLETE','binding':binding(p.contract,file_hash(checkpoint),file_hash(manifest),p.backend().protocol_hash),
            'identity':{'rules_hash':file_hash(path)},'optimizer_step':1,'started_unix':2.,'smoke_or_profile':False,
            'summary':{'scenes':4,'valid_official_scores':4,'nonfinite':0,'PDMS_01':value,'zero_rate':0.},
            'rows':rows,'per_log':{s.source_log_id:{} for s in scenes}})
        return out
    return p,ck,report


def test_failed_then_qualified_history_idempotence_and_selection(tmp_path,monkeypatch):
    p,ck,report=quality_fixture(tmp_path,monkeypatch)
    failed=qualify_base(p,ck,report(ck,.2,'failed'))
    assert failed['status']=='BASE_UNQUALIFIED'
    oldhash=file_hash(failed['qualification_path'])
    state=torch.load(ck,weights_only=False);state['different_weights']='unit_second_checkpoint'
    second=tmp_path/'second.pt';atomic_torch(second,state)
    receipt=read_json(str(ck)+'.COMPLETE.json');receipt['checksum']=file_hash(second);atomic_json(str(second)+'.COMPLETE.json',receipt)
    good_evaluation=report(second,.9,'good')
    good=qualify_base(p,second,good_evaluation)
    assert good['qualification_id'] != failed['qualification_id']
    assert not (p.root/'SELECTED_BASE_QUALIFICATION.json').exists()
    assert qualify_base(p,second,good_evaluation,select=True)==good
    selected=file_hash(p.root/'SELECTED_BASE_QUALIFICATION.json')
    assert require_qualification(p,second)['checksum']==file_hash(good['qualification_path'])
    qualify_base(p,ck,report(ck,.1,'later_failure'))
    assert file_hash(p.root/'SELECTED_BASE_QUALIFICATION.json')==selected
    assert file_hash(failed['qualification_path'])==oldhash
    assert read_json(p.root/'LATEST_QUALIFICATION_ATTEMPT.json')['status']=='BASE_UNQUALIFIED'
    # The selected pointer is not an implicit mutable Base switch after locking.
    atomic_json(p.root/'LOCKED_S0.json',p.contract|{'base_mode':'full','query_checkpoint_hash':file_hash(second),
        'base_qualification':require_qualification(p,second)})
    with pytest.raises(ValueError,match='LOCKED_BASE_IMMUTABLE'):
        qualify_base(p,ck,report(ck,.95,'other_good'),select=True)
    p.mode='full'
    with pytest.raises(ValueError,match='LOCKED_BASE_IMMUTABLE'):p.freeze_base(ck)


def test_explicit_qualification_and_legacy_failed_are_not_promoted(tmp_path,monkeypatch):
    p,ck,report=quality_fixture(tmp_path,monkeypatch)
    failed=qualify_base(p,ck,report(ck,.1,'bad'))
    atomic_json(p.root/'BASE_QUALIFICATION.json',{k:v for k,v in failed.items() if k not in {'qualification_id','qualification_path'}})
    old=file_hash(p.root/'BASE_QUALIFICATION.json')
    with pytest.raises(ValueError,match='BASE_UNQUALIFIED'):require_qualification(p,ck)
    assert file_hash(p.root/'BASE_QUALIFICATION.json')==old
    assert list((p.root/'base_qualification/legacy').glob('*/MIGRATION.json'))
    good=qualify_base(p,ck,report(ck,.9,'good'))
    p.config['base_evaluation']['qualification']=good['qualification_path']
    assert require_qualification(p,ck)['path']==good['qualification_path']
    atomic_json(Path(good['evaluation_path']),read_json(good['evaluation_path'])|{'tampered':True})
    with pytest.raises(ValueError,match='evaluation changed'):require_qualification(p,ck)


def test_freeze_revalidates_input_bytes_with_legacy_evaluation(tmp_path, monkeypatch):
    p, ck, report = quality_fixture(tmp_path, monkeypatch)
    evaluation = report(ck, .9, 'legacy_complete')
    good = qualify_base(p, ck, evaluation, select=True)
    assert require_qualification(p, ck)['checksum'] == file_hash(good['qualification_path'])
    scene = load_scenes(p.config['base_evaluation']['manifest'])[0]
    original = Path(scene.target_ref).read_bytes()
    Path(scene.target_ref).write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='evaluation target changed'):
        require_qualification(p, ck)
    with pytest.raises(ValueError, match='evaluation target changed'):
        qualify_base(p, ck, evaluation)


def test_locked_base_rejects_conflicting_explicit_config(tmp_path, monkeypatch):
    p = pipeline_fixture(tmp_path, monkeypatch)
    atomic_json(p.root/'LOCKED_S0.json', p.contract | {'query_checkpoint':str(tmp_path/'locked.pt')})
    p.raw_config['s0']['query_checkpoint'] = str(tmp_path/'different.pt')
    with pytest.raises(ValueError, match='LOCKED_BASE_IMMUTABLE'):
        p.use_locked_base()


def test_preflight_preserves_explicit_experiment_checkpoint(tmp_path, monkeypatch):
    from iqe.cli import parser, execute
    p = pipeline_fixture(tmp_path, monkeypatch)
    config = deepcopy(p.raw_config)
    config['s0']['query_checkpoint'] = str(tmp_path/'chosen.pt')
    monkeypatch.setattr('iqe.cli.load_config', lambda _:config)
    monkeypatch.setattr('iqe.resources.qualify_distributed', lambda _: {'device':'cpu'})
    monkeypatch.setattr('iqe.preflight.audit', lambda *args:p.contract | {'query_checkpoint':None,'binding_status':'FRAMEWORK_BOUND_QUERY_UNTRAINED'})
    result = execute(parser().parse_args(['preflight','--config','fixture.yaml','--device','cpu']))
    assert result['query_checkpoint'] == config['s0']['query_checkpoint']
    assert read_json(p.root/'preflight_resolved_config.json')['s0']['query_checkpoint'] == config['s0']['query_checkpoint']


def test_timeout_retries_only_failed_score_without_model_load(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    original=p.backend().score
    def flaky(candidate,scene):
        calls['scoring'].append(scene.scene_id)
        if len(calls['scoring'])==2:raise TimeoutError('temporary worker unavailable')
        return original(candidate,scene)
    p.backend().score=flaky
    first=evaluate_base(p,ck)
    assert first['status']=='BLOCKED_RETRYABLE' and first['summary']['valid_official_scores']==3
    assert first['summary']['PDMS_01'] is None and first['summary']['PDMS_valid_subset_01'] is not None
    with pytest.raises(ValueError,match='INCOMPLETE'):qualify_base(p,ck,first['result_path'])
    monkeypatch.setattr('iqe.base_qualification.S0Adapter.load_and_validate_s0',lambda *a:pytest.fail('retry reloaded VLM'))
    second=evaluate_base(p,ck)
    assert second['status']=='COMPLETE' and second['summary']['valid_official_scores']==4
    assert len(calls['scoring'])==5 and calls['inference']==4
    assert evaluate_base(p,ck)['reused'] is True
    assert len(calls['scoring'])==5
    attempts=list(Path(second['result_path']).parent.glob('scenes/*/attempt_*.json'))
    assert len(attempts)==5 and any(read_json(a)['outcome']=='transient_scoring_failure' for a in attempts)
    assert read_json(first['result_path'])['status']=='BLOCKED_RETRYABLE'


def test_exhausted_retry_budget_and_changed_context(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    def timeout(*a):calls['scoring'].append(1);raise TimeoutError('worker down')
    p.backend().score=timeout
    p.config['base_evaluation']['scoring_attempts_per_scene_per_call']=2
    report=evaluate_base(p,ck)
    assert len(calls['scoring'])==8 and report['summary']['unfinished']==4
    p.config['base_evaluation']['scoring_attempts_per_scene_per_call']=0
    assert evaluate_base(p,ck)['status']=='BLOCKED_RETRYABLE' and len(calls['scoring'])==8
    scene=load_scenes(manifest)[0];atomic_json(scene.metric_context_ref,{'changed':True})
    with pytest.raises(ValueError,match='context changed'):evaluate_base(p,ck)


def test_invalid_prediction_is_quality_failure_never_zero_or_retry(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    original=adapter.predict_s0
    def invalid(f):
        prediction=original(f);prediction.physical.fill_(float('nan'));return prediction
    adapter.predict_s0=invalid
    p.backend().score=lambda *a:pytest.fail('nonfinite trajectory sent to scoring')
    first=evaluate_base(p,ck)
    assert first['status']=='COMPLETE' and first['summary']['invalid_predictions']==4
    assert first['summary']['PDMS_01'] is None and first['summary']['valid_official_scores']==0
    assert all(r['score'] is None and r['outcome']=='invalid_prediction' for r in first['rows'])
    assert evaluate_base(p,ck)['reused']
    result=qualify_base(p,ck,first['result_path'])
    assert result['status']!='QUALIFIED' and not result['checks']['all_finite']


def test_legacy_false_complete_is_migrated_and_retried(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    original=p.backend().score
    def timeout(*a):raise TimeoutError('temporary')
    p.backend().score=timeout
    first=evaluate_base(p,ck)
    # Reproduce the actual legacy directory layout, preserve its bogus COMPLETE.
    old_identity={k:v for k,v in first['identity'].items() if k not in {'evaluation_schema','eval_seed'}}
    old=p.root/'base_evaluation'/digest(old_identity)
    atomic_json(old/'EVALUATION.json',first|{'status':'COMPLETE','identity':old_identity})
    for row in first['rows']:
        prediction={k:v for k,v in row.items() if k not in {'outcome','error','attempt_count'}}
        atomic_json(old/'scenes'/(row['scene_id']+'.json'),prediction)
    legacy_hash=file_hash(old/'EVALUATION.json')
    # New evaluation root imports legacy artifacts read-only; no manual deletion.
    from shutil import copytree
    new_root=tmp_path/'new_out';copytree(old,new_root/'base_evaluation'/old.name)
    p.root=new_root;p.backend().score=original
    monkeypatch.setattr('iqe.base_qualification.S0Adapter.load_and_validate_s0',lambda *a:pytest.fail('legacy prediction rerun'))
    second=evaluate_base(p,ck)
    assert second['summary']['valid_official_scores']==4
    assert file_hash(old/'EVALUATION.json')==legacy_hash
    assert len(list(new_root.glob('base_evaluation/*/scenes/*/LEGACY_SOURCE.json')))==4


def test_evaluation_receipt_crash_repaired_without_inference_or_score(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    import iqe.base_evaluation_cache as cache
    original=cache.atomic_json
    def fail(path,*a,**kw):
        if str(path).endswith('EVALUATION.json.COMPLETE.json'):raise OSError('receipt crash')
        return original(path,*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(cache,'atomic_json',fail)
        with pytest.raises(OSError,match='receipt crash'):evaluate_base(p,ck)
    monkeypatch.setattr('iqe.base_qualification.S0Adapter.load_and_validate_s0',lambda *a:pytest.fail('reinference'))
    p.backend().score=lambda *a:pytest.fail('rescoring')
    assert evaluate_base(p,ck)['reused']


def test_engineering_qualification_is_explicit_and_not_convergence(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    path=tmp_path/'engineering.json'
    rules={'status':'PREREGISTERED','registered_unix':1.,'validation_manifest_hash':file_hash(manifest),
        'score_protocol_hash':p.backend().protocol_hash,'qualification_scope':'engineering_incremental_experiment',
        'scientific_baseline_quality':'UNTESTED','permits_deployment':False,'minimum_scenes':4,'minimum_logs':4,
        'candidate_checkpoint_hashes':[file_hash(ck)]}
    atomic_json(path,rules);p.config['base_evaluation']['rules']=str(path)
    evaluation=evaluate_base(p,ck)
    result=qualify_base(p,ck,evaluation['result_path'],select=True)
    assert result['status']=='QUALIFIED' and result['scientific_baseline_quality']=='UNTESTED'
    assert require_qualification(p,ck)['path']==result['qualification_path']


def test_changed_checkpoint_or_protocol_never_reuses_predictions(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    first=evaluate_base(p,ck)
    state=torch.load(ck,weights_only=False);state['new_identity']=True
    next_ck=tmp_path/'second.pt';atomic_torch(next_ck,state)
    receipt=read_json(str(ck)+'.COMPLETE.json');receipt['checksum']=file_hash(next_ck);atomic_json(str(next_ck)+'.COMPLETE.json',receipt)
    second=evaluate_base(p,next_ck)
    assert calls['inference']==8 and first['result_path']!=second['result_path']
    p.backend().protocol_hash='another_protocol'
    third=evaluate_base(p,next_ck)
    assert calls['inference']==12 and second['result_path']!=third['result_path']


def test_reused_prediction_rejects_changed_image_and_target_bytes(tmp_path,monkeypatch):
    p,ck,manifest,adapter,calls=evaluation_fixture(tmp_path,monkeypatch)
    evaluate_base(p,ck)
    scene=load_scenes(manifest)[0]
    image=Path(read_json(scene.observation_ref)['image_paths'][0]);original=image.read_bytes()
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='observation/images changed'):evaluate_base(p,ck)
    image.write_bytes(original)
    Path(scene.target_ref).write_bytes(b'changed')
    with pytest.raises(ValueError,match='target changed'):evaluate_base(p,ck)


def test_explicit_tampered_qualification_cannot_override_failed_assessment(tmp_path,monkeypatch):
    p,ck,report=quality_fixture(tmp_path,monkeypatch)
    failed=qualify_base(p,ck,report(ck,.1,'bad'))
    forged=tmp_path/'forged_qualification.json'
    atomic_json(forged,failed|{'status':'QUALIFIED','checks':{k:True for k in failed['checks']}})
    p.config['base_evaluation']['qualification']=str(forged)
    with pytest.raises(ValueError,match='assessment differs'):require_qualification(p,ck)


def test_preflight_posttrain_migration_preserves_model_hash_checks(tmp_path):
    from iqe.preflight import verify_existing_contract
    previous={'input':'locked','query_model_code_hashes':{'iqe/query_base.py':'original'},'iqe_code_hash':'historical_posttrain',
        'query_checkpoint':None,'query_checkpoint_hash':None,'binding_status':'FRAMEWORK_BOUND_QUERY_UNTRAINED'}
    previous['framework_contract_hash']=digest({k:v for k,v in previous.items() if k not in {'query_checkpoint','query_checkpoint_hash','binding_status'}})
    source=tmp_path/'contract.json';atomic_json(source,previous);checksum=file_hash(source)
    current=previous|{'iqe_code_hash':'new_posttrain','framework_contract_hash':'new_full_tree','workspace':{'commit':'unit'}}
    assert verify_existing_contract(previous,current,source,tmp_path/'out',dependency_hashes={'unit':'explicit_fixture'})==previous
    assert file_hash(source)==checksum and (tmp_path/'out/PREFLIGHT_COMPATIBILITY.json').is_file()
    changed=current|{'query_model_code_hashes':{'iqe/query_base.py':'changed'}}
    with pytest.raises(ValueError,match='model/input/source/metric'):verify_existing_contract(previous,changed,source,tmp_path/'out',dependency_hashes={'unit':'explicit_fixture'})
