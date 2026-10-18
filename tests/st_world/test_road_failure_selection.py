"""Synthetic CPU provenance fixtures, never real scoring or science evidence."""
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.st_world_policy import run_road_direction as runner
from tools.st_world_policy import road_failure_selection as selection
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.flow_posttraining.policy_snapshot import PolicyIdentity
from starVLA.model.modules.flow_posttraining.rollout_contract import RolloutBatch
from starVLA.model.modules.st_world.consequence import (
    ConsequenceRequest, METRICS, _reference_evidence, array_sha256, fingerprint, road_summary)


def put(path,value,*,tensor=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if tensor:torch.save(value,path)
    else:path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    raw=path.read_bytes()
    return {'path':str(path),'sha256':runner.sha(raw),'bytes':len(raw)}


def fixture_frame(tmp_path,cases=2):
    """Two non4095 boundaries, nonzero ranks/slots, all four actual raw heads."""
    run=tmp_path/selection.SOURCE_RUN_NAME
    entries=[put(run/'identity.json',{'fixture':'identity'}),put(run/'RESULT.json',{'fixture':'complete'}),
             put(tmp_path/'registrations'/selection.SOURCE_RUN_NAME/'REGISTRATION.json',{'fixture':'reg'})]
    frame_rows=[];train=[]
    for n in range(cases):
        boundary,rank,slot=(7,2,3) if n==0 else (13+n,1,6)
        sid=f'fixture_scene_{n}';log=f'fixture_log_{n}';train.append({'token':sid,'log':log})
        policy=PolicyIdentity(boundary,boundary,'a'*64,'b'*64,'old')
        raw=torch.randn((8,4,8,4),generator=torch.Generator().manual_seed(90+n));raw[:,:,:,3]+=4
        fields=dict(scene_ids=tuple(sid if s==slot else f'unselected_{n}_{s}' for s in range(8)),
            observation_hashes=tuple('c'*64 for _ in range(8)),condition_hashes=tuple('d'*64 for _ in range(8)),
            policy=policy,initial_noise=torch.zeros_like(raw),raw_actions=raw)
        draft=RolloutBatch(**fields,rollout_id='')
        rollout=RolloutBatch(**fields,rollout_id=draft.content_identity())
        identity={'rollout_id':rollout.rollout_id,'old_policy':asdict(policy),'fixture':'feedback'}
        payload=put(run/'rollouts'/f'boundary_{boundary:06d}'/f'rank_{rank}.pt',
            {'schema':'st_flow_update_feedback_v1','identity':identity,'rollout':rollout.state_dict()},tensor=True)
        receipt=put(Path(payload['path']).with_name(f'rank_{rank}_COMPLETE.json'),
            {'identity':identity,'payload':{k:payload[k] for k in ('path','sha256')}})
        request_rows=[];record_rows=[]
        # Failures deliberately at1/3, safe siblings0/2. Selection is by fixed
        # seeded IDs, so the selected failing candidate need not be candidate0.
        for c in range(4):
            physical=decode_ego(raw[slot,c]).double().numpy()
            origin={'kind':'in_memory_parameter_snapshot','state_sha256':policy.state_hash,
                'optimizer_step':boundary,'policy_version':str(boundary),'parent_checkpoint_sha256':'e'*64,
                'parent_receipt_sha256':'f'*64,'numeric_execution_hash':'1'*64}
            request=ConsequenceRequest(campaign_id='fixture',scene_id=sid,source_log=log,
                candidate_id=f'{rollout.rollout_id}:{slot}:{c}',policy_version=str(boundary),
                generator_checkpoint_sha256=None,generator_kind='policy_snapshot',generator_identity=origin,
                generator_source_hash=fingerprint(origin),inference_runtime_hash='1'*64,
                trajectory_sha256=array_sha256(physical),metric_cache_sha256='2'*64,score_protocol_hash='3'*64,
                metric_cache_path=str(tmp_path/'contexts'/log/'unknown'/sid/'metric_cache.pkl'),
                trajectory=physical,observe=True)
            metrics={key:1. for key in METRICS};metrics.update(score=.9 if c%2==0 else 0.,drivable_area_compliance=1. if c%2==0 else 0.)
            flags=np.zeros(41,dtype=bool);flags[10]=c%2==1
            states=np.zeros((2,41,11),dtype=np.float64)
            record={'identity':request.identity(),'valid':True,'status':'VALID_SCORED','score_units':'0..1',
                'context_unchanged':True,'context_hash':'4'*64,'metrics':metrics,
                'metrics_valid':{k:True for k in METRICS},'requested_trajectory':physical,'executed_states':states[1].copy(),
                'consequences':{'road':road_summary(flags)},
                **_reference_evidence(request,'4'*64,states[0],states,{'dependencies':{'numpy':'fixture'}})}
            request_rows.append(request.identity());record_rows.append(record)
        ledger=f'fixture_ledger_{n}'
        request_ref=put(run/'official_feedback'/ledger/'REQUESTS.json',
            {'policy_version':str(boundary),'requests':request_rows})
        result_ref=put(run/'official_feedback'/ledger/'attempt_00'/'RESULTS.pt',record_rows,tensor=True)
        admitted=put(Path(result_ref['path']).with_name('ADMITTED.json'),{'fixture':'admitted'})
        complete=put(Path(result_ref['path']).with_name('COMPLETE.json'),
            {'admission_sha256':admitted['sha256'],'result':{k:result_ref[k] for k in ('path','sha256')}})
        frame_rows.append({'scene_id':sid,'source_log':log,'role':'train','policy_version':boundary,'rank':rank,
            'local_slot':slot,'global_scene_slot':rank*8+slot,'ledger':ledger,
            'original_rollout_id':rollout.rollout_id,'original_policy_identity':asdict(policy),
            'original_requests':request_ref,'original_feedback_payload':result_ref,
            'original_rank_receipt':receipt,'original_rollout_payload':payload,
            'original_input_hashes':{'REQUESTS':request_ref['sha256'],'RESULTS':result_ref['sha256'],
                                     'ADMITTED':admitted['sha256'],'COMPLETE':complete['sha256']}})
    index=put(tmp_path/'student_train_v1'/'index.json',train);entries.append(index)
    legacy=put(tmp_path/'legacy_evidence.json',{'source_runtime_commit':selection.SOURCE_RUNTIME_COMMIT,
        'source_runtime':'/fixture/unchanged_runtime','sources':[],'budget':{'entries':entries},
        'train_index_identity':{'index_sha256':fingerprint(train)}})
    frame={'schema':selection.FRAME_SCHEMA,'domain':selection.DOMAIN,'seed':42,'run':str(run),
           'selected':frame_rows,'selection_frame_sha256':selection.canonical_hash(frame_rows)}
    frame_ref=put(tmp_path/'FRAME.json',frame)
    return frame_ref,legacy,frame


def bound_fixture(tmp_path,cases=2):
    frame,legacy,content=fixture_frame(tmp_path,cases)
    budget=runner.Budget(tmp_path/'binding',runner.BINDING_LIMITS)
    evidence,audit=selection.bind_frame(frame['path'],frame['sha256'],legacy['path'],budget,runner.decode_discrepancy_gate)
    return evidence,audit,content


def test_binding_and_loading_arbitrary_original_boundary_rank_candidate(tmp_path):
    evidence,audit,frame=bound_fixture(tmp_path)
    assert audit['new_official_calls']==audit['proxy_calls']==audit['model_loads']==0
    assert audit['bound_scenes']==2 and not audit['cuda_initialized']
    assert all(item['failed']['candidate'] in (1,3) for item in evidence['selected'])
    assert [item['boundary'] for item in evidence['selected']]==[7,14]
    loaded=runner.load_inputs(evidence,runner.Budget(tmp_path/'execute'))
    assert [row[0]['scene_id'] for row in loaded]==[row['scene_id'] for row in frame['selected']]
    for item,raw,request,record in loaded:
        assert raw.dtype==torch.float32 and raw.shape==(8,4)
        assert runner.sha(raw.numpy().tobytes())==item['failed']['raw_sha256']
        np.testing.assert_array_equal(decode_ego(raw).double().numpy(),request.trajectory)
        assert record['metrics']['drivable_area_compliance']==0


def test_no_metadata_only_frame_can_prepare_as_selection(tmp_path):
    frame,_,_=fixture_frame(tmp_path)
    with pytest.raises((ValueError,KeyError)):
        runner.registration(frame['path'],tmp_path/'out','failure_cohort_v1')


def test_legacy_schema_cannot_silently_activate_failure_limits():
    with pytest.raises(ValueError,match='execution schema'):
        runner.execution_limits({'schema':runner.SCHEMA,'selection_mode':'failure_cohort_v1'})


def test_failure_prepare_explicit_mode_and_same_path_content_refused(tmp_path):
    evidence,_,_=bound_fixture(tmp_path)
    path=tmp_path/'selection.json';put(path,evidence)
    reg=runner.registration(path,tmp_path/'out','failure_cohort_v1')
    assert reg['limits']['max_official_calls']==96 and reg['requested_scenes']==2
    assert reg['selection_hash']==selection.canonical_hash(evidence['selected'])
    old_hash=reg['evidence']['sha256']
    evidence['selected'][0]['failed']['raw_sha256']='9'*64
    put(path,evidence)
    assert runner.sha(path.read_bytes())!=old_hash
    with pytest.raises(ValueError,match='content hash'):
        runner.registration(path,tmp_path/'out','failure_cohort_v1')


@pytest.mark.parametrize('change',['duplicate','source','candidate','context','sibling','provenance','frame_order'])
def test_negative_selection_contracts(tmp_path,change):
    evidence,_,frame=bound_fixture(tmp_path)
    row=evidence['selected'][0]
    if change=='duplicate':evidence['selected'][1]=deepcopy(row)
    elif change=='source':row['source_log']='foreign_log'
    elif change=='candidate':row['failed']['candidate']=4
    elif change=='context':row['safe_sibling']['context_hash']='9'*64
    elif change=='sibling':row.pop('safe_sibling')
    elif change=='provenance':row['artifacts'].pop('complete')
    else:evidence['selected'].reverse()
    evidence['selection_sha256_before_proxy']=selection.canonical_hash(evidence['selected'])
    with pytest.raises((ValueError,KeyError)):
        selection.validate_selection(evidence,frame)


def test_bound_raw_mismatch_rejects_even_with_selection_hash_updated(tmp_path):
    evidence,_,_=bound_fixture(tmp_path)
    evidence['selected'][0]['failed']['raw_sha256']='9'*64
    evidence['selection_sha256_before_proxy']=selection.canonical_hash(evidence['selected'])
    with pytest.raises(ValueError,match='FP32 raw'):
        runner.load_inputs(evidence,runner.Budget(tmp_path/'execute'))


def test_fixed_candidate_rule_checked_against_all_four_siblings(tmp_path):
    evidence,_,_=bound_fixture(tmp_path)
    row=evidence['selected'][0]
    other=3 if row['failed']['candidate']==1 else 1
    payload=torch.load(row['artifacts']['rollout']['path'],weights_only=False)
    records=torch.load(row['artifacts']['results']['path'],weights_only=False)
    cid=f"{row['rollout_id']}:{row['slot']}:{other}"
    record=next(rec for rec in records if rec['identity']['candidate_id']==cid)
    old=row['failed']
    row['failed']={**old,'candidate':other,'request':record['identity'],
        'raw_sha256':runner.sha(payload['rollout']['raw_actions'][row['slot'],other].numpy().tobytes())}
    evidence['selection_sha256_before_proxy']=selection.canonical_hash(evidence['selected'])
    with pytest.raises(ValueError,match='seeded'):
        runner.load_inputs(evidence,runner.Budget(tmp_path/'execute'))


def test_native_t0_requires_actual_bool_trace_and_dac_consistency():
    record={'consequences':{'road':{'native_flag':np.zeros(41,dtype=bool)}},
            'metrics':{'drivable_area_compliance':1.}}
    assert selection.native_t0(record) is False
    record['consequences']['road']['native_flag'][0]=True
    with pytest.raises(ValueError,match='disagrees'):selection.native_t0(record)
    record['metrics']['drivable_area_compliance']=0.
    assert selection.native_t0(record) is True
    record['consequences']['road']['native_flag']=[0]*41
    with pytest.raises(ValueError,match='bool'):selection.native_t0(record)


def test_summary_has_nonzero_actual_failure_denominator_and_unknowns(tmp_path):
    evidence,_,_=bound_fixture(tmp_path)
    row=evidence['selected'][0];root=tmp_path/'out'
    variants={}
    for name in runner.VARIANTS:
        metrics=deepcopy(row['failed']['metrics'])
        metrics.update(score=.5 if name=='negative' else 0.,drivable_area_compliance=1. if name=='negative' else 0.)
        variants[name]={'metrics':metrics,'geometry':{'custom_executed_cost_m':.1 if name=='negative' else .2}}
    put(root/'scenes'/row['scene_id']/'SCENE_RESULT.json',{'scene_id':row['scene_id'],'status':'COMPLETE',
        'original_metrics':row['failed']['metrics'],'variants':variants})
    other=evidence['selected'][1]
    put(root/'scenes'/other['scene_id']/'SCENE_RESULT.json',{'scene_id':other['scene_id'],
        'status':'UNDEFINED_DIRECTION','error':'ZERO_OR_NONFINITE_GRADIENT','original_metrics':other['failed']['metrics']})
    result=runner.summarize(root,evidence,{'worker_exit_code':0})
    assert result['requested_scenes']==2 and result['road_repair_denominator']==2
    assert result['complete_scenes']==1 and result['unresolved_or_undefined_scenes']==1
    assert result['DAC_repair']['negative']=={'requested_original_failure_denominator':2,
        'common_valid_original_failure_denominator':1,'repaired_DAC_scenes':1,'unresolved_original_failures':1,
        'repair_rate_valid_subset':1.,'known_repairs_over_requested_lower_bound':.5}
    assert result['valid_subset_only_direction_contrasts']['negative_minus_positive']['negative_lower_cost_scenes']==1
    assert not result['population_summary_complete']


def test_per_case_and_global_new_call_guards_no_original_or_retry(tmp_path):
    root=tmp_path/'out';root.mkdir();budget=runner.Budget(root)
    item={'scene_id':'fixture','candidate_id':'origin'}
    for name in runner.VARIANTS[1:]:
        runner.charge_official_call(root,item,{'candidate_id':'origin:road_direction_v1:'+name},budget,
                                   runner.FAILURE_LIMITS,failure_mode=True)
    for candidate_id in ('origin','origin:road_direction_v1:negative'):
        with pytest.raises(runner.LimitError,match='PER_SCENE'):
            runner.charge_official_call(root,item,{'candidate_id':candidate_id},budget,
                                       runner.FAILURE_LIMITS,failure_mode=True)
    for n in range(3,96):put(root/'calls'/f'{n:03d}.json',{'scene_id':str(n),'identity':{'candidate_id':str(n)}})
    with pytest.raises(runner.LimitError,match='OFFICIAL_CALL_CAP'):
        runner.charge_official_call(root,{'scene_id':'new','candidate_id':'new'},
            {'candidate_id':'new:road_direction_v1:negative'},budget,runner.FAILURE_LIMITS,failure_mode=True)
    assert len(list((root/'calls').glob('*.json')))==96


def test_binding_source_mutation_stops_without_replacement_or_proxy(tmp_path):
    frame,legacy,content=fixture_frame(tmp_path)
    Path(content['selected'][0]['original_rollout_payload']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='ASSET_CHANGED'):
        selection.bind_frame(frame['path'],frame['sha256'],legacy['path'],runner.Budget(tmp_path/'out',runner.BINDING_LIMITS),
                             lambda *args:pytest.fail('No decode beyond corrupt provenance'))
    assert [r['scene_id'] for r in content['selected']]==['fixture_scene_0','fixture_scene_1']


def test_new_mode_still_needs_exact_independent_admission(tmp_path,monkeypatch):
    from test_run_road_direction import _admission_fixture
    rp,ap,admission=_admission_fixture(tmp_path,monkeypatch)
    value=json.loads(rp.read_text());value.update(schema=selection.EXECUTION_SCHEMA,
        selection_mode='failure_cohort_v1',limits=runner.FAILURE_LIMITS)
    # A valid legacy approval cannot silently authorize the new schema/population.
    rp.write_text(json.dumps(value))
    with pytest.raises((ValueError,KeyError)):
        runner.check_admission(rp,ap,runner.sha(ap.read_bytes()))


def test_new_mode_exact_approval_and_same_path_replacement_rejected(tmp_path,monkeypatch):
    from test_run_road_direction import _admission_fixture
    rp,ap,admission=_admission_fixture(tmp_path,monkeypatch)
    evidence,_,_=bound_fixture(tmp_path/'sources')
    selection_path=tmp_path/'selection.json';put(selection_path,evidence)
    reg=runner.registration(selection_path,tmp_path/'out','failure_cohort_v1')
    rp.write_text(json.dumps(reg));reg_hash=runner.sha(rp.read_bytes())
    admission.update(registration_sha256=reg_hash)
    for kind,ref in admission['reviews'].items():
        path=Path(ref['path']);review=json.loads(path.read_text())
        review['registration_sha256']=reg_hash;path.write_text(json.dumps(review))
        ref['sha256']=runner.sha(path.read_bytes())
    ap.write_text(json.dumps(admission))
    assert runner.check_admission(rp,ap,runner.sha(ap.read_bytes()))[0]['limits']['max_official_calls']==96
    selection_path.write_text(json.dumps(evidence)+'\n')
    with pytest.raises(ValueError,match='selection changed'):
        runner.check_admission(rp,ap,runner.sha(ap.read_bytes()))


def test_t0_undefined_preserves_requested_failure_population(tmp_path):
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import DiagnosticUndefined
    evidence,_,_=bound_fixture(tmp_path)
    evidence['selected'][0]['failed']['native_t0']=True
    flags=np.zeros(41,dtype=bool);flags[0]=True
    record={'metrics':evidence['selected'][0]['failed']['metrics'],
            'consequences':{'road':{'native_flag':flags}}}
    with pytest.raises(DiagnosticUndefined,match='IRREPARABLE_NATIVE'):
        runner.require_repairable_native_t0(record)
    result=runner.summarize(tmp_path/'never_started',evidence,{})
    assert result['requested_scenes']==result['road_repair_denominator']==2
    assert result['original_t0_failure_scenes']==1 and result['eligible_original_DAC_failures']==1
    assert result['complete_scenes']==0 and len(result['scenes'])==2


def test_binding_cli_subprocess_and_execution_read_plan(tmp_path):
    import os,subprocess
    frame,legacy,_=fixture_frame(tmp_path/'sources')
    # Opaque stand-ins for stat-only cache presence; no fixture cache is opened.
    content=json.loads(Path(frame['path']).read_text())
    for item in content['selected']:
        folder=tmp_path/'sources'/'contexts'/item['source_log']/'unknown'/item['scene_id']
        folder.mkdir(parents=True);(folder/'metric_cache.pkl').write_bytes(b'fixture-not-read')
    output=tmp_path/'out'/'SELECTION.json'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=['/root/miniconda3/envs/ddp/bin/python',str(runner.ROOT/'tools/st_world_policy/run_road_direction.py'),
        '--bind-failure-selection','--frame',frame['path'],'--frame-sha256',frame['sha256'],
        '--evidence',legacy['path'],'--selection-output',str(output)]
    completed=subprocess.run(command,cwd=runner.ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stderr+completed.stdout
    evidence=json.loads(output.read_text());receipt=json.loads(output.with_name('SELECTION_BINDING_REPORT.json').read_text())
    assert receipt['status']=='BOUND_BEFORE_PROXY_NOT_EXECUTION_APPROVAL'
    assert receipt['population']['requested_scene_ids']==receipt['population']['bound_scene_ids']
    assert len(receipt['cpu_affinity'])<=2
    assert receipt['audit']['execution_read_plan']['total_worker_asset_bytes']<runner.FAILURE_LIMITS['asset_read_bytes']
    assert receipt['audit']['execution_read_plan']['metric_cache_content_reads']==0
    repeated=subprocess.run(command,cwd=runner.ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert repeated.returncode==2 and 'already exists' in repeated.stderr
    assert not torch.cuda.is_initialized()


def test_binding_cli_missing_source_retains_failed_attempt_population(tmp_path):
    import os,subprocess
    frame,legacy,content=fixture_frame(tmp_path/'sources')
    Path(content['selected'][1]['original_rollout_payload']['path']).unlink()
    output=tmp_path/'out'/'SELECTION.json'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    completed=subprocess.run(['/root/miniconda3/envs/ddp/bin/python',str(runner.ROOT/'tools/st_world_policy/run_road_direction.py'),
        '--bind-failure-selection','--frame',frame['path'],'--frame-sha256',frame['sha256'],
        '--evidence',legacy['path'],'--selection-output',str(output)],
        cwd=runner.ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert completed.returncode==2 and not output.exists()
    receipt=json.loads(output.with_name('SELECTION_BINDING_REPORT.json').read_text())
    assert receipt['status']=='FAILED_CLOSED'
    assert receipt['population']['requested_scene_ids']==['fixture_scene_0','fixture_scene_1']
    assert receipt['population']['bound_scene_ids']==['fixture_scene_0']
    assert receipt['population']['replacement_scenes']==0
    assert receipt['new_official_calls']==receipt['proxy_calls']==0
