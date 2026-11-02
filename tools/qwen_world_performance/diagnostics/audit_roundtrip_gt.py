"""Finite CPU TRAIN64 GT roundtrip diagnostic. Does not modify S6/teacher banks."""
from collections import Counter,defaultdict
from concurrent.futures import as_completed
from datetime import datetime,timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
P=Path('/mnt/project/qwen-world-performance-rl-20261103')
SOURCE=P/'runtime/formal_v1'
S6SOURCE=P/'runtime/verified_quality_v2'
RUN=P/'runs/S6_VERIFIED_QUALITY'
OLD=P/'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT'
OUT=P/'analysis/S6_FIRST_WINDOW_GT_FALLBACK_AUDIT'
sys.path.insert(0,str(OLD))
import audit_verified_gt as A
import numpy as np
import torch
from starVLA.rl.qwen_world_research_data.manifest import atomic_json,file_sha,fingerprint,read_sealed,physical
from starVLA.rl.qwen_world_research_data.data import ResearchSceneData
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.simwam_recipe.reward_readouts import RewardService

SCHEMA='world_S6_first_window_TRAIN_roundtrip_GT_fallback_audit_v1'
DATA=Path('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN')
RAW=Path('/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval')


def seal(x):
    x=A.native(x);return dict(x,sha256=fingerprint(x))


def status(stage,**kwargs):
    value=dict(schema=SCHEMA,stage=stage,at=datetime.now(timezone.utc).isoformat(),**kwargs)
    atomic_json(OUT/'STATUS.json',value);print(json.dumps(value),flush=True)


def prepare():
    review=read_sealed(P/'analysis/S6_ACTIVATION_REVIEW/REVIEW_u000004.json')
    identity=read_sealed(RUN/'identity.json')
    if review['common_updates']!=4 or review['run_identity_sha256']!=identity['sha256']:raise ValueError('Wrong sealed first window')
    rows={};behaviors={}
    for rank in range(8):
        path=RUN/'behavior/collection000001'/f'rank{rank}.pt';raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=review['evidence']['behavior_file_sha256'][str(path)]:raise ValueError('First behavior bytes changed')
        window=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if window['behavior_version']['update']!=0 or window['behavior_version']['run']!=identity['sha256']:raise ValueError('Frozen initial IL70 window required')
        behaviors[str(path)]=digest
        for leaf in window['samples']:
            t=leaf['token'];mat=leaf['nominal_improvement'];nominal=mat['nominal'];b1=nominal['feedback']
            main_good=any(A.strict_high(r['metrics']) for r in leaf['records'])
            extra=mat['extra'];extra_good=False if extra is None else any(A.strict_high(r['metrics']) for r in extra['records'])
            fallback=(not A.strict_high(b1['metrics'])) and not main_good and not extra_good
            if t in rows or not leaf['valid_scene'] or leaf['group_size']!=16:raise ValueError('64 unique valid main scenes required')
            if bool(extra)!=((not A.strict_high(b1['metrics'])) and not main_good):raise ValueError('Conditional extra gate differs')
            rows[t]=dict(token=t,log=leaf['log'],rank=rank,physical_log=physical(leaf['log']),
                B1_metrics=b1['metrics'],B1_strict_high=A.strict_high(b1['metrics']),
                B1_actual_requested_sha256=A.typed_sha(np.asarray(b1['requested_trajectory'],dtype=np.float32)),
                source_version=leaf['behavior_version'],cache=b1['cache'],
                protocol_sha256=b1['protocol_sha256'],main_G16_has_strict_high=main_good,
                conditional_extra_G16=extra is not None,extra_G16_has_strict_high=extra_good,
                fallback_no_sampled_strict_high=fallback,existing_aux_enabled=mat['enabled'],
                existing_skip_reason=mat['skip_reason'],existing_alpha=mat['active_alpha'],
                existing_teacher_score=None if not mat['enabled'] else mat['teacher']['score'],
                best_main_score=max(float(r['metrics']['score']) for r in leaf['records']))
    if len(rows)!=64 or sum(r['fallback_no_sampled_strict_high'] for r in rows.values())!=7:raise ValueError('Exact first64/7-fallback census changed')
    # Old prepare() is hard-coded to2048. Reuse its audited coordinate/time
    # equations and typed/cache validators; never monkeypatch its population.
    data=ResearchSceneData(DATA,role='NAVTRAIN')
    original_hash={};source_manifests={}
    for src in data.manifest['source_records']:
        path=Path(src['root'])/'manifest.json'
        if file_sha(path)!=src['manifest_sha256']:raise ValueError('Original IL manifest changed')
        m=json.loads(path.read_text());original_hash.update(m['raw_logs_sha256']);source_manifests[str(path)]=file_sha(path)
    protocol=v1_reward.protocol_identity(SOURCE)
    if any(r['protocol_sha256']!=protocol['sha256'] for r in rows.values()):raise ValueError('Actual S6 B1 scorer differs')
    source_files={}
    for n in ('starVLA/model/modules/qwen_sourceflow/action_codec.py','starVLA/rl/simwam_recipe/reward_readouts.py',
              'starVLA/rl/safe_flow_improvement/v1_reward.py','scripts/analysis/constrained_flow_noise/batched_canonical_score.py'):
        f=file_sha(SOURCE/n);s=file_sha(S6SOURCE/n)
        if f!=s:raise ValueError('Audited and S6 numerical scorer/codec sources differ: '+n)
        source_files[n]=dict(audited_formal_v1=f,actual_S6_source8067=s)
    spec=dict(repo=str(SOURCE),maps_root='/mnt/navsim/maps',cache_roots=data.manifest['canonical_metric_cache_roots'],protocol_sha256=protocol['sha256'])
    v1_reward.initialize(spec)
    from navsim.common.dataclasses import Scene,SceneMetadata
    groups=defaultdict(list)
    for t,r in rows.items():
        if t not in data.records or r['log']!=data.logs[t]:raise ValueError('TRAIN token/log join differs')
        groups[r['log']].append(t)
    raw_sources={}
    for log,tokens in groups.items():
        path=RAW/(log+'.pkl');digest=file_sha(path)
        if digest!=original_hash[log]:raise ValueError('Raw TRAIN log changed')
        raw_sources[str(path)]=digest
        with path.open('rb') as f:frames=pickle.load(f)
        indices={f['token']:i for i,f in enumerate(frames)}
        for token in tokens:
            r=rows[token];rec=data.records[token];idx=indices[token];future=frames[idx:idx+9];cur=future[0]
            if len(future)!=9 or rec['token']!=token or rec['log']!=log:raise ValueError('Raw horizon/record join changed')
            scene=Scene(SceneMetadata(log,cur['scene_token'],cur['map_location'],token,1,8),None,
                [SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in future])
            requested=data.target(token).numpy()
            if not np.array_equal(scene.get_future_trajectory(8).poses.astype(np.float32),requested):raise ValueError('GT coordinate provenance differs')
            dt=np.diff([float(f['timestamp']) for f in future])/1e6
            if int(cur['timestamp'])!=rec['timestamp'] or not np.array_equal(dt,np.asarray(rec['raw_future_dt_seconds'])) or not (dt>0).all():raise ValueError('Raw timestamp join differs')
            mask=((torch.tensor(rec['raw_future_dt_seconds']).cumsum(0)-torch.arange(1,9)*.5).abs()<.06).numpy()
            clean=ActionCodec.norm_odo(torch.from_numpy(requested));roundtrip=ActionCodec.denorm_odo(clean).numpy()
            if not np.isfinite(roundtrip).all():raise ValueError('Nonfinite real FP32 codec request')
            r.update(status='VALID_UNSCORED' if bool(mask.all()) else 'INVALID_TIME_NOT_A_TEACHER',
                eligible_for_scoring=bool(mask.all()),IL_action_valid_mask=mask.tolist(),raw_future_dt_seconds=dt.tolist(),
                actual_time_error_seconds=(np.cumsum(dt)-np.arange(1,9)*.5).tolist(),raw_log_sha256=digest,
                timestamp_us=int(cur['timestamp']),rear_axle_origin=scene.frames[0].ego_status.ego_pose.tolist(),
                official_GT_FP32_bitexact=True,raw_GT=requested.tolist(),clean_normalized=clean.numpy().tolist(),
                actual_roundtrip_requested=roundtrip.tolist(),raw_GT_typed_sha256=A.typed_sha(requested),
                roundtrip_requested_typed_sha256=A.typed_sha(roundtrip),
                codec_roundtrip_max_abs_by_dimension=np.max(np.abs(roundtrip-requested),axis=0).tolist())
    fallback=[t for t,r in rows.items() if r['fallback_no_sampled_strict_high']]
    controls=sorted([t for t,r in rows.items() if r['B1_strict_high']],key=lambda t:fingerprint(['S6first_GT_raw_roundtrip_control',42,t]))[:4]
    dual=set(fallback+controls)
    for t,r in rows.items():r['raw_roundtrip_double_score']=t in dual and r['eligible_for_scoring']
    expected=sum((2 if r['raw_roundtrip_double_score'] else 1) for r in rows.values() if r['eligible_for_scoring'])
    if expected>75 or expected>128:raise ValueError('Finite CPU budget exceeded')
    pop=json.loads(Path(data.manifest['source_population_path']).read_text());tt=set(rows)
    if tt&set(pop['navtest_tokens']) or {r['physical_log'] for r in rows.values()}&{physical(pop['token_logs'][t]) for t in pop['navtest_tokens']}:raise ValueError('NAVTEST contamination')
    protocol_doc=seal(dict(schema=SCHEMA,status='PREPARED_NOT_TRAINING',source=str(SOURCE),actual_S6_source=str(S6SOURCE),
        S6_source_commit='8067e13db926ed23b801133921095e99c5c54a3b',script_sha256=file_sha(__file__),
        reused_audit_helper_path=str(OLD/'audit_verified_gt.py'),reused_audit_helper_sha256=file_sha(OLD/'audit_verified_gt.py'),
        source_files=source_files,source_manifests=source_manifests,raw_TRAIN_inputs=raw_sources,
        native_first_window_identity_sha256=identity['sha256'],behavior_file_sha256=behaviors,
        data_manifest_sha256=data.sha,source_population_sha256=data.manifest['population_sha256'],
        current_TRAIN_scenes=64,source_TRAIN_count=len(tt&set(pop['train_tokens'])),old_DEV_in_existing_NAVTRAIN_count=len(tt&set(pop['dev_tokens'])),
        NAVTEST_trajectory_or_label_read=False,coordinate_contract='official rear-axle local x/y/relativeyaw eight poses nominal0.5..4sec, exact IL FP32 targets',
        time_gate='all8 original IL action_valid(abs(FP32 cumulative_dt-nominal)<.06); no resampling or fake timing',
        target='CPU FP32 ActionCodec.denorm_odo(ActionCodec.norm_odo(original_GT)); actually scored request bytes saved',
        strict4='NC=DAC=TTC=Comfort=1',strict_high='strict4 AND canonical>=.9',
        teacher_gates='B1 notstrict-high:gain>=.02; alreadystrict-high:gain>=.005',
        requested_sensitivity_raw_double_score_tokens=sorted(dual),first_fallback_tokens=fallback,
        expected_new_PDM_candidate_queries=expected,maximum=128,workers=2,
        scoring_protocol=protocol,training_updates=0,formal_S6_changes=False,teacher_bank_created=False,
        limits=['This score certifies the exact saved CPU FP32 roundtrip request, not an unmeasured CUDA decode with potentially different rounding.',
            'Human GT is privileged offline TRAIN supervision, not a current-policy sample; never eligible for on-policy advantages.',
            'No teacher threshold is lowered. Invalid-time targets are omitted as teachers, never assigned reward0.',
            'Conditional extra teacher pool is independent G16, not true batch32; first-window findings are finite realized sampling evidence.']))
    atomic_json(OUT/'PROTOCOL.json',protocol_doc);atomic_json(OUT/'PREPARED.json',seal(dict(schema=SCHEMA,rows=rows,protocol_sha256=protocol_doc['sha256'])))
    tokens=list(rows);np.savez_compressed(OUT/'TRAJECTORIES.npz',tokens=np.array(tokens),
        raw_GT_FP32=np.asarray([rows[t]['raw_GT'] for t in tokens],dtype=np.float32),
        clean_normalized_FP32=np.asarray([rows[t]['clean_normalized'] for t in tokens],dtype=np.float32),
        actual_roundtrip_requested_FP32=np.asarray([rows[t]['actual_roundtrip_requested'] for t in tokens],dtype=np.float32),
        whole_teacher_valid=np.array([rows[t]['eligible_for_scoring'] for t in tokens]))
    return rows,spec,protocol_doc


def summarize(rows,protocol,ledger):
    scored=[r for r in rows.values() if r['status']=='SCORED']
    cohorts={'all':scored,'fallback7': [r for r in scored if r['fallback_no_sampled_strict_high']],
        'B1_already_strict_high':[r for r in scored if r['B1_strict_high']],
        'B1_not_strict_high':[r for r in scored if not r['B1_strict_high']],
        'existing_aux_disabled':[r for r in scored if not r['existing_aux_enabled']]}
    results={}
    for name,rr in cohorts.items():
        results[name]=dict(scored_count=len(rr),GT_roundtrip_strict4=sum(r['GT_strict4'] for r in rr),
            GT_roundtrip_strict_high=sum(r['GT_strict_high'] for r in rr),
            teacher_meets_actual_gate=sum(r['qualifying_teacher'] for r in rr),
            new_qualifying_teacher_for_currently_disabled_scene=sum(r['qualifying_teacher'] and not r['existing_aux_enabled'] for r in rr),
            gain_ge02=sum(r['gain']>=.02 for r in rr),gain_ge005=sum(r['gain']>=.005 for r in rr),
            score_worse_lt_minus005=sum(r['gain']<-.005 for r in rr),gain_distribution=A.quantiles([r['gain'] for r in rr]),
            GT_roundtrip_score_distribution=A.quantiles([r['GT_metrics']['score'] for r in rr]))
    sensitivity=[r for r in scored if r['raw_roundtrip_double_score']]
    report=dict(schema=SCHEMA,status='COMPLETE_ALL64_INPUTS_ACCOUNTED',at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=protocol['sha256'],input_scenes=64,scored_roundtrip_scenes=len(scored),
        invalid_time_scenes=[dict(token=r['token'],log=r['log'],IL_action_valid_mask=r['IL_action_valid_mask']) for r in rows.values() if not r['eligible_for_scoring']],
        cohort_statistics=results,
        fallback_cases=[{k:r.get(k) for k in ('token','log','status','B1_metrics','GT_metrics','gain','GT_strict4','GT_strict_high','qualifying_teacher','existing_skip_reason','raw_metrics','raw_roundtrip_metric_delta')} for r in rows.values() if r['fallback_no_sampled_strict_high']],
        raw_roundtrip_sensitivity=dict(double_scored_scenes=len(sensitivity),
            strict_high_changed=sum(r['raw_GT_strict_high']!=r['GT_strict_high'] for r in sensitivity),
            strict4_changed=sum(r['raw_GT_strict4']!=r['GT_strict4'] for r in sensitivity),
            max_abs_score_delta=max((abs(r['raw_roundtrip_metric_delta']['score']) for r in sensitivity),default=0),
            source='first fallback7 plus4 preselected alreadystrict controls; not all64 raw GT rescored'),
        actual_budget=dict(new_roundtrip_queries=len(scored),new_raw_comparison_queries=len(sensitivity),
            actual_new_PDM_candidate_queries=ledger['completed_queries'],reserved=ledger['reserved_queries'],maximum=128,
            scorer_submissions=len(scored),reference_simulations=len(scored),workers=2,new_GPU_queries=0,training_updates=0),
        no_formal_recipe_bank_or_priority_mutation=True,
        limitations=protocol['limits'])
    atomic_json(OUT/'SUMMARY.json',seal(report))
    with (OUT/'SCENES.jsonl').open('w') as f:
        for r in rows.values():f.write(json.dumps(A.native(r),sort_keys=True,allow_nan=False)+'\n')
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'scenes').mkdir(exist_ok=True)
    lock=(OUT/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'BUDGET_LEDGER.json').exists():raise RuntimeError('Preserve existing budget; no automatic duplicate scoring')
    torch.set_num_threads(1);status('GT_COORDINATE_TIME_AND_NATIVE_JOIN_PREFLIGHT')
    rows,spec,protocol=prepare()
    ledger=dict(schema=SCHEMA,maximum=128,reserved_queries=0,completed_queries=0,scenes={})
    atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
    service=RewardService(SOURCE,workers=2,cache_roots=spec['cache_roots'])
    try:
        legal=[t for t,r in rows.items() if r['eligible_for_scoring']]
        futures={service.pool.submit(A.validate_cache,(t,rows[t]['log'],rows[t]['cache'],rows[t]['rear_axle_origin'],rows[t]['timestamp_us'])):t for t in legal}
        for future in as_completed(futures):rows[futures[future]]['cache_origin_validation']=future.result()
        status('ROUNDTRIP_GT_CPU_SCORING',legal=len(legal),expected_new_queries=protocol['expected_new_PDM_candidate_queries'])
        # The fallback cases are queued first, followed by controls and remaining scenes.
        selected=sorted(legal,key=lambda t:(not rows[t]['fallback_no_sampled_strict_high'],not rows[t]['raw_roundtrip_double_score']))
        futures={}
        for t in selected:
            r=rows[t];requests=[r['actual_roundtrip_requested']]
            if r['raw_roundtrip_double_score']:requests.append(r['raw_GT'])
            n=len(requests)
            if ledger['reserved_queries']+n>128:raise RuntimeError('Hard query cap')
            ledger['reserved_queries']+=n;ledger['scenes'][t]=dict(status='RESERVED',candidate_queries=n)
            atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
            futures[service.submit((t,r['log'],np.asarray(requests,dtype=np.float32)))]=t
        for future in as_completed(futures):
            t=futures[future];r=rows[t];feedback=future.result();expected=2 if r['raw_roundtrip_double_score'] else 1
            if len(feedback)!=expected:raise ValueError('Lost GT candidate')
            for index,f in enumerate(feedback):
                expected_sha=r['roundtrip_requested_typed_sha256'] if index==0 else r['raw_GT_typed_sha256']
                if f['status']!='OK' or f['token']!=t or f['log']!=r['log'] or f['cache']['sha256']!=r['cache']['sha256'] or f['protocol_sha256']!=spec['protocol_sha256'] or A.typed_sha(f['requested_trajectory'])!=expected_sha:raise ValueError('Scored request/protocol/cache differs')
            m=feedback[0]['metrics'];r.update(status='SCORED',GT_metrics=m,GT_strict4=A.strict4(m),GT_strict_high=A.strict_high(m),gain=float(m['score']-r['B1_metrics']['score']))
            margin=.005 if r['B1_strict_high'] else .02;r['qualifying_teacher']=bool(r['GT_strict_high'] and r['gain']>=margin)
            if expected==2:
                rm=feedback[1]['metrics'];r.update(raw_metrics=rm,raw_GT_strict4=A.strict4(rm),raw_GT_strict_high=A.strict_high(rm),raw_roundtrip_metric_delta={k:float(m[k]-rm[k]) for k in m})
            atomic_json(OUT/'scenes'/(t+'.json'),seal(dict(r,roundtrip_feedback=feedback[0],raw_feedback=feedback[1] if expected==2 else None)))
            ledger['completed_queries']+=expected;ledger['scenes'][t]['status']='COMPLETE';atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
    except Exception as exc:
        atomic_json(OUT/'FAILED.json',seal(dict(schema=SCHEMA,error_type=type(exc).__name__,message=str(exc),budget=ledger)));raise
    finally:service.close()
    for t,r in rows.items():
        if not r['eligible_for_scoring']:atomic_json(OUT/'scenes'/(t+'.json'),seal(r))
    report=summarize(rows,protocol,ledger)
    if ledger['reserved_queries']!=ledger['completed_queries']:raise ValueError('Unaccounted query reservation')
    atomic_json(OUT/'COMPLETE.json',seal(dict(schema=SCHEMA,status='COMPLETE',input_scenes=64,
        actual_new_PDM_candidate_queries=ledger['completed_queries'],maximum=128,
        files_sha256={name:file_sha(OUT/name) for name in ('audit_roundtrip_gt.py','PROTOCOL.json','PREPARED.json','SUMMARY.json','SCENES.jsonl','TRAJECTORIES.npz','BUDGET_LEDGER.json')})))
    status('COMPLETE',actual_new_PDM_queries=ledger['completed_queries'],cohorts=report['cohort_statistics'])


if __name__=='__main__':main()
