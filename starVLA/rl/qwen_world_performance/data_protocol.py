"""Sealed full-NAVTRAIN membership and TRAIN-only own-policy B1 feedback."""
from pathlib import Path
import json
from starVLA.rl.qwen_world_research_data.manifest import (
    atomic_json, file_sha, fingerprint, physical, read_sealed, sealed)

STREAM_SCHEMA = 'world_performance_scene_stream_v1'
FEEDBACK_SCHEMA = 'world_performance_own_TRAIN_B1_feedback_v1'
DEFAULT_SOURCE = '/mnt/project/qwen-world-rl-optimization-20261030/data/manifests/FULL_103288.json'
DEFAULT_POLICY = dict(schema='world_performance_60_25_15_TRAIN_hard_regression_v1',
    warmup_full_epochs=1, uniform_fraction=.60, nominal_hard_fraction=.25,
    regression_fraction=.15, priority_repeat_cap_per_epoch=4,
    maximum_feedback_age_epochs=1., refresh_fraction_of_epoch=.5,
    hard_below=.9, regression_drop=.05, formerly_high_at_least=.9,
    log_balance='uniform_physical_log_then_weighted_scene')


def source_population(source=DEFAULT_SOURCE, expected_count=103288):
    source=Path(source);m=read_sealed(source)
    if m['mode']!='uniform' or m['seed']!=42 or m['count']!=expected_count:
        raise ValueError('Use the registered complete seed42 uniform source population')
    data_path=Path(m['data_root'])/'manifest.json';pop_path=Path(m['population_manifest'])
    if file_sha(data_path)!=m['data_manifest_sha256'] or file_sha(pop_path)!=m['population_sha256']:
        raise ValueError('Registered source data/population changed')
    data=json.loads(data_path.read_text());pop=json.loads(pop_path.read_text())
    tokens=m['tokens'];legal=pop['train_tokens']+pop['dev_tokens'];test=set(pop['navtest_tokens'])
    if(len(tokens)!=expected_count or len(tokens)!=len(set(tokens)) or len(legal)!=expected_count
       or len(legal)!=len(set(legal)) or len(pop['navtest_tokens'])!=len(test) or set(tokens)!=set(legal)
       or set(data['tokens'])!=set(tokens) or len(data['tokens'])!=len(tokens)):
        raise ValueError('Full NAVTRAIN membership/uniqueness changed')
    logs={t:pop['token_logs'][t]for t in tokens}
    test_logs={pop['token_logs'][t]for t in test};test_phys={physical(x)for x in test_logs}
    if set(tokens)&test or set(logs.values())&test_logs or {physical(v)for v in logs.values()}&test_phys:
        raise ValueError('NAVTEST token/segment/physical-log leakage')
    return m,data,pop,logs


def write_manifest(path, source=DEFAULT_SOURCE, expected_count=103288):
    m,data,pop,logs=source_population(source,expected_count)
    body=dict(schema=STREAM_SCHEMA,source_manifest=str(Path(source).resolve()),
        source_manifest_sha256=file_sha(source),count=len(m['tokens']),tokens=m['tokens'],
        data_root=m['data_root'],data_manifest_sha256=m['data_manifest_sha256'],
        population_manifest=m['population_manifest'],population_sha256=m['population_sha256'],
        seed=42,policy=DEFAULT_POLICY,old_DEV_is_train_diagnostic=True,
        NAVTEST_token_segment_physical_disjoint=True,
        actor_inputs='current L/F/R images, ego, navigation only; future/cache only official scoring',
        padding='ceil(N/globalB)*globalB; invalid slots are shape-only and have zero all-loss/stat weight')
    atomic_json(path,sealed(body));return read_sealed(path)


def load_manifest(path, expected_count=103288):
    m=read_sealed(path)
    if m['schema']!=STREAM_SCHEMA or file_sha(m['source_manifest'])!=m['source_manifest_sha256']:
        raise ValueError('Performance stream source changed')
    source,data,pop,logs=source_population(m['source_manifest'],expected_count)
    for key in ('count','tokens','data_root','data_manifest_sha256','population_manifest','population_sha256','seed'):
        if m[key]!=source[key]:raise ValueError('Performance/source binding changed: '+key)
    if m['policy']!=DEFAULT_POLICY:raise ValueError('Priority recipe must be explicitly versioned')
    return m,logs


def load_feedback(path, *, logs, run_identity):
    b=read_sealed(path)
    if(b.get('schema')!=FEEDBACK_SCHEMA or b.get('status')!='COMPLETE' or b.get('cohort')!='TRAIN_DIAGNOSTIC'
       or b.get('train_feedback_allowed')is not True or b.get('run_identity')!=run_identity
       or b.get('inference')!='single_ODE_K10_shift5_seed2101'
       or not isinstance(b.get('observed_valid_scene_clock'),int)
       or b['observed_valid_scene_clock']<0 or not b.get('checkpoint_identity')):
        raise ValueError('Only same-run completed TRAIN B1 feedback may guide selection')
    import math
    records=b['records']
    if not records or len(records)!=b['count'] or b.get('completed_count')!=b['count']:raise ValueError('Incomplete feedback census')
    history=b.get('history',{})
    for t,h in history.items():
        if(t not in logs or not isinstance(h.get('ever_strict_feasible'),bool)
           or not isinstance(h.get('best_own_score'),(int,float))or not math.isfinite(h['best_own_score'])
           or not 0<=h['best_own_score']<=1
           or not 0<=h.get('last_observed_valid_scene_clock',-1)<=b['observed_valid_scene_clock']):
            raise ValueError('Invalid/forbidden own-run TRAIN history')
    for t,r in records.items():
        if t not in logs or r.get('log')!=logs[t]:raise ValueError('Forbidden/misidentified feedback token')
        score=r['score'];best=r.get('best_own_score',score)
        observed=r.get('observed_valid_scene_clock',b['observed_valid_scene_clock'])
        if not isinstance(observed,int) or not 0<=observed<=b['observed_valid_scene_clock']:
            raise ValueError('Invalid actual per-record TRAIN observation clock')
        if not all(isinstance(v,(int,float))and math.isfinite(v)and 0<=v<=1 for v in (score,best)):
            raise ValueError('Invalid canonical score')
        if best<score:raise ValueError('Own best score cannot be below current score')
        if not isinstance(r.get('ever_strict_feasible'),bool):raise ValueError('Explicit own-run safety history required')
        metrics=r['metrics']
        names=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')
        if not all(isinstance(metrics.get(k),(int,float))and math.isfinite(metrics[k])and 0<=metrics[k]<=1 for k in names):
            raise ValueError('Official canonical readouts missing or invalid')
        nc,dac,ttc,ep,comfort=(metrics[k]for k in names)
        if abs(nc*dac*(5*ep+5*ttc+2*comfort)/12-score)>1e-7:
            raise ValueError('Feedback canonical score/readouts disagree')
        if t in history and (history[t]['best_own_score']!=best or history[t]['ever_strict_feasible']!=r['ever_strict_feasible']):
            raise ValueError('Own best/safety history changed')
    return b


def write_feedback(path, *, run_identity, checkpoint_identity, observed_valid_scene_clock,
                   records, logs, previous=None, observed_history=None):
    """Seal an already COMPLETE own TRAIN B1 result, never create its scores.

    Caller must supply the complete scheduled panel, with official metrics and
    per-scene score/log. A previous same-run bank may only supply historical best.
    """
    previous_bank=load_feedback(previous,logs=logs,run_identity=run_identity)if previous else None
    if previous_bank and previous_bank['observed_valid_scene_clock']>observed_valid_scene_clock:
        raise ValueError('Own feedback history cannot move backwards')
    history={t:dict(h)for t,h in previous_bank.get('history',{}).items()}if previous_bank else{}
    for t,h in (observed_history or {}).items():
        if t not in logs:raise ValueError('Forbidden token in actual own observation history')
        old=history.get(t,{})
        history[t]=dict(best_own_score=max(h['best_own_score'],old.get('best_own_score',0)),
            ever_strict_feasible=bool(h['ever_strict_feasible']or old.get('ever_strict_feasible',False)),
            last_observed_valid_scene_clock=max(h['last_observed_valid_scene_clock'],old.get('last_observed_valid_scene_clock',0)))
    rows={}
    for t,r in records.items():
        row=dict(r);old=history.get(t)or(previous_bank['records'].get(t)if previous_bank else None)
        row.setdefault('observed_valid_scene_clock',observed_valid_scene_clock)
        row['best_own_score']=max(row['score'],old.get('best_own_score',old.get('score',row['score']))if old else row['score'])
        row['ever_strict_feasible']=bool((old and old.get('ever_strict_feasible'))or all(row['metrics'].get(k)==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')))
        history[t]=dict(best_own_score=row['best_own_score'],ever_strict_feasible=row['ever_strict_feasible'],last_observed_valid_scene_clock=row['observed_valid_scene_clock'])
        rows[t]=row
    body=dict(schema=FEEDBACK_SCHEMA,status='COMPLETE',cohort='TRAIN_DIAGNOSTIC',
        train_feedback_allowed=True,run_identity=run_identity,checkpoint_identity=checkpoint_identity,
        observed_valid_scene_clock=observed_valid_scene_clock,inference='single_ODE_K10_shift5_seed2101',
        count=len(rows),completed_count=len(rows),records=rows,history=history,previous_same_run_feedback_sha256=previous_bank['sha256']if previous_bank else None)
    # Validate before publishing. Never replace an already admitted bank.
    path=Path(path)
    if path.exists():raise FileExistsError('Feedback snapshots must be immutable: '+str(path))
    candidate=path.with_name(path.name+'.validation')
    if candidate.exists():raise FileExistsError(candidate)
    atomic_json(candidate,sealed(body))
    try:load_feedback(candidate,logs=logs,run_identity=run_identity)
    except BaseException:
        candidate.unlink();raise
    candidate.rename(path)
    return read_sealed(path)
