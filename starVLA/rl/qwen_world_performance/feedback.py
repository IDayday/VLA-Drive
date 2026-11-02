"""Own TRAIN B1 cache/half-epoch snapshot. Never scores or forwards a model.

Call once after fresh window/scene-stream commit, before four inner updates:
accumulate_nominal_feedback(state,window); flush_feedback(runtime,window,state).
All ranks call flush, including ranks with zero B1 records. Only a half-epoch
crossing gathers primitive rows; no action/feature/optimizer tensor is gathered.
"""
import copy,math
from pathlib import Path
from .data_protocol import fingerprint,write_feedback,read_sealed,file_sha,atomic_json

SCHEMA='world_performance_own_TRAIN_B1_feedback_cache_v1'
SAFE=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def _memory(state):
    aux=state.setdefault('performance_auxiliary',{})
    return aux.setdefault('priority_feedback',dict(schema=SCHEMA,cache={},observed_history={},
        last_accumulated=None,last_flush_bucket=0,last_snapshot=None,previous_snapshot=None,
        snapshot_material=None,actual_local_B1_observations=0,empty_flushes=0,flushes=0))


def accumulate_nominal_feedback(state,window):
    memory=_memory(state);version=window['behavior_version'];marker=(window['window_start_cursor'],version['update'],version['run'])
    if memory['last_accumulated']==marker:raise ValueError('B1 feedback may only accumulate once per fresh collection')
    memory['last_accumulated']=marker
    valid=window['valid_global_scenes'];clock=state['valid_scene_exposures']-valid
    if clock<0:raise ValueError('Accumulate B1 only after actual fresh scene accounting commit')
    for leaf in window['samples']:
        nominal=leaf.get('nominal_improvement',{}).get('nominal')
        if not leaf.get('valid_scene',True)or nominal is None:continue
        feedback=nominal['feedback'];token=leaf['token'];log=leaf['log']
        if(nominal['source_version']!=version or feedback.get('status')!='OK'or feedback.get('token')!=token
           or feedback.get('log')!=log or feedback.get('protocol_sha256')!=leaf['protocol_sha256']):
            raise ValueError('B1 feedback is not this exact own TRAIN scene/policy/protocol')
        metrics={k:float(v)for k,v in feedback['metrics'].items()};score=metrics['score']
        if any(not math.isfinite(v)or not 0<=v<=1 for v in metrics.values()):raise ValueError('Invalid official B1 metrics')
        material=dict(token=token,log=log,metrics=metrics,protocol_sha256=feedback['protocol_sha256'],
            trajectory_hash=feedback.get('trajectory_hash'),cache_sha256=feedback.get('cache',{}).get('sha256'))
        row=dict(log=log,score=score,metrics=metrics,observed_valid_scene_clock=clock,
            source_behavior_update=version['update'],source_model_sha256=version['adapter_sha256'],
            feedback_material_sha256=fingerprint(material))
        from .metrics import strict
        row.update(sampled_strict_high=any(strict(r) for r in leaf['records']),
            nominal_strict_high=strict(feedback),rescue_activated=bool(leaf['nominal_improvement']['enabled']))
        old=memory['observed_history'].get(token,{})
        memory['observed_history'][token]=dict(best_own_score=max(score,old.get('best_own_score',0)),
            ever_strict_feasible=bool(all(metrics[k]==1 for k in SAFE)or old.get('ever_strict_feasible',False)),
            last_observed_valid_scene_clock=clock)
        memory['cache'][token]=row;memory['actual_local_B1_observations']+=1
    return dict(actual_local_B1_observations=memory['actual_local_B1_observations'],local_distinct_cached=len(memory['cache']))


def flush_feedback(runtime,window,state):
    from starVLA.rl.simwam_recipe import checkpoint
    import torch.distributed as dist
    memory=_memory(state);population=runtime.spec['population'];clock=state['valid_scene_exposures'];bucket=(2*clock)//population
    if bucket<=memory['last_flush_bucket']:return dict(flushed=False,reason='before_next_half_epoch')
    header=dict(clock=clock,bucket=bucket,last_bucket=memory['last_flush_bucket'],run=runtime.training_identity['sha256'],
        rows=len(memory['cache']),previous_sha=memory['last_snapshot']['file_sha256']if memory['last_snapshot']else None)
    headers=checkpoint.gather(header)
    keys=('clock','bucket','last_bucket','run','previous_sha')
    if any(any(h[k]!=header[k]for k in keys)for h in headers):raise ValueError('Rank feedback clocks/lineage disagree')
    memory['last_flush_bucket']=bucket
    if sum(h['rows']for h in headers)==0:
        memory['empty_flushes']+=1;return dict(flushed=False,reason='no_actual_B1_uniform_fallback',bucket=bucket)
    payload=checkpoint.gather(dict(cache=memory['cache'],history=memory['observed_history']))
    records=copy.deepcopy(memory['snapshot_material']['records'])if memory['snapshot_material']else{}
    history={}
    for part in payload:
        for t,row in part['cache'].items():
            old=records.get(t)
            if old is None or(row['observed_valid_scene_clock'],row['source_behavior_update'],row['feedback_material_sha256'])>(old['observed_valid_scene_clock'],old['source_behavior_update'],old['feedback_material_sha256']):records[t]=row
        for t,h in part['history'].items():
            old=history.get(t,{})
            history[t]=dict(best_own_score=max(h['best_own_score'],old.get('best_own_score',0)),
                ever_strict_feasible=bool(h['ever_strict_feasible']or old.get('ever_strict_feasible',False)),
                last_observed_valid_scene_clock=max(h['last_observed_valid_scene_clock'],old.get('last_observed_valid_scene_clock',0)))
    # Age uses each original observation, never the publication timestamp.
    records={t:r for t,r in records.items()if 0<=clock-r['observed_valid_scene_clock']<=population}
    if not records:memory['empty_flushes']+=1;return dict(flushed=False,reason='all_B1_records_stale_uniform_fallback',bucket=bucket)
    if any(t not in runtime.stream.logs for t in records)or any(t not in runtime.stream.logs for t in history):raise ValueError('Forbidden TRAIN feedback membership')
    digest=fingerprint(dict(records=records,clock=clock,run=header['run'],history=history))
    path=Path(runtime.args.output)/'TRAIN_B1_feedback'/f'clock{clock:09}_b{bucket:03}_{digest[:16]}.json'
    previous=memory['last_snapshot']['path']if memory['last_snapshot']else None
    error=None
    if not dist.is_initialized()or dist.get_rank()==0:
        try:
            if not path.exists():write_feedback(path,run_identity=header['run'],checkpoint_identity=dict(kind='actual_scored_behavior_policies',latest_behavior=window['behavior_version']),
                observed_valid_scene_clock=clock,records=records,logs=runtime.stream.logs,previous=previous,observed_history=history)
            else:
                existing=read_sealed(path)
                if existing['run_identity']!=header['run']or existing['observed_valid_scene_clock']!=clock:raise ValueError('Unrelated immutable feedback snapshot collision')
        except Exception as exc:error=repr(exc)
    checkpoint.check_errors(error)
    snapshot=read_sealed(path);ref=dict(path=str(path),file_sha256=file_sha(path),sha256=snapshot['sha256'],observed_valid_scene_clock=clock)
    runtime.stream.offer_feedback(path)
    memory['previous_snapshot']=memory['last_snapshot'];memory['last_snapshot']=ref;memory['snapshot_material']=snapshot
    memory['cache']={};memory['observed_history']={};memory['flushes']+=1
    return dict(flushed=True,bucket=bucket,count=len(records),snapshot=ref)


def feedback_state_validation(state):
    """Return shared fields for rank checks; local row counts intentionally differ."""
    memory=state.get('performance_auxiliary',{}).get('priority_feedback')
    if memory is None:return dict(priority_feedback_present=False)
    if memory['schema']!=SCHEMA or memory['last_flush_bucket']<0:raise ValueError('Invalid native B1 feedback cache')
    material=memory['snapshot_material']
    if material is not None:
        body=dict(material);saved=body.pop('sha256')
        if fingerprint(body)!=saved or saved!=memory['last_snapshot']['sha256']:raise ValueError('Native feedback snapshot material changed')
    return dict(priority_feedback_present=True,priority_feedback_last_bucket=memory['last_flush_bucket'],
        priority_feedback_snapshot_sha256=memory['last_snapshot']['sha256']if memory['last_snapshot']else None,
        priority_feedback_cache_material_sha256=fingerprint(memory['cache']))
