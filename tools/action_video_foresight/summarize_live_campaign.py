"""Snapshot actual formal progress and allocation costs without touching trainers."""
import argparse
import datetime
import json
import math
from pathlib import Path
import time

from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def union_seconds(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Negative resource interval')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return sum(end-start for start,end in merged)


def resource_cost(root,now):
    per_device={};allocated=0.;unassigned=0.;kinds={}
    for path in sorted((Path(root)/'runs').glob('*/status.json')):
        record=json.loads(path.read_text())
        count=record['gpu_count'];end=min(record.get('end_unix',now),now)
        seconds=end-record['start_unix']
        if seconds<0:raise ValueError('Resource ledger clock mismatch')
        cost=seconds*count/3600;allocated+=cost
        kinds[record.get('kind','UNSPECIFIED')]=kinds.get(record.get('kind','UNSPECIFIED'),0.)+cost
        devices=[s.strip() for s in (record.get('cuda_visible_devices') or '').split(',') if s.strip()]
        if len(devices)!=count:
            unassigned+=cost
            continue
        for device in devices:
            per_device.setdefault((record['host'],device),[]).append((record['start_unix'],end))
    missing=[]
    if (Path(root)/'clip_nccl_equivalence_v1.log').exists() and not (Path(root)/'runs'/'clip_nccl_equivalence_v1'/'status.json').exists():
        missing.append({'check':'actual two-process NCCL equivalence',
            'GPU_count':2,'GPUh':'NOT_MEASURED','reason':'Standalone validation log has a result but no recorded start/end allocation interval; do not invent elapsed time.'})
    return {'job_allocated_GPUh':allocated,'job_allocated_GPUh_by_kind':kinds,
        'physical_device_interval_union_GPUh':sum(union_seconds(v) for v in per_device.values())/3600,
        'unassigned_allocation_GPUh':unassigned,'known_unmetered_checks':missing,
        'accounting':'Loading/failures/saving/evaluation included. Job allocations overlap when authorized jobs share a GPU; physical union counts each declared host/device interval once. Idle pressure and unrelated tasks are excluded. Unassigned intervals are reported separately.'}


def read_steps(path):
    if not path.exists():return []
    value=path.read_text();lines=value.splitlines()
    # A concurrently appended last line may still be incomplete. Preserve every
    # completed record; malformed completed lines are errors, not filtered data.
    if value and not value.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines]


def development_results(plan):
    root=Path(plan['campaign_root']);results={}
    for arm,spec in plan['runs'].items():
        for update in plan['evaluation_updates']:
            label=spec['run_id']+'_dev'+str(update)+'_seed42'
            path=root/'evaluations'/(label+'_state.json')
            if not path.exists():continue
            state=json.loads(path.read_text())
            recovered_states=[]
            if state['status']!='COMPLETE':
                for recovery in sorted(root.glob('reconciled_development*/'+label+'_state.json')):
                    recovered=json.loads(recovery.read_text())
                    if recovered['status']!='COMPLETE':continue
                    from starVLA.model.modules.vehicle_joint.initialization import file_sha256
                    if recovered['original_failed_state_SHA256']!=file_sha256(path):
                        raise ValueError('Recovered evaluation original state changed')
                    recovered_states.append(recovered)
                if recovered_states:
                    if any(v['identity']!=recovered_states[0]['identity'] or v['PDMS_points']!=recovered_states[0]['PDMS_points'] for v in recovered_states):
                        raise ValueError('Multiple recovery versions disagree; do not choose the favorable score')
                    state=recovered_states[0]
            expected={'plan':plan['identity'],'arm':arm,'update':update,
                'evaluation_source':plan['training_source_sha'],
                'purpose':'registered complete development; not Navtest'}
            if state['identity']!=expected:raise ValueError('Development evaluation identity mismatch')
            entry={'status':state['status'],'identity':expected}
            if state.get('transport_source'):
                entry['transport_source']=state['transport_source']
            if state['status']=='COMPLETE':
                score=json.loads(Path(state['scores']).read_text())
                ego=json.loads(Path(state['ego']).read_text())
                if any(not v['valid'] or v['failed'] or v['scenes']!=plan['dev_scenes'] for v in (score,ego)):
                    raise ValueError('Incomplete development result; preserve failed rows')
                values=[score['PDMS'],*score['metrics'].values(),
                    *(ego['groups']['all'][k] for k in ('ADE','FDE','yaw_MAE_rad'))]
                if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite development result')
                if state['PDMS_points']!=100*score['PDMS']:
                    raise ValueError('Development score scale mismatch')
                entry.update(PDMS_points=state['PDMS_points'],metrics=score['metrics'],
                    ego=ego['groups']['all'],scenes=score['scenes'],failed=score['failed'])
            elif state.get('error'):
                entry['error']=state['error']
            results.setdefault(arm,{})[str(update)]=entry
    return results


def snapshot(plan,now):
    if plan['identity']!=identity_hash({k:v for k,v in plan.items() if k!='identity'}):
        raise ValueError('Changed frozen formal plan')
    root=Path(plan['campaign_root']);runs={};initial={}
    for arm,spec in plan['runs'].items():
        run=root/'students'/spec['run_id'];controller=json.loads((root/'formal_controllers'/spec['run_id']/'status.json').read_text())
        record={'run_id':spec['run_id'],'host':spec['host'],'gpus':spec['gpus'],
                'controller_status':controller['status'],'completed_updates':0,'scene_exposure':0,
                'latest_controller_event':controller['events'][-1]}
        if (run/'status.json').exists():
            actual=json.loads((run/'status.json').read_text())
            if actual['source_sha']!=plan['training_source_sha']:raise ValueError('Unexpected live training source')
            rows=read_steps(run/'steps.jsonl')
            record.update(training_status=actual['status'],completed_updates=actual['completed'],
                scene_exposure=actual['exposure'],effective_data_traversals=actual['exposure']/plan['scene_count'],
                observed_unix=actual['updated_unix'],latest_recorded_update=rows[-1]['update'] if rows else 0,
                parameters=json.loads((run/'parameters.json').read_text()),
                observations=[row for row in rows if row['update'] in {1,100,500,1000,2000}])
            record['all_logged_losses_finite']=all(math.isfinite(v) for row in rows for v in row['raw_losses'].values())
            record['all_logged_master_updates_verified']=all(row['fp32_master_update_verified'] for row in rows)
            if rows:
                recent=rows[-min(100,len(rows)):]
                record['recent_training']={'records':len(recent),
                    'step_mean_seconds':sum(r['seconds'] for r in recent)/len(recent),
                    'raw_loss_means':{key:sum(r['raw_losses'][key] for r in recent)/len(recent) for key in recent[0]['raw_losses']}}
            init=json.loads((run/'initialization.json').read_text());initial[arm]=init
            record['initialization_hash']=identity_hash(init)
            record['initialization_summary']={key:init[key] for key in ('W','driving_weights_loaded','future_teacher_in_model','generic_source_manifest')}
            record['auxiliary_initialization_hash']=identity_hash(json.loads((run/'auxiliary_initialization.json').read_text()))
        runs[arm]=record
    comparisons={}
    names=sorted(initial)
    for arm in names[1:]:
        comparisons[arm+'-'+names[0]]={key:initial[arm][key]==initial[names[0]][key] for key in initial[names[0]]}
    caches={}
    for target in ('video_clip','dino_sequence'):
        directory=root/f'targets_{target}_train_v1'
        complete=directory/'COMPLETE.json'
        caches[target]={'identity':json.loads((directory/'identity.json').read_text())['identity'],
            'complete':complete.exists(),'published_chunks':len(list(directory.glob('chunk_*.json'))),
            'required_scenes':plan['scene_count']}
        if complete.exists():caches[target].update(json.loads(complete.read_text()))
    evaluations=development_results(plan)
    have_scores=any(v['status']=='COMPLETE' for arm in evaluations.values() for v in arm.values())
    return {'snapshot_utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
        'training_source':plan['training_source_sha'],'formal_plan_identity':plan['identity'],
        'scope':'Actual fresh generic-Qwen/random-driving formal training; startup/profile/probe checkpoints excluded',
        'common_updates':plan['updates'],'global_batch':plan['global_batch'],
        'train_scenes':plan['scene_count'],'train_logs':plan['train_logs'],'dev_scenes':plan['dev_scenes'],
        'dev_logs':plan['dev_logs'],'weights':plan['common_weights'],'runs':runs,
        'actual_common_initialization_comparison':comparisons,'full_native_caches':caches,
        'cost':resource_cost(root,now),'development_results':evaluations,
        'new_method_PDMS':'DEVELOPMENT_RESULTS_AVAILABLE' if have_scores else 'NOT_YET_EVALUATED',
        'remaining':'Complete shared100k training/dev; locked final Navtest; key second seed and registered capacity/MAE controls. No early-warmup loss claim substitutes for these.'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('Preserve prior progress snapshots; choose a new output')
    atomic_json(out,snapshot(json.loads(Path(args.plan).read_text()),time.time()))


if __name__=='__main__':main()
