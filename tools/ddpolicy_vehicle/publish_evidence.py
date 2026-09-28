"""Publish aggregate run evidence without weights, images, labels or scene IDs."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--campaign-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.campaign_root);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    now=time.time();rows=[]
    for path in sorted((root/'runs').glob('*/status.json')):
        value=json.loads(path.read_text())
        row={k:value[k] for k in ('run_id','kind','status','source_sha','evaluation_source','gpu_count',
             'real_optimizer_updates','synthetic_optimizer_updates','inference_scenes','failed','parameters',
             'trainable_parameters','load_seconds','forward_seconds','backward_seconds','gradient_l2','losses',
             'peak_allocated_bytes','peak_reserved_bytes') if k in value}
        row.setdefault('run_id',path.parent.name)
        end=now if value['status']=='RUNNING' else value.get('end_unix',value['start_unix'])
        row['gpu_hours_at_snapshot']=(end-value['start_unix'])*value['gpu_count']/3600
        row['raw_status_sha256']=file_sha256(path);rows.append(row)
    for path in sorted((root/'training').glob('*/status.json')):
        value=json.loads(path.read_text());attempts=[];attempt_calls=[];attempt_exposures=[]
        for attempt in sorted(path.parent.glob('attempt_*.json')):
            item=json.loads(attempt.read_text())
            end=now if item['status']=='RUNNING' else item.get('end_unix',item['start_unix'])
            attempts.append((end-item['start_unix'])*item['gpu_count']/3600)
            if 'attempt_optimizer_updates' in item:attempt_calls.append(item['attempt_optimizer_updates'])
            if 'attempt_sample_presentations' in item:attempt_exposures.append(item['attempt_sample_presentations'])
        row={k:value[k] for k in ('kind','status','source_sha','gpu_count','real_optimizer_updates','sample_presentations','epoch','batch_offset')}
        row.update(run_id=path.parent.name,gpu_hours_at_snapshot=sum(attempts),attempts=len(attempts),raw_status_sha256=file_sha256(path));rows.append(row)
        if attempt_calls and len(attempt_calls)==len(attempts):row['actual_optimizer_calls_in_run']=sum(attempt_calls)
        if attempt_exposures and len(attempt_exposures)==len(attempts):
            row['model_progress_scene_presentations']=row['sample_presentations']
            row['sample_presentations']=sum(attempt_exposures)
        fork=path.parent/'fork_provenance.json'
        if fork.exists():row['diagnostic_fork_inherited_updates']=json.loads(fork.read_text())['inherited_updates']
        for log in path.parent.glob('train_rank*.jsonl'):
            # Training log schema contains only aggregate losses/counts/costs.
            values=[json.loads(line) for line in log.read_text().splitlines()]
            allowed={'update','epoch','offset','global_scene_exposure','lr','seconds','rank','losses','coordinates','graphs','roles','peak_allocated_bytes','tasks','role_counter_scope','optimizer_update'}
            if any(set(v)-allowed for v in values):raise ValueError('Unknown training log fields need privacy review')
            (out/(path.parent.name+'_'+log.name)).write_text(''.join(json.dumps(v)+'\n' for v in values))
    invalid={}
    for path in sorted(root.glob('optimizer_stasis_correction*/validity_overlay.json')):
        for row in json.loads(path.read_text())['runs']:invalid[row['run_id']]=row
    for row in rows:
        row['recorded_real_optimizer_calls']=row.get('actual_optimizer_calls_in_run',row.get('real_optimizer_updates',0))
        row['verified_effective_real_updates']=None
        if row.get('run_id') in invalid:
            row['verified_effective_real_updates']=invalid[row['run_id']]['effective_parameter_updates']
            row['training_validity']='INVALID_OPTIMIZER_STASIS; historical call/exposure/cost record preserved'
        else:
            path=root/'training'/row.get('run_id','')/'train_rank0.jsonl'
            if path.exists():
                values=[json.loads(line) for line in path.read_text().splitlines()]
                verified=sum(v.get('optimizer_update',{}).get('fp32_master_update_verified',False) for v in values)
                if verified:row['verified_effective_real_updates']=verified
    atomic_json(out/'RUNS.json',{'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runs':rows,'gpu_hours_at_snapshot':sum(r['gpu_hours_at_snapshot'] for r in rows),
        'recorded_real_optimizer_calls':sum(r.get('recorded_real_optimizer_calls',0) for r in rows),
        'verified_effective_real_updates':sum(r.get('verified_effective_real_updates') or 0 for r in rows),
        'sample_presentations':sum(r.get('sample_presentations',0) for r in rows),
        'accounting':'actual attempt update/exposure deltas when available (diagnostic fork inheritance excluded); GPU occupancy summed over attempts, including load/failure/evaluation; RUNNING costs are lower bounds',
        'formal_training_gpu_hour_cap':8000,'initial_diagnostic_gpu_hour_ceiling':20,'maximum_diagnostic_gpu_hour_ceiling':40})
    for name in ('startup_AB_initialization_comparison.json','startup_BC_initialization_comparison.json',
                 'startup_resume_comparison_initial.json','depth_current_transform_audit.json'):
        if (root/name).exists():shutil.copyfile(root/name,out/name)
    audit=json.loads((root/'vehicle_targets_v1_complete/audit.json').read_text())
    split=json.loads((root/'vehicle_targets_v1_complete/split.json').read_text())
    logs={}
    for path in (root/'vehicle_targets_v1_complete/logs').glob('*.json'):
        for record in json.loads(path.read_text())['records']:logs[record['token']]=record['log']
    atomic_json(out/'DATA.json',{'vehicle_targets_identity':json.loads((root/'vehicle_targets_v1_complete/identity.json').read_text())['identity'],
        'counts':audit,'train_scenes':len(split['train_tokens']),'train_logs':len({logs[t] for t in split['train_tokens']}),
        'dev_scenes':len(split['dev_tokens']),'dev_logs':len({logs[t] for t in split['dev_tokens']}),
        'train_dev_token_overlap':len(set(split['train_tokens']) & set(split['dev_tokens'])),
        'train_dev_log_overlap':len({logs[t] for t in split['train_tokens']} & {logs[t] for t in split['dev_tokens']}),
        'current_dev_identity':json.loads((root/'current_dev_v1/identity.json').read_text())['identity'],
        'current_navtest_identity':json.loads((root/'current_navtest_v1/identity.json').read_text())['identity'],
        'current_navtest_data_only':json.loads((root/'current_navtest_v1/audit.json').read_text()),
        'navtest_model_predictions':'NOT_RUN','raw_log_population_available':True})
    atomic_json(out/'FILES.json',{p.name:file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__=='__main__':main()
