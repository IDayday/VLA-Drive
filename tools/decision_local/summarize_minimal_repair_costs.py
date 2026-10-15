"""Account actual native cost receipts without counting milestones twice.

GPU hours mean elapsed time of an allocated GPU process, including its IO and
initialization. Shared execution is not equivalent to exclusive accelerator
hours. Canonical scorer wall time is not relabelled as CPU process time.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root
    roots={p for p in r.iterdir() if p.is_dir()}
    for pattern in ('stageA*seed*','stageB*seed*','smoke_v*','resume_v*','resume_RAM_v*',
                    'runner_smoke_v*','online_formal_v*','cached_formal4_v*',
                    'online_balanced_v*','cached_balanced4_v*'):
        for parent in r.glob(pattern):
            if parent.is_dir():roots.update(p for p in parent.iterdir() if p.is_dir())
    # Registered nested outputs are enumerated explicitly; never walk the
    # hundreds of thousands of private scene or checkpoint tensor files.
    for registry in list(r.glob('*REGISTRY.json'))+list(r.glob('*EVAL_*.json'))+list(r.glob('*TASKS.json'))+list(r.glob('*MECHANISM_*.json')):
        value=read(registry)
        if not isinstance(value,dict):continue
        for job in value.get('jobs',[]):
            for key in ('output','predictions','scores'):
                if key in job and Path(job[key]).is_dir():roots.add(Path(job[key]))
    gpu=[];labels=[];geometry=[];canonical=[];unmeasured=[]
    for path in sorted(roots):
        complete=path/'COMPLETE.json';status=path/'STATUS.json';meta=path/'identity.json'
        d=read(complete) if complete.exists() else read(status) if status.exists() else None
        if isinstance(d,dict) and 'GPU_hours' in d:
            info=read(meta) if meta.exists() else {}
            gpu.append({'path':str(path),'GPU_process_hours':d['GPU_hours'],
                        'complete':complete.exists(),'scope':info.get('scope',info.get('schema')),
                        'kind':info.get('kind'),'receipt_sha256':digest(complete if complete.exists() else status)})
        if path.name.startswith('targets_'):
            if complete.exists():reports=[complete]
            elif (path/'INCOMPLETE.json').exists():reports=[path/'INCOMPLETE.json']
            else:reports=sorted(path.glob('COMPLETE_rank*.json'))+sorted(path.glob('INCOMPLETE_rank*.json'))
            if reports:
                values=[read(f) for f in reports]
                labels.append({'path':str(path),'scenes':sum(x['scenes'] for x in values),
                               'CPU_process_seconds':sum(x['CPU_process_seconds'] for x in values),
                               'worker_wall_seconds':sum(x['worker_wall_seconds'] for x in values),
                               'produced_npz_bytes':sum(x['bytes'] for x in values),
                               'errors':sum(len(x.get('errors',[])) for x in values),
                               'complete_global':complete.exists(),
                               'note':'Includes preserved superseded/failed label versions; each output counted once'})
            else:unmeasured.append({'path':str(path),'reason':'No completed producer cost receipt'})
        elif isinstance(d,dict) and 'CPU_process_seconds' in d:
            geometry.append({'path':str(path),'CPU_process_seconds':d['CPU_process_seconds'],
                             'worker_wall_seconds':d.get('worker_wall_seconds'),
                             'elapsed_seconds':d.get('elapsed_seconds')})
        if isinstance(d,dict) and all(k in d for k in ('q0','q_final','seconds')):
            usage=path/'CPU_RESOURCE_USAGE.json'
            timed=read(usage) if usage.exists() else None
            if timed is not None and timed['exit_code']!=0:raise ValueError('Failed canonical timing receipt')
            canonical.append({'path':str(path),'job_wall_seconds':d['seconds'],
                              'CPU_process_seconds':(timed['user_seconds']+timed['system_seconds']) if timed is not None else None,
                              'note':(timed.get('mechanism','GNU time child-inclusive user+system seconds') if timed else
                                      'Unknown child CPU time, never inferred from wall')})
    prereads=[]
    for path in sorted(r.glob('PREREAD_*.json')):
        d=read(path)
        if isinstance(d,dict) and 'bytes_verified' in d:
            prereads.append({'path':str(path),'seconds':d['seconds'],'bytes_verified':d['bytes_verified'],
                             'CPU_process_seconds':d.get('CPU_process_seconds'),'new_persistent_data_bytes':0})
    # Read-only support attribution is not another label-production pass.
    audits=[]
    audit_paths=set(r.glob('SUPPORT_ATTRIBUTION_*.json'))|set(r.glob('LOW_PROGRESS_TARGET_AUDIT_*.json'))|set(r.glob('RAM_LABELS_*.json'))|set(r.glob('TEMP_RAM_RELEASE_*.json'))
    for path in sorted(audit_paths):
        d=read(path)
        if 'CPU_process_seconds' in d:audits.append({'path':str(path),'CPU_process_seconds':d['CPU_process_seconds']})
    standalone_gpu=[]
    for path in sorted(r.glob('TRAINING_CONDITIONAL_FIT_*.json')):
        d=read(path)
        if 'GPU_process_hours' in d:
            standalone_gpu.append({'path':str(path),'GPU_process_hours':d['GPU_process_hours'],
                                   'scope':'Frozen-model training diagnostic, no optimizer updates'})
    # Acceptance files summarize the same native runs; do not add them again.
    measured_cpu=sum(x['CPU_process_seconds'] for x in labels+geometry+audits)+sum(x['CPU_process_seconds'] or 0 for x in prereads+canonical)
    measured_gpu=sum(x['GPU_process_hours'] for x in gpu+standalone_gpu)
    budget=read(r/'RESOURCE_BUDGET.json')
    result={'GPU_process_hours_recorded':measured_gpu,'CPU_process_hours_recorded_lower_bound':measured_cpu/3600,
            'GPU_runs_and_exports':gpu,'standalone_GPU_diagnostics':standalone_gpu,'label_versions':labels,'geometry_evaluation':geometry,
            'read_only_audits':audits,'pre_read_IO':prereads,'canonical_job_wall_times':canonical,
            'unmeasured':unmeasured,'budget':budget,'tool_sha256':digest(__file__),
            'exceeds_recorded_cap':measured_gpu>budget['GPU_process_hours_cap'] or measured_cpu/3600>budget['CPU_process_hours_cap'],
            'historical_shared_S0_GPU_hours':483.481,'new_full_VLA_encoding_runs':0,
            'historical_S0_cost_source':'Previous decision_local frozen S0 source and training receipts; outside current incremental budget',
            'limits':['Native unfinished-run costs are lower bounds at the last saved checkpoint',
                      'Uninstrumented trainer CPU, failed startup, generic tests, SHA finalization and early scorer child CPU are not invented; the CPU cap cannot be certified from this lower bound',
                      'GPU process-hours under sharing are not exclusive GPU-hours or measured kernel active time',
                      'Label worker busy wall and process CPU time are different quantities',
                      'No historical S0 feature-cache bytes or prior study GPU hours charged again']}
    atomic_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('GPU_process_hours_recorded','CPU_process_hours_recorded_lower_bound','exceeds_recorded_cap')}))


if __name__=='__main__':main()
