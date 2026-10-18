"""Finite fixed-stage Navtest handoff using existing qualified workers/helpers.

Candidate hosts are disjoint from the DEV queue. Individual visible GPUs are
admitted from two fresh observations; active training may share when qualified.
There is no trainer, score-based checkpoint selection or automatic failure retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from scripts.analysis.fm_transfer.final_handoff import (
    read,stage,probe,available_devices,allocate,model_state,validate_summary,verify_source_snapshot,PYTHON)
from scripts.analysis.fm_transfer.milestone_controller import worker_allowed
from scripts.analysis.fm_transfer.source_process_navtest import ready,STAGES
from scripts.analysis.fm_transfer.final_lock import authenticated
from starVLA.rl.fm_transfer.source_process import ROOT,MODES
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

REFERENCE='/mnt/project/fm-transfer-20261016/evaluations_scaleout/B_SOURCE_ACTION_DIT_SFT_final_NAVTEST'
REFERENCE_V10='/mnt/project/fm-transfer-20261016/pdms_versions/v1.0/rescores_scaleout/B_SOURCE_ACTION_DIT_SFT'
V10_SOURCE='/mnt/project/fm-transfer-20261016/upstream/navsim_versions/9f3ea68ffda0c6afe37b51b402f3245ea9936e18'
V10_CACHE='/mnt/project/fm-transfer-20261016/pdms_versions/v1.0/metric_cache_navtest'


def tick(update,value,source):
    missing=ready(update)
    if missing:return dict(status='WAITING_FOR_REGISTERED_MILESTONES',missing=missing,Navtest_opened=False)
    root=ROOT/'navtest'/f'update{update:06d}';control=root/'controller';control.mkdir(parents=True,exist_ok=True)
    lock=root/'final_navtest/LOCK.json'
    stage(control,'lock',[PYTHON,'-m','scripts.analysis.fm_transfer.source_process_navtest',
        '--update',str(update),'--source',value['evaluation_source'],'--output',str(lock)],source=source)
    selection=authenticated(lock);routes_path=control/'routes.json'
    routes=read(routes_path) if routes_path.exists() else {}
    states={mode:model_state(root,mode,plan,evaluation_source=value['evaluation_source']) for mode,plan in routes.items()}
    reserved={u for s in states.values() for u in s['reserved']}
    absent=[m for m in MODES if m not in routes]
    if absent:
        def two(host):
            try:
                worker_allowed(host);a=probe(host);time.sleep(2);return a,probe(host)
            except Exception as exc:return dict(unavailable=str(exc),capacity_not_inferred=True)
        with ThreadPoolExecutor(max_workers=len(value['hosts'])) as pool:
            observations=dict(zip(value['hosts'],pool.map(two,value['hosts'])))
        samples={h:v for h,v in observations.items() if isinstance(v,tuple)}
        devices,excluded=available_devices(samples,reserved)
        atomic_json(control/f'inventory_{time.time_ns()}.json',dict(observations=observations,excluded=excluded,admitted=devices))
        planned=allocate(absent,devices);routes.update(planned);atomic_json(routes_path,routes)
        for mode,plan in planned.items():
            for host in dict.fromkeys(d['host'] for d in plan['assignments']):
                rows=[d for d in plan['assignments'] if d['host']==host]
                stage(control,mode+'_'+host,[PYTHON,'-m','scripts.analysis.fm_transfer.dispatch_source_navtest',
                    '--lock',str(lock),'--source',value['evaluation_source'],'--model',mode,'--shards',str(plan['shards']),
                    '--ranks',','.join(str(d['rank']) for d in rows),'--gpus',','.join(str(d['gpu']) for d in rows)],
                    host=host,source=source)
        if planned:return dict(status='PARTITIONS_STARTED',routes=routes,Navtest_opened=True)
    completed={}
    for mode,s in states.items():
        if s['status']!='WORKERS_EXITED_COMPLETE':continue
        output=root/'evaluations'/(mode+'_final_NAVTEST');summary=root/'analysis'/(mode+'.json')
        stage(control,'analyze_'+mode,[PYTHON,'-m','scripts.analysis.fm_transfer.analyze_evaluation',
            '--evaluation',str(output),'--reference',REFERENCE,'--output',str(summary)],source=source)
        validate_summary(read(summary),authenticated(output/'manifest.json'),selection,mode)
        rescore=root/'v1.0'/mode
        stage(control,'rescore_v10_'+mode,[PYTHON,'-m','scripts.analysis.fm_transfer.rescore_saved_navtest',
            '--evaluation',str(output),'--output',str(rescore),'--version','v1.0','--source',V10_SOURCE,
            '--cache-root',V10_CACHE,'--workers','8'],source=source)
        v10_summary=root/'analysis'/(mode+'_v1.0.json')
        stage(control,'analyze_v10_'+mode,[PYTHON,'-m','scripts.analysis.fm_transfer.analyze_versioned_navtest',
            '--rescored',str(rescore),'--reference',REFERENCE_V10,'--output',str(v10_summary)],source=source)
        completed[mode]=dict(v11=str(summary),v10=str(v10_summary),v11_sha256=file_sha(summary),v10_sha256=file_sha(v10_summary))
    return dict(status='COMPLETE' if len(completed)==2 else 'WAITING_FOR_WORKERS_OR_CAPACITY',
        states=states,completed=completed,Navtest_opened=bool(routes),final_lock_sha256=file_sha(lock))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation-source',required=True)
    p.add_argument('--hosts',nargs='+',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    require_dispatch()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only controller')
    source=Path(__file__).resolve().parents[3];verify_source_snapshot(source);verify_source_snapshot(a.evaluation_source)
    if len(set(a.hosts))!=len(a.hosts):raise ValueError('Duplicate hosts')
    for host in a.hosts:worker_allowed(host)
    # The two queue owners have disjoint candidate hosts, including pre-CUDA loading.
    dev_identity=read(ROOT/'milestone_controller/identity.json')
    if set(a.hosts)&set(dev_identity['hosts']):raise ValueError('DEV/Navtest admission hosts must be disjoint')
    control=ROOT/'navtest_controller';control.mkdir(exist_ok=True)
    value=dict(schema='source_process_fixed_navtest_controller_v1',hosts=a.hosts,evaluation_source=a.evaluation_source,
        controller_source=str(source),updates=list(STAGES),models=list(MODES),seed=2101,
        registration_sha256=file_sha(ROOT/'registration.json'),no_automatic_retry=True)
    value['sha256']=fingerprint(value)
    with (control/'owner.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (control/'identity.json').exists() and read(control/'identity.json')!=value:raise ValueError('Explicit handoff required')
        atomic_json(control/'identity.json',value)
        owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),unix=time.time())
        atomic_json(control/'owner.json',owner)
        try:
            while True:
                require_dispatch();results={}
                for update in STAGES:
                    result=tick(update,value,source);results[str(update)]=result
                    if result['status']!='COMPLETE':break
                complete=len(results)==2 and all(r['status']=='COMPLETE' for r in results.values())
                atomic_json(control/'status.json',dict(owner=owner,observed_unix=time.time(),results=results,
                    status='ALL_FIXED_OBSERVATIONS_COMPLETE' if complete else 'WAITING'))
                if complete or a.once or (control/'STOP_AFTER_TICK').exists():break
                time.sleep(60)
        except BaseException:
            atomic_json(control/'failure.json',dict(owner=owner,error=traceback.format_exc(),unix=time.time(),workers_preserved=True));raise

if __name__=='__main__':main()
