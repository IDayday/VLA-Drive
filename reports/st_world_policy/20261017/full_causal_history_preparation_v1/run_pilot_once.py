#!/usr/bin/env python3
"""Lead-only history64 owned CPU cost pilot. No approval, claim or retry creation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[3]
BASE=Path('/mnt/project/st-world-policy-20261016')
ROOT=BASE/'audits/history_data/causal_history_cost64_v2'
CP={'path':str(BASE/'smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json'),
    'sha256':'13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3'}
LIMITS=dict(wall_seconds=900,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30,
            cpu_affinity=[6,7],max_disk_bytes=32<<20,max_invocations=1)
PYTHON='/root/miniconda3/envs/ddp/bin/python'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def read(pin):
    p=Path(pin['path'])
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)) or p.stat().st_size>4<<20:raise ValueError('Small canonical control required')
    before=p.stat();raw=p.read_bytes();after=p.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('Control changed')
    return json.loads(raw)

def validate(a,now,child=False):
    if a.get('schema')!='st_history64_owned_admission_v1' or a.get('status')!='APPROVED_HISTORY64_PILOT_ONLY' or a.get('limits')!=LIMITS:raise ValueError('Exact independently reviewed finite admission required')
    if a.get('reviewer') in (None,'','/root/st_temporal_readiness_inventory') or a.get('hostname')!=socket.gethostname():raise ValueError('Independent reviewer and actual host required')
    if a.get('checkpoint_receipt')!=CP or a.get('output_root')!=str(ROOT) or a.get('runtime')!=str(RUNTIME):raise ValueError('Fixed CP/runtime/output identity required')
    for k in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(a.get(k)) not in (float,int) or not math.isfinite(a[k]):raise ValueError('Finite lease required')
    if not a['issued_at_unix']<=now<=a['latest_start_unix'] or a['valid_until_unix']<now+(0 if child else 940):raise ValueError('Expired or insufficient lease')

def controls(args):
    a=read({'path':args.admission,'sha256':args.admission_sha256});validate(a,time.time(),args.child)
    for pin in a['controller_sources']:
        p=Path(pin['path'])
        if not p.is_absolute() or any(x.is_symlink() for x in (p,*p.parents)) or p.stat().st_size>1<<20:raise ValueError('Small canonical controller required')
        raw=p.read_bytes()
        if p.stat().st_size>1<<20 or hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('Controller source changed')
    required={str(HERE/'run_pilot_once.py'),str(HERE.parent/'full_future_metadata_preparation_v1/run_build_once.py')}
    helper=HERE.parent/'spatial_probe_actual_preparation_v1'
    required|={str(helper/n) for n in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}
    if {p['path'] for p in a['controller_sources']}!=required:raise ValueError('Exact monitor dependencies required')
    sys.path.insert(0,str(helper));sys.path.insert(0,str(RUNTIME))
    import supervise_once as s
    import observed_cli as obs
    from tools.st_world_policy import metadata_storage as m,cache_storage_budget as b
    core=load('history_owned_core',HERE.parent/'full_future_metadata_preparation_v1/run_build_once.py')
    reg=read(a['registration']);command=reg['command']
    selection_pin={'path':str(BASE/'audits/history_data/smoke_train64_v1/SELECTED_SCENES.json'),'sha256':'dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01'}
    read(selection_pin)
    if command[:3]!=[PYTHON,'-m','tools.st_world_policy.prepare_causal_history']:raise ValueError('Fixed builder command required')
    if reg.get('writer_kind')!='history_preparation' or reg['output_root']!=str(ROOT) or reg['cwd']!=str(RUNTIME):raise ValueError('Wrong history registration')
    from tools.st_world_policy import prepare_causal_history as writer
    parsed=writer.parser().parse_args(command[3:])
    expected={'max_scenes':64,'split':'train','target_times':[0.], 'mode':'registered','resume':False,
      'scene_list':str(BASE/'audits/history_data/smoke_train64_v1/SELECTED_SCENES.json'),
      'student_root':'/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1',
      'raw_log_root':'/mnt/navsim/trainval_navsim_logs/trainval',
      'sensor_root':'/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval',
      'partition':'/mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json',
      'output':str(ROOT)}
    if any(getattr(parsed,k)!=v for k,v in expected.items()) or parsed.pin_selection:raise ValueError('Fixed pilot population/recipe changed')
    guard=dict(policy_path=a['policy']['path'],policy_sha256=a['policy']['sha256'],registration_path=a['registration']['path'],registration_sha256=a['registration']['sha256'],checkpoint_receipt=CP['path'],checkpoint_sha256=CP['sha256'])
    def storage_guard():
        cp=read(CP)
        if cp.get('status')!='CURRENT' or cp.get('remaining_bytes')!=2303*(1<<30) or cp.get('free_floor_bytes')!=1<<40 or cp.get('valid_until_unix',0)<a['valid_until_unix']:raise ValueError('Checkpoint obligations/lease changed')
        free=os.statvfs(BASE);available=free.f_bavail*free.f_frsize
        if available<2303*(1<<30)+(1<<40):raise ValueError('Shared free floor')
        return available
    storage_guard()
    return a,reg,guard,s,obs,m,b,core,storage_guard

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True);p.add_argument('--child',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    os.sched_setaffinity(0,{6,7})
    a,reg,guard,s,obs,m,b,core,storage_guard=controls(args)
    if args.child:
        os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
        if os.getpid()!=os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Owned CPU child required')
        before=obs.proc_io();result=None;error=None
        try:result=m.run_registered_metadata(**guard)
        except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
        after=obs.proc_io()
        print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':result,'error':error,'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
        return 2 if error else 0
    start=ROOT/'CONTROL_HISTORY_STARTED.json';result_path=ROOT/'CONTROL_HISTORY_RESULT.json'
    core.publish_guarded(m,b,guard,start,{'admission':args.admission,'sha256':args.admission_sha256,'automatic_retry':False})
    # Existing one-shot marker prevents duplicate invocation; inherited read-only FD
    # proves that child belongs to this live parent, not a reconstructed old PID.
    with start.open('rb') as lock:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(RUNTIME),ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()))
        command=[PYTHON,str(Path(__file__).resolve()),'--child','--admission',args.admission,'--admission-sha256',args.admission_sha256]
        observed=core.monitor(command,runtime=str(RUNTIME),env=env,fd=lock.fileno(),root=ROOT,s=s,m=m,obs=obs,budget=LIMITS,cpu_affinity=(6,7),storage_guard=storage_guard,lease_valid_until=a['valid_until_unix'])
    envelope=observed.get('child_accounting') or {};result=envelope.get('cli_result') or {}
    charged=core.charge_after_reap(observed['read_accounting'],obs.proc_io())
    ok=observed['returncode']==0 and not observed['failure'] and result.get('artifact',{}).get('status')=='COMPLETE' and charged<=LIMITS['read_bytes']
    core.publish_guarded(m,b,guard,result_path,{'status':'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY','resources':observed,'charged_rchar_bytes':charged,'scientific_result':False})
    print(json.dumps({'status':'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY','result':str(result_path)}));return 0 if ok else 2

if __name__=='__main__':raise SystemExit(main())
