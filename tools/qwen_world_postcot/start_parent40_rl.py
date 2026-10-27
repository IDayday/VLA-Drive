"""Finite IL40k-user-parent qualification -> canonical RL owner.

Uses the existing trainer, complete frozen runtime snapshot, resource admission
worker, four-inner-update lifecycle and native checkpoint transactions. No old
IL campaign is resumed and no other experiment is stopped. One owner per root.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import allowed_hosts, file_sha
from starVLA.rl.qwen_world_postcot.resources import MODEL_PYTHON, OPERATOR_PYTHON, probe


def verify_qualification(root, parent):
    rows=[]
    identity=json.loads((root/'identity.json').read_text())
    if (identity['purpose']!='qualification_not_formal_initialization'
            or identity['SFT_sha256']!=parent['model_sha256']):
        raise ValueError('Qualification parent/purpose changed')
    for rank in range(8):
        result=json.loads((root/f'result_rank{rank}.json').read_text())
        if (not result['qualification'] or result['updates']!=16
                or result['status']!='STOPPED_AT_SAVED_BOUNDARY'
                or result['frozen_SHA_before']!=result['frozen_SHA_after']
                or result['frozen_observation_before']!=result['frozen_observation_after']
                or not any(result['changed_adapters'].values())
                or result['collections']!=4 or result['new_candidates']!=256
                or result['prefill_calls_this_invocation']!=4):
            raise ValueError('Actual-parent eight-GPU RL qualification failed on rank'+str(rank))
        rows.append(result)
    return dict(status='PASS',updates=16,fresh_global_rollouts=4,new_candidates=256,
        qualification_weights_used_for_formal_initialization=False,
        peak_GPU_GiB=max(r['peak_GPU_bytes'] for r in rows)/2**30,
        slowest_worker_wall_seconds=max(r['wall_seconds_this_invocation'] for r in rows),
        logical_update_seconds_per_worker=[r['seconds_update'] for r in rows],
        result_files={f'result_rank{rank}.json':file_sha(root/f'result_rank{rank}.json') for rank in range(8)})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration',required=True)
    args=parser.parse_args()
    registration=Path(args.registration).resolve()
    cfg=json.loads(registration.read_text());root=Path(cfg['root'])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'OWNER_RESULT.json').exists():raise ValueError('Owner already finished; inspect its outcome rather than repeat it')
    source=Path(cfg['source']);host=cfg['host']
    if host not in allowed_hosts(admit_long_job=True):raise PermissionError('Host authorization changed')
    if (file_sha(source/'configs/qwen_world_postcot/rl.yaml')!=cfg['recipe_sha256']
            or file_sha(cfg['parent_selection'])!=cfg['parent_selection_file_sha256']):
        raise ValueError('Registered recipe/parent receipt changed')
    selected=json.loads(Path(cfg['parent_selection']).read_text())['selected']
    state=dict(status='STARTING',pid=os.getpid(),host=socket.gethostname(),start_unix=time.time(),
        registration_sha256=file_sha(registration),stages=[],formal_run=cfg['formal_run'])
    def publish():atomic_json(root/'OWNER_STATE.json',state)
    publish()

    def run_stage(name,command):
        if host not in allowed_hosts(admit_long_job=True):raise PermissionError('Host authorization withdrawn')
        receipt=root/'jobs'/name
        if receipt.exists():raise FileExistsError('Do not duplicate an earlier invocation: '+str(receipt))
        first=probe(host);time.sleep(2);second=probe(host)
        physical=[row['uuid'] for row in second['gpus']]
        if physical!=cfg['gpu_uuids'] or physical!=[row['uuid'] for row in first['gpus']]:
            raise ValueError('Registered GPU mapping changed')
        # Native worker rechecks owners/total/free/utilization under UUID locks.
        # Refuse on changed capacity; never evict another task or silently share.
        if any(row['free_MiB']<cfg['minimum_free_MiB'] for row in second['gpus']):
            raise RuntimeError('Eight-GPU capacity changed before dispatch')
        receipt.mkdir(parents=True)
        row=dict(stage=name,host=host,source=str(source),command=command,
            receipt=str(receipt),dispatch_unix=time.time(),status='RUNNING')
        atomic_json(receipt/'INTENT.json',row)
        state['status']=name;state['stages'].append(row);publish()
        worker=[OPERATOR_PYTHON,'-m','tools.qwen_world_postcot.job_worker',
            '--receipt-root',str(receipt),'--gpus','0,1,2,3,4,5,6,7',
            '--expected-gpu-uuids-json',json.dumps(physical),
            '--minimum-free-mib',str(cfg['minimum_free_MiB']),
            '--command-json',json.dumps(command),'--cwd',str(source)]
        remote=['env','PYTHONPATH='+str(source),*worker]
        transport=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(remote)]
        with (receipt/'transport.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(transport,stdout=log,stderr=subprocess.STDOUT)
            row['transport_pid']=child.pid;publish()
            code=child.wait()
        result=json.loads((receipt/'RESULT.json').read_text()) if (receipt/'RESULT.json').exists() else None
        row.update(returncode=code,end_unix=time.time(),result=result,
            status='EXITED' if code==0 and result and result['returncode']==0 else 'FAILED')
        publish()
        if row['status']!='EXITED':raise RuntimeError('Native training stage failed; last valid checkpoint retained: '+name)

    base=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        '-m','tools.qwen_world_postcot.train_rl','--config','configs/qwen_world_postcot/rl.yaml',
        '--manifest',cfg['manifest'],'--data',cfg['data'],'--il-checkpoint',selected['checkpoint']]
    try:
        run_stage('QUALIFICATION16',base+['--output',cfg['qualification_run'],
            '--qualification','--stop-after-update','16','--save-every','8'])
        acceptance=verify_qualification(Path(cfg['qualification_run']),selected)
        atomic_json(root/'RL_ENGINEERING_ACCEPTANCE.json',acceptance)
        run_stage('FORMAL50000',base+['--output',cfg['formal_run'],
            '--parent-selection',cfg['parent_selection']])
        results=[json.loads((Path(cfg['formal_run'])/f'result_rank{rank}.json').read_text()) for rank in range(8)]
        if any(r['status']!='COMPLETE' or r['qualification'] or r['updates']!=50000
               or r['collections']!=12500 or r['new_candidates']!=800000 for r in results):
            raise ValueError('Formal terminal budget is not complete')
        state.update(status='COMPLETE',end_unix=time.time())
        publish();atomic_json(root/'OWNER_RESULT.json',state)
    except BaseException as error:
        state.update(status='FAILED_REVIEW_REQUIRED',error=repr(error),end_unix=time.time())
        publish();atomic_json(root/'OWNER_RESULT.json',state)
        raise


if __name__=='__main__':main()
