"""Finite per-arm native handoff owner; never signals a GPU process.

Waits for each actual eight-rank equivalence proof. Stops only the old CPU
campaign owner, then requests one learner at a time to save/exit through its
existing STOP_REQUESTED protocol. Other learners/evaluators remain running.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PY='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)


def alive(pid,start):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split()
        return s[2]not in('Z','X')and int(s[21])==start
    except (OSError,ValueError,IndexError):return False


def wait_until(predicate,seconds,message):
    start=time.monotonic()
    while not predicate():
        if time.monotonic()-start>seconds:raise TimeoutError(message)
        time.sleep(5)


def signal_verified_cpu(pid,start,cwd,entry):
    # The qualified model Python was built without os.pidfd_open. Use the
    # existing system operations interpreter, retaining PID/start/cwd/argv
    # checks under the pidfd rather than falling back to a racy os.kill(pid).
    code='''import os,sys,signal,json
from pathlib import Path
pid,start=int(sys.argv[1]),int(sys.argv[2]);cwd,entry=sys.argv[3:5]
fd=os.pidfd_open(pid)
try:
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split()
 args=(p/'cmdline').read_bytes().decode().split('\\0')
 assert s[2]not in('Z','X')and int(s[21])==start
 assert str((p/'cwd').resolve())==cwd and entry in args
 signal.pidfd_send_signal(fd,signal.SIGTERM)
finally:os.close(fd)
'''
    subprocess.run(['/usr/bin/python3','-c',code,str(pid),str(start),str(cwd),entry],check=True)


def available_native_slot(uuids):
    for slot,folder in [('shared','runtime_gpu_shared_workers'),('primary','runtime_gpu_leases')]:
        held=[]
        try:
            for uuid in uuids:
                f=(Path('/mnt/project')/folder/(uuid+'.lock')).open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:f.close();break
                held.append(f)
            if len(held)==len(uuids):return slot
        finally:
            for f in held:f.close()
    return None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True);p.add_argument('--root',required=True)
    p.add_argument('--performance-root',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--controller-source',required=True);p.add_argument('--controller-migration',required=True)
    args=p.parse_args();source=Path(args.source).resolve();root=Path(args.root).resolve();perf=Path(args.performance_root).resolve()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    own=(out/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(source);sys.path.insert(0,str(source));os.environ['WORLD_COMPARE_ROOT']=str(root)
    os.environ['WORLD_COMPARE_TASK']='qwen_world_compare_20261029'
    from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint,require_dispatch
    from starVLA.rl.qwen_world_compare.resources import remote,probe
    from tools.qwen_world_compare import dispatch
    require_dispatch()
    state_path=out/'STATE.json'
    if state_path.exists():
        if not args.resume:raise ValueError('Existing handoff requires explicit resume')
        state=read(state_path)
        if state['source']!=str(source):raise ValueError('Source changed during handoff')
    else:state=dict(status='WAITING_NATIVE_PROOFS',source=str(source),architectures={},started=time.time())
    if state.get('error'):
        state.setdefault('error_history',[]).append(dict(error=state.pop('error'),at=state.get('failure_at')))
    state.update(owner_pid=os.getpid(),owner_start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        owner_file=str(Path(__file__).resolve()),owner_sha256=file_sha(__file__))
    write(state_path,state)
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONUTF8='1',LC_ALL='C.UTF-8',CUDA_VISIBLE_DEVICES='')
    owner_lock=None
    def progress():state['at']=time.time();write(state_path,state)
    def tool(arguments):subprocess.run([PY,'-m','tools.qwen_world_compare.prepare_execution_handoff',*arguments],cwd=source,env=env,check=True)
    def stop_cpu_owner():
        nonlocal owner_lock
        if not state.get('original_owner'):
            original=read(root/'campaign/OWNER.json');state['original_owner']=original;progress()
        original=state['original_owner'];pid=original['pid'];ticks=original['start_ticks']
        if alive(pid,ticks):
            argv=(Path(f'/proc/{pid}/cmdline')).read_bytes().decode().split('\0')
            cwd=Path(f'/proc/{pid}/cwd').resolve()
            if 'tools.qwen_world_compare.campaign'not in argv or cwd!=Path(original['controller_source']).resolve():
                raise ValueError('Refusing to signal an unverified CPU owner')
            signal_verified_cpu(pid,ticks,cwd,'tools.qwen_world_compare.campaign')
            wait_until(lambda:not alive(pid,ticks),180,'Old CPU owner did not finish; native workers untouched')
        owner_lock=(root/'campaign/owner.lock').open('a');fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'campaign/OWNER.json',dict(pid=os.getpid(),start_ticks=state['owner_start_ticks'],
            started_unix=time.time(),owner_kind='validated_per_arm_execution_handoff',
            source=original['source'],controller_source=str(source),target_source=str(source),progress_file=str(state_path),
            native_workers_preserved=True,mixed_execution_until_all_four_committed=True))
        state['controller_stopped']=True;progress()
    try:
        for arch in 'ABCD':
            row=state['architectures'].setdefault(arch,{})
            if row.get('status')=='PASS':continue
            proof=perf/'native_qualification'/arch/'COMPARISON.json'
            state['status']='WAITING_NATIVE_EQUIVALENCE_'+arch;progress()
            while not proof.exists():
                result=perf/'native_qualification/job_attempt001/RESULT.json'
                if result.exists()and read(result).get('returncode')!=0:raise RuntimeError('Native qualification failed; formal learners retained')
                time.sleep(15)
            value=read(proof)
            if value['status']!='PASS'or value['world']!=8 or value['architecture_id']!=arch:
                raise ValueError('Not an actual eight-rank arm qualification')
            if owner_lock is None:stop_cpu_owner()
            state['status']='HANDOFF_'+arch;progress()
            current=read(root/'campaign/STATE.json')
            if 'old_job'not in row:
                jobs=[(name,j)for name,j in current['jobs'].items()if j.get('family')==arch+'_IL']
                name,job=max(jobs,key=lambda item:(item[1].get('at',0),item[1].get('attempt',0)))
                if job.get('execution_source')!=state['original_owner']['source']:
                    raise ValueError('Unexpected original learner source; do not stop it')
                row.update(old_job_name=name,old_job=job);progress()
            job=row['old_job'];receipt=Path(job['receipt']);run=root/'runs'/('ARCH_'+arch)/'IL_seed42'
            if 'migration'not in row:
                old_result=receipt/'RESULT.json'
                if not old_result.exists():
                    (receipt/'STOP_REQUESTED').touch();row['stop_requested_at']=time.time();progress()
                    wait_until(old_result.exists,600,'Native save/exit incomplete: '+arch)
                ended=read(old_result)
                if ended.get('returncode')!=0 or ended.get('status')!='EXITED':raise ValueError('Original learner did not stop normally')
                latest=read(run/'LATEST.json');parent=Path(latest['checkpoint'])
                if not(parent/'COMPLETE').exists():raise ValueError('Original stop lacks complete native state')
                old_id=read(parent/'identity.json')
                if old_id!=read(run/'IDENTITY.json'):raise ValueError('Run and sealed boundary disagree')
                row.update(old_boundary=latest['updates'],old_checkpoint=str(parent),host=job['host']);progress()
                migration=out/arch
                tool(['--checkpoint',str(parent),'--output',str(migration),'--checkpoint-policy','all_off','--cpu-prefetch-depth','2',
                    '--proof',str(proof),'--proof',str(perf/'all_graphs'/arch/'RESULT.json'),
                    '--proof',str(perf/'CPU_PREFETCH_TRANSFER.json')])
                row['migration']=str(migration/'MIGRATION.json');progress()
            migration=read(row['migration']);host=row['host'];devices=probe(host)
            if [g['uuid']for g in devices['gpus']]!=job['uuids']:raise ValueError('Original GPU mapping changed')
            if 'new_job'not in row:
                command=[PY,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m','tools.qwen_world_compare.train_il',
                    '--config',str(source/'configs/qwen_world_compare'/('arch_'+arch.lower()+'.yaml')),'--output',str(run),
                    '--resume',migration['parent_checkpoint'],'--checkpoint-policy','all_off','--cpu-prefetch-depth','2',
                    '--execution-receipt',row['migration']]
                slot=available_native_slot(job['uuids'])
                if slot is None:raise RuntimeError('No all-rank native slot is currently free; no worker started')
                row['lease_slot']=slot;progress()
                previous=list((root/'jobs').glob(f'execution_fast_{source.name}_{arch}_resume{row["old_boundary"]:06d}_attempt*'))
                first=max([int(x.name.rsplit('attempt',1)[1])for x in previous],default=0)+1
                for attempt in range(first,first+3):
                    name=f'execution_fast_{source.name}_{arch}_resume{row["old_boundary"]:06d}_attempt{attempt:03d}'
                    entry=dict(architecture=arch,logical=arch+'_IL_execution_resume'+str(row['old_boundary']),family=arch+'_IL',
                        attempt=attempt,command=command,host=host,scientific_source=str(source),execution_source=str(source),
                        execution_migration_sha256=migration['sha256'],kind='training',stage='IL',uuids=job['uuids'],gpus=list(range(8)),
                        receipt=str(root/'jobs'/name),status='LAUNCH_INTENT',at=time.time())
                    current=read(root/'campaign/STATE.json');current['jobs'][name]=entry;write(root/'campaign/STATE.json',current)
                    directory=dispatch.launch(host,source,command,name,gpus='0,1,2,3,4,5,6,7',expected_uuids=job['uuids'],
                        share_minimum=40000,lease_slot=slot)
                    wait_until(lambda:(directory/'RUNNING.json').exists()or(directory/'RESULT.json').exists(),60,'Missing admission receipt')
                    result=dispatch.job_state(directory)
                    if result.get('child_not_started')is True and result.get('returncode')==75:
                        entry.update(status='REFUSED_RESOURCE',result=result);current=read(root/'campaign/STATE.json');current['jobs'][name]=entry;write(root/'campaign/STATE.json',current)
                        time.sleep(10);continue
                    if result.get('status')not in('RUNNING','EXITED'):raise ValueError('Native launch failed')
                    entry['status']='RUNNING';current=read(root/'campaign/STATE.json');current['jobs'][name]=entry;write(root/'campaign/STATE.json',current)
                    row['new_job']=str(directory);row['new_job_name']=name;progress();break
                else:raise RuntimeError('Bounded pre-child resource retries exhausted')
            directory=Path(row['new_job']);complete=run/'EXECUTION_HANDOFF_COMPLETE.json'
            def committed():
                if(directory/'RESULT.json').exists()and read(directory/'RESULT.json').get('returncode')!=0:
                    raise RuntimeError('New invocation failed; pinned original native state retained')
                return complete.exists()and read(complete).get('migration_sha256')==migration['sha256']
            wait_until(committed,900,'First native handoff boundary incomplete: '+arch)
            committed_value=read(complete)
            if committed_value['from_updates']!=row['old_boundary']or committed_value['updates']!=row['old_boundary']+2:
                raise ValueError('Native continuation skipped or reset updates')
            actual_slot=read(directory/'RUNNING.json')['lease_slot']
            row.update(status='PASS',committed=committed_value,completed_at=time.time(),lease_slot=actual_slot);progress()
        state['status']='ADOPTING_STEADY_CAMPAIGN';progress()
        owner_lock.close();owner_lock=None
        controller=Path(args.controller_source).resolve();controller_env=dict(env,PYTHONPATH=str(controller))
        subprocess.run([PY,str(controller/'tools/qwen_world_compare/prepare_execution_handoff.py'),'--adopt-campaign',str(root),
            '--native-proof-root',str(perf/'native_qualification'),'--output',str(out/'campaign_adoption'),
            '--training-source',str(source),'--controller-migration',args.controller_migration],
            cwd=source,env=controller_env,check=True)
        for arch,row in state['architectures'].items():
            tool(['--activate-architecture',arch,'--root',str(root),'--receipt',row['migration'],'--host',row['host'],
                '--lease-slot',row['lease_slot'],'--share-min-free-mib','40000'])
        command=[PY,'-u','-m','tools.qwen_world_compare.campaign','run','--config',str(source/'configs/qwen_world_compare/campaign.yaml'),
            '--source',str(source),'--architectures','A','B','C','D','--stages','il','rl','evaluate','--resume','--poll-seconds','30']
        command+=['--execution-migration',args.controller_migration]
        with(out/'steady_campaign.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(command,cwd=controller,env=controller_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['steady_campaign_pid']=child.pid;state['steady_campaign_command']=command;progress()
        wait_until(lambda:read(root/'campaign/OWNER.json').get('pid')==child.pid,90,'Steady owner did not acquire its lock')
        state.update(status='PASS',completed=time.time(),all_original_committed_updates_preserved=True,all_four_actually_resumed=True)
        progress();write(out/'COMPLETE.json',state)
    except BaseException as error:
        state.update(status='FAILED_REVIEW_REQUIRED',error=repr(error),failure_at=time.time());progress();raise
    finally:
        if owner_lock is not None:owner_lock.close()


if __name__=='__main__':main()
