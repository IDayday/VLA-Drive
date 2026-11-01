"""Exact TIME_RESIDUAL native recovery; no model/configuration modification."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

ROOT=Path('/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/TIME_RESIDUAL')
SOURCE=Path('/mnt/project/VLA-Drive-qwen-world-loss-from-init-20261029')
WORKER=Path('/mnt/project/VLA-Drive-abcd-il-c-recovery-20261101')
HOST='training-vlawm-tsinghua-zt5'


def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    out=ROOT/'operations/recovery_20261101';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'operation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'LAUNCH.json').exists():raise FileExistsError('Do not duplicate a dispatched recovery')
    run=ROOT/'runs/IL_full_seed42';parent=Path(json.loads((run/'LATEST.json').read_text())['checkpoint'])
    identity=json.loads((parent/'identity.json').read_text());seal=json.loads((parent/'COMPLETE').read_text())
    if seal['updates']!=79000:raise ValueError('Review a different recovery boundary explicitly')
    for name,digest in identity['source_files'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:raise ValueError('Scientific source changed: '+name)
    if hashlib.sha256((parent/'files.json').read_bytes()).hexdigest()!=seal['files_sha256']:raise ValueError('Checkpoint inventory seal changed')
    old=json.loads((ROOT/'OWNER.json').read_text());p=Path('/proc')/str(old['pid'])
    if p.exists():
        actual=(p/'stat').read_text().split();argv=[x for x in(p/'cmdline').read_bytes().decode().split(chr(0))if x]
        if int(actual[21])!=old['start_ticks']or '--arm'not in argv or argv[argv.index('--arm')+1]!='TIME_RESIDUAL':raise ValueError('Owner identity changed')
        cwd=str((p/'cwd').resolve());fd=os.pidfd_open(old['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
    else:raise RuntimeError('Review missing old CPU owner before recovery')
    owner_lock=(ROOT/'owner.lock').open('a');deadline=time.monotonic()+40
    while True:
        try:fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('Old owner lock remains held')
            time.sleep(.5)
    state=json.loads((ROOT/'STATE.json').read_text());write(out/'STATE_BEFORE.json',state);write(out/'OWNER_BEFORE.json',old)
    oldjob=state['jobs']['full_IL_attempt005'];write(out/'OLD_JOB.json',oldjob)
    locks=[]
    for rank in range(8):
        stream=(run/f'rank{rank}.lock').open('a');fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
    archive=out/'aborted_journals';archive.mkdir(exist_ok=False);journals=[]
    for rank in range(8):
        path=run/f'metrics_rank{rank}.jsonl';original=archive/path.name;os.link(path,original)
        tmp=path.with_name(path.name+'.recovery.tmp');kept=aborted=0;last=None;worker_seconds=0.
        with original.open('rb')as source,tmp.open('wb')as target:
            for line in source:
                row=json.loads(line)
                if row['updates']<=seal['updates']:target.write(line);kept+=1
                else:aborted+=1;worker_seconds+=row.get('seconds',0.)
                last=row['updates']
            target.flush();os.fsync(target.fileno())
        if last!=79779 or aborted!=779:raise ValueError('Interrupted suffix differs from reviewed 79779 endpoint')
        os.replace(tmp,path);journals.append(dict(rank=rank,archived=str(original),prefix_rows=kept,
            replay_updates=[79001,79779],replay_count=aborted,aborted_rank_worker_seconds=worker_seconds))
        print('archived rank',rank,'prefix',kept,'replay',aborted,flush=True)
    write(out/'JOURNAL_ROLLBACK.json',dict(parent=str(parent),ranks=journals,raw_files_preserved=True))
    for stream in locks:stream.close()
    command=list(json.loads((Path(oldjob['receipt'])/'RUNNING.json').read_text())['command']);command[command.index('--resume')+1]=str(parent)
    query="import subprocess,json;print(json.dumps(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()))"
    uuids=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',HOST,'/usr/bin/python3','-c',shlex.quote(query)],text=True))
    job=ROOT/'jobs/full_IL_attempt006';job.mkdir(exist_ok=False)
    entry=dict(name='full_IL',role='training',receipt=str(job),host=HOST,uuids=uuids,gpus=list(range(8)),command=command,
        status='RUNNING',dispatch_unix=time.time(),lease_slot='primary',scientific_source_unchanged=True)
    oldjob.update(status='REVIEWED_INFRASTRUCTURE_FAILURE',recovery_evidence=str(out/'JOURNAL_ROLLBACK.json'))
    state['jobs']['full_IL_attempt006']=entry;write(ROOT/'STATE.json',state)
    routing=json.loads((ROOT/'EXECUTION_ROUTING.json').read_text());write(out/'ROUTING_BEFORE.json',routing)
    routing.update(training_hosts=[HOST],request='User withdrew vla-zt3; recover same native TIME IL on fixed whitelist')
    write(ROOT/'EXECUTION_ROUTING.json',routing)
    write(job/'CAMPAIGN_INTENT.json',dict(source=str(SOURCE),invocation='full_IL_attempt006',job=entry))
    envs=dict(WORLD_COMPARE_ROOT=str(ROOT),WORLD_COMPARE_TASK='qwen_world_fresh_loss_IL_20261029',
        WORLD_POSTCOT_ROOT=str(ROOT),WORLD_POSTCOT_TASK='qwen_world_fresh_loss_IL_20261029',
        PYTHONUTF8='1',LC_ALL='C.UTF-8',PYTHONPATH=str(WORKER))
    invocation=['/usr/bin/python3','-u','-m','tools.qwen_world_compare.job_worker','--receipt-root',str(job),
        '--gpus','0,1,2,3,4,5,6,7','--expected-gpu-uuids-json',json.dumps(uuids),'--command-json',json.dumps(command),
        '--cwd',str(SOURCE),'--share-min-free-mib','32768','--lease-slot','primary','--admission-wait-seconds','1800']
    code='import subprocess,os,json;f=open('+repr(str(out/'worker.log'))+',"ab",buffering=0);env=dict(os.environ);env.update('+repr(envs)+');p=subprocess.Popen('+repr(invocation)+',cwd='+repr(str(WORKER))+',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(pid=p.pid)))'
    actual=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',HOST,'/usr/bin/python3','-c',shlex.quote(code)],text=True))
    write(out/'LAUNCH.json',dict(at_unix=time.time(),job=str(job),worker=actual,host=HOST,command=command,
        native_parent=str(parent),source=str(SOURCE),resource_worker=str(WORKER),resource_task=envs['WORLD_COMPARE_TASK'],minimum_free_MiB=32768))
    owner_lock.close()
    log=(out/'owner.log').open('ab',buffering=0);env=dict(os.environ,WORLD_POSTCOT_ROOT=str(ROOT),WORLD_POSTCOT_TASK='qwen_world_fresh_loss_IL_20261029',PYTHONUTF8='1',LC_ALL='C.UTF-8')
    new=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'CPU_ADOPTION.json',dict(old_owner=old,new_pid=new.pid,argv=argv,cwd=cwd,at_unix=time.time(),GPU_signals=0))
    print(json.dumps(dict(worker=actual,cpu_owner=new.pid,host=HOST,legal_parent_update=79000)),flush=True)


if __name__=='__main__':main()
