"""Finite same-run recovery of the six authorized ILs after shared-disk exhaustion.

Uses the existing native trainer, checkpoint validators, campaign dispatch and
owners. Never resumes V12, screened losses, completed ILs or closed RL arms.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .health import atomic,alive,read,rewind_journal
from .lifecycle import user_termination

BASE=Path('/mnt/project/qwen-world-il-research-20261030/maintenance/disk_recovery_20261031')
FOUR=Path('/mnt/project/qwen-world-four-frameworks-20261029')
FRESH=Path('/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029')
OPERATOR=Path('/mnt/project/VLA-Drive-qwen-world-time-owner-adoption-20261029')
SCIENCE=Path('/mnt/project/VLA-Drive-qwen-world-loss-from-init-20261029')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewind_at_boundary(path,boundary,archive):
    """Reuse the audited path; retain a disk-truncated *uncommitted* tail raw."""
    try:return rewind_journal(path,boundary,archive)
    except json.JSONDecodeError:
        path=Path(path);archive=Path(archive);raw=path.read_bytes();prefix=[];previous=None
        for line in raw.splitlines(keepends=True):
            row=json.loads(line) # A malformed committed prefix must still fail.
            step=row['updates']
            if previous is not None and step!=previous+1:raise ValueError('Committed journal is not continuous')
            prefix.append(line);previous=step
            if step==boundary:break
        if previous!=boundary:raise ValueError('Missing committed boundary')
        committed=b''.join(prefix);suffix=raw[len(committed):];archive.mkdir(parents=True,exist_ok=True)
        saved=archive/path.name
        if saved.exists():raise ValueError('Refuse to overwrite aborted evidence')
        saved.write_bytes(suffix)
        receipt=dict(file=str(path),aborted_file=str(saved),resume_boundary=boundary,
            original_sha256=hashlib.sha256(raw).hexdigest(),committed_sha256=hashlib.sha256(committed).hexdigest(),
            aborted_sha256=digest(saved),uncommitted_tail_contains_invalid_JSON=True,
            physical_seconds='Uncommitted raw bytes preserved; damaged final row cost unknown')
        atomic(archive/(path.name+'.receipt.json'),receipt)
        tmp=path.with_name(path.name+'.disk_recovery.'+str(os.getpid()));tmp.write_bytes(committed);os.replace(tmp,path)
        return receipt


def verify_exit(remote,job,run):
    recorded=read(Path(job['receipt'])/'RUNNING.json')
    code='''import json,sys;from pathlib import Path
r=json.loads(sys.argv[1]);target=sys.argv[2];live=[]
known={x['pid']:x['start_ticks']for x in r.get('known_rank_processes',[])}
for key in('worker','child'):
 if r.get(key+'_pid'):known[r[key+'_pid']]=r[key+'_start_ticks']
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  s=p.joinpath('stat').read_text().split()
  if s[2]=='Z':continue
  a=[x for x in p.joinpath('cmdline').read_bytes().decode().split('\\0')if x]
  if known.get(int(p.name))==int(s[21])or('--output'in a and a[a.index('--output')+1]==target and '-m'in a and a[a.index('-m')+1]in('tools.qwen_world_compare.train_il','tools.qwen_world_postcot.train_il')):live.append(dict(pid=int(p.name),start_ticks=int(s[21])))
 except(OSError,ValueError,IndexError):pass
print(json.dumps(live))'''
    active=json.loads(remote(job['host'],['/usr/bin/python3','-c',code,json.dumps(recorded),str(run)]))
    if active:raise RuntimeError('Old owned native process remains: '+str(active))
    return dict(host=job['host'],recorded=recorded,verified_no_old_process=True,at=time.time())


def prepare(run,source,job,record,remote,stack):
    if user_termination(run):raise PermissionError('User-terminated run: never restore')
    record.mkdir(parents=True,exist_ok=True)
    if(record/'NEW_INVOCATION.json').exists():raise RuntimeError('Recovery already dispatched; inspect it, do not duplicate')
    atomic(record/'OLD_EXIT.json',verify_exit(remote,job,run))
    for rank in range(8):
        lock=stack.enter_context((run/f'rank{rank}.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=Path(read(run/'LATEST.json')['checkpoint']);seal=read(checkpoint/'COMPLETE');identity=read(checkpoint/'identity.json');files=read(checkpoint/'files.json')
    if (seal.get('status')!='COMPLETE'or digest(checkpoint/'files.json')!=seal['files_sha256']or
        identity['world']!=8 or len([n for n in files if n.endswith('_optim_states.pt')])!=8):raise ValueError('Invalid complete native checkpoint')
    for name in files:
        path=checkpoint/name
        if not path.resolve().is_relative_to(checkpoint.resolve())or not path.is_file()or path.stat().st_size==0:raise ValueError('Missing checkpoint payload: '+name)
    for name,value in identity['source_files'].items():
        if digest(source/name)!=value:raise ValueError('Scientific source changed: '+name)
    import torch
    clients=[torch.load(checkpoint/f'rank{r:02d}_state.pt',map_location='cpu',weights_only=False)for r in range(8)]
    if any(c['state']!=clients[0]['state']or c['identity_sha256']!=seal['identity_sha256']for c in clients):raise ValueError('Rank recovery state mismatch')
    for key in('updates','cursor','phase'):
        if clients[0]['state'][key]!=seal[key]:raise ValueError('Committed state and seal differ')
    receipts=[rewind_at_boundary(run/f'metrics_rank{r}.jsonl',seal['updates'],record/'aborted_journals')for r in range(8)]
    # Preserve failure bytes; the immutable trainer does not inherit a stale
    # FAILURE marker that would block a later legitimate continuation.
    failures=record/'prior_rank_failures';failures.mkdir(exist_ok=True)
    for path in run.glob('FAILURE_rank*.json'):
        destination=failures/path.name
        if destination.exists():raise ValueError('Failure archive already exists')
        os.rename(path,destination)
    previous=record/'previous_progress.json'
    if not previous.exists():previous.write_bytes((run/'progress.json').read_bytes())
    atomic(run/'progress.json',dict(clients[0]['state'],status='RECOVERY_PENDING',checkpoint=str(checkpoint),recovery_receipt=str(record)))
    atomic(record/'PREPARED.json',dict(checkpoint=str(checkpoint),seal=seal,source=str(source),
        identity_sha256=seal['identity_sha256'],rank_states_validated=8,journal_rewinds=receipts,
        original_worker=job,full_file_hash_validation='immutable native restore on startup',
        optimizer_scheduler_RNG_cursor_reset=False,at=time.time()))
    command=list(job['command'])
    if '--resume'in command:command[command.index('--resume')+1]=str(checkpoint)
    else:command+=['--resume',str(checkpoint)]
    return checkpoint,seal,command


def start_owner(command,cwd,record,env):
    with(record/'restored_owner.log').open('ab')as log:
        p=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),command=command,cwd=str(cwd),at=time.time())
    atomic(record/'RESTORED_OWNER.json',value);return value


def eligible_sample(snapshot,minimum):
    return (len(snapshot['gpus'])==8 and min(g['free_MiB']for g in snapshot['gpus'])>=minimum and
        all(o.get('visible')and(o.get('output')or'').startswith('/mnt/project/qwen-world-postcot-20261025/periodic_NAVTEST_')for o in snapshot['owners']))


class SharedDriver:
    """Same immutable worker, explicit qualified shared admission and UTF8 env."""
    def __init__(self,delegate,remote):self.delegate,self.remote=delegate,remote
    def __getattr__(self,name):return getattr(self.delegate,name)
    def launch(self,host,source,command,name,*,gpus='',expected_uuids=None,minimum=None):
        directory=FOUR/'jobs'/name;directory.mkdir(parents=True,exist_ok=True)
        if(directory/'DISPATCH.json').exists():raise ValueError('Invocation already dispatched')
        executor=Path(source)/'tools/qwen_world_compare/job_worker.py'
        args=[PYTHON,str(executor),'--receipt-root',str(directory),'--gpus',gpus,
            '--command-json',json.dumps(command),'--cwd',str(source),'--expected-gpu-uuids-json',json.dumps(expected_uuids),
            '--share-min-free-mib',str(minimum),'--lease-slot','shared']
        env=dict(WORLD_COMPARE_ROOT=str(FOUR),WORLD_COMPARE_TASK='qwen_world_compare_20261029',PYTHONPATH=str(source),
                 PYTHONUTF8='1',LC_ALL='C.UTF-8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        code=('import subprocess,os,json;from pathlib import Path;p=subprocess.Popen('+repr(args)+',cwd='+repr(str(source))+',stdout=open('+repr(str(directory/'worker.log'))+',"ab"),stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**'+repr(env)+'));print(json.dumps(dict(worker_pid=p.pid,worker_start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]))))')
        started=json.loads(self.remote(host,['/usr/bin/env','PYTHONUTF8=1','LC_ALL=C.UTF-8',PYTHON,'-c',code]))
        atomic(directory/'DISPATCH.json',dict(host=host,source=str(source),command=command,invocation=name,
            dispatched_unix=time.time(),executor=str(executor),executor_sha256=digest(executor),lease_slot='shared',**started))
        return directory


def recover_four():
    manifest=read(FOUR/'RUNS.json');plan=manifest['plan'];operator=Path(plan['controller_source']);source=Path(plan['source'])
    os.environ.update(WORLD_COMPARE_ROOT=str(FOUR),WORLD_COMPARE_TASK='qwen_world_compare_20261029')
    sys.path.insert(0,str(operator))
    from tools.qwen_world_compare import campaign as c
    from starVLA.rl.qwen_world_compare.resources import probe,remote
    record=BASE/'FOUR';record.mkdir(parents=True,exist_ok=True)
    if alive(read(FOUR/'campaign/OWNER.json')):raise RuntimeError('Existing campaign alive; do not replace')
    with(FOUR/'campaign/owner.lock').open('a')as control:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from tools.qwen_world_compare import dispatch
        instance=c.Campaign(plan,manifest['source_files'],driver=SharedDriver(dispatch,remote))
        for a in 'ABCD':
            arm_record=record/a;run=Path(plan['architectures'][a]['IL'])
            if(arm_record/'NEW_INVOCATION.json').exists():
                prior=read(arm_record/'NEW_INVOCATION.json')['entry']
                if not any(j['receipt']==prior['receipt']and j['status']=='RUNNING'for j in instance.state['jobs'].values()):raise RuntimeError('Prior recovery needs explicit review')
                continue # Adopt the durable dispatch, never generate another.
            old=instance._last_family(a+'_IL')
            if old is None:raise ValueError('Missing original architecture invocation')
            host='training-vlawm-tsinghua-zt6'if a=='A'else old['host'];snapshot=probe(host)
            if not eligible_sample(snapshot,65000):raise RuntimeError('No qualified idle/known-evaluation headroom: '+host)
            with ExitStack()as locks:
                cp,seal,command=prepare(run,source,old,arm_record,remote,locks)
                # Invalid/missing old RESULT remains as evidence; a verified
                # administrative failure is not relabeled a successful exit.
                old.update(status='FAILED',recovery_review=str(arm_record/'PREPARED.json'))
                instance.persist()
            command=['/usr/bin/env','PYTHONUTF8=1','LC_ALL=C.UTF-8',*command]
            entry=instance._launch(a,f'{a}_IL_diskresume{seal["updates"]:06d}',command,host,
                gpus=range(8),uuids=[g['uuid']for g in snapshot['gpus']],minimum=65000,
                family=a+'_IL',kind='training',stage='IL')
            instance.state['architectures'][a]['status']='IL_RUNNING';instance.persist()
            atomic(arm_record/'NEW_INVOCATION.json',dict(entry=entry,checkpoint=str(cp),boundary=seal['updates']))
    old_owner=read(FOUR/'campaign/OWNER.json')
    argv=[PYTHON,'-u','-m','tools.qwen_world_compare.campaign','run','--config',str(source/'configs/qwen_world_compare/campaign.yaml'),
          '--source',str(source),'--execution-migration',plan['execution_migration'],'--architectures','A','B','C','D','--stages','il','rl','evaluate','--resume']
    start_owner(argv,operator,record,os.environ.copy())


def recover_fresh(arm):
    root=FRESH/arm;record=BASE/arm;record.mkdir(parents=True,exist_ok=True)
    os.environ.update(WORLD_POSTCOT_ROOT=str(root),WORLD_POSTCOT_TASK='qwen_world_fresh_loss_IL_20261029')
    sys.path.insert(0,str(OPERATOR))
    from tools.qwen_world_postcot import campaign as c
    from starVLA.rl.qwen_world_postcot.resources import probe,remote
    if alive(read(root/'OWNER.json')):raise RuntimeError('Existing arm owner alive')
    with(root/'owner.lock').open('a')as control:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'STATE.json')
        jobs=[j for j in state['jobs'].values()if j['role']=='training'];old=max(jobs,key=lambda j:j['dispatch_unix'])
        host=old['host'];snapshot=probe(host)
        if not eligible_sample(snapshot,49152):raise RuntimeError('Original arm host has no qualified evaluation-sharing capacity')
        run=root/'runs/IL_full_seed42'
        with ExitStack()as locks:
            cp,seal,command=prepare(run,SCIENCE,old,record,remote,locks)
            old.update(status='REVIEWED_INFRASTRUCTURE_FAILURE',recovery_review=str(record/'PREPARED.json'))
            # Reconcile refresh workers without discarding successful scenes.
            for job in state['jobs'].values():
                if job is old or job['status']not in('RUNNING','WAITING_OWNED_EXIT'):continue
                try:result=read(Path(job['receipt'])/'RESULT.json')
                except json.JSONDecodeError:result=dict(status='UNREADABLE_METADATA')
                if result and result.get('status')=='EXITED'and result.get('returncode')==0:
                    verify_exit(remote,job,run)
                    job.update(status='COMPLETE',result=result,old_process_exit_verified=True)
                else:
                    proof=verify_exit(remote,job,run)
                    job.update(status='REVIEWED_INFRASTRUCTURE_FAILURE',result=result,
                        recovery_review=str(record/'PREPARED.json'),old_exit_review=proof)
            atomic(root/'STATE.json',state)
        invocation=c.dispatch(state,str(SCIENCE),'full_IL',command,dict(host=host,gpus=list(range(8)),
            uuids=[g['uuid']for g in snapshot['gpus']],minimum_free_MiB=49152,lease_slot='shared',share=49152),role='training')
        atomic(root/'STATE.json',state);atomic(record/'NEW_INVOCATION.json',dict(entry=state['jobs'][invocation],checkpoint=str(cp),boundary=seal['updates']))
    start_owner([PYTHON,'-u','-m','tools.qwen_world_postcot.fresh_loss_owner','--arm',arm,'--runtime-source',str(SCIENCE),'--recover-owner'],OPERATOR,record,os.environ.copy())


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['four','fresh'],required=True);p.add_argument('--arm',choices=['TIME_RESIDUAL','HUBER']);a=p.parse_args()
    os.environ.update(PYTHONUTF8='1',LC_ALL='C.UTF-8');BASE.mkdir(parents=True,exist_ok=True)
    target=BASE/('FOUR'if a.mode=='four'else a.arm);target.mkdir(parents=True,exist_ok=True)
    with(target/'recovery.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            recover_four()if a.mode=='four'else recover_fresh(a.arm)
            atomic(target/'DISPATCH_COMPLETE.json',dict(status='DISPATCHED_AWAIT_REAL_UPDATES',at=time.time(),science_or_budget_changed=False))
        except BaseException:
            atomic(target/'FAILURE.json',dict(at=time.time(),error=traceback.format_exc(),no_blind_retry=True));raise


if __name__=='__main__':main()
