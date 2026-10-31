"""Resume only the still-authorized SafetyEP run from native28k, not a new RL arm."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .health import atomic,read,alive
from .disk_recovery import verify_exit,rewind_at_boundary,digest

ROOT=Path('/mnt/project/qwen-world-rl-optimization-20261030')
OPERATOR=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030')
RECORD=Path('/mnt/project/qwen-world-il-research-20261030/maintenance/disk_recovery_20261031/SAFETY_EP')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def main():
    RECORD.mkdir(parents=True,exist_ok=True)
    with ExitStack()as locks:
        for p in(RECORD/'recovery.lock',ROOT/'owners/CAMPAIGN/owner.lock'):
            stream=locks.enter_context(p.open('a'));fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if alive(read(ROOT/'owners/CAMPAIGN/OWNER.json')):raise RuntimeError('Campaign owner still alive')
        if(RECORD/'NEW_INVOCATION.json').exists():raise RuntimeError('Already dispatched; never duplicate')
        os.environ.update(WORLD_POSTCOT_TASK='qwen_world_rl_optimization_20261030',PYTHONUTF8='1',LC_ALL='C.UTF-8')
        sys.path.insert(0,str(OPERATOR))
        from tools.qwen_world_research import dispatch
        from starVLA.rl.qwen_world_postcot.resources import remote
        row=next(x for x in read(ROOT/'RUNS.json')['runs']if x['id']=='PPO_SAFETY_EP')
        if row['status']!='TRAINING_DISPATCHED'or row['research_budget_updates']!=50000:raise ValueError('Safety arm is not approved to continue')
        run=Path(row['root']);old=Path(row['job']);running=read(old/'RUNNING.json')
        atomic(RECORD/'OLD_EXIT.json',verify_exit(remote,dict(receipt=str(old),host=row['host']),run))
        for rank in range(8):
            lock=locks.enter_context((run/f'rank{rank}.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cp=Path(read(run/'latest.json')['checkpoint']);seal=read(cp/'COMPLETE');identity=read(cp/'identity.json');files=read(cp/'files.json')
        if seal['phase']!='READY'or seal['next_inner']!=0 or seal['updates']!=28000 or digest(cp/'files.json')!=seal['files_sha256']:raise ValueError('Unexpected native transaction boundary')
        if any(not(cp/name).is_file()or(cp/name).stat().st_size==0 for name in files):raise ValueError('Missing native RL payload')
        for name,value in identity['source_files'].items():
            if digest(Path(row['source'])/name)!=value:raise ValueError('Scientific source changed: '+name)
        receipts=[rewind_at_boundary(run/f'metrics_rank{r}.jsonl',28000,RECORD/'aborted_journals')for r in range(8)]
        state_path=ROOT/'owners/CAMPAIGN/STATE.json';previous=state_path.read_bytes();(RECORD/'PREVIOUS_CAMPAIGN_STATE.json').write_bytes(previous)
        (RECORD/'PREVIOUS_RUNS.json').write_bytes((ROOT/'RUNS.json').read_bytes())
        atomic(RECORD/'PREPARED.json',dict(checkpoint=str(cp),seal=seal,rank_journal_rewinds=receipts,
            source=row['source'],old_job=str(old),phase='READY',next_inner=0,original_target=35000,original_final_budget=50000,
            source_or_recipe_changed=False,full_tensor_optimizer_rng_validation='native restore',at=time.time()))
        # rank locks must be free for the real immutable entry.
    job=dispatch.dispatch_training(ROOT,'PPO_SAFETY_EP',35000,resume=str(cp),source=row['source'])
    state=json.loads(previous);arm=state['runs']['PPO_SAFETY_EP'];node=arm['stages']['35000']
    node.setdefault('previous_train_jobs',[]).append(dict(job=str(old),status='UNREADABLE_RESULT_AFTER_DISK_FAILURE',recovery_review=str(RECORD/'PREPARED.json')))
    node.update(train_job=str(job),recovery_native_checkpoint=str(cp));node.pop('train_result',None)
    arm.update(status='TRAINING',job=str(job),recovery_review=str(RECORD/'PREPARED.json'))
    atomic(state_path,state);atomic(RECORD/'NEW_INVOCATION.json',dict(job=str(job),checkpoint=str(cp),boundary=28000,target=35000))
    argv=[PYTHON,'-u','-m','tools.qwen_world_research.campaign_owner','--root',str(ROOT)]
    with(RECORD/'restored_owner.log').open('ab')as log:
        p=subprocess.Popen(argv,cwd=OPERATOR,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(RECORD/'RESTORED_OWNER.json',dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),command=argv,at=time.time()))


if __name__=='__main__':
    RECORD.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException:
        atomic(RECORD/'FAILURE.json',dict(error=traceback.format_exc(),at=time.time(),no_blind_retry=True));raise
