"""Finish only three reviewed, unstarted Huber evaluation shards; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest, finalize
from tools.qwen_world_postcot.recover_admission_only import admission_refusal, alive, finish_model
from tools.qwen_world_postcot.user_interim_owner import read, write


ROOT=Path('/mnt/project/qwen-world-postcot-20261025/loss_designs_20261028/NAVTEST')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    owner=read(ROOT/'owner.json');source=Path(owner['source']);directory=ROOT/'HUBER_065000'
    if alive(owner) or alive(read(ROOT/'ADMISSION_RECOVERY_OWNER.json')):
        raise RuntimeError('Original evaluation/recovery owner still alive')
    if (ROOT/'COMPLETE.json').exists():raise ValueError('Already complete')
    def git(*argv):return subprocess.check_output(['git',*argv],cwd=source,text=True)
    if git('rev-parse','HEAD').strip()!=owner['source_sha'] or git('status','--porcelain'):
        raise ValueError('Original runtime must remain clean and unchanged')
    for job in directory.glob('jobs/*'):
        result=read(job/'result.json')
        if result.get('status')=='FAILED' and not admission_refusal(job):
            raise ValueError('An executed/scientific failure needs separate review: '+str(job))
    for rank in (4,5):
        if not admission_refusal(directory/'jobs'/f'rank{rank}_admission_recovery001'):
            raise ValueError('Expected the specifically reviewed pre-CUDA refusals')
    if (directory/'jobs/rank7').exists():raise ValueError('Unexpected original rank7 invocation')
    for rank in (0,1,2,3,6):
        tag=f'rank{rank}'if rank in(0,1)else f'rank{rank}_admission_recovery001'
        result=read(directory/'jobs'/tag/'result.json')
        if result.get('status')!='PROCESS_COMPLETE' or result.get('exit_code')!=0:
            raise ValueError('Previously successful shard is incomplete')
    if args.check_only:
        print('PASS: ranks0/1/2/3/6 complete; ranks4/5 refused before child; rank7 never dispatched; runtime unchanged.')
        return
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=ROOT/'admission_recovery002';attempt.mkdir(exist_ok=False)
    write(attempt/'OWNER.json',dict(pid=os.getpid(),at=time.time(),ranks=[4,5,7],
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source=str(source),source_sha=owner['source_sha'],old_failures_preserved=True))
    write(ROOT/'ADMISSION_RECOVERY_LATEST.json',dict(attempt=str(attempt),at=time.time(),ranks=[4,5,7]))
    pin=read(ROOT/'PINNED.json')['models']['HUBER_065000']
    def new_request():
        r=SnapshotRequest(directory,pin);r.source=source;r.source_sha=owner['source_sha'];return r
    def run_rank(rank):
        request=new_request();spec=read(directory/'jobs/rank1/spec.json');module=spec['module']
        command=list(spec['command']);command[command.index('--rank')+1]=str(rank)
        argv=command[command.index(module)+1:]
        for number in (2,3,4):
            route=None
            while route is None:
                routes,inventory=request.resources.available(min_free_MiB=32768)
                route=request.route(routes,inventory)
                if route is None:time.sleep(15)
            name=f'rank{rank}_admission_recovery{number:03d}'
            request.submit(name,module,argv,route)
            while request.done(name)is None:time.sleep(15)
            if request.done(name)is True:return
            if not admission_refusal(directory/'jobs'/name):
                raise RuntimeError('An executed model failure cannot be retried as admission contention')
            write(attempt/f'rank{rank}_refusal{number:03d}.json',dict(job=str(directory/'jobs'/name),
                child_started=False,scoring_calls=0,at=time.time()))
            time.sleep(15)
        raise RuntimeError('Bounded pre-CUDA retry budget exhausted')
    try:
        with ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(run_rank,(4,5,7)))
        if git('rev-parse','HEAD').strip()!=owner['source_sha'] or git('status','--porcelain'):
            raise ValueError('Changed original source during worker execution')
        # The reused finisher validates ALL12146 scene receipts and identities.
        # No new candidates, scorer semantics, or partial means are introduced.
        request=new_request();finish_model(request,[])
        requests={}
        for label,pin in read(ROOT/'PINNED.json')['models'].items():
            r=SnapshotRequest(ROOT/label,pin);r.source=source;r.source_sha=owner['source_sha'];requests[label]=r
        report=source/'reports/qwen_world_IL_loss_designs_20261028';finalize(ROOT,requests,report)
        prefix=str(report.relative_to(source))
        for row in git('status','--porcelain=v1','-z','--untracked-files=all').split('\0'):
            if row and not row[3:].startswith(prefix+'/'):raise ValueError('Unrelated publication change')
        subprocess.run(['git','add','--',prefix],cwd=source,check=True)
        subprocess.run(['git','commit','-m','Complete independent IL-loss Test after reviewed admission-only recovery'],cwd=source,check=True)
        branch=git('branch','--show-current').strip();commit=git('rev-parse','HEAD').strip()
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
        if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=commit:raise ValueError('Remote SHA mismatch')
        publication=dict(status='PUBLISHED',commit=commit,remote_SHA=commit,at=time.time())
        write(ROOT/'PUBLICATION.json',publication)
        write(ROOT/'ADMISSION_RECOVERY_COMPLETE.json',dict(status='COMPLETE',attempt=str(attempt),at=time.time(),commit=commit))
        write(ROOT/'COMPLETE.json',dict(status='COMPLETE',models=list(requests),all_owned_workers_exited=True,at=time.time()))
        write(ROOT.parent/'COMPLETE.json',dict(status='COMPLETE',publication=publication,at=time.time()))
        write(attempt/'COMPLETE.json',dict(status='COMPLETE',publication=publication,at=time.time()))
    except BaseException:
        write(attempt/'FAILURE.json',dict(error=traceback.format_exc(),at=time.time(),automatic_retry=False));raise


if __name__=='__main__':main()
