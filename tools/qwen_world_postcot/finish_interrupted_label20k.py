"""Resume only two scene-boundary-stopped shards; retain11593 valid scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest
from tools.qwen_world_postcot.recover_admission_only import alive,admission_refusal
from tools.qwen_world_postcot.user_interim_owner import read,write,sha,validate_complete_summary

ROOT=Path('/mnt/project/qwen-world-postcot-20261025/original_vs_optimized20k_NAVTEST_20261029T0740Z')
LABEL='ORIGINAL_GT_IL_020000'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    owner=read(ROOT/'owner.json');source=Path(owner['source']);directory=ROOT/LABEL
    if alive(owner)or(ROOT/'COMPLETE.json').exists():raise ValueError('Existing live/completed owner')
    def git(*argv):return subprocess.check_output(['git',*argv],cwd=source,text=True)
    if git('rev-parse','HEAD').strip()!=owner['source_sha']or git('status','--porcelain'):
        raise ValueError('Original source must remain clean and unchanged')
    identity=read(directory/'evaluation/identity.json');tokens=identity['tokens']
    for relative,digest in identity['source'].items():
        if sha(source/relative)!=digest:raise ValueError('Original evaluator dependency changed: '+relative)
    for rank in range(8):
        result=read(directory/f'jobs/rank{rank}/result.json');progress=read(directory/f'evaluation/rank{rank}.json')
        if result['status']!='PROCESS_COMPLETE'or result['exit_code']!=0:
            raise ValueError('Old workers must have drained successfully')
        expected='STOPPED_AT_SCENE_BOUNDARY'if rank in(5,6)else'COMPLETE'
        if progress['status']!=expected:raise ValueError('Unexpected old rank termination')
    got={p.stem for p in(directory/'evaluation/scenes').glob('*.json')};missing=set(tokens)-got
    if got-set(tokens)or not missing or not missing.issubset(set(tokens[5::8])|set(tokens[6::8])):
        raise ValueError('Missing scenes do not match the reviewed two stopped shards')
    if args.check_only:
        print(json.dumps(dict(status='PASS',missing_scenes=len(missing),reused_scenes=len(got),ranks=[5,6],runtime=owner['source_sha'])));return
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=ROOT/'scene_boundary_recovery001';attempt.mkdir(exist_ok=False)
    write(attempt/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        at=time.time(),ranks=[5,6],source=str(source),source_sha=owner['source_sha'],missing_scenes=len(missing),
        old_failure_preserved=True,controller_source=str(Path(__file__).resolve().parents[2])))
    for rank in(5,6):write(attempt/f'previous_rank{rank}.json',read(directory/f'evaluation/rank{rank}.json'))
    pin=dict(read(ROOT/'PINNED.json')['models'][LABEL],evaluation_hosts=['training-vla-zt3'])
    def request():
        r=SnapshotRequest(directory,pin);r.source=source;r.source_sha=owner['source_sha'];return r
    def run_rank(rank):
        r=request();spec=read(directory/f'jobs/rank{rank}/spec.json');cmd=spec['command'];module=spec['module'];argv=cmd[cmd.index(module)+1:]
        for number in range(1,4):
            route=None
            while route is None:
                routes,inventory=r.resources.available(min_free_MiB=32768);route=r.route(routes,inventory)
                if route is None:time.sleep(15)
            name=f'rank{rank}_scene_resume{number:03d}';r.submit(name,module,argv,route)
            while r.done(name)is None:time.sleep(15)
            if r.done(name)is True:return
            if not admission_refusal(directory/'jobs'/name):raise RuntimeError('Executed worker failure; no automatic scientific retry')
        raise RuntimeError('Bounded admission retries exhausted')
    try:
        with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(run_rank,(5,6)))
        with(attempt/'summary.log').open('x')as log:
            subprocess.run([request().resources.MODEL_PYTHON,'-m','tools.qwen_world_postcot.summarize',
                '--evaluation',str(directory/'evaluation'),'--output',str(directory/'SUMMARY.json')],
                cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=read(directory/'SUMMARY.json');counts=validate_complete_summary(row,identity,
            read('/mnt/project/flow-safe-improvement-20261013/population/manifest.json'))
        write(directory/'COMPLETE.json',dict(status='COMPLETE',PDMS=row['PDMS'],scenes=12146,
            logs=counts['manifest_logs'],physical_logs=counts['physical_logs'],all_owned_workers_exited=True,
            registration_sha256=read(directory/'registration.json')['sha256'],at=time.time(),
            recovery='Only missing scenes from ranks5/6 stopped on exact Tsinghua-zt4 withdrawal; old costs and failure retained.'))
        request().publish(row)
        # Use the original generalized20k comparison helper and its audit.
        code="""from pathlib import Path
from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest,finalize
from tools.qwen_world_postcot.user_interim_owner import read
root=Path(__import__('sys').argv[1]);source=Path.cwd();requests={}
for label,pin in read(root/'PINNED.json')['models'].items():
 r=SnapshotRequest(root/label,pin);r.source=source;requests[label]=r
finalize(root,requests,source/'reports/qwen_world_label_source20k_20261029')
"""
        subprocess.run([request().resources.MODEL_PYTHON,'-c',code,str(ROOT)],cwd=source,check=True)
        if git('rev-parse','HEAD').strip()!=owner['source_sha']:raise ValueError('Changed source before publication')
        prefix='reports/qwen_world_label_source20k_20261029'
        for entry in git('status','--porcelain=v1','-z','--untracked-files=all').split('\0'):
            if entry and not entry[3:].startswith(prefix+'/'):raise ValueError('Unrelated Git changes')
        subprocess.run(['git','add','--',prefix],cwd=source,check=True)
        subprocess.run(['git','commit','-m','Finish matched20k Test after checkpoint-safe host withdrawal'],cwd=source,check=True)
        branch=git('branch','--show-current').strip();commit=git('rev-parse','HEAD').strip()
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
        if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=commit:raise ValueError('Remote SHA mismatch')
        write(ROOT/'PUBLICATION.json',dict(status='PUBLISHED',commit=commit,remote_SHA=commit,at=time.time()))
        write(ROOT/'COMPLETE.json',dict(status='COMPLETE',models=[LABEL],all_owned_workers_exited=True,at=time.time(),recovery=str(attempt)))
        write(attempt/'COMPLETE.json',dict(status='COMPLETE',commit=commit,reused_scenes=len(got),new_scenes=len(missing),at=time.time()))
    except BaseException:
        write(attempt/'FAILURE.json',dict(error=traceback.format_exc(),at=time.time(),automatic_retry=False));raise


if __name__=='__main__':main()
