"""Explicit operator review of only the four user-withdrawn host shards."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OPS=Path('/mnt/project/VLA-Drive-flow-eval-resource-pauses-20261101')
WITHDRAWN={'training-vla-zt3','training-vlawm-zt2','training-vlawm-zt3','training-vlawm-zt5'}


def acquire(path):
    f=path.open('a');end=time.monotonic()+70
    while True:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
        except BlockingIOError:
            if time.monotonic()>end:raise RuntimeError('Existing owner lock not released: '+str(path))
            time.sleep(.5)


def main():
    sys.path.insert(0,str(OPS))
    from tools.qwen_world_safety_exploration.resource_migration import requeue_clean_scene_pause
    from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts
    for study,campaign,native in [('standard_training_20261101','STANDARD_CAMPAIGN','STANDARD_NATIVE'),('revival_training_20261101','REVIVAL_CAMPAIGN','REVIVAL_NATIVE')]:
        root=Path('/mnt/project/qwen-world-safety-exploration-20261031')/study;out=root/'operations/WITHDRAWN_EVAL_RECOVERY_20261101';out.mkdir(parents=True,exist_ok=True)
        if(out/'LAUNCH.json').exists():raise FileExistsError('Reviewed adoption already completed')
        parent=json.loads((root/'owners'/campaign/'OWNER.json').read_text());p=Path('/proc')/str(parent['pid']);stat=(p/'stat').read_text().split()
        if int(stat[21])!=parent['start_ticks']:raise ValueError('CPU parent identity changed')
        argv=[x for x in(p/'cmdline').read_bytes().decode().split(chr(0))if x]
        if not any(x in argv for x in ['tools.qwen_world_safety_exploration.standard_campaign','tools.qwen_world_safety_exploration.revival_campaign']):raise ValueError('Foreign CPU parent')
        (out/'PARENT_BEFORE.json').write_text(json.dumps(parent,indent=2));fd=os.pidfd_open(parent['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
        native_lock=acquire(root/'owners'/native/'owner.lock');directory=root/'owners/RESEARCH_EVALUATION'
        stop=directory/'STOP_OWNER';stop.write_text(json.dumps(dict(reason='User resource withdrawal; owner-only metadata review; preserve GPU workers',parent=parent['pid'])))
        evaluation_lock=acquire(directory/'owner.lock');path=directory/'STATE.json';state=json.loads(path.read_text());(out/'STATE_BEFORE.json').write_text(json.dumps(state,indent=2));review=[]
        for label,entry in state['evaluations'].items():
            changed=False
            for rank,node in entry.get('ranks',{}).items():
                if node.get('host')not in WITHDRAWN or node.get('status')=='COMPLETE'or not node.get('job'):continue
                job=Path(node['job']);rp=job/'RESULT.json';receipt=json.loads(rp.read_text())if rp.exists()else None
                if receipt and requeue_clean_scene_pause(node,receipt,allowed_hosts()):
                    review.append(dict(label=label,rank=rank,mode='VERIFIED_SCENE_BOUNDARY_PAUSE'));changed=True;continue
                # The withdrawn vla-zt3 endpoint is inaccessible over SSH.
                # Its rank's actual exclusive output lock is shared-filesystem
                # evidence; no ordinary successful exit receipt is fabricated.
                if node['host']=='training-vla-zt3'and receipt is None and(job/'STOP_REQUESTED').exists():
                    output=Path(entry['evaluation']);rank_path=output/f'rank{rank}.json';rank_lock=output/f'rank{rank}.lock'
                    if not rank_path.exists()or time.time()-rank_path.stat().st_mtime<600:continue
                    with rank_lock.open('a')as lock:
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                        original=json.loads(rank_path.read_text());command=node['command']
                        if (original['rank']!=int(rank)or original['world']!=entry['world']or
                                command[command.index('--output')+1]!=str(output)):raise ValueError('Orphan shard differs from recorded population')
                        (out/(label+'_rank'+rank+'_original.json')).write_bytes(rank_path.read_bytes())
                        event=dict(job=str(job),host=node['host'],reason='Withdrawn host; stale rank metadata and positively released output lock; no fabricated exit receipt')
                        node.setdefault('operator_failures',[]).append(event)
                        for key in ['job','result','error']:node.pop(key,None)
                        node['status']='RESOURCE_REQUEUE';review.append(dict(label=label,rank=rank,mode='REVIEWED_RELEASED_ORPHAN_LOCK'));changed=True
            if changed and not any(n.get('status')=='FAILED_REVIEW_REQUIRED'for n in entry.get('ranks',{}).values()):
                entry['status']='RUNNING';entry.pop('failure_rank',None);entry.pop('error',None)
        tmp=path.with_suffix('.review.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,path)
        (out/'REVIEW.json').write_text(json.dumps(dict(rows=review,original_scene_cache_preserved=True,world_and_seed_unchanged=True),indent=2))
        stop.rename(out/'STOP_OWNER_APPLIED');evaluation_lock.close();native_lock.close()
        log=(out/'cpu_owner.log').open('ab',buffering=0);child=subprocess.Popen(argv,cwd=OPS,env=dict(os.environ,PYTHONPATH=str(OPS),CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'LAUNCH.json').write_text(json.dumps(dict(old_parent=parent,new_pid=child.pid,source=str(OPS),source_commit='c5b86f1',at_unix=time.time(),reviewed_shards=len(review),GPU_signals=0),indent=2))
        print(study,'reviewed',len(review),'new CPU',child.pid,flush=True)


if __name__=='__main__':main()
