"""One explicit review of expired G/H evaluation attempts; preserve scene caches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BASE = Path('/mnt/project/qwen-world-safety-exploration-20261031')
OUT = Path('/mnt/project/qwen-world-il-research-20261030/maintenance/evaluation_resource_recovery_20261102')


def write(path, value):
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def capture(owner):
    p=Path('/proc')/str(owner['pid']);st=(p/'stat').read_text().split()
    if int(st[21])!=owner['start_ticks'] or st[2]in('Z','X'):raise ValueError('CPU identity changed')
    args=[x for x in(p/'cmdline').read_bytes().decode().split(chr(0))if x]
    if not any(x.startswith('tools.qwen_world_safety_exploration.')for x in args):raise ValueError('Foreign CPU owner')
    return dict(pid=owner['pid'],start_ticks=owner['start_ticks'],args=args,cwd=str((p/'cwd').resolve()))


def stop_cpu(record):
    fd=os.pidfd_open(record['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)


def lock(path):
    f=path.open('a');end=time.monotonic()+70
    while True:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
        except BlockingIOError:
            if time.monotonic()>end:raise RuntimeError('Owner did not release lock')
            time.sleep(.5)


def start(record, log):
    f=log.open('ab',buffering=0)
    p=subprocess.Popen(record['args'],cwd=record['cwd'],env=dict(os.environ,PYTHONPATH=record['cwd'],CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(pid=p.pid,start_ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]),source=record['cwd'])


def main():
    OUT.mkdir(exist_ok=True)
    for study,native_name,evaluation_name,helper_name in [
        ('revival_training_20261101','REVIVAL_NATIVE','RESEARCH_EVALUATION','STANDARD_LAUNCH.json'),
        ('improvement_training_20261101','IMPROVEMENT_NATIVE','IMPROVEMENT_EVAL','HELPER_LAUNCH.json')]:
        root=BASE/study;out=OUT/study;out.mkdir(exist_ok=True)
        if(out/'COMPLETE.json').exists():raise FileExistsError('Already reviewed; do not duplicate owners')
        nd=root/'owners'/native_name;ed=root/'owners'/evaluation_name
        if(ed/'STOP_OWNER').exists():raise ValueError('A different owner adoption is already active')
        parent=capture(json.loads((nd/'OWNER.json').read_text()));evaluation=capture(json.loads((ed/'OWNER.json').read_text()))
        helper=json.loads((ed/helper_name).read_text())
        if helper['pid']!=evaluation['pid']:raise ValueError('Helper receipt does not bind evaluator')
        write(out/'OWNERS_BEFORE.json',dict(parent=parent,evaluation=evaluation,helper=helper))
        stop_cpu(parent);native_lock=lock(nd/'owner.lock')
        write(ed/'STOP_OWNER',dict(reason='Explicit expired evaluation attempt review; keep GPU children',at=time.time()))
        evaluation_lock=lock(ed/'owner.lock');state=json.loads((ed/'STATE.json').read_text());write(out/'STATE_BEFORE.json',state)
        reviewed=[]
        for label,entry in state['evaluations'].items():
            for rank,node in entry.get('ranks',{}).items():
                status=node.get('status')
                if status not in('FAILED_REVIEW_REQUIRED','DISPATCH_UNCERTAIN_REVIEW_REQUIRED'):continue
                job=Path(node['job']);reason=None
                if study.startswith('revival') and status=='FAILED_REVIEW_REQUIRED':
                    receipt=json.loads((job/'RESULT.json').read_text())
                    if node['host']!='training-vlawm-zt4' or receipt.get('returncode')!=1 or not 1793571300<=receipt['end_unix']<1793571360:raise ValueError('Not reviewed expiry failure')
                    with(job/'stdout.log').open('rb')as f:f.seek(max(0,f.seek(0,2)-32768));error=f.read().decode()
                    if 'PermissionError: Host not authorized: training-vlawm-zt4'not in error:raise ValueError('Different evaluator failure')
                    reason='EXPIRED_HOST_PERMISSION_BEFORE_INFERENCE'
                elif study.startswith('improvement') and status=='DISPATCH_UNCERTAIN_REVIEW_REQUIRED':
                    if ('PermissionError: Task host lease does not cover remote invocation: training-vlawm-tsinghua-zt2'not in node.get('dispatch_error','')or node['host']!='training-vlawm-tsinghua-zt2'):raise ValueError('Different ambiguous dispatch')
                    if any((job/name).exists()for name in ['RESULT.json','RUNNING.json','worker.log']):raise ValueError('Ambiguous attempt may have started')
                    reason='LOCAL_PERMISSION_REJECTED_BEFORE_SSH_AND_POPEN'
                if reason is None:raise ValueError('Unreviewed failure; no automatic retry')
                record=dict(label=label,rank=rank,job=str(job),host=node['host'],reason=reason,
                            same_world_seed_command=True,scene_cache_preserved=True)
                node.setdefault('reviewed_expiry_attempts',[]).append(record)
                for key in ['job','result','dispatch_error','error']:node.pop(key,None)
                node['status']='RESOURCE_REQUEUE';reviewed.append(record)
            if entry.get('status')=='FAILED_REVIEW_REQUIRED' and not any(n.get('status')in('FAILED_REVIEW_REQUIRED','DISPATCH_UNCERTAIN_REVIEW_REQUIRED')for n in entry.get('ranks',{}).values()):
                entry['status']='RUNNING';entry.pop('failure_rank',None);entry.pop('error',None)
        if len(reviewed)!=(3 if study.startswith('revival')else 2):raise ValueError('Expected exact known expired shards')
        write(ed/'STATE.json',state);write(out/'REVIEW.json',reviewed)
        (ed/'STOP_OWNER').rename(out/'STOP_OWNER_APPLIED')
        evaluation_lock.close();new_eval=start(evaluation,out/'evaluation_cpu.log')
        write(out/'HELPER_BEFORE.json',helper);helper.update(new_eval,at_unix=time.time());write(ed/helper_name,helper)
        native_lock.close();new_parent=start(parent,out/'parent_cpu.log')
        write(out/'COMPLETE.json',dict(reviewed=reviewed,new_evaluator=new_eval,new_parent=new_parent,
                                      GPU_signals=0,scientific_source_changes=0))
        print(study,'reviewed',len(reviewed),'CPU',new_parent['pid'],'Eval',new_eval['pid'],flush=True)


if __name__=='__main__':main()
