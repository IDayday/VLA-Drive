"""Resume D34 ranks30/31 after verified text-I/O failures; adopt one cadence."""
import hashlib
import json
import os
from pathlib import Path
import time
from .adopt_periodic_queues_20261102 import capture, stop, lock, launch

SOURCE=Path(__file__).resolve().parents[2]
CADENCE=Path('/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031')
OUT=CADENCE.parent/'maintenance/completion_event_and_D34_20261102'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def write(p,x):
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)


def main():
    OUT.mkdir(exist_ok=True)
    if(OUT/'RECOVERY_LAUNCH.json').exists():raise FileExistsError('Already resumed; inspect existing owner')
    state=read(CADENCE/'STATE.json')
    jobs=[j for j in state['jobs']if j['label']=='D_IL_034000']
    if len(jobs)!=1 or jobs[0]['status']!='FAILED_REVIEW_REQUIRED':raise ValueError('Not exact failed D34 point')
    job=jobs[0];root=Path(job['root']);directory=root/job['label'];bad=directory/'jobs/rank31_attempt001'
    failed=[]
    for index in (30,31):
        attempt=directory/f'jobs/rank{index}_attempt001';result=read(attempt/'RESULT.json');rank=read(directory/f'evaluation/rank{index}.json')
        if(result.get('returncode')!=1 or rank.get('status')!='FAILED' or rank['rank']!=index or rank['world']!=32
                or 'UnicodeDecodeError'not in rank.get('error','') or "encodings/ascii.py"not in rank['error']
                or 'qwen_world_compare/contracts.py'not in rank['error']):raise ValueError('Different inference/scorer failure; no retry')
        write(OUT/f'FAILED_RANK{index}_BEFORE.json',rank);write(OUT/f'FAILED_RECEIPT{index}_BEFORE.json',result);failed.append(attempt)
    if Path('/proc',str(read(root/'owner.json')['pid'])).exists():raise ValueError('Old failed owner is unexpectedly alive')
    remaining=[]
    for i in range(30):
        p=directory/'evaluation'/f'rank{i}.json';r=read(p)
        if r.get('status')!='COMPLETE'or r['world']!=32:remaining.append(i)
    if remaining:raise ValueError('Other uncompleted shards need explicit review: '+str(remaining))
    cadence=capture(read(CADENCE/'OWNER.json'))
    if 'tools.il_report.navtest_cadence'not in cadence['args']:raise ValueError('Foreign cadence owner')
    write(OUT/'CADENCE_BEFORE.json',cadence);stop(cadence);cadence_lock=lock(CADENCE/'owner.lock');burst_lock=lock(root/'owner.lock')
    state=read(CADENCE/'STATE.json');write(OUT/'STATE_BEFORE.json',state)
    job=next(j for j in state['jobs']if j['label']=='D_IL_034000')
    for attempt in failed:
        spec=attempt/'spec.json';write(attempt/'OPERATOR_REQUEUE.json',dict(status='REVIEWED_TERMINAL_REQUEUE',
            spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),reason='Exact policy text decoding failure; unchanged model/scorer/rank/world/seed; preserve valid cached scene',at=time.time()))
    script=SOURCE/'tools/il_report/navtest_burst.py';digest=hashlib.sha256(script.read_bytes()).hexdigest();operator=root/('BURST_OPERATOR_ops_'+digest[:12]+'.py')
    if not operator.exists():operator.write_bytes(script.read_bytes())
    args=[cadence['args'][0],'-u',str(operator),'--root',str(root),'--source',job['source'],'--family','four','--workers','32','--resume',
          '--worker-source','/mnt/project/VLA-Drive-abcd-il-c-recovery-20261101']
    burst_lock.close();process=launch(args,root/'owner.log')
    job.update(status='RUNNING',process=process,command=args,operator_sha256=digest,dispatched_unix=time.time(),
               recovery_review=str(OUT/'FAILED_RECEIPT31_BEFORE.json'))
    for key in ['failure','error']:job.pop(key,None)
    write(root/'LAUNCH.json',job);write(CADENCE/'STATE.json',state)
    cadence_lock.close();new_cadence=launch(cadence['args'],OUT/'cadence_cpu.log')
    write(OUT/'RECOVERY_LAUNCH.json',dict(burst=process,cadence=new_cadence,execution_source=str(SOURCE),
        original_scientific_source=job['source'],retained_completed_shards=30,retry_ranks=[30,31],world=32,seed=2101,
        valid_scene_cache_preserved=True,UTF8_IO_only=True,GPU_signals=0))
    print('D34 CPU',process['pid'],'cadence',new_cadence['pid'],'only ranks30/31 requeued',flush=True)


if __name__=='__main__':main()
