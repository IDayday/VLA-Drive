"""Adopt live IL burst owners with qualified admission; no scientific worker stops."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SOURCE=Path(__file__).resolve().parents[2]
ROOT=Path('/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031')
OUT=ROOT.parent/'maintenance/evaluation_resource_recovery_20261102/IL_ADOPTION'


def write(p,x):
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)


def capture(x):
    p=Path('/proc')/str(x['pid']);s=(p/'stat').read_text().split()
    if int(s[21])!=x['start_ticks']or s[2]in('Z','X'):raise ValueError('CPU owner changed')
    return dict(x,args=[v for v in(p/'cmdline').read_bytes().decode().split(chr(0))if v],cwd=str((p/'cwd').resolve()))


def stop(x):
    fd=os.pidfd_open(x['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)


def lock(p):
    f=p.open('a');end=time.monotonic()+75
    while True:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
        except BlockingIOError:
            if time.monotonic()>end:raise RuntimeError('Old CPU owner lock still held')
            time.sleep(.5)


def launch(args,log):
    f=log.open('ab',buffering=0);p=subprocess.Popen(args,cwd=SOURCE,env=dict(os.environ,PYTHONPATH=str(SOURCE)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(pid=p.pid,start_ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]))


def main():
    OUT.mkdir(exist_ok=True)
    if(OUT/'COMPLETE.json').exists():raise FileExistsError('Already adopted; inspect owners')
    cadence=capture(json.loads((ROOT/'OWNER.json').read_text()))
    if 'tools.il_report.navtest_cadence'not in cadence['args']:raise ValueError('Foreign cadence')
    write(OUT/'CADENCE_BEFORE.json',cadence);stop(cadence);cadence_lock=lock(ROOT/'owner.lock')
    state=json.loads((ROOT/'STATE.json').read_text());write(OUT/'STATE_BEFORE.json',state)
    script=SOURCE/'tools/il_report/navtest_burst.py';digest=hashlib.sha256(script.read_bytes()).hexdigest();adopted=[]
    for job in state['jobs']:
        if job['status']!='RUNNING'or job['family']!='four':continue
        root=Path(job['root']);old=capture(json.loads((root/'owner.json').read_text()))
        if '--root'not in old['args']or old['args'][old['args'].index('--root')+1]!=str(root):raise ValueError('Foreign burst CPU')
        write(OUT/(job['label']+'_OWNER_BEFORE.json'),old);stop(old);owner_lock=lock(root/'owner.lock')
        if job['method']=='ARCH_B'and job['updates']==36000:
            directory=root/job['label'];spec=directory/'jobs/parity_attempt001/spec.json'
            result=json.loads((spec.parent/'RESULT.json').read_text());rank=json.loads((directory/'evaluation/rank0.json').read_text())
            if(result['returncode']!=0 or rank['status']!='STOPPED_AT_SCENE_BOUNDARY'or rank['new']!=0 or (directory/'native_export_parity.json').exists()):raise ValueError('B36 parity is not the verified abandoned startup')
            write(spec.parent/'OPERATOR_REQUEUE.json',dict(status='REVIEWED_TERMINAL_REQUEUE',spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),reason='Withdrawn old host exited before native parity PASS; zero scene inferences; retry same fixed protocol',at=time.time()))
        operator=root/('BURST_OPERATOR_ops_'+digest[:12]+'.py')
        if not operator.exists():operator.write_bytes(script.read_bytes())
        args=[old['args'][0],'-u',str(operator),'--root',str(root),'--source',job['source'],'--family','four','--workers','32','--resume',
              '--worker-source','/mnt/project/VLA-Drive-abcd-il-c-recovery-20261101']
        owner_lock.close();process=launch(args,root/'owner.log')
        job.update(process=process,command=args,operator_sha256=digest,dispatched_unix=time.time(),status='RUNNING')
        write(root/'LAUNCH.json',job);write(ROOT/'STATE.json',state)
        adopted.append(dict(label=job['label'],old_pid=old['pid'],process=process,scientific_source=job['source']))
        print(job['label'],'CPU',process['pid'],flush=True)
    cadence_lock.close();new_cadence=launch(cadence['args'],OUT/'cadence.log')
    write(OUT/'COMPLETE.json',dict(adopted=adopted,cadence=new_cadence,new_operator_sha256=digest,science_unchanged=True,GPU_signals=0,
                                 completed_scene_records_retained=True))
    print('cadence CPU',new_cadence['pid'],flush=True)


if __name__=='__main__':main()
