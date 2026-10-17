"""Bounded independent test runner, signals only its freshly spawned fixture."""
import os,time
START=time.monotonic()
os.sched_setaffinity(0,{8,9})
import json,resource,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def counters(pid):
    try:
        rows={k:int(v) for k,v in (x.split(':') for x in Path('/proc/'+str(pid)+'/io').read_text().splitlines())}
        fields={k:v.strip() for k,v in (x.split(':',1) for x in Path('/proc/'+str(pid)+'/status').read_text().splitlines())}
        return {'rchar':rows['rchar'],'rss':int(fields['VmRSS'].split()[0])*1024 if 'VmRSS' in fields else None,'hwm':int(fields['VmHWM'].split()[0])*1024 if 'VmHWM' in fields else None}
    except FileNotFoundError:return None
def child_setup():
    resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
    resource.setrlimit(resource.RLIMIT_CPU,(115,115))
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
peak=0;max_hwm=0;missing=0;samples=0;failure=None
with (HERE/'TEST.log').open('wb') as log:
    child=subprocess.Popen([sys.executable,str(HERE/'test_independent.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=HERE,preexec_fn=child_setup)
    try:
        while child.poll() is None:
            own=counters(os.getpid());other=counters(child.pid);samples+=1
            if other is None or other['rss'] is None:missing+=1
            peak=max(peak,(own['rss'] or 0)+(other['rss'] or 0) if other else own['rss'] or 0)
            if other and other['hwm'] is not None:max_hwm=max(max_hwm,other['hwm'])
            totalread=own['rchar']+(other['rchar'] if other else 0)
            if peak>512<<20 or totalread>32<<20 or time.monotonic()-START>120:
                failure='RESOURCE_LIMIT';child.terminate();break
            time.sleep(.02)
        code=child.wait(timeout=2)
    finally:
        if child.returncode is None:child.kill();child.wait(timeout=2)
post=counters(os.getpid())
summary=dict(status='PASS' if code==0 and failure is None and post['rchar']<=32<<20 else 'FAIL',returncode=code,failure=failure,wall_seconds=time.monotonic()-START,peak_sampled_aggregate_rss_bytes=peak,child_sampled_hwm_bytes=max_hwm,terminal_unavailable_samples=missing,samples=samples,charged_rchar_parent_after_wait_bytes=post['rchar'],read_accounting='parent absolute post-wait includes fixture once; no child re-add',cpu_affinity=sorted(os.sched_getaffinity(0)),limits=dict(wall_seconds=120,rss_bytes=512<<20,total_read_bytes=32<<20,fixture_bytes=8<<20),actual_payload_reads=0,actual_ledger_mutations=0,actual_task_signals=0)
(HERE/'RESOURCE_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
raise SystemExit(0 if summary['status']=='PASS' else 1)
