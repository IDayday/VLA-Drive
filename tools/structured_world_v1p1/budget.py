"""V1.1 locked step reservations and live GPU wall time; stopped jobs stay charged."""
import fcntl,json,time,os
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def locked(path):
    p=Path(path)
    with open(str(p)+'.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(p.read_text());yield d
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)


def totals(d):
    runs=d['runs'];now=time.time()
    # Initial ledger includes P1 charge as a run. Each active process accrues wall time.
    hours=sum(r.get('gpu_hours',0)+(now-r['heartbeat'])/3600 if r.get('status')=='running' else r.get('gpu_hours',0) for r in runs)
    steps=sum(r.get('reserved_steps',r.get('optimizer_steps',0)) if r.get('status')=='running' else r.get('optimizer_steps',0) for r in runs)
    return hours,steps


def start(path,run_id,steps,metadata,resume=False):
    with locked(path) as d:
        found=next((r for r in d['runs'] if r['id']==run_id),None)
        if found and not resume:raise ValueError('Existing run; explicit resume required')
        if found and found['status']=='running':raise ValueError('Cannot resume a live/unreconciled run')
        h,s=totals(d)
        old=found.get('optimizer_steps',0) if found else 0
        if h>=d['gpu_hour_cap'] or s+steps-old>d['step_cap']:raise ValueError('Campaign cap exceeded')
        if found is None:
            found={'id':run_id,'optimizer_steps':0,'gpu_hours':0};d['runs'].append(found)
        found.update(status='running',reserved_steps=steps,metadata=metadata,heartbeat=time.time(),pid=os.getpid())


def record(path,run_id,step,status='running'):
    with locked(path) as d:
        r=next(r for r in d['runs'] if r['id']==run_id);now=time.time()
        if step<r['optimizer_steps'] or step>r['reserved_steps']:raise ValueError('Invalid update accounting')
        r['gpu_hours']+=(now-r['heartbeat'])/3600;r.update(heartbeat=now,optimizer_steps=step,status=status)
        h,s=totals(d);d['gpu_hours']=h;d['optimizer_steps']=sum(x.get('optimizer_steps',0) for x in d['runs'])
        return h<d['gpu_hour_cap'] and d['optimizer_steps']<d['step_cap']
