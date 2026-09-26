"""Locked optimizer-step reservations shared across processes and hosts via one ledger."""
import fcntl
import json
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(str(path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=json.loads(path.read_text()) if path.exists() else {'cap':12000,'runs':{}}
        yield value
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def reserve(path,run_id,steps,metadata):
    with locked(path) as ledger:
        if run_id in ledger['runs']:raise ValueError('Run already reserved; use explicit resume')
        used=sum(r['reserved'] if r['status']=='running' else r['consumed'] for r in ledger['runs'].values())
        if steps<0 or used+steps>ledger['cap']:raise ValueError('Optimizer budget exceeded')
        ledger['runs'][run_id]={'reserved':steps,'consumed':0,'status':'running','metadata':metadata}


def record(path,run_id,step,status='running'):
    with locked(path) as ledger:
        run=ledger['runs'][run_id]
        if step<run['consumed'] or step>run['reserved']:raise ValueError('Invalid optimizer-step accounting')
        run.update(consumed=step,status=status)
