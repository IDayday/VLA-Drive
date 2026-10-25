"""Read-only repeated GPU/owner inventory; no process evictions."""
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import socket
import subprocess
import time
from .contracts import allowed_hosts, require_dispatch

MODEL_PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
OPERATOR_PYTHON = '/usr/bin/python3'


def remote(host, args, *, timeout=45):
    if host not in allowed_hosts():
        raise PermissionError('Task host lease does not cover remote invocation: '+host)
    if host == socket.gethostname().removesuffix('-worker-0'):
        command = args
    else:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join(args)]
    return subprocess.check_output(command, text=True, timeout=timeout)


def probe(host):
    code = """import json,os,time,subprocess,hashlib
from pathlib import Path
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[dict(index=int(i),name=n.strip(),uuid=u.strip(),free_MiB=int(f),utilization=int(v)) for i,n,u,f,v in (r.split(',') for r in g.splitlines())]
o=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
owners=[]
for line in o.splitlines():
 p,u=line.split(',');pid=int(p);path=Path('/proc')/str(pid);row=dict(pid=pid,uuid=u.strip(),visible=False)
 try:
  args=[a for a in (path/'cmdline').read_bytes().decode().split('\\0') if a];row.update(visible=True,start_ticks=int((path/'stat').read_text().split()[21]),entry=args[args.index('-m')+1] if '-m' in args else Path(args[0]).name,output=args[args.index('--output')+1] if '--output' in args else None)
 except (OSError,ValueError,IndexError):pass
 owners.append(row)
grant=Path('/mnt/project/qwen-world-postcot-20261025/RESOURCE_AUTHORIZATION.json');s=os.statvfs(grant.parent)
print(json.dumps(dict(at=time.time(),gpus=gpus,owners=owners,CPU_affinity=len(os.sched_getaffinity(0)),load1=os.getloadavg()[0],MemAvailable_bytes=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')),project_free_bytes=s.f_bavail*s.f_frsize,grant_sha256=hashlib.sha256(grant.read_bytes()).hexdigest(),worker_python_exists=Path('/mnt/project/flow-safe-improvement-20261013/venv/bin/python').is_file())))
"""
    return json.loads(remote(host, [OPERATOR_PYTHON, '-c', code]))


def inventory(*, minimum_free_MiB=0):
    hosts = allowed_hosts()
    def read(host):
        try:
            return host, probe(host), None
        except Exception as error:
            return host, None, str(error)
    samples, errors = [], {}
    for iteration in range(2):
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
            rows = list(pool.map(read, hosts))
        samples.append({host: row for host, row, error in rows if error is None})
        errors.update({host: error for host, row, error in rows if error is not None})
        if iteration == 0:
            time.sleep(2)
    routes, seen = [], set()
    for host, after in samples[1].items():
        if host not in samples[0]:
            continue
        before = samples[0][host]
        if [v['uuid'] for v in before['gpus']] != [v['uuid'] for v in after['gpus']]:
            raise RuntimeError('GPU physical mapping changed during inventory: '+host)
        for a, b in zip(before['gpus'], after['gpus']):
            owners = [o for sample in [before, after] for o in sample['owners'] if o['uuid'] == b['uuid']]
            if b['uuid'] in seen or owners or min(a['free_MiB'], b['free_MiB']) < minimum_free_MiB:
                continue
            seen.add(b['uuid'])
            routes.append(dict(host=host, gpu=b['index'], uuid=b['uuid'], name=b['name'], free_MiB=min(a['free_MiB'], b['free_MiB'])))
    return routes, dict(first=samples[0], second=samples[1], errors=errors)
