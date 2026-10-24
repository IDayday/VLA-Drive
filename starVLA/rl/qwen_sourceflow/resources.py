"""Live native-task admission; physical UUIDs, owned receipts and no evictions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
from .contracts import require_dispatch

OPERATOR='/usr/bin/python3'
MODEL_PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def allowed_hosts():
    from .contracts import ROOT
    from .resource_authorization import authorized_hosts
    p=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    return authorized_hosts(p,ROOT)


def remote(host,args):
    if host not in allowed_hosts():raise PermissionError('Withdrawn native job host: '+host)
    if host==socket.gethostname().removesuffix('-worker-0'):
        return subprocess.check_output(args,cwd='/',text=True,timeout=60)
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(args)],text=True,timeout=60)


def probe(host):
    program="""import subprocess,json,os
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[dict(index=int(i),uuid=u.strip(),free_MiB=int(f),utilization=int(v)) for i,u,f,v in (x.split(',') for x in rows.splitlines())]
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
owners=[dict(pid=int(p),uuid=u.strip(),visible=(Path('/proc')/p.strip()/'cmdline').is_file()) for p,u in (x.split(',') for x in rows.splitlines())]
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
free=memory['MemAvailable']
limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
if limit.exists() and current.exists() and limit.read_text().strip()!='max':free=min(free,max(0,int(limit.read_text())-int(current.read_text())))
print(json.dumps(dict(gpus=gpus,owners=owners,CPU_affinity=len(os.sched_getaffinity(0)),load1=os.getloadavg()[0],available_memory_bytes=free)))"""
    return json.loads(remote(host,[OPERATOR,'-c',program]))


def available(exclude=(),min_free_MiB=32000):
    hosts=allowed_hosts();first={};second={};errors={}
    def read(host):
        try:return host,probe(host),None
        except Exception as exc:return host,None,str(exc)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        for host,value,error in pool.map(read,hosts):
            if error:errors[host]=error
            else:first[host]=value
    time.sleep(2)
    with ThreadPoolExecutor(max_workers=max(1,len(first))) as pool:
        for host,value,error in pool.map(read,first):
            if error:errors[host]=error
            else:second[host]=value
    routes=[];seen=set(exclude)
    for host,b in second.items():
        a=first[host]
        if [g['uuid'] for g in a['gpus']]!=[g['uuid'] for g in b['gpus']]:raise ValueError('Physical GPU mapping changed during admission')
        for old,new in zip(a['gpus'],b['gpus']):
            uuid=new['uuid']
            invisible=any(not owner['visible'] and owner['uuid']==uuid for sample in [a,b] for owner in sample['owners'])
            if uuid in seen or invisible or min(old['free_MiB'],new['free_MiB'])<min_free_MiB:continue
            seen.add(uuid);routes.append(dict(host=host,gpu=new['index'],uuid=uuid,free_MiB=new['free_MiB']))
    return routes,dict(first=first,second=second,errors=errors,min_free_MiB=min_free_MiB)


def launch(spec_path,source):
    spec=json.loads(Path(spec_path).read_text());require_dispatch()
    code="""import subprocess,json,sys
from pathlib import Path
command=[sys.executable,'-m','scripts.analysis.qwen_sourceflow.job_worker','--spec',sys.argv[2]]
p=subprocess.Popen(command,cwd=sys.argv[1],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'operator_pid':p.pid,'start_ticks':int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21])}))"""
    return json.loads(remote(spec['host'],[OPERATOR,'-c',code,str(source),str(spec_path)]))


def operator_alive(spec,receipt):
    code="""from pathlib import Path
import json,sys
p=Path('/proc')/sys.argv[1]
try:
 fields=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().decode().split('\\0')
 alive=int(fields[21])==int(sys.argv[2]) and fields[2]!='Z' and sys.argv[3] in cmd and 'scripts.analysis.qwen_sourceflow.job_worker' in cmd
except FileNotFoundError:alive=False
print(json.dumps(alive))"""
    return json.loads(remote(spec['host'],[OPERATOR,'-c',code,str(receipt['operator_pid']),str(receipt['start_ticks']),str(Path(spec['receipt_dir'])/'spec.json')]))
