#!/usr/bin/env python3
"""One read-only sample, existing allowed SSH aliases, two concurrent hosts."""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROBE = r'''
import csv,io,json,os,socket,subprocess,time
from pathlib import Path
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=4)
 if p.returncode: raise RuntimeError('read_command_failed:'+args[0])
 return p.stdout
def small(path):
 try: return Path(path).read_text()[:4096].strip()
 except OSError: return None
def classify(pid):
 try:
  p=Path('/proc')/str(pid)
  argv=(p/'cmdline').read_bytes()[:8192].split(b'\0')
  args=[x.decode(errors='replace') for x in argv if x]
  same_ns=os.readlink(p/'ns/pid')==os.readlink('/proc/self/ns/pid')
  same_uid=p.stat().st_uid==os.getuid()
  pressure=[x for x in args if 'pressure' in Path(x).name.lower()]
  approved=any(x.startswith('/tmp/zt-idle-pressure-v1/') and x.endswith('.py') for x in pressure)
  if approved and same_ns and same_uid: kind='CONFIRMED_OWNED_PRESSURE'
  elif pressure: kind='PRESSURE_CANDIDATE_REQUIRES_IDENTITY_CHECK'
  else: kind='REAL_OR_UNKNOWN_PROTECTED'
  return {'pid':pid,'kind':kind,'same_pid_namespace':same_ns,'same_uid':same_uid,
          'pressure_script_basename':Path(pressure[0]).name if pressure else None}
 except OSError: return {'pid':pid,'kind':'UNRESOLVED_PROTECTED'}
r={'hostname':socket.gethostname(),'observed_unix':time.time(),'cpu_affinity_count':len(os.sched_getaffinity(0)),
   'loadavg':small('/proc/loadavg'),'cpu_max':small('/sys/fs/cgroup/cpu.max'),
   'cpu_cfs_quota_us':small('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
   'cpu_cfs_period_us':small('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
   'memory_max':small('/sys/fs/cgroup/memory.max'),'memory_current':small('/sys/fs/cgroup/memory.current')}
mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','MemTotal')}
r.update(mem)
quotas=[]
cg={}
for line in Path('/proc/self/cgroup').read_text().splitlines():
 _,controllers,rel=line.split(':',2)
 for controller in controllers.split(','):cg[controller]=rel
quota_checks=[]
for mount in ('cpu,cpuacct','cpu'):
 base=Path('/sys/fs/cgroup')/mount
 current=base/cg.get('cpu','').lstrip('/')
 if not current.exists():current=base
 for _ in range(16):
  q=small(current/'cpu.cfs_quota_us');p=small(current/'cpu.cfs_period_us')
  if q and p:
   quota_checks.append({'quota_us':q,'period_us':p})
   if int(q)>0:quotas.append(int(q)/int(p))
  if current==base or base not in current.parents:break
  current=current.parent
r['cpu_ancestor_quota_checks']=quota_checks
mbase=Path('/sys/fs/cgroup/memory');mc=mbase/cg.get('memory','').lstrip('/')
if not mc.exists():mc=mbase
r['memory_limit_in_bytes']=small(mc/'memory.limit_in_bytes');r['memory_usage_in_bytes']=small(mc/'memory.usage_in_bytes')
if r['cpu_max']:
 q,p=r['cpu_max'].split()
 if q!='max': quotas.append(int(q)/int(p))
if r['cpu_cfs_quota_us'] and int(r['cpu_cfs_quota_us'])>0:
 quotas.append(int(r['cpu_cfs_quota_us'])/int(r['cpu_cfs_period_us']))
r['effective_cpu_quota']=min([r['cpu_affinity_count']]+quotas)
r['cpu_quota_verified']=bool(quota_checks or r['cpu_max'])
d=os.statvfs('/mnt/project');r['shared_disk_available_bytes']=d.f_bavail*d.f_frsize;r['shared_disk_total_bytes']=d.f_blocks*d.f_frsize
gpu=run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
apps=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
by={}
for row in csv.reader(io.StringIO(apps)):
 if len(row)<3: continue
 uuid,pid,used=[x.strip() for x in row]
 try: rec=classify(int(pid))
 except ValueError: rec={'kind':'UNRESOLVED_PROTECTED'}
 rec['used_gpu_memory_mib']=used;by.setdefault(uuid,[]).append(rec)
r['gpus']=[]
for row in csv.reader(io.StringIO(gpu)):
 if len(row)!=6: continue
 idx,uuid,name,total,used,util=[x.strip() for x in row];procs=by.get(uuid,[])
 state='EMPTY_SNAPSHOT' if not procs and int(used)<=128 else ('OWNED_PRESSURE_ONLY' if procs and all(p['kind']=='CONFIRMED_OWNED_PRESSURE' for p in procs) else 'OCCUPIED_PROTECTED_OR_UNRESOLVED')
 if r['hostname']=='training-vla-zt-worker-0' and int(idx) in (0,7): state='LEAD_RESERVED_PROTECTED'
 r['gpus'].append({'index':int(idx),'uuid':uuid,'model':name,'memory_total_mib':int(total),'memory_used_mib':int(used),'utilization_percent':util,'state':state,'processes':procs})
print(json.dumps(r))
'''

def main():
    root=Path(__file__).parent
    policy_path=Path('/mnt/project/server_dispatch_policy.json')
    policy_bytes=policy_path.read_bytes();policy=json.loads(policy_bytes)
    auth_path=Path('/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json')
    auth=json.loads(auth_path.read_text())
    hosts=list(policy['allowed_hosts'])
    expected={'training-vla-zt','training-rl-zt2','training-vlawm-tsinghua-zt',*[f'training-vlawm-tsinghua-zt{x}' for x in range(2,7)]}
    assert set(hosts)==expected and set(hosts)<=set(auth['allowed_hosts'])
    started=time.monotonic()
    def one(host):
        cmd=[sys.executable,'-B','-c',PROBE] if host=='training-vla-zt' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ConnectionAttempts=1',host,'python3','-B','-']
        try:
            p=subprocess.run(cmd,input=None if host=='training-vla-zt' else PROBE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
            if p.returncode: return {'requested_host':host,'status':'ENTRY_FAILED','exit_code':p.returncode,'error_class':'SSH_OR_REMOTE_READ_FAILURE','stderr_tail':p.stderr[-500:]}
            r=json.loads(p.stdout);r.update(requested_host=host,status='OBSERVED');return r
        except subprocess.TimeoutExpired: return {'requested_host':host,'status':'ENTRY_FAILED','error_class':'HOST_TIMEOUT_15_SECONDS'}
        except Exception as e: return {'requested_host':host,'status':'ENTRY_FAILED','error_class':type(e).__name__}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(one,hosts))
    report={'schema':'consequence_readonly_capacity_snapshot_v1','observed_unix':time.time(),'elapsed_seconds':time.monotonic()-started,
            'policy_sha256':hashlib.sha256(policy_bytes).hexdigest(),'authorization_path':str(auth_path),
            'maximum_parallel_hosts':2,'per_host_timeout_seconds':15,'dispatch_performed':False,'processes_signalled':0,
            'raw_command_lines_recorded':False,'hosts':rows}
    (root/'CAPACITY_SNAPSHOT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'elapsed_seconds':report['elapsed_seconds'],'hosts':[{'host':x['requested_host'],'status':x['status'],'gpu_states':[g['state'] for g in x.get('gpus',[])],'cpu_quota':x.get('effective_cpu_quota'),'mem_available_gib':round(x.get('MemAvailable',0)/(1<<30),1)} for x in rows]}))

if __name__=='__main__':main()
