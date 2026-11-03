#!/usr/bin/env python3
"""One-minute read-only NVML/known-rank CPU samples. No profiling or CUDA context."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os,pathlib,subprocess,time
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103');OUT=R/'analysis/THROUGHPUT_RESOURCE_SAMPLING_20261103.json'
JOBS=[]
for arm,stage in [('D0','D0_WORLD16_PRIORITY_v2'),('D3_FPO','FPO_WORLD16_PRIORITY_v1')]:
 s=json.loads((R/'operations/WORLD16_STAGES'/stage/'STATE.json').read_text())
 for j in s['active_jobs']:JOBS.append(dict(arm=arm,host=j['host'],job=j['job']))
CODE=r'''
import json,os,sys,time,subprocess,hashlib,importlib.util
from pathlib import Path
JOB=Path(__JOB__);ARM=__ARM__;state=json.loads((JOB/'RUNNING.json').read_text())
worker=Path(state['execution_worker']);sys.path.insert(0,str(worker.parents[2]));os.environ['WORLD_COMPARE_TASK']='d46_priority_rl_20261103'
spec=importlib.util.spec_from_file_location('audited_readonly_worker',worker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ranks=module.task_ranks(state);HZ=os.sysconf('SC_CLK_TCK')
def proc(row):
 try:
  v=(Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  if int(v[19])!=row['start_ticks']:return dict(pid=row['pid'],identity_changed=True)
  return dict(pid=row['pid'],start_ticks=int(v[19]),utime_ticks=int(v[11]),stime_ticks=int(v[12]),threads=int(v[17]),rss_pages=int(v[21]),state=v[0])
 except OSError:return dict(pid=row['pid'],exited=True)
def number(v):
 try:return float(v.strip())
 except ValueError:return None
samples=[];start=time.monotonic()
for index in range(13):
 wait=start+index*5-time.monotonic()
 if wait>0:time.sleep(wait)
 at=time.time();s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
 g=[]
 for line in s.splitlines():
  i,u,util,mu,total,used,free,power,limit=line.split(',');g.append(dict(index=int(i),uuid=u.strip(),utilization_pct=number(util),memory_utilization_pct=number(mu),total_MiB=number(total),used_MiB=number(used),free_MiB=number(free),power_W=number(power),power_limit_W=number(limit)))
 samples.append(dict(at_unix=at,elapsed_seconds=time.monotonic()-start,gpus=g,known_rank_processes=[proc(r)for r in ranks],CPU_load=os.getloadavg()[0]))
print(json.dumps(dict(status='READ_COMPLETE',arm=ARM,host=state['host'],job=str(JOB),worker_source=str(worker),worker_source_file_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
 RUNNING_file_sha256=hashlib.sha256((JOB/'RUNNING.json').read_bytes()).hexdigest(),exact_known_ranks=ranks,clock_ticks_per_second=HZ,
 samples=samples,elapsed_seconds=time.monotonic()-start,unknown_process_command_or_environment_emitted=False,no_signal_or_lock_operation=True)))
'''
PROBE=r'''
import subprocess,json,os,time
from pathlib import Path
SELECTED=__INDICES__
s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
g=[]
for line in s.splitlines():
 i,u,t,f,v=line.split(',')
 if int(i)in SELECTED:g.append(dict(index=int(i),uuid=u.strip(),total_MiB=int(t),free_MiB=int(f),utilization_pct=int(v)))
o=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
for x in g:
 owners=[]
 for line in o.splitlines():
  pid,u,m=line.split(',')
  if u.strip()!=x['uuid']:continue
  p=Path('/proc')/pid.strip();owners.append(dict(pid=int(pid),used_MiB=int(m)if m.strip().isdigit()else None,visible=(p/'cmdline').is_file()))
 x['owners']=owners;x['unexplained_MiB']=x['total_MiB']-x['free_MiB']-sum(y['used_MiB']or 0for y in owners)
 x['idle_candidate']=not owners and x['utilization_pct']<=10 and x['unexplained_MiB']<=2048 and x['free_MiB']>=65000
print(json.dumps(dict(status='READ_COMPLETE',at_unix=time.time(),hostname=os.uname().nodename,gpus=g,unknown_cmdline_or_environ_emitted=False)))
'''
def remote(host,code,timeout):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=7',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=timeout)
 if p.returncode:return dict(status='READ_FAILED',host=host,error=p.stderr[-2500:],returncode=p.returncode)
 return json.loads(p.stdout)
def sample(j):return j['arm']+'@'+j['host'],remote(j['host'],CODE.replace('__JOB__',repr(j['job'])).replace('__ARM__',repr(j['arm'])),80)
def candidate(h,indices):return h,remote(h,PROBE.replace('__INDICES__',repr(indices)),20)
results={};probes={};started=time.time()
with ThreadPoolExecutor(max_workers=6)as pool:
 future_types={pool.submit(sample,j):'sample'for j in JOBS}
 future_types.update({pool.submit(candidate,h,ix):'probe'for h,ix in [('training-vlawm-zt2',[0,1]),('training-rl-zt2',[1,7])]})
 for f in as_completed(future_types):
  k,v=f.result();(results if future_types[f]=='sample'else probes)[k]=v;print(k,v['status'],v.get('elapsed_seconds'),len(v.get('exact_known_ranks',[])),flush=True)
value=dict(schema='d46_one_minute_actual_resource_readonly_samples_v1',started_unix=started,finished_unix=time.time(),
 generated_utc=datetime.now(timezone.utc).isoformat(),sample_interval_seconds=5,requested_duration_seconds=60,native_node_samples=results,qualification_candidate_GPU_probes=probes,
 source_script_file_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),no_live_profiler_or_model_forward=True)
b=json.dumps(value,sort_keys=True,separators=(',',':')).encode();value['sha256']=hashlib.sha256(b).hexdigest();tmp=OUT.with_suffix('.writing');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,OUT)
print(json.dumps(dict(output=str(OUT),file_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())),flush=True)
