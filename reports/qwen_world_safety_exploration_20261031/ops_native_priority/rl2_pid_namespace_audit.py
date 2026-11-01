import json,time,hashlib
from pathlib import Path
from starVLA.rl.qwen_world_safety_exploration.resources import remote,OPERATOR_PYTHON
from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts,fingerprint
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');host='training-rl-zt2'
if host not in allowed_hosts(admit_long_job=True):raise RuntimeError('Not authorized')
jobs=[E/'standard_training_20261101/jobs/SAFETY_G8_CONTROL/formal_to032000_attempt001',E/'improvement_training_20261101/jobs/IMPROVEMENT_EARLY_DECAY_G8/formal_to010000_attempt001'];receipts=[json.load(open(j/'RUNNING.json'))for j in jobs];outputs=[]
for r in receipts:
 c=r['command'];outputs.append(c[c.index('--output')+1])
code=r'''import json,os,subprocess,time
from pathlib import Path
outputs=OUTPUTS;known=KNOWN

def rtext(p):
 try:return Path(p).read_text()
 except(OSError,PermissionError)as e:return dict(error=repr(e))
def inspect(pid):
 p=Path('/proc')/str(pid);row=dict(pid=pid,exists=p.exists())
 try:
  row['argv']=[a for a in(p/'cmdline').read_bytes().decode().split('\0')if a];row['start_ticks']=int((p/'stat').read_text().split()[21]);row['status']={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in(p/'status').read_text().splitlines()if line.startswith(('Name:','State:','Pid:','PPid:','NSpid:','Uid:','NStgid:'))};row['cgroup']=(p/'cgroup').read_text();row['ns']={n:os.readlink(p/'ns'/n)for n in ['pid','pid_for_children','mnt','user']};row['cwd']=str((p/'cwd').resolve())
  env=(p/'environ').read_bytes().decode().split('\0');row['rank_environment']={v.split('=',1)[0]:v.split('=',1)[1]for v in env if v.startswith(('LOCAL_RANK=','RANK=','WORLD_SIZE=','CUDA_VISIBLE_DEVICES=','TORCHELASTIC_RUN_ID='))}
 except(OSError,ValueError,IndexError)as e:row['error']=repr(e)
 return row
nvml=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv,noheader,nounits'],text=True)
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode().split('\0')
 except(OSError,UnicodeDecodeError):continue
 if any(out in args for out in outputs):rows.append(inspect(int(p.name)))
procfields={'ssh_self':inspect(os.getpid()),'pid1':inspect(1),'known_worker_torchrun':[inspect(p)for p in known],'nvml_pids':[inspect(int(line.split(',',1)[0]))for line in nvml.splitlines()]}
extra={}
for path in ['/proc/driver/nvidia/clients','/proc/1/root/proc/662948/status','/proc/1/root/proc/664374/status']:
 v=rtext(path);extra[path]=v[:8192]if isinstance(v,str)else v
print(json.dumps(dict(at_unix=time.time(),hostname=os.uname().nodename,nvml_raw=nvml,gpu_raw=gpus,own_exact_output_processes=rows,proc=procfields,proc_mounts=[s for s in Path('/proc/mounts').read_text().splitlines()if' proc 'in s],extra_readonly=extra)))'''.replace('OUTPUTS',repr(outputs)).replace('KNOWN',repr([r[k]for r in receipts for k in ['worker_pid','child_pid']]))
result=json.loads(remote(host,[OPERATOR_PYTHON,'-c',code]));out=dict(schema='world_rl2_nvml_pid_namespace_readonly_audit_v1',status='COMPLETE_READ_ONLY_NO_ADMISSION_CHANGE',created_unix=time.time(),host=host,job_receipts=[dict(job=str(j),running=r,running_file_sha256=hashlib.sha256((j/'RUNNING.json').read_bytes()).hexdigest())for j,r in zip(jobs,receipts)],evidence=result,no_signals=True,no_unknown_owner_rule_relaxation=True)
# Pairing is only certified if actual NSpid contains an NVML PID; memory/rank count alone is insufficient.
nvmlpids={int(s.split(',',1)[0])for s in result['nvml_raw'].splitlines()};pairs=[]
for r in result['own_exact_output_processes']:
 ids=[int(v)for v in r.get('status',{}).get('NSpid','').split()]
 for pid in ids:
  if pid in nvmlpids:pairs.append(dict(nvml_pid=pid,visible_pid=r['pid'],nspid=ids,rank_environment=r.get('rank_environment'),argv=r.get('argv')))
out['exact_nspid_mapping']=pairs;out['exact_all16_nvml_mapping_proven']=len({p['nvml_pid']for p in pairs})==len(nvmlpids)==16;out['sha256']=fingerprint(out)
p=E/'resources/RL2_NVML_PID_NAMESPACE_AUDIT_20261101.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(p),sha256=out['sha256'],own_exact_output_processes=len(result['own_exact_output_processes']),nvml_count=len(nvmlpids),pairs=len(pairs),sample=[dict(pid=r['pid'],nspid=r.get('status',{}).get('NSpid'),rank=r.get('rank_environment'),ns=r.get('ns'),cgroup=r.get('cgroup'))for r in result['own_exact_output_processes']],extra=result['extra_readonly']),indent=2))
