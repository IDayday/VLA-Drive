import hashlib,json,os,pathlib,socket,subprocess,time
r=pathlib.Path('/mnt/project/minimal-repair-artifacts/20261015')
host=socket.gethostname().removesuffix('-worker-0')
p=pathlib.Path('/mnt/project/server_dispatch_policy.json'); policy=json.loads(p.read_text())
assert host in policy['allowed_hosts'] and host not in policy['global_dispatch_prohibitions']
assert host != 'training-vla-zt2'
readers=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
  relevant=[v for v in args if v.endswith('.py') and (v.startswith(str(r)+'/') or 'VLA-Drive-minimal-repair' in v)]
  relevant=[v for v in relevant if not v.endswith('/audit_resource_closure_v15.py')]
  if relevant:
   readers.append({'pid':int(proc.name),'scripts':relevant,'start_ticks':proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]})
 except (OSError,IndexError):pass
out={'host':host,'observed_unix':time.time(),'policy_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'owned_research_processes':readers,'owned_queue_empty':not readers,'unrelated_jobs_not_stopped':True,'G2_host_not_accessed':True,'no_pressure_scripts_preempted_or_restarted':True,'scope':'Only this study process presence; not a claim the entire host is idle','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(r/f'RESOURCE_CLOSURE_{host}_v15.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
