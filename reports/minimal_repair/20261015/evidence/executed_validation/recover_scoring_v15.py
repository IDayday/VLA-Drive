import copy,json,os,shlex,subprocess,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');s=Path('/mnt/project/VLA-Drive-minimal-repair-scoring-v15-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import read,atomic_json,check_host,digest
r=Path('/mnt/project/minimal-repair-artifacts/20261015');check_host();paths={}
assert read(r/'CPU_WRAPPER_REAL_CHECK_v15.json')['exit_code']==0
for phase in ['EVAL','MECHANISM']:
 for dist in ['NATURAL','RECOVERY']:
  old=r/(f'BALANCED_V14_{phase}_{dist}.json');reg=read(old);reg['runtime']+= '_scoring_recovery_v15'
  reg['supersedes_execution_registry']=str(old);reg['scientific_outputs_unchanged']=True
  for job in reg['jobs']:
   original=job['commands']['score'];assert original[0]=='/usr/bin/time';child=original[5:]
   # Preserve the exact canonical child argv; replace only absent timing executable.
   assert child[2]==str(w/'tools/structured_world/score_navsim.py')
   output=Path(job['scores'])/'CPU_RESOURCE_USAGE.json'
   job['commands']['score']=[sys.executable,'-u',str(s/'tools/decision_local/measure_scoring_process.py'),'--output',str(output),'--']+child
   job['CPU_resource_usage']=str(output)
   if phase=='EVAL' and dist=='RECOVERY' and job['kind']=='MIN_CV':
    # Its earlier single-GPU export is still valid and running. Do not create a second writer.
    job['requires']+= [str(Path(job['predictions'])/'COMPLETE.json')]
  path=r/(f'BALANCED_V15_{phase}_{dist}.json');assert not path.exists();atomic_json(path,reg);paths[(phase,dist)]=path
records=[]
def launch(host,gpus,phase,dist):
 args=[sys.executable,'-u',str(w/'tools/decision_local/evaluate_milestones.py'),'--registry',str(paths[(phase,dist)]),'--gpus',*map(str,gpus),'--cpu-parallel','1','--max-seconds','10800']
 log=r/f'balanced_v15_{phase}_{dist}_{host}.log'
 if host=='training-vla-zt':
  with log.open('x') as f:p=subprocess.Popen(args,cwd=w,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  pid=p.pid
 else:
  code='import subprocess;f=open('+repr(str(log))+',"x");p=subprocess.Popen('+repr(args)+',cwd='+repr(str(w))+',stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
  pid=int(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join([sys.executable,'-c',code])],text=True,timeout=30).strip())
 records.append(dict(host=host,pid=pid,argv=args,started_unix=time.time(),log=str(log)))
for host,gpus,phase,dist in [('training-vla-zt',[3,5,7],'EVAL','NATURAL'),('training-vlawm-tsinghua-zt4',list(range(8)),'EVAL','NATURAL'),('training-rl-zt2',list(range(8)),'EVAL','RECOVERY'),('training-vlawm-tsinghua-zt2',list(range(8)),'EVAL','RECOVERY'),('training-rl-zt2',list(range(8)),'MECHANISM','NATURAL'),('training-vla-zt',[2,4,6],'MECHANISM','RECOVERY')]:launch(host,gpus,phase,dist)
atomic_json(r/'BALANCED_V15_SCORING_RUNNERS.json',records);print(json.dumps([dict(host=x['host'],pid=x['pid']) for x in records]))
