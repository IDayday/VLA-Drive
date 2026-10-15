import json,os,subprocess,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import read,atomic_json,source_sha,check_host
r=Path('/mnt/project/minimal-repair-artifacts/20261015');check_host()
registries=[]
for distribution in ['NATURAL','RECOVERY']:
 old=read(r/('STAGE_A_V5_EVAL_'+distribution+'.json'));jobs=[];base=r/'development_balanced_v14'/distribution.lower()
 for kind in ['MIN_NONE','MIN_CV','MIN_FUT']:
  original=next(j for j in old['jobs'] if j['kind']==kind and j['seed']==42 and j['epoch']==24)
  job=dict(original);name='stageA_balanced_v14_seed42_'+kind+'_epoch24';checkpoint=r/'stageA_balanced_v14_seed42'/kind/'epoch_24.pt'
  job.update(id=name,checkpoint=str(checkpoint),predictions=str(base/'predictions'/name),scores=str(base/'scores'/name),requires=[str(checkpoint.parent/'COMPLETE.json')])
  export=list(original['commands']['export']);score=list(original['commands']['score'])
  for key,value in {'--checkpoint':str(checkpoint),'--output':job['predictions']}.items():export[export.index(key)+1]=value
  for key,value in {'--predictions':job['predictions'],'--output':job['scores']}.items():score[score.index(key)+1]=value
  measure=Path(job['scores'])/'CPU_RESOURCE_USAGE.json'
  score=['/usr/bin/time','-f','{"user_seconds": %U, "system_seconds": %S, "max_rss_kb": %M, "wall_seconds": %e, "exit_code": %x}','-o',str(measure)]+score
  job['commands']={'export':export,'score':score};job['CPU_resource_usage']=str(measure);jobs.append(job)
 registry={**old,'jobs':jobs,'runtime':str(base/'runtime'),'selection':'Exactly common24; one bounded post-primary class-balance diagnostic, no Navtest selection','training_source_sha':source_sha(Path('/mnt/project/VLA-Drive-minimal-repair-balanced-v14-20261015'))}
 path=r/('BALANCED_V14_EVAL_'+distribution+'.json');assert not path.exists();atomic_json(path,registry);registries.append(path)
records=[]
# On this host, leave training2/4/6 and current-image checks0/1 unclaimed.
# tsinghua uses only current policy-authorized host; runner rechecks on launch.
for host,gpus,path in [('training-vla-zt',[3,5,7],registries[0]),('training-rl-zt2',list(range(8)),registries[1])]:
 argv=[sys.executable,'-u',str(w/'tools/decision_local/evaluate_milestones.py'),'--registry',str(path),'--gpus',*map(str,gpus),'--cpu-parallel','2','--max-seconds','10800']
 log=r/('balance_eval_v14_'+host+'.log')
 if host=='training-vla-zt':
  with log.open('x') as stream:p=subprocess.Popen(argv,cwd=w,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  pid=p.pid
 else:
  # Structured argv transmitted to a short Python launcher, no shell interpolation of data.
  program='import json,subprocess;from pathlib import Path; a=json.loads('+repr(json.dumps(argv))+'); f=open('+repr(str(log))+',"x"); p=subprocess.Popen(a,cwd='+repr(str(w))+',stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)'
  import shlex
  answer=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join([sys.executable,'-c',program])],text=True,timeout=30);pid=int(answer.strip())
 records.append(dict(host=host,pid=pid,argv=argv,log=str(log),started_unix=time.time()))
atomic_json(r/'BALANCED_V14_EVAL_RUNNERS.json',records);print(json.dumps(records))
