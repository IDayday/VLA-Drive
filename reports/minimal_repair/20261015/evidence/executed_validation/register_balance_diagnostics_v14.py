import copy,json,os,shlex,subprocess,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import read,atomic_json,check_host
r=Path('/mnt/project/minimal-repair-artifacts/20261015');check_host()
# Exactly four fixed dependence probes and six full behavior analyses.
mechanisms=[];behavior=[]
for dist in ['NATURAL','RECOVERY']:
 original=read(r/('BALANCED_V14_EVAL_'+dist+'.json'));jobs=[]
 for job in original['jobs']:
  name=dist.lower()+'_'+job['id'];out=r/'behavior_balanced_v14'/dist.lower()/job['id']
  behavior.append(dict(id=name,predictions=job['predictions'],scores=job['scores'],output=str(out),distribution=dist,kind=job['kind'],seed=42,requires=[str(Path(job['predictions'])/'COMPLETE.json'),str(Path(job['scores'])/'COMPLETE.json')],max_seconds=600,workers=8))
 template=next(j for j in original['jobs'] if j['kind']=='MIN_FUT')
 for mode in ['zero','reverse_time']:
  j=copy.deepcopy(template);j['id']+='_'+mode;base=r/'mechanism_balanced_v14'/dist.lower()
  j['predictions']=str(base/'predictions'/j['id']);j['scores']=str(base/'scores'/j['id']);j['reference_scores']=template['scores'];j['relations_ablation']=mode
  j['requires']=[str(Path(template['scores'])/'COMPLETE.json')]
  cmd=j['commands']['export'];cmd[cmd.index('--output')+1]=j['predictions'];cmd+=['--relations-ablation',mode]
  score=j['commands']['score'];score[score.index('--predictions')+1]=j['predictions'];score[score.index('--output')+1]=j['scores'];score[score.index('-o')+1]=str(Path(j['scores'])/'CPU_RESOURCE_USAGE.json')
  jobs.append(j)
 reg={**original,'jobs':jobs,'runtime':str(r/'mechanism_balanced_v14'/dist.lower()/'runtime'),'diagnostic_only':True,'selection':'Fixed24 dependence only, not candidate execution selection','intervention_scope':'All five relation channels zeroed or time-reversed; original support retained; not pure future-only intervention'}
 p=r/('BALANCED_V14_MECHANISM_'+dist+'.json');atomic_json(p,reg);mechanisms.append(p)
old=read(r/'STAGE_A_V5_BEHAVIOR_REGISTRY.json');old.update(jobs=behavior,runtime=str(r/'behavior_balanced_v14/runtime'),scope='Six fixed24 complete analyses for bounded class-balance follow-up',CPU_worker_allocation_hour_upper_bound=8)
atomic_json(r/'BALANCED_V14_BEHAVIOR_REGISTRY.json',old)
s=(r/'run_behavior_v12.py').read_text().replace('Finite18 common-endpoint analyses','Finite6 balanced common-endpoint analyses').replace('STAGE_A_V5_BEHAVIOR_REGISTRY.json','BALANCED_V14_BEHAVIOR_REGISTRY.json').replace("host+'_FINISHED.json'","host+'_'+str(os.getpid())+'_FINISHED.json'")
script=r/'run_behavior_balanced_v14.py';script.write_text(s)
records=[]
def launch(host,argv,log):
 if host=='training-vla-zt':
  with log.open('x') as f:p=subprocess.Popen(argv,cwd=w,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  pid=p.pid
 else:
  code='import subprocess;f=open('+repr(str(log))+',"x");p=subprocess.Popen('+repr(argv)+',cwd='+repr(str(w))+',stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
  pid=int(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join([sys.executable,'-c',code])],text=True,timeout=30).strip())
 records.append(dict(host=host,pid=pid,argv=argv,log=str(log),started_unix=time.time()))
for host,reg in [('training-rl-zt2',mechanisms[0]),('training-vlawm-tsinghua-zt',mechanisms[1])]:
 launch(host,[sys.executable,'-u',str(w/'tools/decision_local/evaluate_milestones.py'),'--registry',str(reg),'--gpus',*map(str,range(8)),'--cpu-parallel','1','--max-seconds','10800'],r/('balanced_mechanism_v14_'+host+'.log'))
for host in ['training-vla-zt','training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5']:
 for index in range(2):launch(host,[sys.executable,'-u',str(script)],r/(f'balanced_behavior_v14_{host}_{index}.log'))
atomic_json(r/'BALANCED_V14_DIAGNOSTIC_RUNNERS.json',records);print(json.dumps([{'host':x['host'],'pid':x['pid']} for x in records]))
