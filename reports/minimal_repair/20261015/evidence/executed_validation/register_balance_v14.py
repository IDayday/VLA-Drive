import json,os,subprocess,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-balanced-v14-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import read,atomic_json,digest,source_sha,check_host
r=Path('/mnt/project/minimal-repair-artifacts/20261015');check_host()
proof=read(r/'resume_v14/ACCEPTANCE.json');assert proof['exact_balanced_resume'] and proof['exact_default_old_state']
accept=read(r/'STAGE_A_ACCEPTANCE_v5.json');sha=source_sha(w)
accept.update(training_source_sha=sha,kinds=['MIN_NONE','MIN_CV','MIN_FUT'],seeds=[42],registered_runs=[[k,42] for k in ['MIN_NONE','MIN_CV','MIN_FUT']],repair_class_balance_sha256=digest(r/'REPAIR_CLASS_BALANCE_v14.json'),
    followup='One bounded post-primary learning correction: fixed training-exposure balanced KEEP/REPAIR loss; main six groups retained',
    stage_b_training_allowed=False,evidence=accept['evidence']+[str(r/'resume_v14/ACCEPTANCE.json'),str(r/'TRAINING_CONDITIONAL_FIT_v13.json')])
accept_path=r/'BALANCED_MIN_ACCEPTANCE_v14.json';assert not accept_path.exists();atomic_json(accept_path,accept)
old=read(r/'STAGE_A_V5_TASKS.json');jobs=[]
for kind in ['MIN_NONE','MIN_CV','MIN_FUT']:
 orig=next(j for j in old['jobs'] if j['kind']==kind and j['seed']==42)
 j=dict(orig);j['id']='stageA_balanced_v14_seed42_'+kind;j['output']=str(r/'stageA_balanced_v14_seed42'/kind);j['source']=str(w);j['training_source_sha']=sha;j['max_seconds']=3600
 cmd=list(j['command']);cmd[2]=str(w/'tools/decision_local/train_small_model.py')
 for key,value in {'--output':j['output'],'--target-acceptance':str(accept_path),'--max-seconds':'3600','--features':'/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1','--dev-features':'/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_dev_full_v1','--labels':'/dev/shm/decision-local-repair-labels-v12-20261015/targets_foldB_v5','--dev-labels':'/dev/shm/decision-local-repair-labels-v12-20261015/targets_dev1696_v5'}.items():cmd[cmd.index(key)+1]=value
 cmd+=['--repair-class-balance',str(r/'REPAIR_CLASS_BALANCE_v14.json')];j['command']=cmd;j['requires']=[str(accept_path),str(r/'resume_v14/ACCEPTANCE.json'),str(r/'targets_foldB_v5/COMPLETE.json')];jobs.append(j)
registry={**old,'formal_acceptance':str(accept_path),'allowed_hosts':{'training-vla-zt':[2,4,6]},'jobs':jobs,'runtime':str(r/'stageA_balanced_v14_runtime'),'registered_revision_reason':accept['followup']}
p=r/'STAGE_A_BALANCED_V14_TASKS.json';assert not p.exists();atomic_json(p,registry)
argv=[sys.executable,'-u',str(w/'tools/decision_local/run_registered_heads.py'),'--registry',str(p),'--gpus','2','4','6','--max-seconds','3900']
log=r/'balanced_heads_v14.log'
with log.open('x') as stream:process=subprocess.Popen(argv,cwd=w,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
record={'host':'training-vla-zt','pid':process.pid,'argv':argv,'log':str(log),'source_sha':sha,'started_unix':time.time(),'max_GPU_process_hours':3,'max_additional_CPU_hours_allocation':8,'no_cap_increase':True}
atomic_json(r/'BALANCED_V14_RUNNER.json',record);print(json.dumps(record))
