import json,os,subprocess,sys,time
from pathlib import Path
import torch
w=Path('/mnt/project/VLA-Drive-minimal-repair-balanced-v14-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,check_host,digest,read,identity,source_sha
from tools.decision_local.verify_resume import equal
from starVLA.dataloader.recovery_plan_dataset import RecoveryPlanData
from tools.decision_local.minimal_repair_training import label_semantics
r=Path('/mnt/project/minimal-repair-artifacts/20261015');out=r/'resume_v14';out.mkdir(exist_ok=False);check_host()
features=Path('/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1')
labels=Path('/dev/shm/decision-local-repair-labels-v12-20261015/targets_train64_v5')
d=RecoveryPlanData(features,labels);receipt=read(r/'REPAIR_CLASS_BALANCE_v14.json')
receipt.update(labels=d.label_identity['identity'],fold=None,training_index_sha256=identity(d.index),label_semantics=label_semantics(d.label_identity),scope='Real64 training-chain smoke only, using same fixed weights')
atomic_json(out/'BALANCE.json',receipt)
base=[sys.executable,'-u',str(w/'tools/decision_local/train_small_model.py'),'--protocol','minimal_repair_v1','--features',str(features),'--labels',str(labels),'--kind','MIN_FUT','--scope','smoke','--epochs','12','--batch','16','--microbatch','8','--data-reader','mmap_tensors','--max-seconds','300']
records=[]
for name,extra,balanced in [('default',[],False),('continuous',[],True),('interrupted',['--max-updates','7'],True),('interrupted',['--resume'],True)]:
 command=base+['--output',str(out/name)]+extra+(['--repair-class-balance',str(out/'BALANCE.json')] if balanced else [])
 t=time.time();log=out/(name+('_resume' if '--resume' in extra else '')+'.log')
 with log.open('x') as stream:p=subprocess.run(command,cwd=w,stdout=stream,stderr=subprocess.STDOUT,timeout=330)
 records.append(dict(argv=command,returncode=p.returncode,seconds=time.time()-t,log=str(log)))
 atomic_json(out/'JOBS.json',records);p.check_returncode()
 if name=='interrupted' and '--resume' not in extra:os.link(out/name/'latest.pt',out/name/'origin7.pt')
keys=['model','optimizer','scheduler','rng','completed','epoch','offset','exposure','order','effective_optimizer_updates','exposure_counts']
a=torch.load(out/'continuous/latest.pt',map_location='cpu',weights_only=False);b=torch.load(out/'interrupted/latest.pt',map_location='cpu',weights_only=False)
for key in ['identity']+keys:equal(a[key],b[key],key)
old=torch.load(r/'resume_RAM_v11/continuous_A/latest.pt',map_location='cpu',weights_only=False)
new=torch.load(out/'default/latest.pt',map_location='cpu',weights_only=False)
for key in keys:equal(old[key],new[key],key)
atomic_json(out/'ACCEPTANCE.json',dict(exact_balanced_resume=True,exact_default_old_state=True,compared_keys=keys,completed=a['completed'],source_sha=source_sha(w),script_sha256=digest(__file__),GPU_process_hours=sum(x['seconds'] for x in records)/3600))
print('balanced resume and default old training states exact through48',flush=True)
