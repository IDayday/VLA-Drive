import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,read,digest,source_sha,check_host
r=Path('/mnt/project/minimal-repair-artifacts/20261015');o=Path('/mnt/project/decision-local-consequence-artifacts/20261014');check_host()
start=time.time();runs=[]
def run(kind,gpu):
 checkpoint=r/'stageA_v5_seed42'/kind/'epoch_24.pt';complete=read(checkpoint.parent/'COMPLETE.json')
 assert complete['epoch']==24 and complete['fold']['fold']==1 and complete['fold']['scenes']==48881
 online=r/'online_formal_v14'/kind;cached=r/'cached_formal4_v14'/kind
 commands=[['--anchor',str(o/'ANCHOR.json'),'--refiner',str(checkpoint),'--index',str(r/'ONLINE_DEV4.json'),'--output',str(online),'--split','dev','--max-seconds','900'],['--features',str(o/'anchor_dev_full_v1'),'--checkpoint',str(checkpoint),'--index',str(r/'ONLINE_DEV4.json'),'--output',str(cached),'--distribution','NATURAL','--mode','camera_only','--max-seconds','300']]
 records=[]
 for entry,args in zip(['predict_current.py','evaluate_minimal_repair_planning.py'],commands):
  argv=[sys.executable,'-u',str(w/'tools/decision_local'/entry)]+args;t=time.time();log=r/(kind+'_'+entry+'_formal_v14.log')
  with log.open('x') as stream:
   p=subprocess.run(argv,cwd=w,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,timeout=960)
  records.append(dict(argv=argv,returncode=p.returncode,seconds=time.time()-t,log=str(log)));p.check_returncode()
 comparisons={}
 for row in read(r/'ONLINE_DEV4.json'):
  token=row['token']
  with np.load(online/'predictions'/(token+'.npz')) as a,np.load(cached/'predictions'/(token+'.npz')) as b:
   comparisons[token]={k:dict(exact=bool(np.array_equal(a[k],b[k])),max_abs=float(np.max(np.abs(a[k]-b[k])))) for k in ['q0','q_final']}
 assert all(v['exact'] for d in comparisons.values() for v in d.values())
 record=dict(kind=kind,source_sha=source_sha(w),checkpoint_sha256=digest(checkpoint),fold=complete['fold'],comparisons=comparisons,online_receipt=read(online/'COMPLETE.json'),commands=records)
 atomic_json(r/(kind+'_FORMAL_ONLINE_ACCEPTANCE_v14.json'),record);return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,k,g) for k,g in [('LOG_NONE',0),('MIN_NONE',1)]]
 for f in futures:runs.append(f.result())
atomic_json(r/'FORMAL_ONLINE_ACCEPTANCE_v14.json',dict(passed=True,runs=runs,script_sha256=digest(__file__),elapsed_seconds=time.time()-start,scope='Real current camera-only, completed B-fold common24 controls; no test-set selection'))
print('Both formal controls match real online/current-only cache output exactly',flush=True)
