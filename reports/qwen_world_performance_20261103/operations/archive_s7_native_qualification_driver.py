"""One finite engineering run; own S0 yields to a fresh-IL70 qualification."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
P=Path('/mnt/project/qwen-world-performance-rl-20261103')
D=P/'operations/S7_ENGINEERING_AND_FORMAL'
OPS=P/'runtime/ops_v8'
SCI=P/'runtime/structured_progress_v1'
QUAL=P/'qualification/S7_STRUCTURED_PROGRESS_NATIVE_V1'
HOST='training-vlawm-tsinghua-zt2'
MODEL='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
os.environ.update(WORLD_COMPARE_ROOT=str(P),WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(OPS))
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_safety_exploration.resources import probe,gpu_activity_issue
from tools.qwen_world_safety_exploration import dispatch
from tools.qwen_world_safety_exploration.eval_owner import lease_free,LEASES
from tools.qwen_world_performance.continuation import shared_sample_ready,PROTOCOL_V2

def read(p):return json.loads(Path(p).read_text())
def update(**kw):
 state.update(kw,heartbeat_unix=time.time());atomic_json(D/'QUALIFICATION_DRIVER_STATE.json',state)

def run():
 wait=read(D/'S0_OWN_YIELD_REQUEST.json');job=Path(wait['job'])
 while not(job/'RESULT.json').exists():
  if(D/'STOP_QUALIFICATION_DRIVER').exists():raise RuntimeError('Explicit CPU stop before dispatch')
  if time.time()>1793660400:raise RuntimeError('No new engineering admission after23UTC')
  update(status='WAITING_OWN_NATIVE_SAVE');time.sleep(2)
 result=read(job/'RESULT.json')
 if result['returncode']!=0 or result.get('forced_expiry'):raise RuntimeError('Own native return needs review')
 update(status='CHECKING_ACTUAL_CAPACITY',yielded_native_result=result)
 plan=read(QUAL/'PLAN.json')
 for stage in plan['stages']:
  if time.time()>1793660400:raise RuntimeError('No new engineering admission after23UTC')
  sample=probe(HOST);gpus=sample['gpus']
  if(len(gpus)!=8 or min(g['free_MiB']for g in gpus)<32768 or any(gpu_activity_issue(g,sample['owners'])for g in gpus)):
   raise RuntimeError('Actual physical capacity changed; no GPU dispatched')
  ready,reason=shared_sample_ready(sample,dict(root=str(QUAL),resource_protocol=PROTOCOL_V2))
  if not ready:raise RuntimeError('Known-native capacity changed: '+reason)
  if not all(lease_free(LEASES['primary']/(g['uuid']+'.lock'))for g in gpus):
   raise RuntimeError('Primary UUID lane changed; do not evict its owner')
  actual=Path(stage['transport_result']).parent;name=str(actual.relative_to(P/'jobs'))
  if(actual/'DISPATCH.json').exists():raise RuntimeError('Existing transport requires explicit review')
  with(P/'resources/NATIVE_DISPATCH.lock').open('a')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);claims=read(P/'resources/NATIVE_HOST_CLAIMS.json')
   claims[HOST]=dict(job=str(actual),arm='S7_ENGINEERING_QUALIFICATION',at_unix=time.time(),reason='Finite actual world8 qualification')
   atomic_json(P/'resources/NATIVE_HOST_CLAIMS.json',claims)
  update(status='DISPATCHING',stage=stage['id'],job=str(actual),actual_sample={k:v for k,v in sample.items()if k!='_exact_peer_validation'})
  dispatch.launch(HOST,str(SCI),stage['command'],name,gpus='0,1,2,3,4,5,6,7',
      expected_uuids=[g['uuid']for g in gpus],share_minimum=32768,lease_slot='primary',worker_source=str(OPS),
      known_peer_receipts=sample.get('_exact_peer_validation',{}).get('value',{}).get('paths',[]))
  while not(actual/'RESULT.json').exists():
   update(status='RUNNING',stage=stage['id'],job=str(actual));time.sleep(5)
  result=read(actual/'RESULT.json');state.setdefault('results',[]).append(dict(stage=stage['id'],job=str(actual),result=result))
  if result['returncode']!=0:raise RuntimeError('Actual qualification child failure/refusal retained: '+str(actual))
 update(status='COMPARING_ACTUAL_NATIVE_STATES')
 command=[MODEL,'-m','tools.qwen_world_structured_progress.qualify_native','compare','--plan',str(QUAL/'PLAN.json'),'--output',str(QUAL/'COMPLETE.json')]
 with(D/'qualification_compare.log').open('ab',buffering=0)as log:
  compared=subprocess.run(command,cwd=SCI,env=dict(os.environ,PYTHONPATH=str(SCI),CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 if compared.returncode:raise RuntimeError('Actual native comparison failed; original evidence retained')
 proof=read(QUAL/'COMPLETE.json')
 if proof.get('status')!='PASS':raise RuntimeError('No actual PASS proof')
 update(status='PASS',complete=str(QUAL/'COMPLETE.json'))

if __name__=='__main__':
 with(D/'qualification_driver.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(schema='world_S7_finite_engineering_execution_v1',source=str(SCI),worker_source=str(OPS),host=HOST,started_unix=time.time())
  try:run()
  except Exception:
   update(status='REVIEW_REQUIRED',error=traceback.format_exc());raise
