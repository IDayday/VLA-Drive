"""Finite CPU dispatcher. Default prints plan; --run admits at most one job/task.

Waits for B/D evaluation and its shared lease to clear naturally. Never stops
existing tasks. No new GPU admission at/after2026-11-02T23:00:00UTC.
"""
import argparse,datetime,fcntl,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
P=Path('/mnt/project/qwen-world-performance-rl-20261103');TASK='qwen_world_safety_exploration_20261031'
SOURCE=P/'runtime/formal_v1';WORKER=P/'runtime/ops_v5';HOST='training-vlawm-zt2'
REQUEST=P/'diagnostics/RL_TRAIN2048_DIAGNOSTIC_REQUEST.json';PLAN=P/'diagnostics/dispatch_plans/RL_TRAIN2048_CPU_DISPATCH_PLAN.json'
D=P/'diagnostics/owners/RL_TRAIN2048_PAIR';CUTOFF=1793660400.
MODEL='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
BASELINE_OWNER=str(P/'diagnostics/IL70_FULL_NAVTRAIN')
TERMINAL={'COMPLETE','RESULT_REVIEW_REQUIRED','RESOURCE_REFUSED_NO_CHILD_REVIEW','FULL_SUMMARY_REVIEW_REQUIRED','DEFERRED_AFTER_2300_ADMISSION_CUTOFF','DISPATCH_UNCERTAIN_REVIEW_REQUIRED'}
ROUTES={'S0_LARGE_SAFETYEP':(3,'GPU-92337d32-0247-da22-077c-a19994af0d37'),'S5_PERFORMANCE_MAX':(5,'GPU-69a69533-8950-70a7-6971-b8891b199ca2')}
os.environ.update(WORLD_COMPARE_ROOT=str(P),WORLD_COMPARE_TASK=TASK)
sys.path.insert(0,str(WORKER))
from starVLA.rl.qwen_world_safety_exploration.resources import probe,gpu_activity_issue
from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts
from tools.qwen_world_safety_exploration.dispatch import launch,job_state

def read(p,default=None):return json.loads(Path(p).read_text())if Path(p).exists()else default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def free(slot,uuid):
 folder={'primary':'runtime_gpu_leases','shared':'runtime_gpu_shared_workers','research':'runtime_gpu_research_slots','evaluation':'runtime_gpu_evaluation_slots'}[slot]
 path=Path('/mnt/project')/folder/(uuid+'.lock')
 if not path.exists():return True
 with path.open('r')as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return True
  except BlockingIOError:return False

def command(task,parity):
 req=read(REQUEST);meta=read(Path(task['export'])/'export.json');roots=meta['RL_parent']['recipe']['research']['scoring_cache_roots']
 c=[MODEL,'-m','tools.qwen_world_safety_exploration.evaluate','--parent-checkpoint',task['parent_checkpoint'],'--rl-checkpoint',task['checkpoint'],
    '--export',task['export'],'--data',task['data'],'--manifest',req['manifest'],'--cohort','TRAIN_DIAGNOSTIC','--output',task['output'],
    '--rank','0','--world','1','--seed','2101','--cpu-threads','2']
 for root in roots:c+=['--cache-root',root]
 if parity:c+=['--native-parity-output',str(Path(task['output'])/'native_export_parity.json')]
 return c

def validate_task(task):
 req=read(REQUEST);r=next(x for x in req['runs']if x['arm']==task['arm'])
 if any(r[k]!=task[k]for k in ('updates','checkpoint','export','export_job'))or req['manifest']!=task['manifest']or sha(req['manifest'])!=task['manifest_sha256']:raise ValueError('Fixed diagnostic request changed')
 result=read(Path(r['export_job'])/'RESULT.json',{});meta=read(Path(r['export'])/'export.json');seal=read(Path(r['export'])/'COMPLETE')
 if result.get('returncode')!=0 or seal['export_sha256']!=sha(Path(r['export'])/'export.json')or meta['model_sha256']!=task['model_sha256']or meta['RL_updates']!=r['updates']or Path(meta['RL_checkpoint']).resolve()!=Path(r['checkpoint']).resolve():raise ValueError('Export/native metadata binding changed')
 if meta.get('qualification')is not False or meta['IL_parent']['model_sha256']!='7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40':raise ValueError('Wrong formal pinnedIL70 parent')
 proof=read(Path(r['export']).with_name(Path(r['export']).name+'.native_validation.json'),{})
 cp=read(Path(r['checkpoint'])/'COMPLETE',{})
 if proof.get('status')!='PASS'or proof.get('checkpoint',{}).get('files_sha256')!=cp.get('files_sha256'):raise ValueError('Missing strict native export verification')

def capacity(task,sample):
 gpu=next((g for g in sample['gpus']if g['index']==task['gpu']),None);uuid=task['uuid']
 if gpu is None or gpu['uuid']!=uuid:return False,'physical_GPU_mapping_changed',False
 if task['gpu']==6:return False,'GPU6_forbidden',False
 if gpu_activity_issue(gpu,sample['owners']):return False,'unaccounted_GPU_owner_or_allocation',False
 owners=[x for x in sample['owners']if x['uuid']==uuid]
 for x in owners:
  if not x.get('visible')or x.get('entry')!='tools.qwen_world_safety_exploration.evaluate'or not isinstance(x.get('output'),str)or Path(x['output']).resolve()!=Path(BASELINE_OWNER):
   return False,'waiting_B_D_or_nonbaseline_owner_natural_exit',False
 # Shared-slot holder can be an IL model still loading on CPU, absent in NVML.
 if not free('shared',uuid):return False,'waiting_pending_B_D_shared_lease_natural_exit',False
 if not free('research',uuid)or not free('evaluation',uuid):return False,'independent_diagnostic_lease_unavailable',False
 if not owners and not free('primary',uuid):return False,'unresolved_primary_loading_window',False
 if gpu['free_MiB']<32768:return False,'less_than_32GiB_actual_headroom',False
 if sample['MemAvailable_bytes']<64*2**30 or sample['load1']>sample['CPU_affinity']*1.25:return False,'CPU_headroom_busy',False
 parity=gpu['free_MiB']>=65536
 return True,'actual_only_baseline_or_empty;32GiB_gate;parity_requires64GiB',parity

def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args();plan=read(PLAN)
 if not args.run:print(json.dumps(plan,indent=2));return
 D.mkdir(parents=True,exist_ok=True);guard=(D/'owner.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=read(D/'STATE.json',dict(schema='world_finite_TRAIN2048_pair_dispatcher_v1',tasks={}))
 binding={x['arm']:{k:x[k]for k in ('updates','checkpoint','export','export_job','output','gpu','uuid','manifest_sha256','model_sha256')}for x in plan['tasks']}
 if state.get('binding',binding)!=binding:raise ValueError('Diagnostic CPU adoption binding changed')
 state.update(binding=binding,status='RUNNING',admission_cutoff_unix=CUTOFF,max_successful_dispatches_per_task=1)
 write(D/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),started_unix=time.time()))
 while True:
  if(D/'STOP_OWNER').exists():state['status']='CPU_QUIESCED_CHILDREN_PRESERVED';write(D/'STATE.json',state);return
  pending=[]
  for task in plan['tasks']:
   arm=task['arm'];node=state['tasks'].setdefault(arm,{})
   if node.get('status')in TERMINAL:continue
   if node.get('job'):
    try:result=job_state(node['job'])
    except (FileNotFoundError,ValueError)as error:
     node.update(status='DISPATCH_UNCERTAIN_REVIEW_REQUIRED',reason='Saved attempt has no valid published receipt; never start a duplicate',error=repr(error));continue
    node['last_job_status']=result.get('status');node['result']=result if'returncode'in result else None
    if'returncode'in result:
     rank=read(Path(task['output'])/'rank0.json',{})
     if result['returncode']==0 and rank.get('status')=='COMPLETE'and rank.get('rank')==0 and rank.get('world')==1:
      if not(Path(task['output'])/'SUMMARY.json').exists():
       cmd=[MODEL,'-m','tools.qwen_world_safety_exploration.evaluate','--summarize','--output',task['output']]
       done=subprocess.run(cmd,cwd=SOURCE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(SOURCE)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
       node['summary_CPU_returncode']=done.returncode
       if done.returncode:node['summary_error_tail']=done.stdout[-2500:]
      summary=read(Path(task['output'])/'SUMMARY.json',{})
      node['status']='COMPLETE'if summary.get('status')=='COMPLETE'and summary.get('scenes')==2048 and summary.get('invalid_scenes')==0 else'FULL_SUMMARY_REVIEW_REQUIRED'
      node['summary']=str(Path(task['output'])/'SUMMARY.json')
     elif result.get('returncode')==75 and result.get('child_not_started')is True:node['status']='RESOURCE_REFUSED_NO_CHILD_REVIEW';node['scientific_failure']=False
     else:node['status']='RESULT_REVIEW_REQUIRED';node['partial_population_never_published']=True
    else:node['status']='DISPATCHED_RUNNING_OR_LOADING'
   elif time.time()>=CUTOFF:node['status']='DEFERRED_AFTER_2300_ADMISSION_CUTOFF'
   else:pending.append(task)
  if pending:
   if HOST not in allowed_hosts(admit_long_job=True):
    for task in pending:state['tasks'][task['arm']].update(status='WAITING_AUTHORIZED_RESOURCE')
   else:
    try:sample=probe(HOST)
    except Exception as error:
     for task in pending:state['tasks'][task['arm']].update(status='WAITING_PROBE',reason=repr(error))
    else:
     for task in pending:
      node=state['tasks'][task['arm']];validate_task(task);ok,reason,parity=capacity(task,sample)
      node.update(status='WAITING_RESOURCE',reason=reason,last_sample_unix=sample['at'])
      if not ok:continue
      lockpath=P/'resources/EVALUATION_DISPATCH.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
      with lockpath.open('a')as lock:
       fcntl.flock(lock,fcntl.LOCK_EX)
       if time.time()>=CUTOFF or(D/'STOP_OWNER').exists():continue
       fresh=probe(HOST);ok,reason,parity=capacity(task,fresh)
       if not ok:node['reason']=reason;continue
       name='TRAIN_DIAG/'+task['arm']+f"_u{task['updates']:06}_TRAIN2048_world1_attempt001";job=P/'jobs'/name;c=command(task,parity)
       if(job/'DISPATCH.json').exists():
        old=read(job/'DISPATCH.json')
        if old['host']!=HOST or old['source']!=str(SOURCE)or old['command']not in(command(task,True),command(task,False)):raise ValueError('Existing diagnostic job binding differs')
        node.update(job=str(job),status='ADOPTED_EXISTING_DISPATCH');continue
       node.update(job=str(job),command=c,source=str(SOURCE),worker_source=str(WORKER),status='DISPATCHING',
        native_parity_enabled=parity,native_parity_skip_reason=None if parity else'actual_free<64GiB;strict_existing_native/export proof retained',
        measured_free_MiB=next(g['free_MiB']for g in fresh['gpus']if g['index']==task['gpu']),successfully_dispatched_tasks=0)
       write(D/'STATE.json',state)
       # One transport attempt per task; ambiguous failures are retained and
       # never launch a second GPU process automatically.
       try:
        launch(HOST,str(SOURCE),c,name,gpus=str(task['gpu']),expected_uuids=[task['uuid']],share_minimum=65536 if parity else 32768,lease_slot='research',worker_source=str(WORKER))
        node.update(status='DISPATCHED',successfully_dispatched_tasks=1,dispatched_unix=time.time())
       except Exception as error:node.update(status='DISPATCH_UNCERTAIN_REVIEW_REQUIRED',dispatch_error=repr(error))
       claims=read(P/'resources/EVALUATION_CLAIMS.json',{});claims[task['uuid']]=dict(receipt=str(job),owner='RL_TRAIN2048_PAIR',arm=task['arm'],cohort='TRAIN_DIAGNOSTIC',rank=0,world=1,host=HOST,updated_unix=time.time());write(P/'resources/EVALUATION_CLAIMS.json',claims)
  state['heartbeat_unix']=time.time();write(D/'STATE.json',state)
  statuses=[state['tasks'][t['arm']].get('status')for t in plan['tasks']]
  if all(s in TERMINAL for s in statuses):
   state['status']='COMPLETE'if all(s=='COMPLETE'for s in statuses)else'FINISHED_WITH_REVIEW_OR_RESOURCE_DEFERRED';write(D/'STATE.json',state);return
  time.sleep(30)
if __name__=='__main__':main()
