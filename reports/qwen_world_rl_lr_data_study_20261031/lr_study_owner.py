"""Independent exact-native Safety10k four-arm owner, global12k->20k.

Root first-dispatches12k; this owner adopts it, never owns CAMPAIGN/FIRST14.
Extra12/14/16/18k NAVTEST belongs to cadence; only final20k NAVTEST/TRAIN
is dispatched here. No training waits for evaluations before reaching20k.
"""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,os,select,subprocess,sys,threading,traceback
from concurrent.futures import ThreadPoolExecutor,Future
from pathlib import Path
import yaml
A=Path(__file__).resolve().parent;SOURCE=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030')
sys.path.insert(0,str(A));sys.path.insert(0,str(SOURCE));os.environ['WORLD_POSTCOT_TASK']='qwen_world_rl_optimization_20261030'
from tools.qwen_world_research.campaign_owner import CampaignOwner,native_checkpoint,read_json,atomic_json,now,result_code,ArmHeldForResourceLease
from tools.qwen_world_research import dispatch as D
from navtest_cadence_owner_v2 import CadenceDispatch,Events,NoLongerAuthorized
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');O=R/'owners/LR_STUDY'
ARMS=('PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS','PPO_SAFETY_DATA','PPO_SAFETY_LRDATA')
TARGETS=(12000,14000,16000,18000,20000);CAP=20000;ORIGIN=10000
SCHEMA='independent_exact_native_Safety10k_LR_data_study_owner_v1'
PLAN=A/'LR_STUDY_OWNER_LAUNCH_PLAN_20261031.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def terminal(row):return any(s in str(row.get('status','')).upper() for s in ('STOP','FAIL','CANCEL','HELD','PAUSED','DEFER'))
def registered():
 rows={x['id']:x for x in read_json(R/'RUNS.json')['runs']};out={}
 for arm in ARMS:
  if arm not in rows:raise ValueError('Root must register and first-dispatch '+arm)
  row=rows[arm];cfg=yaml.safe_load(Path(row['config']).read_text());q=cfg['research'];f=q.get('native_fork',{})
  expected_policy='static_rescue_retention' if arm in ARMS[2:] else 'uniform_full'
  if (row.get('module')!='starVLA.rl.qwen_world_research_algorithms.safety_lr_entry' or not row.get('job')
      or int(row.get('stage_budget_updates',0)) not in TARGETS or int(row.get('research_budget_updates',0))!=CAP
      or sha(row['config'])!=row['config_sha256'] or q.get('arm')!=arm or cfg.get('seed')!=42
      or cfg.get('max_steps')!=50000 or q.get('data_policy')!=expected_policy
      or f.get('schema')!='world_Safety10k_exact_native_LR_fork_v1' or f.get('expected_updates')!=ORIGIN
      or f.get('optimizer')!='inherited_AdamW' or f.get('rng')!='inherited_all_native_per_rank'
      or f.get('reference')!='original_frozen_IL70' or cfg['grpo']['sample']['sde_step_indices'] is not None
      or cfg['grpo']['sample']['train_sde_steps']!=3 or cfg['grpo']['train']['kl_beta']!=50):
   raise ValueError('Exact-native Safety10k/W3/credit/LR/data/cap registration changed '+arm)
  out[arm]=row
 return out


class StudyDispatch:
 def __init__(self,owner):self.owner=owner;self.nav=CadenceDispatch(owner)
 def __getattr__(self,name):return getattr(D,name)
 def dispatch_evaluation(self,**args):
  if args['cohort']=='NAVTEST':return self.nav.dispatch_evaluation(**args)
  arm=args['label'].rsplit('_u',1)[0]
  if arm not in ARMS or args['label']!=f'{arm}_u020000_TRAIN1024':raise NoLongerAuthorized('Only final20k TRAIN permitted')
  self.owner.authorized(arm,CAP)
  return D.dispatch_evaluation(**args)


class StudyOwner(CampaignOwner):
 def __init__(self):
  self.root=R;self.owner_root=O;O.mkdir(parents=True,exist_ok=True)
  self.lock=(O/'owner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.mutex=threading.RLock();self.capacity_changed=threading.Condition(self.mutex);self.capacity_epoch=0
  self.runs=registered();self.dispatcher=StudyDispatch(self);self.resource_retries=0
  self.parent=str(R/'IL_PARENT');self.data=str(R/'data/NAVTRAIN');self.panel=str(R/'data/panels/TRAIN_DIAGNOSTIC_001024.json')
  self.navtest_data='/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST';self.navtest_manifest=str(R/'data/panels/NAVTEST_012146.json')
  self.state=read_json(O/'STATE.json',{})
  if self.state and self.state.get('schema')!=SCHEMA:raise ValueError('Different LR study state')
  self.state.setdefault('schema',SCHEMA);self.state.setdefault('runs',{});self.state.setdefault('events',[]);self.state.setdefault('results',[])
  self.state.update(status='RUNNING_NATIVE_LR_DATA_STUDY',initial_global_updates=ORIGIN,global_budget_cap=CAP,
      native_optimizer_RNG_cursor_inherited=True,phase_reset=False,main_campaign_lock_used=False,training_waits_for_intermediate_evaluation=False,
      targets=list(TARGETS),extra_NAVTEST_12_18k_owner='independent existing NAVTEST_CADENCE',owner_evaluation_gates=[20000])
  for arm,row in self.runs.items():
   n=self.state['runs'].setdefault(arm,dict(id=arm,updates=ORIGIN,stages={},status='ADOPTING_ROOT_FIRST12000_JOB'))
   n.update(run_root=row['root'],source=row['source'],research_budget_updates=CAP,initial_global_updates=ORIGIN,
       phase_reset=False,additional_updates=int(n['updates'])-ORIGIN)
  self.eval_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='lr_study_eval');self.evaluation_futures={}
  self.baseline_future=Future();base=read_json(R/'evaluation_specs/PPO_SAFETY_EP_u010000_TRAIN1024.json')
  self.baseline_future.set_result(dict(output=base['output'],training_bank=base['training_bank']))
  self.navbase=read_json(R/'evaluation_specs/PPO_SAFETY_EP_u010000_NAVTEST.json')['output']
  atomic_json(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),started_at=now(),
      source_file=str(Path(__file__).resolve()),source_sha256=sha(__file__),scope='CPU owner; root-adopted nativeLR/data forks global10k->20k; no main controller ownership'))
  self.flush()
 def update_arm(self,arm,**values):
  if 'updates' in values:values['additional_updates']=values['updates']-ORIGIN
  super().update_arm(arm,**values)
 def authorized(self,arm,target):
  row=next(x for x in read_json(R/'RUNS.json')['runs'] if x['id']==arm);binding=self.runs[arm]
  if terminal(row) or (O/'STOP_AFTER_CURRENT_STAGE').exists():raise NoLongerAuthorized('No new LR-study work after stop/hold: '+arm)
  if target not in TARGETS or int(row.get('research_budget_updates',0))!=CAP:raise NoLongerAuthorized('LR study cap/target changed')
  if any(row[k]!=binding[k] for k in ('root','source','module','config','config_sha256')):raise ValueError('Registered scientific binding changed '+arm)
  return row
 def require_allowed(self,label):
  # CadenceDispatch callback is exact to final20k NAVTEST for these new arms.
  arm=label.rsplit('_u',1)[0]
  if arm not in ARMS or label!=f'{arm}_u020000_NAVTEST':raise NoLongerAuthorized('LR owner must not dispatch intermediate cadence points')
  self.authorized(arm,CAP)
 def wait_capacity(self,node,*,status,label,arm=None,since=None):
  if since is not None and since!=self.capacity_epoch:return
  node['status']=status
  if arm:self.state['runs'][arm]['status']=status
  self.event('waiting_real_capacity',label=label,status=status)
  ev=Events();fds=[]
  try:
   ev.add(R,{'RUNS.json'});ev.add(R/'resources',{'EVALUATION_CLAIMS.json'});ev.add(O,{'STOP_AFTER_CURRENT_STAGE'});ev.add(Path('/mnt/project'),{'server_dispatch_policy.json'})
   paths={row['job'] for row in read_json(R/'RUNS.json')['runs'] if row.get('job')}
   paths.update(row['receipt'] for row in read_json(R/'resources/EVALUATION_CLAIMS.json',{}).values() if row.get('receipt'))
   for job in paths:
    p=Path(job);ev.add(p,{'RESULT.json'})
    if (p/'RESULT.json').exists():continue
    launch=read_json(p/'LAUNCH.json',{})
    try:
     pid=int(launch['transport_pid']);st=Path(f'/proc/{pid}/stat').read_text().split()
     if int(st[21])!=launch['transport_start_ticks'] or st[2] in ('Z','X'):continue
     fd=os.pidfd_open(pid) if hasattr(os,'pidfd_open') else int(ctypes.CDLL(None,use_errno=True).syscall(434,pid,0))
     if fd>=0:fds.append(fd)
    except (OSError,KeyError,ValueError):pass
   if ev.wait(1800,fds):self.capacity_released()
   else:self.event('capacity_1800s_heartbeat',label=label)
  finally:
   ev.close()
   for fd in fds:os.close(fd)
  if (O/'STOP_AFTER_CURRENT_STAGE').exists():raise NoLongerAuthorized('Owner holds new work; preserve existing children')
 def train_to(self,arm,target):
  node=self.stage(arm,target);epoch=None
  if node.get('train_completed'):return node['checkpoint']
  while True:
   row=self.authorized(arm,target)
   if not node.get('train_job'):
    if node.get('resource_capacity_wait') and epoch is not None:self.wait_capacity(node,status='WAITING_TRAIN_CAPACITY',label=arm,arm=arm,since=epoch)
    if int(row.get('stage_budget_updates',0))==target and row.get('job') and not node.get('resource_requeue_pending'):
     node.update(train_job=row['job'],adopted_published_dispatch=True)
    else:
     # First12k must already be root-dispatched. Only proven75 pre-child
     # refusals allow creating a new12k receipt, never a duplicate model child.
     if target==TARGETS[0] and not node.get('previous_train_jobs'):raise ValueError('Root must first-dispatch12k; no implicit initial dispatch')
     try:node['train_job']=D.dispatch_training(R,arm,target,resume=self.state['runs'][arm].get('checkpoint'),source=self.runs[arm]['source'])
     except (BlockingIOError,PermissionError) as exc:
      node['capacity_reason']=str(exc);epoch=self.capacity_epoch;self.wait_capacity(node,status='WAITING_TRAIN_CAPACITY',label=arm,arm=arm,since=epoch);continue
    node.pop('resource_requeue_pending',None);self.flush()
   job=str(node['train_job']);self.update_arm(arm,status='TRAINING',stage_target_updates=target,job=job)
   result=D.wait_job(job);node['train_result']=result;self.flush();code=result_code(result)
   if code==75 and result.get('child_not_started') is True:
    D.wait_transport_exit(job);node.setdefault('previous_train_jobs',[]).append(dict(job=job,result=result));node.pop('train_job',None)
    node.update(resource_capacity_wait=True,resource_requeue_pending=True);epoch=self.capacity_epoch;self.flush();continue
   self.capacity_released()
   if code!=0:raise RuntimeError('Actual model child failure, no scientific retry: '+json.dumps(result)[:2000])
   checkpoint=native_checkpoint(self.runs[arm]['root'])
   if checkpoint['updates']<target:
    if (Path(job)/'STOP_REQUESTED').exists() or result.get('expiry_guard_requested') is True:
     D.wait_transport_exit(job);self.update_arm(arm,status='HELD_COOPERATIVE_NATIVE_BOUNDARY',updates=checkpoint['updates'],next_inner=checkpoint.get('next_inner'),checkpoint=checkpoint['path'],cooperative_stop_result=result)
     raise ArmHeldForResourceLease()
    raise ValueError('Clean job ended below target without declared cooperative stop')
   checkpoint=native_checkpoint(self.runs[arm]['root'],target)
   node.update(checkpoint=checkpoint['path'],train_completed=True,updates=target,train_completed_at=now())
   self.update_arm(arm,updates=target,checkpoint=checkpoint['path'])
   # Preserve an operator stop published while the model child was running.
   self.authorized(arm,target)
   self.update_run_registry(arm,completed_updates=target,last_checkpoint=checkpoint['path'],status='STAGE_TRAIN_COMPLETE');return checkpoint['path']
 def aggregate_priorities(self):
  # Scientific analysis is a separate finite event process; training never
  # blocks on scene reads/bootstrap or awaits an intermediate NAVTEST.
  return None
 def run_evaluation(self,label,checkpoint,*,cohort,node,baseline=None):
  if cohort=='TRAIN_DIAGNOSTIC':
   arm=label.rsplit('_u',1)[0];self.authorized(arm,CAP)
  return super().run_evaluation(label,checkpoint,cohort=cohort,node=node,baseline=baseline or (self.navbase if cohort=='NAVTEST' else self.baseline_future.result()['output']))
 def queue_final(self,arm,checkpoint):
  for cohort,suffix in [('NAVTEST','NAVTEST'),('TRAIN_DIAGNOSTIC','TRAIN1024')]:
   key=(arm,cohort);node=self.stage(arm,CAP).setdefault(cohort,{})
   if key in self.evaluation_futures:continue
   def evaluate(arm=arm,cohort=cohort,suffix=suffix,node=node,checkpoint=checkpoint):
    try:return self.run_evaluation(f'{arm}_u020000_{suffix}',checkpoint,cohort=cohort,node=node)
    except Exception:
     node.update(status='FAILED_REQUIRES_REVIEW',evaluation_error=traceback.format_exc());self.flush();return None
   self.evaluation_futures[key]=self.eval_pool.submit(evaluate)
 def run_one(self,arm):
  if self.state['runs'][arm]['status'] in ('FAILED_REQUIRES_REVIEW','HELD_COOPERATIVE_NATIVE_BOUNDARY','STUDY_REVIEW','HELD_BY_STOP_REQUEST'):return
  try:
   for target in TARGETS:
    row=self.authorized(arm,target);node=self.stage(arm,target)
    if not node.get('train_completed'):
     if target<int(row.get('stage_budget_updates',0)):
      cp=native_checkpoint(self.runs[arm]['root'],target);node.update(checkpoint=cp['path'],train_completed=True,updates=target,adopted_existing_checkpoint=True)
      self.update_arm(arm,updates=target,checkpoint=cp['path'])
     else:self.train_to(arm,target)
   self.queue_final(arm,self.stage(arm,CAP)['checkpoint']);self.update_arm(arm,status='EVALUATIONS_PENDING',updates=CAP)
   for (name,_),future in list(self.evaluation_futures.items()):
    if name==arm:future.result()
   final=self.stage(arm,CAP)
   if all(final[c].get('summary') for c in ('NAVTEST','TRAIN_DIAGNOSTIC')):
    self.update_arm(arm,status='STUDY_REVIEW',updates=CAP);self.update_run_registry(arm,status='STUDY_REVIEW',completed_updates=CAP)
   else:self.update_arm(arm,status='STUDY_EVALUATION_REQUIRES_REVIEW',updates=CAP)
  except NoLongerAuthorized as exc:self.update_arm(arm,status='HELD_BY_STOP_REQUEST',reason=str(exc))
  except ArmHeldForResourceLease:pass
  except Exception:
   error=traceback.format_exc();self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error);self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
 def run_study(self):
  self.event('native_study_adopted',arms=list(ARMS),targets=list(TARGETS),initial_global_updates=ORIGIN,cap=CAP,firstjobs={a:self.runs[a]['job'] for a in ARMS})
  try:
   with ThreadPoolExecutor(max_workers=4,thread_name_prefix='lr_native') as pool:list(pool.map(self.run_one,ARMS))
   self.state['status']='STUDY_REVIEW' if all(n['status']=='STUDY_REVIEW' for n in self.state['runs'].values()) else 'STUDY_REVIEW_WITH_FAILURE_OR_HOLD'
   self.flush();return self.state
  finally:self.eval_pool.shutdown(wait=True)


def bindings():return {str(p):sha(p) for p in [Path(__file__).resolve(),A/'navtest_cadence_owner_v2.py',SOURCE/'tools/qwen_world_research/campaign_owner.py',SOURCE/'tools/qwen_world_research/dispatch.py']}
def prepare():
 try:rows=registered();ready=True;reason=None
 except (ValueError,FileNotFoundError,KeyError) as e:rows={};ready=False;reason=str(e)
 value=dict(schema='LR_study_native_owner_launch_plan_v1',status='READY_FOR_ROOT_REVIEW' if ready else 'WAITING_ROOT_REGISTRATION_FIRST_DISPATCH',prepared_at=now(),started=False,
   script=str(Path(__file__).resolve()),bindings=bindings(),arms=list(ARMS),global_targets=list(TARGETS),cap=CAP,origin_updates=ORIGIN,
   launch_argv=[D.MODEL,str(Path(__file__).resolve()),'--launch'],owner=str(O/'OWNER.json'),state=str(O/'STATE.json'),launch=str(O/'LAUNCH.json'),
   adopt_only_first12000=True,ready_for_owner=ready,waiting_reason=reason,registered_firstjobs={a:r['job'] for a,r in rows.items()},
   native_parallel_arms=4,max_active_final_evaluations=2,NAVTEST20k='sharedCadenceDispatch up-to32; same-label locks/claims/frozenworld intact',
   NAVTEST12_18k='owned by already-running cadence, not submitted by thisowner',TRAIN20k='originaldispatcher,full1024diagnostic',
   evaluation_gates_do_not_block_native=True,scientific_cfg_modified=False,main_controller_or_CAMPAIGN_lock_used=False,auto50k=False)
 atomic_json(PLAN,value);return value

def launch():
 plan=read_json(PLAN);registered()
 if plan['bindings']!=bindings():raise ValueError('Reviewed LR owner dependency bindings changed; prepare/review again')
 O.mkdir(parents=True,exist_ok=True)
 with (O/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for old in (read_json(O/'OWNER.json',{}),read_json(O/'LAUNCH.json',{})):
   if not old:continue
   try:
    st=Path(f"/proc/{old['pid']}/stat").read_text().split()
    if int(st[21])==old['start_ticks'] and st[2] not in ('Z','X'):raise RuntimeError('Existing exactLR study owner')
   except FileNotFoundError:pass
  cmd=[D.MODEL,'-u',str(Path(__file__).resolve()),'--run']
  with (O/'stdout.log').open('ab',buffering=0) as log:proc=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(pid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().split()[21]),argv=cmd,source_sha256=sha(__file__),started_at=now(),plan=str(PLAN),bindings=plan['bindings'])
  atomic_json(O/'LAUNCH.json',record);print(json.dumps(record))


def self_check():
 import tempfile
 from unittest.mock import patch
 # Fake exact-boundary first job: adoption must not invoke GPU dispatch; sealed
 # targets remain global12k..20k, with retainedparent10k counter.
 with tempfile.TemporaryDirectory() as tmp:
  owner=object.__new__(StudyOwner);owner.runs={ARMS[0]:dict(root=tmp,source='frozen')};owner.state=dict(runs={ARMS[0]:dict(updates=10000,stages={})});owner.events=[];owner.mutex=threading.RLock()
  owner.authorized=lambda a,t:dict(stage_budget_updates=12000,job=str(Path(tmp)/'root_firstjob'))
  owner.flush=lambda:None;owner.capacity_released=lambda:None;owner.update_arm=lambda a,**v:owner.state['runs'][a].update(v);owner.update_run_registry=lambda *a,**v:None
  def cp(root,expected=None):return dict(path=str(Path(tmp)/'checkpoint12000'),updates=12000,next_inner=0)
  with patch.object(D,'wait_job',return_value=dict(returncode=0)),patch.object(D,'dispatch_training',side_effect=AssertionError('duplicate initialjob')),patch(__name__+'.native_checkpoint',side_effect=cp):
   assert owner.train_to(ARMS[0],12000).endswith('checkpoint12000')
  assert owner.stage(ARMS[0],12000)['adopted_published_dispatch'] is True
 assert len(ARMS)==4 and TARGETS==(12000,14000,16000,18000,20000) and CAP==20000
 assert terminal(dict(status='STOPPED_FUTILITY')) and not terminal(dict(status='STAGE_TRAIN_COMPLETE'))
 result=dict(status='PASS',at=now(),script_sha256=sha(__file__),scope='CPU mockfirst12k adoption/no duplicate dispatch; globalcaps4arm; stopguard; noGPU/model/probe')
 atomic_json(A/'LR_STUDY_OWNER_CPU_CHECK_20261031.json',result);return result


def main():
 p=argparse.ArgumentParser(__doc__);g=p.add_mutually_exclusive_group(required=True)
 for arg in ('prepare','self-check','launch','run'):g.add_argument('--'+arg,action='store_true')
 q=p.parse_args()
 if q.prepare:print(json.dumps(prepare(),indent=2))
 elif q.self_check:print(json.dumps(self_check()))
 elif q.launch:launch()
 else:print(json.dumps(StudyOwner().run_study(),default=str))
if __name__=='__main__':main()
