"""Independent freshIL70 Safety SuccessMemory/FPO owner, capped10k.

Root dispatches1k; owner adopts exactly, resumes5k/10k. NAVTEST andTRAIN
are asynchronous atall3gates. No oldRLarm/optimizerwarmstart orCAMPAIGNlock.
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
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');O=R/'owners/SAFETY_FRESH_PAIR'
ARMS=('PPO_SAFETY_SUCCESS_MEMORY','PPO_SAFETY_FPO')
MODULES={ARMS[0]:'starVLA.rl.qwen_world_research_algorithms.success_memory_entry',ARMS[1]:'starVLA.rl.qwen_world_research_algorithms.fpo_entry'}
REGISTRATION=R/'EXPERIMENT_SAFETY_FRESH_PAIR_REGISTRATION_20261031.json'
TARGETS=(1000,5000,10000);CAP=10000;ORIGIN=0
SCHEMA='independent_fresh_IL70_Safety_success_FPO_pair_owner_v1'
PLAN=A/'SAFETY_FRESH_PAIR_OWNER_LAUNCH_PLAN_20261031.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def terminal(row):return any(s in str(row.get('status','')).upper() for s in ('STOP','FAIL','CANCEL','HELD','PAUSED','DEFER'))
def registration_rows():
 registration=read_json(REGISTRATION)
 declared=registration.get('arms',registration.get('runs',[]))
 if isinstance(declared,dict):declared=[dict(v,arm=k) for k,v in declared.items()]
 out={x if isinstance(x,str) else x.get('arm',x.get('id')):x for x in declared}
 if set(out)!=set(ARMS):raise ValueError('Onlythe2registeredfreshSafetyarmsareauthorized')
 if terminal(registration):raise NoLongerAuthorized('Freshpairregistrationstopped/held')
 return out

def registered():
 registered_arms=registration_rows();rows={x['id']:x for x in read_json(R/'RUNS.json')['runs']};out={}
 for arm in ARMS:
  if arm not in rows:raise ValueError('Root must register and first-dispatch '+arm)
  row=rows[arm];cfg=yaml.safe_load(Path(row['config']).read_text());q=cfg['research'];s=cfg['grpo']['sample'];t=cfg['grpo']['train']
  proof=registered_arms[arm]
  if isinstance(proof,dict) and proof.get('config_sha256') not in (None,row['config_sha256']):raise ValueError('Rootregistrationcfgbindingchanged')
  if (row.get('module')!=MODULES[arm] or not row.get('job') or not Path(row['source']).is_dir()
      or int(row.get('stage_budget_updates',0)) not in TARGETS or int(row.get('research_budget_updates',0))!=CAP
      or sha(row['config'])!=row['config_sha256'] or q.get('arm')!=arm or cfg.get('seed')!=42
      or cfg.get('max_steps')!=50000 or cfg.get('lr_scheduler_type')!='constant'
      or q.get('data_policy')!='uniform_full' or q.get('policy_objective')!='ppo'
      or q.get('structured_advantage')!='candidate_metric_safety_feasible_EP_global_std_v1'
      or cfg.get('learning_rate')!=5e-5 or cfg.get('memory_learning_rate')!=5e-6
      or cfg.get('lr_warmup_steps')!=100 or cfg.get('world_size')!=8
      or any(q.get(k) is not None for k in ('native_fork','warmstart','lr_schedule','sampler_ablation','noise_level_ablation'))
      or s.get('sde_step_indices') is not None or s.get('train_sde_steps')!=3
      or s.get('sde_exclude_last_step') is not False or s.get('noise_level')!=.1 or t.get('kl_beta')!=50):
   raise ValueError('FreshIL70/W3/Safety/anchor/data/LR/capregistrationchanged '+arm)
  out[arm]=dict(row)
 return out


class StudyDispatch:
 def __init__(self,owner):self.owner=owner;self.nav=CadenceDispatch(owner)
 def __getattr__(self,name):return getattr(D,name)
 def dispatch_evaluation(self,**args):
  arm,target,suffix=self.owner.parse_label(args['label'])
  self.owner.authorized(arm,target)
  if args['cohort']=='NAVTEST':return self.nav.dispatch_evaluation(**args)
  if suffix!='TRAIN1024' or args['cohort']!='TRAIN_DIAGNOSTIC':raise NoLongerAuthorized('Unregisteredpaircohort')
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
  if self.state and self.state.get('schema')!=SCHEMA:raise ValueError('Differentfreshpair ownerstate')
  self.state.setdefault('schema',SCHEMA);self.state.setdefault('runs',{});self.state.setdefault('events',[]);self.state.setdefault('results',[])
  self.state.update(status='RUNNING_FRESH_IL70_SAFETY_PAIR',initial_global_updates=ORIGIN,global_budget_cap=CAP,
      fresh_IL70_Adam_RNG_seed42=True,initial_phase_update0=True,main_campaign_lock_used=False,training_waits_for_intermediate_evaluation=False,
      targets=list(TARGETS),owner_evaluation_gates=list(TARGETS),final_native_status='COMPLETED_AT_REGISTERED_10K',scientific_registration=str(REGISTRATION),old_arms_revivable=False,automatic_scientific_early_stop=False)
  for arm,row in self.runs.items():
   n=self.state['runs'].setdefault(arm,dict(id=arm,updates=ORIGIN,stages={},status='ADOPTING_ROOT_FIRST1000_JOB'))
   n.update(run_root=row['root'],source=row['source'],research_budget_updates=CAP,initial_global_updates=ORIGIN,
       fresh_from_IL=True,RL_checkpoint_warmstart=False,additional_updates=int(n['updates'])-ORIGIN)
  self.eval_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='freshpair_eval');self.evaluation_futures={}
  self.review_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='freshpair_CPUreview');self.review_futures=[]
  self.baseline_future=Future();base=read_json(R/'IL_BASELINE_EVALUATION.json')
  self.baseline_future.set_result(dict(output=base['output'],training_bank=base['training_bank']))
  atomic_json(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),started_at=now(),
      source_file=str(Path(__file__).resolve()),source_sha256=sha(__file__),scope='CPU owner; exactrootfirst1k adoption; freshIL70 SuccessMemory/FPO native1k->5k->10k; nootherRLarms'))
  self.flush()
 def update_arm(self,arm,**values):
  if 'updates' in values:values['additional_updates']=values['updates']-ORIGIN
  super().update_arm(arm,**values)
 def authorized(self,arm,target):
  if arm not in ARMS:raise NoLongerAuthorized('No otheroldornewarm isauthorized')
  registration_rows()
  row=next(x for x in read_json(R/'RUNS.json')['runs'] if x['id']==arm);binding=self.runs[arm]
  if terminal(row) or (O/'STOP_AFTER_CURRENT_STAGE').exists():raise NoLongerAuthorized('No newfreshpairwork afterstop/hold: '+arm)
  if target not in TARGETS or int(row.get('research_budget_updates',0))!=CAP:raise NoLongerAuthorized('Freshpair10kcap/targetchanged')
  if sha(row['config'])!=row['config_sha256'] or any(row[k]!=binding[k] for k in ('root','source','module','config','config_sha256')):raise ValueError('Registered scientific binding changed '+arm)
  return row
 def parse_label(self,label):
  try:arm,rest=label.rsplit('_u',1);update,suffix=rest.split('_',1);target=int(update)
  except (ValueError,TypeError):raise NoLongerAuthorized('Invalidfreshpairevaluationlabel')
  if arm not in ARMS or target not in TARGETS or suffix not in ('NAVTEST','TRAIN1024') or label!=f'{arm}_u{target:06}_{suffix}':raise NoLongerAuthorized('Notthispair/gate/cohort')
  return arm,target,suffix
 def require_allowed(self,label):
  arm,target,suffix=self.parse_label(label)
  if suffix!='NAVTEST':raise NoLongerAuthorized('NAVadapteronly')
  self.authorized(arm,target)
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
     # First1k must already be root-dispatched. Only proven75 pre-child
     # refusals allow creating a new1k receipt, never a duplicate model child.
     if target==TARGETS[0] and not node.get('previous_train_jobs'):raise ValueError('Root must first-dispatch1k; no implicit initial dispatch')
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
  def review():
   try:
    from safety_fresh_pair_review import analyze_ready
    out=analyze_ready()
    with self.mutex:self.state['matched_review']=dict(path=str(A/'SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json'),status=out['status'],progress=out['progress']);self.flush()
   except Exception:self.event('CPU_review_requires_attention',error=traceback.format_exc()[-2500:])
  self.review_futures.append(self.review_pool.submit(review))
 def run_evaluation(self,label,checkpoint,*,cohort,node,baseline=None):
  arm,target,suffix=self.parse_label(label);self.authorized(arm,target)
  if baseline is None:
   name=f'PPO_SAFETY_EP_u{target:06}_'+('NAVTEST' if cohort=='NAVTEST' else 'TRAIN1024')
   baseline=read_json(R/'evaluation_specs'/(name+'.json'),{}).get('output')
   if baseline is None and cohort=='TRAIN_DIAGNOSTIC':baseline=self.baseline_future.result()['output']
  return super().run_evaluation(label,checkpoint,cohort=cohort,node=node,baseline=baseline)
 def queue_boundary(self,arm,target,checkpoint):
  for cohort,suffix in [('NAVTEST','NAVTEST'),('TRAIN_DIAGNOSTIC','TRAIN1024')]:
   key=(arm,target,cohort)
   with self.mutex:
    node=self.stage(arm,target).setdefault(cohort,{})
    if key in self.evaluation_futures:continue
    def evaluate(arm=arm,target=target,cohort=cohort,suffix=suffix,node=node,checkpoint=checkpoint):
     try:
      if node.get('evaluation_error'):raise ValueError('Previousactualevaluationfailurerequiresreview')
      result=self.run_evaluation(f'{arm}_u{target:06}_{suffix}',checkpoint,cohort=cohort,node=node)
      node['status']='COMPLETE';self.flush();self.aggregate_priorities();return result
     except NoLongerAuthorized as exc:node.update(status='HELD_BY_STOP_REQUEST',reason=str(exc));self.flush();return None
     except Exception:
      node.update(status='FAILED_REQUIRES_REVIEW',evaluation_error=traceback.format_exc());self.flush()
      self.event('asynchronous_evaluation_failed_no_scientific_retry',arm=arm,target=target,cohort=cohort,error=node['evaluation_error'][-1500:]);return None
    self.evaluation_futures[key]=self.eval_pool.submit(evaluate)
 def run_one(self,arm):
  if self.state['runs'][arm]['status'] in ('FAILED_REQUIRES_REVIEW','HELD_COOPERATIVE_NATIVE_BOUNDARY','HELD_BY_STOP_REQUEST'):return
  try:
   for target in TARGETS:
    row=self.authorized(arm,target);node=self.stage(arm,target)
    if not node.get('train_completed'):
     if target<int(row.get('stage_budget_updates',0)):
      cp=native_checkpoint(self.runs[arm]['root'],target);node.update(checkpoint=cp['path'],train_completed=True,updates=target,adopted_existing_checkpoint=True);self.update_arm(arm,updates=target,checkpoint=cp['path'])
     else:self.train_to(arm,target)
    self.queue_boundary(arm,target,self.stage(arm,target)['checkpoint'])
    # No1k/5kreview/evaluation barrier here; nextnativejob may startnow.
   self.update_arm(arm,status='COMPLETED_AT_REGISTERED_10K',training_status='COMPLETED_AT_REGISTERED_10K',updates=CAP,evaluation_status='PENDING_REGISTERED_GATES',budget_wait=False)
   self.update_run_registry(arm,status='COMPLETED_AT_REGISTERED_10K',completed_updates=CAP)
   with self.mutex:futures=[v for (a,_,_),v in self.evaluation_futures.items() if a==arm]
   for future in futures:future.result()
   complete=all(self.stage(arm,u).get(c,{}).get('summary') for u in TARGETS for c in ('NAVTEST','TRAIN_DIAGNOSTIC'))
   self.update_arm(arm,evaluation_status='ALL_REGISTERED_EVALUATIONS_COMPLETE' if complete else 'REQUIRES_EVALUATION_REVIEW')
   self.aggregate_priorities()
  except NoLongerAuthorized as exc:self.update_arm(arm,status='HELD_BY_STOP_REQUEST',reason=str(exc))
  except ArmHeldForResourceLease:pass
  except Exception:
   error=traceback.format_exc();self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error);self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
 def run_study(self):
  self.event('freshpair_adopted',arms=list(ARMS),targets=list(TARGETS),initial_updates=0,cap=CAP,registration=str(REGISTRATION),firstjobs={a:self.runs[a]['job'] for a in ARMS})
  try:
   with ThreadPoolExecutor(max_workers=2,thread_name_prefix='freshpair_native') as pool:list(pool.map(self.run_one,ARMS))
   self.eval_pool.shutdown(wait=True);self.aggregate_priorities();self.review_pool.shutdown(wait=True)
   alltrained=all(n.get('training_status')=='COMPLETED_AT_REGISTERED_10K' for n in self.state['runs'].values())
   self.state['status']='COMPLETED_AT_REGISTERED_10K' if alltrained else 'FINISHED_WITH_FAILURE_OR_COOPERATIVE_HOLD'
   self.state['budget_wait']=False;self.flush();return self.state
  finally:
   self.eval_pool.shutdown(wait=True);self.review_pool.shutdown(wait=True)


def bindings():return {str(p):sha(p) for p in [Path(__file__).resolve(),A/'navtest_cadence_owner_v2.py',A/'safety_fresh_pair_review.py',SOURCE/'tools/qwen_world_research/campaign_owner.py',SOURCE/'tools/qwen_world_research/dispatch.py']}
def prepare():
 try:rows=registered();ready=True;reason=None
 except (ValueError,FileNotFoundError,KeyError,NoLongerAuthorized) as e:rows={};ready=False;reason=str(e)
 value=dict(schema='fresh_IL70_Safety_pair_owner_launch_plan_v1',status='READY_FOR_ROOT_REVIEW' if ready else 'WAITING_ROOT_REGISTRATION_FIRST_DISPATCH',prepared_at=now(),started=False,
   script=str(Path(__file__).resolve()),bindings=bindings(),arms=list(ARMS),targets=list(TARGETS),cap=CAP,origin_updates=0,registration=str(REGISTRATION),
   launch_argv=[D.MODEL,str(Path(__file__).resolve()),'--launch'],owner=str(O/'OWNER.json'),state=str(O/'STATE.json'),launch=str(O/'LAUNCH.json'),
   adopt_only_first1000=True,ready_for_owner=ready,waiting_reason=reason,registered_firstjobs={a:r['job'] for a,r in rows.items()},
   native_parallel_arms=2,max_active_model_evaluations=2,evaluations='NAVTESTup-to32sharedCadenceDispatch+TRAIN1024 ateach1k/5k/10k; frozenexistingworldadopted',
   baseline='samebudgetoriginalSafetyEP; sharedILTRAINbaseline andfixedILhighcohort',
   evaluation_and_review_do_not_block_native=True,scientific_cfg_modified=False,main_controller_or_CAMPAIGN_lock_used=False,
   finite_end='COMPLETED_AT_REGISTERED_10K',budget_wait=False,automatic_early_stop=False,early_stop='CPU evidenceonly forroot; noextraeffectmargin guessed')
 atomic_json(PLAN,value);return value

def launch():
 plan=read_json(PLAN);registered()
 if plan['bindings']!=bindings():raise ValueError('Reviewed freshpair dependencybindingschanged; prepare/review again')
 O.mkdir(parents=True,exist_ok=True)
 with (O/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for old in (read_json(O/'OWNER.json',{}),read_json(O/'LAUNCH.json',{})):
   if not old:continue
   try:
    st=Path(f"/proc/{old['pid']}/stat").read_text().split()
    if int(st[21])==old['start_ticks'] and st[2] not in ('Z','X'):raise RuntimeError('Existingfreshpair owner')
   except FileNotFoundError:pass
  cmd=[D.MODEL,'-u',str(Path(__file__).resolve()),'--run']
  with (O/'stdout.log').open('ab',buffering=0) as log:proc=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(pid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().split()[21]),argv=cmd,source_sha256=sha(__file__),started_at=now(),plan=str(PLAN),bindings=plan['bindings'])
  atomic_json(O/'LAUNCH.json',record);print(json.dumps(record))


def self_check():
 import tempfile
 from unittest.mock import patch
 with tempfile.TemporaryDirectory() as tmp:
  owner=object.__new__(StudyOwner);owner.runs={ARMS[0]:dict(root=tmp,source='root-frozen-source')};owner.state=dict(runs={ARMS[0]:dict(updates=0,stages={})});owner.mutex=threading.RLock()
  owner.authorized=lambda a,t:dict(stage_budget_updates=1000,job=str(Path(tmp)/'root_first1000'))
  owner.flush=lambda:None;owner.capacity_released=lambda:None;owner.update_arm=lambda a,**v:owner.state['runs'][a].update(v);owner.update_run_registry=lambda *a,**v:None
  with patch.object(D,'wait_job',return_value=dict(returncode=0)),patch.object(D,'dispatch_training',side_effect=AssertionError('duplicaterootfirstjob')),patch(__name__+'.native_checkpoint',return_value=dict(path=str(Path(tmp)/'checkpoint1000'),updates=1000,next_inner=0)):
   assert owner.train_to(ARMS[0],1000).endswith('checkpoint1000')
  assert owner.stage(ARMS[0],1000)['adopted_published_dispatch'] is True
  assert owner.parse_label(ARMS[1]+'_u005000_NAVTEST')==(ARMS[1],5000,'NAVTEST')
  for bad in ('PPO_FULL_u001000_NAVTEST',ARMS[0]+'_u020000_NAVTEST'):
   try:owner.parse_label(bad)
   except NoLongerAuthorized:pass
   else:raise AssertionError('Old/unregisteredarm/gate admitted')
 # Asynchronous boundary jobs must notblock5k/10knative. Future.result
 # asserts allnativegateswerevisited beforeany evaluationwaitoccurs.
 owner=object.__new__(StudyOwner);owner.mutex=threading.RLock();owner.state=dict(runs={ARMS[0]:dict(status='ADOPTING_ROOT_FIRST1000_JOB',updates=0,stages={})});owner.evaluation_futures={};seen=[]
 owner.authorized=lambda a,t:dict(stage_budget_updates=1000)
 owner.update_arm=lambda a,**v:owner.state['runs'][a].update(v);owner.update_run_registry=lambda *a,**v:None;owner.aggregate_priorities=lambda:None
 def train(a,t):
  seen.append(t);owner.stage(a,t).update(train_completed=True,checkpoint='cp'+str(t));owner.update_arm(a,updates=t);return 'cp'+str(t)
 owner.train_to=train
 class Pending:
  def result(self):assert seen==list(TARGETS)
 def queue(a,t,cp):
  for cohort in ('NAVTEST','TRAIN_DIAGNOSTIC'):
   owner.stage(a,t).setdefault(cohort,{})['summary']=dict(status='COMPLETE')
   owner.evaluation_futures[(a,t,cohort)]=Pending()
 owner.queue_boundary=queue;owner.run_one(ARMS[0])
 assert seen==list(TARGETS) and owner.state['runs'][ARMS[0]]['status']=='COMPLETED_AT_REGISTERED_10K'
 from safety_fresh_pair_review import early_stop_evidence
 pair=dict(delta_PDMS=-1.1,CI95_PDMS=[-1.4,-.8],fixed_IL70_high_ge09=dict(CI95_PDMS=[-1.3,-.2]))
 assert early_stop_evidence(pair)['qualifies_for_root_review'] and not early_stop_evidence(pair)['automatic_stop']
 pair['CI95_PDMS']=[-1.4,.1];assert not early_stop_evidence(pair)['qualifies_for_root_review']
 assert len(ARMS)==2 and TARGETS==(1000,5000,10000) and CAP==10000 and ORIGIN==0
 assert terminal(dict(status='STOPPED_USER_SCOPE_CHANGE')) and not terminal(dict(status='COMPLETED_AT_REGISTERED_10K'))
 result=dict(status='PASS',at=now(),script_sha256=sha(__file__),scope='CPU exactfirst1kadoption/noGPUdispatch; noevalbarrierbefore5k/10k; only2newarms/finite10k/stopguard; earlystopCPUevidenceonly; noGPU/model')
 atomic_json(A/'SAFETY_FRESH_PAIR_OWNER_CPU_CHECK_20261031.json',result);return result


def main():
 p=argparse.ArgumentParser(__doc__);g=p.add_mutually_exclusive_group(required=True)
 for arg in ('prepare','self-check','launch','run'):g.add_argument('--'+arg,action='store_true')
 q=p.parse_args()
 if q.prepare:print(json.dumps(prepare(),indent=2))
 elif q.self_check:print(json.dumps(self_check()))
 elif q.launch:launch()
 else:print(json.dumps(StudyOwner().run_study(),default=str))
if __name__=='__main__':main()
