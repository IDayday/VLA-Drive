"""Independent warmstarted Rescue pair owner; adopt first1k, cap5k, then review.

Never owns CAMPAIGN lock or edits its code/state. Ordinary dispatch APIs are used
only after root registers/dispatches both initial jobs and explicitly launches
this owner. No native/GPU kill; STOP_REQUESTED is a native cooperative boundary.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,select,subprocess,sys,threading,time,traceback,ctypes
from concurrent.futures import ThreadPoolExecutor,Future,as_completed
from pathlib import Path
import yaml
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A));SOURCE=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030');sys.path.insert(0,str(SOURCE))
from tools.qwen_world_research.campaign_owner import CampaignOwner,native_checkpoint,read_json,atomic_json,now,result_code,ArmHeldForResourceLease
from tools.qwen_world_research import dispatch
from finite_matched12k_analyzer import MetadataEvents
from rescue_pair_analyzer import analyze_ready
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');ARMS=('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM');O=R/'owners/RESCUE_PAIR'

def registered():
 rows={x['id']:x for x in read_json(R/'RUNS.json')['runs']};out={}
 for arm in ARMS:
  if arm not in rows:raise ValueError('Root must register and first-dispatch '+arm)
  r=rows[arm];c=yaml.safe_load(Path(r['config']).read_text());q=c['research'];s=c['grpo']['sample']
  if (r.get('module')!='starVLA.rl.qwen_world_research_algorithms.rescue_entry' or not r.get('job')
   or int(r.get('stage_budget_updates',0)) not in (1000,5000) or int(r.get('research_budget_updates',5000))!=5000
   or hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()!=r['config_sha256'] or q['arm']!=arm or c['seed']!=42
   or q['warmstart']['schema']!='world_verified_RL_weights_only_warmstart_v1' or q['warmstart']['expected_updates']!=10000
   or s['noise_level']!=.1 or s['sde_step_indices']!=[7,8] or c['grpo']['train']['kl_beta']!=50):raise ValueError('Rescue warmstart/W2/seed/cap registration changed '+arm)
  out[arm]=r
 return out

class PairOwner(CampaignOwner):
 def __init__(self):
  self.root=R;self.owner_root=O;O.mkdir(parents=True,exist_ok=True)
  self.lock=(O/'owner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.mutex=threading.RLock();self.capacity_changed=threading.Condition(self.mutex);self.capacity_epoch=0
  self.dispatcher=dispatch;self.resource_retries=2;self.runs=registered()
  self.parent=str(R/'IL_PARENT');self.data=str(R/'data/NAVTRAIN');self.panel=str(R/'data/panels/TRAIN_DIAGNOSTIC_001024.json');self.navtest_data='/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST';self.navtest_manifest=str(R/'data/panels/NAVTEST_012146.json')
  self.state=read_json(O/'STATE.json',{})
  if self.state and self.state.get('schema')!='independent_rescue_pair_owner_v1':raise ValueError('Different pair state')
  self.state.setdefault('schema','independent_rescue_pair_owner_v1');self.state.setdefault('runs',{});self.state.setdefault('events',[]);self.state.setdefault('results',[])
  self.state.update(status='RUNNING',budget_cap_phase_updates=5000,initial_RL_knowledge_updates=10000,automatic_promotion=False,main_campaign_lock_used=False)
  for arm,r in self.runs.items():
   n=self.state['runs'].setdefault(arm,dict(id=arm,updates=0,stages={},status='ADOPTING_ROOT_FIRST_JOB'))
   n.update(run_root=r['root'],source=r['source'],research_budget_updates=5000,initial_RL_knowledge_updates=10000,phase_updates_are_additional=True)
  self.eval_pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix='rescue_eval');self.feedback_pool=ThreadPoolExecutor(max_workers=2);self.evaluation_futures={}
  self.baseline_future=Future();base=read_json(R/'evaluation_specs/PPO_SAFETY_EP_u010000_TRAIN1024.json');self.baseline_future.set_result(dict(output=base['output'],training_bank=base['training_bank']))
  self.navbase=read_json(R/'evaluation_specs/PPO_SAFETY_EP_u010000_NAVTEST.json')['output']
  atomic_json(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source_file=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started_at=now(),scope='independent Rescue pair,phase1k/5k only,adopt existing firstjobs'))
  self.flush()
 def run_evaluation(self,label,checkpoint,*,cohort,node,baseline=None):
  return super().run_evaluation(label,checkpoint,cohort=cohort,node=node,baseline=baseline or (self.navbase if cohort=='NAVTEST' else self.baseline_future.result()['output']))
 def aggregate_priorities(self):
  try:analyze_ready()
  except Exception:self.event('matched_analysis_requires_review',error=traceback.format_exc()[-2000:])
 def wait_capacity(self,node,*,status,label,arm=None,since=None):
  if since is not None and since!=self.capacity_epoch:return
  node['status']=status
  if arm:self.state['runs'][arm]['status']=status
  self.event('waiting_real_capacity',label=label,status=status)
  paths=set()
  for r in read_json(R/'RUNS.json')['runs']:
   if r.get('job'):paths.add(r['job'])
  for r in read_json(R/'resources/EVALUATION_CLAIMS.json',{}).values():
   if r.get('receipt'):paths.add(r['receipt'])
  fds=[];events=MetadataEvents()
  try:
   for p in (R,R/'resources',R/'evaluation_specs'):events.add(p)
   for path in paths:
    if (Path(path)/'RESULT.json').exists():continue
    launch=read_json(Path(path)/'LAUNCH.json',{})
    try:
     pid=launch['transport_pid'];s=Path(f'/proc/{pid}/stat').read_text().split()
     if int(s[21])!=launch['transport_start_ticks'] or s[2] in ('Z','X'):continue
     fd=os.pidfd_open(pid) if hasattr(os,'pidfd_open') else ctypes.CDLL(None,use_errno=True).syscall(434,pid,0)
     if fd>=0:fds.append(fd)
    except (OSError,KeyError,ValueError):pass
   ready,_,_=select.select([events.fd,*fds],[],[],1800)
   if ready:self.capacity_released()
   else:self.event('capacity_1800s_heartbeat',label=label)
  finally:
   for fd in fds:os.close(fd)
   events.close()
 def train_to(self,arm,target,*,resume=None,adopt_first=False):
  node=self.stage(arm,target);epoch=None
  if node.get('train_completed'):return node['checkpoint']
  while True:
   if not node.get('train_job'):
    if node.get('resource_capacity_wait') and epoch is not None:self.wait_capacity(node,status='WAITING_TRAIN_CAPACITY',label=arm,arm=arm,since=epoch)
    current=next(x for x in read_json(R/'RUNS.json')['runs'] if x['id']==arm)
    if current.get('stage_budget_updates')==target and current.get('job') and not node.get('resource_requeue_pending'):
     node['train_job']=current['job'];node['adopted_published_dispatch']=True
    else:
     try:node['train_job']=dispatch.dispatch_training(R,arm,target,resume=self.state['runs'][arm].get('checkpoint') or resume,source=self.runs[arm]['source'])
     except (BlockingIOError,PermissionError) as error:
      node['capacity_reason']=str(error);epoch=self.capacity_epoch;self.wait_capacity(node,status='WAITING_TRAIN_CAPACITY',label=arm,arm=arm,since=epoch);continue
    node.pop('resource_requeue_pending',None);self.flush()
   job=str(node['train_job']);self.update_arm(arm,status='TRAINING',phase_updates=self.state['runs'][arm].get('updates',0),stage_target_updates=target,job=job)
   result=dispatch.wait_job(job);node['train_result']=result;self.flush();code=result_code(result)
   if code==75 and result.get('child_not_started') is True:
    dispatch.wait_transport_exit(job);node.setdefault('previous_train_jobs',[]).append(dict(job=job,result=result));node.pop('train_job',None);node['resource_capacity_wait']=True;node['resource_requeue_pending']=True;epoch=self.capacity_epoch;self.flush();continue
   self.capacity_released()
   if code!=0:raise RuntimeError('Actual model child failure, no scientific retry: '+json.dumps(result)[:2000])
   checkpoint=native_checkpoint(self.runs[arm]['root'])
   if checkpoint['updates']<target:
    if (Path(job)/'STOP_REQUESTED').exists() or result.get('expiry_guard_requested') is True:
     dispatch.wait_transport_exit(job);self.update_arm(arm,status='HELD_COOPERATIVE_NATIVE_BOUNDARY',updates=checkpoint['updates'],phase_updates=checkpoint['updates'],next_inner=checkpoint.get('next_inner'),checkpoint=checkpoint['path'],cooperative_stop_result=result);raise ArmHeldForResourceLease()
    raise ValueError('Clean job ended below target without a declared cooperative stop')
   checkpoint=native_checkpoint(self.runs[arm]['root'],target);node.update(checkpoint=checkpoint['path'],train_completed=True,updates=target,train_completed_at=now());self.update_arm(arm,updates=target,phase_updates=target,checkpoint=checkpoint['path']);self.update_run_registry(arm,completed_updates=target,last_checkpoint=checkpoint['path'],status='STAGE_TRAIN_COMPLETE');return checkpoint['path']
 def queue_boundary(self,arm,u,checkpoint):
  for cohort,suffix,baseline in [('TRAIN_DIAGNOSTIC','TRAIN1024',self.baseline_future.result()['output']),('NAVTEST','NAVTEST',self.navbase)]:
   key=(arm,u,cohort)
   if key in self.evaluation_futures:continue
   node=self.stage(arm,u).setdefault(cohort,{})
   def evaluate(node=node,cohort=cohort,suffix=suffix,baseline=baseline,u=u):
    try:
     value=self.run_evaluation(f'{arm}_u{u:06}_{suffix}',checkpoint,cohort=cohort,node=node,baseline=baseline);node.update(status='COMPLETE',summary=value);self.flush();self.aggregate_priorities();return value
    except Exception:
     node.update(status='FAILED_REQUIRES_REVIEW',evaluation_error=traceback.format_exc());self.flush();raise
   self.evaluation_futures[key]=self.eval_pool.submit(evaluate)
 def run_one(self,arm):
  if self.state['runs'][arm].get('status') in ('FAILED_REQUIRES_REVIEW','HELD_COOPERATIVE_NATIVE_BOUNDARY','PILOT_REVIEW'):return
  try:
   for target in (1000,5000):
    if (O/'STOP_AFTER_CURRENT_STAGE').exists():self.update_arm(arm,status='HELD_BY_STOP_REQUEST');return
    node=self.stage(arm,target)
    if not node.get('train_completed'):
     current=next(x for x in read_json(R/'RUNS.json')['runs'] if x['id']==arm)
     if target<int(current.get('stage_budget_updates',0)):
      cp=native_checkpoint(self.runs[arm]['root'],target);node.update(checkpoint=cp['path'],train_completed=True,updates=target);self.flush()
     else:self.train_to(arm,target,resume=self.state['runs'][arm].get('checkpoint'),adopt_first=target==1000)
    self.queue_boundary(arm,target,node['checkpoint'])
   self.update_arm(arm,status='EVALUATIONS_PENDING')
   for key,f in list(self.evaluation_futures.items()):
    if key[0]==arm:f.result()
   self.update_arm(arm,status='PILOT_REVIEW',updates=5000,phase_updates=5000);self.update_run_registry(arm,status='PILOT_REVIEW',completed_updates=5000)
  except ArmHeldForResourceLease:pass
  except Exception:
   error=traceback.format_exc();self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error);self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
 def run_pair(self):
  self.event('pair_adopted',arms=list(ARMS),firstjobs={a:self.runs[a]['job'] for a in ARMS},cap=5000,baseline='Safety10k')
  try:
   with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(self.run_one,ARMS))
   self.aggregate_priorities();self.state['status']='PILOT_REVIEW' if all(self.state['runs'][a]['status']=='PILOT_REVIEW' for a in ARMS) else 'PILOT_REVIEW_WITH_FAILURE_OR_HOLD';self.flush();return self.state
  finally:self.eval_pool.shutdown(wait=True);self.feedback_pool.shutdown(wait=True)

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--launch',action='store_true');a=p.parse_args();registered()
 if a.launch:
  O.mkdir(exist_ok=True);owner=read_json(O/'OWNER.json',{})
  if owner:
   stat=Path(f"/proc/{owner['pid']}/stat")
   if stat.exists() and int(stat.read_text().split()[21])==owner['start_ticks']:raise RuntimeError('Existing exact pair owner')
  cmd=[sys.executable,'-u',str(Path(__file__).resolve())]
  with (O/'stdout.log').open('ab',buffering=0) as f:child=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=cmd,cap_phase_updates=5000,scope='exact initialjobs adoption only; native/eval jobs preserved',started_at=now());atomic_json(O/'LAUNCH.json',record);print(json.dumps(record));return
 print(json.dumps(PairOwner().run_pair(),default=str))
if __name__=='__main__':main()
