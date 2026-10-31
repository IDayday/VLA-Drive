"""Finite metadata-event futility guard for exactly two freshSafety pilots.

Root-authorized conjunction: samearm1k AND5k actual-complete samebudgetSafety
PDMSdelta<-1,paired95CIupper<0 ANDfixedIL70highcohort95CIupper<0 atBOTH.
Only co-operative STOP_REQUESTED; no process signals, GPU, scoring or evalcancel.
"""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,math,os,select,subprocess,sys,time,uuid,traceback
from datetime import datetime,timezone
from pathlib import Path
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from navtest_cadence_owner_v2 import Events
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');O=R/'owners/SAFETY_FRESH_PAIR_FUTILITY_GUARD'
REVIEW=R/'analysis/SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json'
REG=R/'EXPERIMENT_SAFETY_FRESH_PAIR_REGISTRATION_20261031.json'
PAIR=R/'owners/SAFETY_FRESH_PAIR';PLAN=A/'SAFETY_FRESH_PAIR_FUTILITY_GUARD_LAUNCH_PLAN_20261031.json'
ARMS=('PPO_SAFETY_SUCCESS_MEMORY','PPO_SAFETY_FPO')
MODULES={ARMS[0]:'starVLA.rl.qwen_world_research_algorithms.success_memory_entry',ARMS[1]:'starVLA.rl.qwen_world_research_algorithms.fpo_entry'}
REVIEW_SCHEMA='world_fresh_IL70_Safety_success_FPO_matched_review_v1'
RULE_SCHEMA='world_freshSafety_two_checkpoint_futility_rule_v1'
STATE_SCHEMA='world_finite_freshSafety_evidence_futility_guard_v1'
PARENT_SHA='7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40'

def now():return datetime.now(timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.writing-'+str(os.getpid()))
 with q.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)

def sealed(p):
 v=read(p);b=dict(v);s=b.pop('sha256',None)
 if s!=fp(b):raise ValueError('Changed sealed proof '+str(p))
 return v

def policy():
 return dict(schema=RULE_SCHEMA,authorized_by='root parent instruction,2026-10-31,preauthorized beforefirstcompletepilotresults',
  arms=list(ARMS),gates=[1000,5000],cap=10000,
  conjunction='samearm1k AND5k: actualcomplete samebudgetSafetyNAVTEST deltaPDMS<-1 ANDpairedCIupper<0 ANDfixedIL70score>=.9cohortCIupper<0 atBOTH',
  one_checkpoint_action=False,partial_scores_imputed=False,expected_scenes=12146,physical_logs=44,bootstrap=2000,seed=42,
  action='auditdecisionthenRUNSlock onlythisarm new_training_authorized=False/statusSTOP_REQUESTED_FUTILITY +exactcurrenttrainingjob STOP_REQUESTED',
  finalize='nativeexit0 +exacttransportexit +atomicfullcheckpointinventory/8rankRNG/bankreceipt +ownerackhold, thenSTOPPED_FUTILITY',
  no_stop_if_natural10k=True,no_process_signals=True,no_evaluation_cancellation=True,no_scoring_or_GPU=True,
  classification='clear sustained degradation of this registered recipe,notalgorithmfamilyrejected; one trainingseed,correlatedcheckpoints,notindependentpvalues',
  lifetime='botharms10kcompleted/stopped/failedor24hours; inotify+1800smetadataheartbeat;nojournalpolling')


def point_conditions(arm,u,p):
 if p is None:return None
 if (p.get('status')!='COMPLETE' or p.get('target')!=f'{arm}_u{u:06}_NAVTEST'
     or p.get('baseline')!=f'PPO_SAFETY_EP_u{u:06}_NAVTEST' or p.get('scenes')!=12146
     or p.get('physical_log_clusters')!=44 or p.get('bootstrap')!=2000 or p.get('seed')!=42):
  raise ValueError('Not the actual complete declared samebudgetpair')
 high=p.get('fixed_IL70_high_ge09',{})
 vals=[p['delta_PDMS'],*p['CI95_PDMS'],*high['CI95_PDMS']]
 if len(p['CI95_PDMS'])!=2 or len(high['CI95_PDMS'])!=2 or not all(math.isfinite(v) for v in vals) or high.get('scenes',0)<=0:
  raise ValueError('Invalidpoint/highcohortinterval')
 return dict(delta_below_minus1=p['delta_PDMS'] < -1,paired_CIupper_negative=p['CI95_PDMS'][1]<0,ILhigh_CIupper_negative=high['CI95_PDMS'][1]<0)


def qualifies(arm,doc):
 if doc is None:return False,{}
 if doc.get('schema')!=REVIEW_SCHEMA:raise ValueError('Unknownreviewactualschema')
 conditions={}
 for u in (1000,5000):
  p=doc.get('contrasts',{}).get(f'{arm}_u{u:06}_minus_samebudgetSafetyEP')
  c=point_conditions(arm,u,p)
  if c is None:return False,conditions
  conditions[str(u)]=c
 return all(all(c.values()) for c in conditions.values()),conditions


def load_bindings():
 reg=sealed(REG);rows=reg.get('arms',[])
 if set(r.get('id') for r in rows)!=set(ARMS) or reg.get('parent_model_sha256')!=PARENT_SHA or reg.get('optimizer_update_cap_per_arm')!=10000:
  raise ValueError('Onlyrootregisteredfreshpair/cap/parentallowed')
 out={}
 for row in rows:
  arm=row['id']
  if row['module']!=MODULES[arm] or sha(row['config'])!=row['config_sha256']:raise ValueError('Scientificregistrationchanged')
  p=Path(row['root'])/'checkpoints/update000000_inner0';i=sealed(p/'identity.json');s=read(p/'COMPLETE')
  if (i['recipe']['research']['arm']!=arm or i['IL_parent']['model_sha256']!=PARENT_SHA or s['updates']!=0 or s['world']!=8):raise ValueError('NotfreshIL70nativeorigin')
  out[arm]={k:row[k] for k in ('id','module','root','config','config_sha256','source','manifest')}
  out[arm].update(native_identity_sha256=i['sha256'],registration_file_sha256=sha(REG),registration_seal=reg['sha256'])
 return out


def validate_row(row,binding):
 if row.get('id') not in ARMS or row['id']!=binding['id']:raise ValueError('Refuseotherarm')
 for k in ('module','root','config','config_sha256','source','manifest'):
  if row.get(k)!=binding[k]:raise ValueError('Current scientificbindingchanged '+k)
 if sha(row['config'])!=binding['config_sha256'] or int(row.get('research_budget_updates',0))!=10000:raise ValueError('Cfg/capchanged')


def arg(command,name):return command[command.index(name)+1] if name in command else None

def validate_job(root,row,binding):
 validate_row(row,binding);job=Path(row['job']).resolve()
 if job.parent!=(Path(root)/'jobs'/row['id']).resolve():raise ValueError('Currentjoboutsideexactregisteredarm')
 launch=read(job/'LAUNCH.json');cmd=launch['command']
 if (launch['source']!=binding['source'] or binding['module'] not in cmd
     or arg(cmd,'--config')!=binding['config'] or arg(cmd,'--manifest')!=binding['manifest'] or arg(cmd,'--output')!=binding['root']
     or arg(cmd,'--il-checkpoint')!=str(Path(root)/'IL_PARENT') or arg(cmd,'--parent-selection')!=str(Path(root)/'PARENT_SELECTION.json')
     or int(arg(cmd,'--stop-after-update') or -1) not in (1000,5000,10000)):
  raise ValueError('Currentjobisnotexactregisteredscientificinvocation')
 return job,launch


def transport_alive(launch):
 try:
  s=Path(f"/proc/{int(launch['transport_pid'])}/stat").read_text().split()
  return int(s[21])==int(launch['transport_start_ticks']) and s[2] not in ('Z','X')
 except (OSError,KeyError,ValueError):return False


def verify_pair_provenance(arm,doc,binding):
 # Reuse CPU review/bootstrap, verify its alreadysealedcompleteinputmetadata.
 # No scene/weight/NPZ scan or bootstrap repetition here.
 if doc.get('source_sha256')!=sha(A/'safety_fresh_pair_review.py') or doc.get('high_cohort_definition')!='fixedIL70canonicalscore>=.9; NAVTESTdiagnosticonly,nottrainingselection':raise ValueError('Reviewwriter/cohortchanged')
 evidence={}
 for u in (1000,5000):
  key=f'{arm}_u{u:06}_minus_samebudgetSafetyEP';p=doc['contrasts'][key]
  for role,label in [('target',f'{arm}_u{u:06}_NAVTEST'),('baseline',f'PPO_SAFETY_EP_u{u:06}_NAVTEST')]:
   spec=read(R/'evaluation_specs'/(label+'.json'));s=sealed(Path(spec['output'])/'SUMMARY.json');i=sealed(Path(spec['output'])/'identity.json')
   proof=p['provenance'][role]
   if (s['status']!='COMPLETE' or s['scenes']!=12146 or s.get('invalid_scenes')!=0 or s['RL_updates']!=u
       or s['evaluation_identity_sha256']!=i['sha256'] or i['seed']!=2101 or i['cohort']!='NAVTEST'
       or i.get('qualification') is not False or i.get('count')!=12146
       or i.get('inference')!=dict(sampler='ODE',steps=10,shift=5.,seed=2101,scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)
       or proof['summary_seal']!=s['sha256'] or proof['identity_seal']!=i['sha256']):raise ValueError('Pairedcompleteinputprovenancechanged')
   if role=='target':
    cp=Path(spec['rl_checkpoint']).resolve()
    if not cp.is_relative_to((Path(binding['root'])/'checkpoints').resolve()):raise ValueError('Targetcheckpointoutsidecurrentrun')
    native=sealed(cp/'identity.json')
    if native['sha256']!=binding['native_identity_sha256']:raise ValueError('Targetscientificidentitychanged')
  evidence[str(u)]=copy_json(p)
 return evidence


def copy_json(x):return json.loads(json.dumps(x,allow_nan=False))


def checkpoint_receipt(row,binding):
 from tools.qwen_world_research.campaign_owner import native_checkpoint
 cp=native_checkpoint(row['root']);p=Path(cp['path'])
 if not p.is_relative_to((Path(row['root'])/'checkpoints').resolve()):raise ValueError('Nativeboundaryoutsidebinding')
 seal=read(p/'COMPLETE');files=read(p/'files.json')
 required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{k}.pt' for k in range(8)}
 if seal.get('schema')!=2 or seal.get('world')!=8 or sha(p/'files.json')!=seal['files_sha256'] or not required<=set(files):raise ValueError('Incompleteatomicnativeboundary')
 for n in required:
  if not (p/n).is_file() or (p/n).stat().st_size<=0 or len(files[n])!=64:raise ValueError('Missingfullnativepayload '+n)
 identity=sealed(p/'identity.json');tx=read(p/'transaction.json')
 if sha(p/'identity.json')!=files['identity.json'] or sha(p/'transaction.json')!=files['transaction.json'] or identity['sha256']!=binding['native_identity_sha256'] or len(tx['ranks'])!=8:raise ValueError('Nativeidentity/transactionchanged')
 for rank,s in enumerate(tx['ranks']):
  if s['updates']!=seal['updates'] or s['next_inner']!=seal['next_inner'] or s['phase']!=seal['phase'] or s['run']!=identity['sha256']:raise ValueError('Eight-rankboundaryinconsistent')
  result=read(Path(row['root'])/f'result_rank{rank}.json')
  if not result or result['updates']!=seal['updates'] or Path(result['checkpoint']).resolve()!=p:raise FileNotFoundError('Waitingall8nativeendresults')
  if row['id']==ARMS[0]:
   m=s.get('success_memory',{});public=result.get('success_memory_summary',{})
   if (m.get('rank')!=rank or m.get('collections')!=s['collections'] or not 0<=m.get('bank_size',-1)<=1024
       or len(m.get('RNG_sha256',''))!=64 or len(m.get('ordered_bank_ids_sha256',''))!=64
       or any(public.get(k)!=m.get(k) for k in ('RNG_sha256','ordered_bank_ids_sha256','bank_size'))):raise ValueError('Fullsuccessbank/privateRNGreceiptmissing')
 return dict(checkpoint=str(p),COMPLETE=seal,files_manifest_sha256=sha(p/'files.json'),files=files,
  rank_native_summaries=tx['ranks'],full_optimizer_and_adapter_payload='optimizer.pt retained',
  per_rank_native_RNG_payloads='all rank0..7.pt retained by source save contract:Python/NumPy/torch/CUDA/rolloutgenerator',
  full_bank_state='SuccessMemoryfullbank/privateRNG ineachnative rankstate' if row['id']==ARMS[0] else 'notapplicable;FPOpendingMCmaterial retainedwhenpending',
  full_tensor_payload_rehashed=False,tensors_loaded=False,verification='atomicmanifest+smallidentity/transactionseals+8endresultmetadata; strictfulltensorhash/contentvalidationremainsnative resume')


def natural_10k_complete(row):
 if int(row.get('completed_updates',0))>=10000 or row.get('status')=='COMPLETED_AT_REGISTERED_10K':return True
 run=Path(row['root'])
 for name in ('latest.json','progress.json'):
  record=read(run/name,{})
  if not record.get('checkpoint'):continue
  path=Path(record['checkpoint']).resolve()
  if not path.is_relative_to((run/'checkpoints').resolve()):raise ValueError('Publishedcheckpointoutsideregisteredroot')
  seal=read(path/'COMPLETE',{})
  if seal.get('updates',0)>=10000 and seal.get('world')==8 and seal.get('schema')==2:return True
 return False


def request_stop(root,owner,arm,binding,decision):
 root=Path(root);owner=Path(owner);dest=owner/'decisions'/f'{arm}.json';atomic(dest,decision)
 with (root/'RUNS.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);reg=read(root/'RUNS.json');row=next(x for x in reg['runs'] if x['id']==arm)
  validate_row(row,binding)
  if natural_10k_complete(row):
   atomic(dest,dict(decision,disposition='REVIEW_ONLY_NATURAL_10K_COMPLETE',stop_action=False))
   return dict(status='REVIEW_ONLY_NATURAL_10K_COMPLETE',decision=str(dest))
  job,launch=validate_job(root,row,binding)
  previous=row.get('futility_guard_decision')
  if previous not in (None,str(dest)):raise ValueError('Anotherstopdecisionexists;manualreview')
  row.update(new_training_authorized=False,status='STOP_REQUESTED_FUTILITY',futility_guard_decision=str(dest),futility_guard_rule=RULE_SCHEMA)
  # Save decision first,thenregistry denial,thenexactstopfile. AllunderRUNSlock.
  atomic(root/'RUNS.json',reg);stop_preexisting=(job/'STOP_REQUESTED').exists();done=(job/'RESULT.json').exists()
  if not done:(job/'STOP_REQUESTED').touch(exist_ok=True)
  return dict(status='STOP_REQUESTED_FUTILITY',decision=str(dest),job=str(job),launch_sha256=sha(job/'LAUNCH.json'),
      stopfile_written=not done,stopfile_preexisting=stop_preexisting,job_result_already_exists=done,at=now(),transport=launch)


class Guard:
 def __init__(self):
  self.bindings=load_bindings();self.state=read(O/'STATE.json',dict(schema=STATE_SCHEMA,arms={a:dict(status='OBSERVING') for a in ARMS},events=[]))
  if self.state.get('schema')!=STATE_SCHEMA:raise ValueError('Differentguardstate')
  if self.state.get('scientific_bindings') not in (None,self.bindings):raise ValueError('Previousguardstatebelongstoanotherregisteredrun')
  self.state['scientific_bindings']=self.bindings
  self.state.update(status='WATCHING_ACTUAL_COMPLETE_EVIDENCE',policy=str(O/'POLICY.json'),policy_sha256=sha(O/'POLICY.json'),rule=policy(),max_hours=24,process_signals=0,evaluation_actions=0)
 def flush(self):self.state['updated_at']=now();atomic(O/'STATE.json',self.state)
 def refresh(self):atomic(R/'reporting/refresh_request.json',dict(id='freshpairfutility-'+uuid.uuid4().hex,scientific_result=False,evidence_kind='completed_supplemental_analysis',at=now(),evidence=str(O/'STATE.json')))
 def step(self):
  raw=REVIEW.read_bytes() if REVIEW.exists() else None;doc=json.loads(raw) if raw else None
  registry=read(R/'RUNS.json');rows={x['id']:x for x in registry['runs']};owner=read(PAIR/'STATE.json',{}).get('runs',{})
  for arm in ARMS:
   node=self.state['arms'][arm]
   if node['status'] in ('STOPPED_FUTILITY','NATURAL_10K_COMPLETE_REVIEW_ONLY','EXTERNALLY_STOPPED_OR_FAILED'):continue
   binding=self.bindings[arm];row=rows[arm];validate_row(row,binding)
   if sha(REG)!=binding['registration_file_sha256'] or sha(O/'POLICY.json')!=self.state['policy_sha256']:raise ValueError('Authorizedregistration/rulechanged')
   if natural_10k_complete(row) or owner.get(arm,{}).get('training_status')=='COMPLETED_AT_REGISTERED_10K':
    node.update(status='NATURAL_10K_COMPLETE_REVIEW_ONLY',completed_updates=10000,review_if_available=qualifies(arm,doc),stop_requested=False);continue
   if 'request' not in node and any(s in row.get('status','').upper() for s in ('STOP','FAIL','CANCEL','HELD')):
    node.update(status='EXTERNALLY_STOPPED_OR_FAILED',row_status=row.get('status'),guard_action=False);continue
   ok,conditions=qualifies(arm,doc);node['observed_conditions']=conditions
   if 'request' not in node:
    if not ok:continue
    evidence=verify_pair_provenance(arm,doc,binding)
    decision=dict(schema='world_freshSafety_recipe_futility_decision_v1',disposition='STOP_FUTILITY',arm=arm,at=now(),
      authorization=policy(),policy_path=str(O/'POLICY.json'),policy_sha256=sha(O/'POLICY.json'),
      registration=str(REG),registration_file_sha256=sha(REG),registration_seal=binding['registration_seal'],
      scientific_binding=binding,conditions=conditions,evidence=evidence,review_path=str(REVIEW),review_file_sha256=hashlib.sha256(raw).hexdigest(),
      classification='thisregisteredrecipe sustaineddegradation,notalgorithmfamilyrejection',
      preserve='allmodels/nativeoptimizer/allrankRNG/pendingrollout/successbank andallevaluations',no_evaluation_cancellation=True,no_process_signals=True)
    node['request']=request_stop(R,O,arm,binding,decision);node['decision']=decision;node['requests']=[node['request']];node['status']=node['request']['status'];self.flush();self.refresh()
   if node['status']=='REVIEW_ONLY_NATURAL_10K_COMPLETE':node['status']='NATURAL_10K_COMPLETE_REVIEW_ONLY';continue
   # Route/start_worker canhavebegun beforetheRUNSlock. Re-readcurrentjob on
   # eachpublication; denyagain/touchonlyit andwaitownerackbeforefinalizing.
   current=next(x for x in read(R/'RUNS.json')['runs'] if x['id']==arm)
   validate_row(current,binding)
   if current.get('job')!=node['request'].get('job') or current.get('status') not in ('STOP_REQUESTED_FUTILITY','STOPPED_FUTILITY'):
    latest=request_stop(R,O,arm,binding,node['decision']);node['request']=latest;node['requests'].append(latest);self.flush()
   job,launch=validate_job(R,current,binding);result=read(job/'RESULT.json')
   if not result or transport_alive(launch):continue
   if int(result.get('returncode',result.get('exit_code',-999)))!=0 or result.get('forced_expiry') is True:
    node.update(status='STOP_REQUESTED_NATIVE_FAILURE_REQUIRES_REVIEW',native_result=result);continue
   ack=read(PAIR/'STATE.json',{}).get('runs',{}).get(arm,{})
   if ack.get('status') not in ('HELD_BY_STOP_REQUEST','HELD_COOPERATIVE_NATIVE_BOUNDARY') or ack.get('job')!=str(job):
    node['waiting_owner_ack_to_exclude_inflight_nextstage']=True;continue
   try:receipt=checkpoint_receipt(current,binding)
   except FileNotFoundError:continue
   with (R/'RUNS.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);r=read(R/'RUNS.json');rr=next(x for x in r['runs'] if x['id']==arm);validate_row(rr,binding)
    if rr['job']!=str(job):continue
    rr.update(new_training_authorized=False,status='STOPPED_FUTILITY',completed_updates=receipt['COMPLETE']['updates'],last_checkpoint=receipt['checkpoint'],futility_stop_receipt=str(O/'receipts'/f'{arm}.json'))
    final=dict(schema='world_freshSafety_verified_native_futility_stop_v1',arm=arm,at=now(),status='STOPPED_FUTILITY',decision=str(O/'decisions'/f'{arm}.json'),
      exact_job=str(job),launch_sha256=sha(job/'LAUNCH.json'),native_exit=result,transport_exited=True,owner_ack=ack,full_native_recovery=receipt,
      no_process_signals=True,no_evaluation_cancellation=True,not_algorithm_family_rejected=True)
    atomic(O/'receipts'/f'{arm}.json',final);atomic(R/'RUNS.json',r)
   node.update(status='STOPPED_FUTILITY',receipt=str(O/'receipts'/f'{arm}.json'),updates=receipt['COMPLETE']['updates']);self.flush();self.refresh()
  self.flush()
  terminal={'STOPPED_FUTILITY','NATURAL_10K_COMPLETE_REVIEW_ONLY','EXTERNALLY_STOPPED_OR_FAILED'}
  return all(n['status'] in terminal for n in self.state['arms'].values())
 def watches(self,ev):
  ev.add(A,{REVIEW.name});ev.add(R,{'RUNS.json'});ev.add(PAIR,{'STATE.json'})
  for arm in ARMS:
   row=next(x for x in read(R/'RUNS.json')['runs'] if x['id']==arm);job=Path(row['job']);run=Path(row['root'])
   ev.add(job,{'RESULT.json','LAUNCH.json'});ev.add(job.parent,None);ev.add(run,{'latest.json','progress.json',*[f'result_rank{k}.json' for k in range(8)]})
   ev.add(run/'checkpoints',None)
 def run(self):
  deadline=time.monotonic()+24*3600;ev=Events()
  try:
   while True:
    self.watches(ev)
    if self.step():self.state['status']='FINISHED_ALL_NATIVE_ARMS_TERMINAL';break
    left=deadline-time.monotonic()
    if left<=0:self.state['status']='TIMED_OUT24H_PRESERVE_ANY_PENDING_STOP';break
    # Exacttransportpidfds wakepostexit evenonNFS; nojournalpolling.
    fds=[]
    for arm in ARMS:
     row=next(x for x in read(R/'RUNS.json')['runs'] if x['id']==arm);launch=read(Path(row['job'])/'LAUNCH.json',{})
     if not transport_alive(launch):continue
     try:
      pid=int(launch['transport_pid']);fd=os.pidfd_open(pid) if hasattr(os,'pidfd_open') else int(ctypes.CDLL(None,use_errno=True).syscall(434,pid,0))
      if fd>=0:fds.append(fd)
     except OSError:pass
    try:ev.wait(min(left,1800),fds)
    finally:
     for fd in fds:os.close(fd)
  except Exception:
   self.state.update(status='GUARD_FAILED_REQUIRES_ROOT_REVIEW',error=traceback.format_exc());raise
  finally:ev.close();self.flush();self.refresh()
  return self.state


def bindings():return {str(p):sha(p) for p in [Path(__file__).resolve(),A/'safety_fresh_pair_owner.py',A/'safety_fresh_pair_review.py',A/'navtest_cadence_owner_v2.py']}
def prepare():
 O.mkdir(parents=True,exist_ok=True);p=O/'POLICY.json'
 if p.exists() and read(p)!=policy():raise ValueError('Existingauthorizedrulechanged')
 if not p.exists():atomic(p,policy())
 b=load_bindings();v=dict(schema='freshSafety_evidence_guard_launch_plan_v1',status='READY_FOR_ROOT_REVIEW_NOT_STARTED',started=False,at=now(),script=str(Path(__file__).resolve()),bindings=bindings(),
  arms=b,policy=str(p),policy_sha256=sha(p),owner=str(O/'OWNER.json'),launch=str(O/'LAUNCH.json'),state=str(O/'STATE.json'),max_hours=24,
  launch_argv=[sys.executable,str(Path(__file__).resolve()),'--launch'],no_GPU_or_evaldispatch=True,cooperative_stop_only=True,first1k_no_stop=True,root_registration=str(REG))
 atomic(PLAN,v);return v

def launch():
 plan=read(PLAN)
 if plan['bindings']!=bindings() or plan['arms']!=load_bindings() or plan['policy_sha256']!=sha(O/'POLICY.json'):raise ValueError('Reviewedguardbindingchanged')
 with (O/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for old in (read(O/'OWNER.json',{}),read(O/'LAUNCH.json',{})):
   if not old:continue
   try:
    s=Path(f"/proc/{old['pid']}/stat").read_text().split()
    if int(s[21])==old['start_ticks'] and s[2] not in ('Z','X'):raise RuntimeError('Existingexactguardowner')
   except FileNotFoundError:pass
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run']
  with (O/'stdout.log').open('ab',buffering=0) as log:p=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  r=dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),argv=cmd,source_sha256=sha(__file__),started_at=now(),plan=str(PLAN));atomic(O/'LAUNCH.json',r);return r

def self_check():
 import tempfile
 arm=ARMS[1]
 def point(u):return dict(status='COMPLETE',target=f'{arm}_u{u:06}_NAVTEST',baseline=f'PPO_SAFETY_EP_u{u:06}_NAVTEST',scenes=12146,physical_log_clusters=44,bootstrap=2000,seed=42,delta_PDMS=-1.1,CI95_PDMS=[-1.5,-.7],fixed_IL70_high_ge09=dict(scenes=8724,CI95_PDMS=[-1.2,-.2]))
 d=dict(schema=REVIEW_SCHEMA,contrasts={f'{arm}_u001000_minus_samebudgetSafetyEP':point(1000)})
 assert not qualifies(arm,d)[0]
 d['contrasts'][f'{arm}_u005000_minus_samebudgetSafetyEP']=point(5000);assert qualifies(arm,d)[0]
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);owner=root/'guard';job=root/'jobs'/arm/'formal_to010000_attempt001';job.mkdir(parents=True);cfg=root/'cfg.yaml';cfg.write_text('scientifictoy')
  b=dict(id=arm,module=MODULES[arm],root=str(root/'run'),source=str(root/'frozen'),config=str(cfg),config_sha256=sha(cfg),manifest=str(root/'manifest'))
  row=dict(b,job=str(job),research_budget_updates=10000,completed_updates=5000,status='TRAINING_DISPATCHED',new_training_authorized=True,new_evaluation_authorized=True)
  cmd=['python','-m',b['module'],'--config',b['config'],'--manifest',b['manifest'],'--output',b['root'],'--il-checkpoint',str(root/'IL_PARENT'),'--parent-selection',str(root/'PARENT_SELECTION.json'),'--stop-after-update','10000']
  atomic(job/'LAUNCH.json',dict(source=b['source'],command=cmd));atomic(root/'RUNS.json',dict(runs=[dict(id='PPO_SAFETY_EP',status='PROTECTED'),row]))
  # Noactionis calledfor1konly. Onlyafter2pointsdoesexactstopfunctionrun.
  response=request_stop(root,owner,arm,b,dict(rule=policy(),evidence=d['contrasts']))
  after=read(root/'RUNS.json');assert response['stopfile_written'] and (job/'STOP_REQUESTED').exists()
  assert after['runs'][0]==dict(id='PPO_SAFETY_EP',status='PROTECTED')
  assert after['runs'][1]['new_training_authorized'] is False and after['runs'][1]['new_evaluation_authorized'] is True
  (job/'STOP_REQUESTED').unlink()
  row['completed_updates']=10000;atomic(root/'RUNS.json',dict(runs=[row]))
  natural=request_stop(root,owner,arm,b,dict(rule=policy(),evidence=d['contrasts']))
  assert natural['status']=='REVIEW_ONLY_NATURAL_10K_COMPLETE' and not (job/'STOP_REQUESTED').exists()
  wrong=dict(row,module='originalSafety')
  try:validate_job(root,wrong,b)
  except ValueError:pass
  else:raise AssertionError('Mismatchedmoduleaccepted')
 r=dict(status='PASS',at=now(),script_sha256=sha(__file__),scope='CPU1konly=false/1kAND5k=true, exacttmpjobstop+onlytargetRUNSchange,originalSafety/evalauthuntouched,wrongmoduledenied; noactualGPUorjobaction')
 atomic(A/'SAFETY_FRESH_PAIR_FUTILITY_GUARD_CPU_CHECK_20261031.json',r);return r

def main():
 p=argparse.ArgumentParser(__doc__);g=p.add_mutually_exclusive_group(required=True)
 for a in ('prepare','launch','self-check','run'):g.add_argument('--'+a,action='store_true')
 x=p.parse_args()
 if x.prepare:out=prepare()
 elif x.launch:out=launch()
 elif x.self_check:out=self_check()
 else:
  O.mkdir(parents=True,exist_ok=True)
  with (O/'owner.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source_sha256=sha(__file__),started_at=now()))
   out=Guard().run()
 print(json.dumps(out,ensure_ascii=False,default=str))
if __name__=='__main__':main()
