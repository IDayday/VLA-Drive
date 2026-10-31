"""Finite CPU evidence guard: SAFETY20k/35k full NAVTEST, at most24hours.

No GPU dispatch, model loading, process signals, checkpoint access, or journal
polling. --allow-cooperative-stop only permits the predeclared35k conjunctive
futility rule to publish an arm decision and touch its exact registered job.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import struct
import time
import traceback
import uuid
import numpy as np

ARM='PPO_SAFETY_EP';GATES=(20000,35000)
METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
SAFETY=METRICS[1:4]
INFERENCE=dict(sampler='ODE',steps=10,shift=5.,seed=2101,scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)
POLICY=dict(schema='finite_SAFETY_20k35k_futility_review_v1',gates=list(GATES),maximum_hours=24,
 bootstrap=2000,seed=42,unit='PDMS or metric percentage points',physical_logs=44,full_NAVTEST=12146,
 refresh='unique id + scientific_result=False',budget_is_ceiling_not_commitment=True,
 at20k='Recommendation only: CI-negative PDMS vs10k and no clear safety gain -> stop-review recommended; no automatic stop.',
 at35k='STOP iff both20k/35k do not exceed10k PDMS AND35k-minus10k PDMS95CI upper<0 AND no NC/DAC/TTC95CI lower>0.',
 clear_safety_improvement='35k-minus-own10k, paired physical-log95CI lower>0; mean-only gain is not clear.',
 matched_FULL='same-budget contrasts are evidence, never substitute a missing own10k comparison.',
 highest_route_protection='Safety decline alone is never a stop; a new PDMS best at20k or35k blocks this stop rule.',
 stop='standard STOP_FUTILITY arm decision + only current registered Safety training job STOP_REQUESTED; native owner seals/resumes state; never kill.',
 exclusions='No single-seed certainty claim, no NAVTEST training token selection, no new50k commitment.')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
 v=read(p);b=dict(v);s=b.pop('sha256',None)
 if s!=fp(b):raise ValueError('Changed sealed evidence '+str(p))
 return v

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)

def refresh(root,evidence):
 atomic(root/'reporting/refresh_request.json',dict(id='safety-evidence-review-'+uuid.uuid4().hex,at=now(),scientific_result=False,evidence_kind='completed_supplemental_analysis',evidence=str(evidence)))

def complete_input(root,arm,gate):
 specpath=root/'evaluation_specs'/f'{arm}_u{gate:06}_NAVTEST.json'
 if not specpath.exists():return None
 spec=read(specpath);out=Path(spec['output']);summarypath=out/'SUMMARY.json'
 if not summarypath.exists():return None
 s=sealed(summarypath);i=sealed(out/'identity.json')
 if s.get('status')!='COMPLETE':return None
 if (s.get('scenes')!=12146 or s.get('invalid_scenes')!=0 or i.get('qualification') is not False
  or i.get('cohort')!='NAVTEST' or i.get('count')!=12146 or i.get('seed')!=2101
  or i.get('inference')!=INFERENCE or s.get('RL_updates')!=gate
  or s['evaluation_identity_sha256']!=i['sha256'] or s['protocol_sha256']!=i['protocol']['sha256']):
  raise ValueError('Incomplete/mismatched full formal NAVTEST: '+str(summarypath))
 if Path(i['native_RL_checkpoint']).resolve()!=Path(spec['rl_checkpoint']).resolve():raise ValueError('Evaluation checkpoint lineage mismatch')
 return dict(label=spec['label'],output=out,summarypath=summarypath,summary=s,identity=i)


def load_metrics(root,info):
 """Exact compact cache reuse; a genuinely new complete population read once."""
 cache=root/'analysis/evaluation_metric_cache';cache.mkdir(exist_ok=True);label=info['label'];s=info['summary'];identity=info['identity']
 p=cache/(label+'.npz');meta=cache/(label+'.json')
 with (cache/(label+'.review.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if p.exists() and meta.exists():
   m=read(meta)
   if m['identity_sha256']!=identity['sha256'] or m['summary_sha256']!=s['sha256'] or sha(p)!=m['npz_sha256']:raise ValueError('Changed compact cache '+label)
   with np.load(p,allow_pickle=False) as z:values=z['metrics'].copy();tokens=z['tokens'].tolist();logs=z['logs'].tolist();hashes=z['cache_hashes'].tolist()
   reused=True
  else:
   tokens=identity['tokens'];values=np.empty((len(tokens),7),np.float64);logs=[];hashes=[];inventory={}
   for j,t in enumerate(tokens):
    scene=info['output']/'scenes'/f'{t}.json';raw=scene.read_bytes();row=json.loads(raw);body=dict(row);seal=body.pop('sha256',None)
    if fp(body)!=seal or row['identity']!=identity['sha256'] or row['token']!=t or row['status']!='OK':raise ValueError('Changed scene receipt '+str(scene))
    feedback=row['feedback'];values[j]=[feedback['metrics'][k] for k in METRICS];logs.append(row['log']);hashes.append(feedback['cache']['sha256']);inventory[t]=hashlib.sha256(raw).hexdigest()
    if row['log']!=identity['token_logs'][t] or feedback['protocol_sha256']!=identity['protocol']['sha256']:raise ValueError('Changed scene/log/scorer join')
   if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():raise ValueError('Invalid complete metric values')
   expected=values[:,1]*values[:,2]*(5*values[:,4]+5*values[:,3]+2*values[:,5])/12
   if not np.allclose(expected,values[:,0],atol=5e-15,rtol=0):raise ValueError('Changed canonical aggregation')
   q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
   with q.open('wb') as f:np.savez_compressed(f,metrics=values,tokens=np.array(tokens),logs=np.array(logs),cache_hashes=np.array(hashes),metric_keys=np.array(METRICS))
   os.replace(q,p);m=dict(schema='exact_canonical_evaluation_metric_cache_v1',label=label,identity_sha256=identity['sha256'],summary_sha256=s['sha256'],npz_sha256=sha(p),scene_raw_hash_inventory_sha256=fp(inventory),shape=list(values.shape),metric_keys=METRICS,units='official0..1 exact JSON FP64 values')
   atomic(meta,m);reused=False
 if values.shape!=(12146,7) or tokens!=identity['tokens'] or len(set(tokens))!=12146:raise ValueError('Changed full census')
 for k,j in zip(METRICS,range(7)):
  if abs(values[:,j].mean()-s['means'][k])>1e-10:raise ValueError('Summary/compact mean mismatch '+k)
 logs=np.array([re.sub(r'_\d+_\d+$','',l) for l in logs])
 if len(set(logs))!=44:raise ValueError('Expected44 physical log clusters')
 return dict(info=info,values=values,tokens=tokens,logs=logs,hashes=hashes,
  provenance=dict(summary=str(info['summarypath']),summary_file_sha256=sha(info['summarypath']),summary_seal=s['sha256'],identity_seal=identity['sha256'],cache=str(p),cache_sha256=m['npz_sha256'],cache_reused=reused))


def paired(target,base):
 a,b=target['info']['identity'],base['info']['identity']
 for k in ['cohort','tokens','count','token_sha256','data_sha256','population_sha256','seed','numerics','qualification','inference']:
  if a[k]!=b[k]:raise ValueError('Paired contract mismatch '+k)
 if a['protocol']['sha256']!=b['protocol']['sha256'] or target['hashes']!=base['hashes'] or not np.array_equal(target['logs'],base['logs']):raise ValueError('Paired physical context mismatch')
 delta=target['values']-base['values'];logs=target['logs'];names=list(dict.fromkeys(logs));index=np.array([names.index(l) for l in logs]);n=np.bincount(index,minlength=44);sums=np.zeros((44,7));np.add.at(sums,index,delta)
 sample=np.random.default_rng(42).integers(0,44,size=(2000,44));boots=100*sums[sample].sum(1)/n[sample].sum(1)[:,None];lo,hi=np.quantile(boots,[.025,.975],axis=0)
 metrics={k:dict(delta_pp=float(100*delta[:,j].mean()),CI95_pp=[float(lo[j]),float(hi[j])],clear_improvement=bool(lo[j]>0)) for j,k in enumerate(METRICS)}
 return dict(target=target['info']['label'],baseline=base['info']['label'],target_PDMS=target['info']['summary']['PDMS'],baseline_PDMS=base['info']['summary']['PDMS'],
  metrics=metrics,physical_logs=44,scenes=12146,bootstrap=2000,seed=42,estimator='resample44 physical logs, scene-weighted ratio of sums; same draws all metrics',
  uncertainty='fixed training/evaluation seed; physical-log variability, not across-seed uncertainty; component CIs unadjusted, used conservatively to protect any clear safety gain')


def decision(gate,current,prior20=None):
 own=current['vs_own10k'];score=own['metrics']['score'];newbest=own['target_PDMS']>own['baseline_PDMS']+1e-12
 safety_gain=any(own['metrics'][k]['CI95_pp'][0]>0 for k in SAFETY)
 if gate==20000:
  rec='STOP_REVIEW_RECOMMENDED' if not newbest and score['CI95_pp'][1]<0 and not safety_gain else 'CONTINUE_EVIDENCE_REVIEW'
  return dict(recommendation=rec,automatic_stop_eligible=False,reason='20k is recommendation only; second gate required for automatic futility',new_PDMS_best=newbest,clear_safety_gain=safety_gain)
 if gate!=35000 or prior20 is None:raise ValueError('35k requires complete20k evidence')
 g20=prior20['vs_own10k'];conditions=dict(gate20_complete=True,gate35_complete=True,
  gate20_no_new10k_best=g20['target_PDMS']<=g20['baseline_PDMS']+1e-12,
  gate35_no_new10k_best=not newbest,gate35_PDMS_CI_upper_negative=score['CI95_pp'][1]<0,
  no_clear_NC_DAC_TTC_gain=not safety_gain)
 eligible=all(conditions.values())
 return dict(recommendation='STOP_FUTILITY' if eligible else 'CONTINUE_OR_REVIEW',automatic_stop_eligible=eligible,conditions=conditions,
  reason='all registered futility conjuncts satisfied' if eligible else 'futility conjunction not satisfied; no stop or50k commitment',new_PDMS_best=newbest,clear_safety_gain=safety_gain)


def request_stop(root,state,evidence):
 """No process/weight access: standard decision plus exact registered stopfile."""
 directory=root/'owners/CAMPAIGN/arm_decisions';directory.mkdir(exist_ok=True);dest=directory/(ARM+'.json')
 with (root/'RUNS.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX);row=next(x for x in read(root/'RUNS.json')['runs'] if x['id']==ARM)
  binding=state['arm_binding']
  if any(str(row[k])!=binding[k] for k in ['root','source','module','config']):raise ValueError('Current registered Safety scientific/run binding changed; no stop')
  job=Path(row['job']).resolve()
  if job.parent!= (root/'jobs'/ARM).resolve():raise ValueError('Current registered Safety job outside its arm')
  launch=read(job/'LAUNCH.json');cmd=launch['command']
  def arg(name):return cmd[cmd.index(name)+1] if name in cmd else None
  if (row['module'] not in cmd or arg('--output')!=row['root'] or arg('--config')!=row['config'] or launch['source']!=row['source']):raise ValueError('Exact current job is not the bound Safety training invocation')
  receipt=dict(schema='world_rl_evidence_based_arm_decision_v1',arm=ARM,disposition='STOP_FUTILITY',created_at=now(),
   reason='Completed20k and35k did not exceed own10k best;35k-minus10k paired44-log PDMS95CI upper<0; no NC/DAC/TTC clear improvement. Preserve best/native states;50k is ceiling.',
   evidence=[str(root/'owners/SAFETY_EVIDENCE_REVIEW/GATE_020000.json'),str(evidence),
     *[state['gates'][str(g)]['provenance'][k]['summary'] for g in GATES for k in ['target','own10k','matched_FULL']]],Test_selection_used=True,NAVTEST_used_for_training_scene_selection=False,
   scope='stop only current registered Safety continuation; preserve10k/20k/35k and all native recovery assets',
   stop_mode='cooperative native checkpoint retaining optimizer RNG and pending rollout',
   guarded_by=state['policy'],evidence_reviewer_source=str(Path(__file__).resolve()),evidence_reviewer_source_sha256=sha(__file__),current_job=str(job),job_launch_sha256=sha(job/'LAUNCH.json'))
  if dest.exists():
   old=read(dest)
   if old.get('evidence_reviewer_source')!=str(Path(__file__).resolve()) or old.get('guarded_by')!=state['policy']:
    return dict(status='EXISTING_ARM_DECISION_PRESERVED_REVIEW_REQUIRED',decision=str(dest),existing=old)
  else:atomic(dest,receipt)
  completed=(job/'RESULT.json').exists();preexisting=(job/'STOP_REQUESTED').exists()
  if not completed:(job/'STOP_REQUESTED').touch(exist_ok=True)
  return dict(status='STOP_FUTILITY_REQUESTED' if not completed else 'STOP_FUTILITY_RECORDED_JOB_ALREADY_COMPLETE',decision=str(dest),job=str(job),stopfile_written=not completed,stopfile_preexisting=preexisting,
   native_exit_not_yet_confirmed=True,checkpoint_files_touched=False,GPU_process_signals_sent=False)


class Events:
 """Bounded inotify wait on evidence publication; no journals or scene polling."""
 def __init__(self):
  self.lib=ctypes.CDLL(None,use_errno=True);self.fd=self.lib.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
  if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
  self.paths=set();self.wd={}
 def add(self,p):
  p=Path(p)
  if str(p) in self.paths or not p.is_dir():return
  wd=self.lib.inotify_add_watch(self.fd,os.fsencode(p),0x8|0x80|0x100|0x400|0x800)
  if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch '+str(p))
  self.paths.add(str(p));self.wd[wd]=str(p)
 def wait(self,seconds):
  deadline=time.monotonic()+min(1800,max(0,seconds))
  while True:
   ready,_,_=select.select([self.fd],[],[],max(0,deadline-time.monotonic()))
   if not ready:return True  # bounded heartbeat, not journal polling
   relevant=False
   while True:
    try:raw=os.read(self.fd,1024*1024)
    except BlockingIOError:break
    at=0
    while at<len(raw):
     wd,mask,cookie,length=struct.unpack_from('iIII',raw,at);at+=16;name=raw[at:at+length].split(b'\0',1)[0].decode();at+=length
     relevant|=bool(mask&0x4000 or name in ('SUMMARY.json','STATE.json','RUNS.json') or
       ((name.startswith(ARM+'_u') or name.startswith('PPO_FULL_u')) and (name.endswith('_NAVTEST.json') or name.endswith('_NAVTEST'))))
   if relevant:return True
   # Ignore temporary/rank scene progress events until a real publication.
 def close(self):os.close(self.fd)


def run(a):
 root=Path(a.root).resolve();owner=root/'owners/SAFETY_EVIDENCE_REVIEW';owner.mkdir(exist_ok=True)
 lock=(owner/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statepath=owner/'STATE.json';start=time.time();state=read(statepath) if statepath.exists() else {}
 if state.get('status') in ['COMPLETE','STOP_FUTILITY_REQUESTED','STOP_FUTILITY_RECORDED_JOB_ALREADY_COMPLETE','COMPLETE_DRY_REVIEW_STOP_RECOMMENDED','EXISTING_ARM_DECISION_PRESERVED_REVIEW_REQUIRED','REVIEW_REQUIRED','TIME_LIMIT_REACHED']:return state
 if not state:
  row=next(x for x in read(root/'RUNS.json')['runs'] if x['id']==ARM)
  state=dict(schema=POLICY['schema'],status='WAITING_20K',started_unix=start,deadline_unix=start+min(a.hours,24)*3600,
   policy=POLICY,allow_cooperative_stop=a.allow_cooperative_stop,arm_binding={k:str(row[k]) for k in ['root','source','module','config']},gates={},events=[])
 elif state['allow_cooperative_stop']!=a.allow_cooperative_stop:raise ValueError('Restart cannot silently change stop authorization')
 atomic(owner/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),start_unix=time.time(),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),state=str(statepath),deadline_unix=state['deadline_unix'],CPU_only=True))
 atomic(statepath,state);events=Events();cache={}
 def publish():
  state['updated_at']=now();atomic(statepath,state);atomic(root/'analysis/SAFETY_EVIDENCE_REVIEW_20261031.json',state)
  refresh(root,root/'analysis/SAFETY_EVIDENCE_REVIEW_20261031.json')
 def finish():
  rec=state['gates']['35000']['decision'];gatefile=owner/'GATE_035000.json'
  if rec['automatic_stop_eligible'] and a.allow_cooperative_stop:
   state['cooperative_stop']=request_stop(root,state,gatefile);state['status']=state['cooperative_stop']['status']
  elif rec['automatic_stop_eligible']:state['status']='COMPLETE_DRY_REVIEW_STOP_RECOMMENDED'
  else:state['status']='COMPLETE'
  publish();return state
 try:
  baseline=complete_input(root,ARM,10000)
  if baseline is None:raise ValueError('Missing full own10k reference; no review/stop permitted')
  cache[10000]=load_metrics(root,baseline)
  if '35000' in state['gates']:return finish()
  while True:
   for p in [root,root/'evaluation_specs',root/'owners'/ARM,root/'owners/PPO_FULL']:events.add(p)
   for gate in GATES:
    info=complete_input(root,ARM,gate)
    if info:events.add(info['output'])
    else:
     spec=root/'evaluation_specs'/f'{ARM}_u{gate:06}_NAVTEST.json'
     if spec.exists():
      output=Path(read(spec)['output']);events.add(output.parent);events.add(output)
    if str(gate) in state['gates'] or info is None:continue
    full=complete_input(root,'PPO_FULL',gate)
    if full is None:
     spec=root/'evaluation_specs'/f'PPO_FULL_u{gate:06}_NAVTEST.json'
     if spec.exists():
      path=Path(read(spec)['output']);events.add(path.parent);events.add(path)
     state['status']='WAITING_MATCHED_CONTROL_EVIDENCE';continue
    if gate==35000 and '20000' not in state['gates']:continue
    current=load_metrics(root,info);control=load_metrics(root,full)
    result=dict(gate=gate,complete_at=now(),vs_own10k=paired(current,cache[10000]),vs_matched_FULL=paired(current,control),
      provenance=dict(target=current['provenance'],own10k=cache[10000]['provenance'],matched_FULL=control['provenance']))
    rec=decision(gate,result,state['gates'].get('20000'));result['decision']=rec
    state['gates'][str(gate)]=result;gatefile=owner/f'GATE_{gate:06}.json';atomic(gatefile,result)
    state['status']='WAITING_35K' if gate==20000 else 'FINAL_EVIDENCE_READY_ACTION_PENDING';publish()
    if gate==35000:return finish()
   if time.time()>=state['deadline_unix']:state['status']='TIME_LIMIT_REACHED';publish();return state
   state['heartbeat_at']=now();atomic(statepath,state)
   if events.wait(state['deadline_unix']-time.time()):continue
 except Exception:
  state.update(status='REVIEW_REQUIRED',error=traceback.format_exc(),no_stop_after_error=True);publish();return state
 finally:events.close()


def self_test(root):
 # One targeted decision replay; no live owner/registered job is opened.
 metrics={k:dict(CI95_pp=[-.2,.2]) for k in METRICS};metrics['score']={'CI95_pp':[-.5,-.1]}
 v=dict(vs_own10k=dict(target_PDMS=91.,baseline_PDMS=91.3,metrics=metrics))
 at20=decision(20000,v);at35=decision(35000,v,v);assert not at20['automatic_stop_eligible'] and at35['automatic_stop_eligible']
 gain=json.loads(json.dumps(v));gain['vs_own10k']['metrics'][SAFETY[0]]['CI95_pp']=[.01,.1]
 assert not decision(35000,gain,v)['automatic_stop_eligible']
 best=json.loads(json.dumps(v));best['vs_own10k']['target_PDMS']=91.4
 assert not decision(35000,best,v)['automatic_stop_eligible']
 # Exercise inotify against a temporary CPU-only evidence directory.
 import tempfile,threading
 with tempfile.TemporaryDirectory() as t:
  e=Events();e.add(t);thread=threading.Thread(target=lambda:atomic(Path(t)/'SUMMARY.json',dict(status='COMPLETE')));thread.start();relevant=e.wait(2);thread.join();e.close();assert relevant
 result=dict(status='PASS',CPU_only=True,test='20k recommendation-only;35k conjunction;clear safety gain/new best protect;inotify event works',live_job_or_checkpoint_touched=False,source_sha256=sha(__file__),at=now())
 atomic(Path(root)/'analysis/SAFETY_EVIDENCE_REVIEW_CPU_CHECK.json',result);return result


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default='/mnt/project/qwen-world-rl-optimization-20261030');p.add_argument('--hours',type=float,default=24)
 p.add_argument('--allow-cooperative-stop',action='store_true');p.add_argument('--print-policy',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if not 0<a.hours<=24:p.error('--hours must be >0 and <=24')
 result=POLICY if a.print_policy else self_test(a.root) if a.self_test else run(a)
 print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
