"""Finite complete-only CPU review of exact-native Safety10k LR/data forks.

No dispatch, scoring, model load, signals, journals, training barriers or stop
policy. One new complete scene population read into the shared compact cache.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from safety_evidence_reviewer import complete_input,load_metrics,atomic,sha,now,METRICS
from navtest_cadence_owner_v2 import Events
R=Path('/mnt/project/qwen-world-rl-optimization-20261030')
O=R/'owners/LR_STUDY_ANALYSIS';OUT=A/'LR_STUDY_MATCHED_REVIEW_20261031.json'
PLAN=A/'LR_STUDY_ANALYZER_LAUNCH_PLAN_20261031.json'
ARMS=('PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS','PPO_SAFETY_DATA','PPO_SAFETY_LRDATA')
CONTROL='PPO_SAFETY_EP';GATES=(12000,14000,16000,18000,20000)
SCHEMA='world_exact_native_Safety10k_LR_data_matched_review_v1'
CACHE={}

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default

def label(a,u):return f'{a}_u{u:06}_NAVTEST'

def tasks():
 t={}
 for u in GATES:
  for a in ARMS:
   t[f'{a}_u{u:06}_minus_SAFETY_same_updates']=(label(a,u),label(CONTROL,u))
   t[f'{a}_u{u:06}_minus_fixed_SAFETY10k']=(label(a,u),label(CONTROL,10000))
  t[f'LRDATA_minus_LRLOW_u{u:06}_data_increment']=(label(ARMS[3],u),label(ARMS[0],u))
  t[f'LRDATA_minus_DATA_u{u:06}_LR_increment']=(label(ARMS[3],u),label(ARMS[2],u))
  # Factorial difference: [LRDATA-DATA]-[LRLOW-Safety] = combo-low-data+control.
  t[f'LR_data_interaction_u{u:06}']=(label(ARMS[3],u),label(ARMS[0],u),label(ARMS[2],u),label(CONTROL,u))
 return t
TASKS=tasks();LABELS=sorted({x for v in TASKS.values() for x in v})

def context_equal(a,b):
 x,y=a['info']['identity'],b['info']['identity']
 for k in ('cohort','tokens','count','token_sha256','data_sha256','population_sha256','seed','numerics','qualification','inference'):
  if x[k]!=y[k]:raise ValueError('Matched evaluation context mismatch: '+k)
 if x['protocol']['sha256']!=y['protocol']['sha256'] or a['hashes']!=b['hashes'] or not np.array_equal(a['logs'],b['logs']):
  raise ValueError('Matched physical logs/canonical cache/protocol mismatch')

def bootstrap_delta(delta,logs,mask):
 names=list(dict.fromkeys(logs))
 if len(names)!=44:raise ValueError('Expected44 physical logs')
 idx=np.array([names.index(x) for x in logs]);n=np.bincount(idx[mask],minlength=44);sums=np.zeros((44,7))
 np.add.at(sums,idx[mask],delta[mask]);draws=np.random.default_rng(42).integers(0,44,(2000,44));den=n[draws].sum(1)
 valid=den>0
 if not mask.any() or not valid.any():raise ValueError('Empty prespecified cohort')
 boot=100*sums[draws[valid]].sum(1)/den[valid,None];lo,hi=np.quantile(boot,[.025,.975],axis=0)
 return {k:dict(delta_pp=float(delta[mask,j].mean()*100),CI95_pp=[float(lo[j]),float(hi[j])]) for j,k in enumerate(METRICS)},int(valid.sum())

def failure_changes(a,b,mask):
 old=b['values'][mask];new=a['values'][mask];x=old[:,0];y=new[:,0];eps=1e-12
 components={k:dict(baseline_failed=int((old[:,j]<1-eps).sum()),target_failed=int((new[:,j]<1-eps).sum()),
  repaired=int(((old[:,j]<1-eps)&(new[:,j]>=1-eps)).sum()),new_failure=int(((old[:,j]>=1-eps)&(new[:,j]<1-eps)).sum())) for j,k in enumerate(METRICS) if j in (1,2,3)}
 return dict(baseline=int((x<=eps).sum()),target=int((y<=eps).sum()),repaired=int(((x<=eps)&(y>eps)).sum()),
  new_zero=int(((x>eps)&(y<=eps)).sum()),high_to_zero=int(((x>=.9)&(y<=eps)).sum()),components=components)

def contrast(a,b,high):
 context_equal(a,b);delta=a['values']-b['values'];allmask=np.ones(len(a['tokens']),bool)
 metrics,n=bootstrap_delta(delta,a['logs'],allmask);hm,hn=bootstrap_delta(delta,a['logs'],high)
 def high_stats(v):
  x=v['values'][high,0]
  return dict(PDMS=float(x.mean()*100),remaining_ge09=int((x>=.9).sum()),below09=int((x<.9).sum()),zero=int((x<=1e-12).sum()))
 d=delta[:,0]
 return dict(target=a['info']['label'],baseline=b['info']['label'],scenes=len(d),physical_logs=44,bootstrap=2000,seed=42,valid_bootstrap_draws=n,
  target_PDMS=float(a['values'][:,0].mean()*100),baseline_PDMS=float(b['values'][:,0].mean()*100),
  delta_PDMS=metrics['score']['delta_pp'],CI95_PDMS=metrics['score']['CI95_pp'],metrics=metrics,
  zeros=failure_changes(a,b,allmask),gain_loss=dict(G_plus_PDMS=float(np.maximum(d,0).mean()*100),G_minus_PDMS=float(-np.minimum(d,0).mean()*100),
   improved=int((d>1e-12).sum()),degraded=int((d < -1e-12).sum()),unchanged=int((np.abs(d)<=1e-12).sum())),
  fixed_SAFETY10k_high_ge09=dict(definition='fixed parent Safety10k canonical score>=.9',scenes=int(high.sum()),metrics=hm,
   target=high_stats(a),baseline=high_stats(b),zeros=failure_changes(a,b,high),valid_bootstrap_draws=hn),
  provenance=dict(target=a['provenance'],baseline=b['provenance']),matched_tokens_context_protocol_cache_hashes=True)

def interaction(vs,high):
 a,b,c,d=vs
 for v in (b,c,d):context_equal(a,v)
 diff=a['values']-b['values']-c['values']+d['values'];m,n=bootstrap_delta(diff,a['logs'],np.ones(len(a['tokens']),bool));h,hn=bootstrap_delta(diff,a['logs'],high)
 return dict(formula='[LRDATA-DATA]-[LRLOW-originalSafety] = LRDATA-LRLOW-DATA+originalSafety',
  labels=[v['info']['label'] for v in vs],scenes=12146,physical_logs=44,bootstrap=2000,seed=42,valid_bootstrap_draws=n,
  metrics=m,delta_PDMS=m['score']['delta_pp'],CI95_PDMS=m['score']['CI95_pp'],
  fixed_SAFETY10k_high_ge09=dict(scenes=int(high.sum()),metrics=h,valid_bootstrap_draws=hn),
  provenance=[v['provenance'] for v in vs],matched_tokens_context_protocol_cache_hashes=True,
  interpretation='加性尺度上的交互；不以各arm单独CI相减代替同44log配对重采样。非跨训练seed结论。')

def info_once(lab):
 if lab in CACHE:return CACHE[lab]
 a,u=lab.rsplit('_u',1);u=int(u.split('_',1)[0]);info=complete_input(R,a,u)
 if info is None:return None
 CACHE[lab]=load_metrics(R,info);return CACHE[lab]

def real_scientific_failure(lab):
 """Actual model-child failures only. Refused resource75 is waiting, not failure."""
 spec=read(R/'evaluation_specs'/(lab+'.json'),{})
 summary=read(Path(spec['output'])/'SUMMARY.json',{}) if spec else {}
 if spec and summary.get('status')!='COMPLETE':
  for j in spec.get('jobs',[]):
   receipt=Path(j['receipt'] if isinstance(j,dict) else j);r=read(receipt/'RESULT.json',{})
   code=r.get('returncode')
   if code not in (None,0) and not (code==75 and r.get('child_not_started') is True):return dict(kind='actual_evaluation_child_failure',receipt=str(receipt),result=r)
 a,u=lab.rsplit('_u',1);u=int(u.split('_',1)[0])
 if a in ARMS:
  state=read(R/'owners'/a/'STATE.json',{})
  if state.get('status')=='FAILED_REQUIRES_REVIEW' and int(state.get('updates',10000))<u:
   return dict(kind='actual_native_arm_failed_before_gate',owner=str(R/'owners'/a/'STATE.json'),status=state['status'],updates=state.get('updates'),error=state.get('error'))
 return None

def initial():
 return dict(schema=SCHEMA,status='WAITING_ACTUAL_COMPLETE_RESULTS',updated_at=now(),contrasts={},factorial_interactions={},waiting={},unavailable={},
  study=dict(arms=list(ARMS),global_update_gates=list(GATES),native_parent='PPO_SAFETY_EP_u010000',initial_global_updates=10000,cap=20000,
   optimizer='inherited_AdamW',rng='inherited_all_native_per_rank',phase_reset=False,reference='original_frozen_IL70',W3_unchanged=True,
   designs=dict(LRLOW='0.2constant/full',LRCOS='1.0->0.1relativecosine/full',DATA='1constant/static80full10low10high',LRDATA='.2constant/static80full10low10high'),
   sampler_data_fork='nativecursor20000/stage2; scientific policy change explicitly registered; noCFM',expected_contrasts=50,expected_interactions=5),
  protocol=dict(population='full12146 NAVTEST',physical_logs=44,bootstrap=2000,seed=42,eval_seed=2101,inference='singleODE10 shift5',
   estimate='44physical-log cluster resampling; scene weighted ratio-of-sums, same draws for every metric and interaction',
   high_cohort='fixed Safety10k score>=.9, never selected from a new target',same_budget_control_required=True,no_missing_control_substitution=True,
   no_partial_or_training_reward_ranking=True,uncertainty='single training/eval seeds, no multiplecomparison correction; exploratory research selection usesTest'),
  guardrails_zh=['不把候选metrics条件信用分配表述为已具备场景风险推理。','不把低LR或sampler改变等同知识已充分；单项收益与交互须分开。',
   'Safety10k native Adam/RNG/cursor完整继承，global12k是追加2k，不是freshIL12k。','原Safety同u未完整就等待，固定10k仅单独保留比较。',
   '未齐全不记COMPLETE，不自动停止/延长/派发。'],
  scope=dict(CPU_only=True,new_forwards=0,new_scores=0,new_seeds=0,journal_reads=0,GPU_process_actions=0,actual_complete_populations_only=True),
  provenance=dict(script=str(Path(__file__).resolve()),script_sha256=sha(__file__),registration=str(R/'EXPERIMENT_LR_STUDY_REGISTRATION_20261031.json')))

def analyze_once(doc):
 parent=info_once(label(CONTROL,10000))
 if parent is None:raise ValueError('Verified fixed Safety10k baseline missing')
 high=parent['values'][:,0]>=.9;changed=False;waiting={}
 for key,labs in TASKS.items():
  dest=doc['factorial_interactions'] if len(labs)==4 else doc['contrasts']
  if key in dest or key in doc['unavailable']:continue
  vals=[info_once(x) for x in labs];missing=[lab for lab,v in zip(labs,vals) if v is None]
  bad={lab:x for lab in missing if (x:=real_scientific_failure(lab)) is not None}
  if bad:
   doc['unavailable'][key]=dict(status='UNAVAILABLE_SCIENTIFIC_FAILURE_NO_SUBSTITUTION',failures=bad,labels=labs,at=now());changed=True;continue
  if missing:waiting[key]=dict(labels=labs,missing_actual_complete_labels=missing);continue
  for v in vals:context_equal(v,parent)
  dest[key]=interaction(vals,high) if len(labs)==4 else contrast(*vals,high)
  dest[key]['completed_at']=now();changed=True
 doc['waiting']=waiting;done=len(doc['contrasts'])+len(doc['factorial_interactions']);missing=len(waiting)
 processed={label(CONTROL,10000)}
 for k in (*doc['contrasts'],*doc['factorial_interactions']):processed.update(TASKS[k])
 doc['progress']=dict(complete_contrasts=len(doc['contrasts']),complete_interactions=len(doc['factorial_interactions']),actual_complete_results_processed=len(processed),
  expected_tasks=len(TASKS),unavailable_tasks=len(doc['unavailable']),waiting_tasks=missing)
 if done==len(TASKS):doc['status']='COMPLETE'
 elif done+len(doc['unavailable'])==len(TASKS):doc['status']='FINISHED_WITH_SCIENTIFIC_FAILURE_MISSING_RESULTS'
 else:doc['status']='PARTIAL_ACTUAL_RESULTS' if done else 'WAITING_ACTUAL_COMPLETE_RESULTS'
 doc['updated_at']=now();atomic(OUT,doc)
 return changed

def refresh():
 atomic(R/'reporting/refresh_request.json',dict(id='lr-study-matched-'+uuid.uuid4().hex,at=now(),scientific_result=False,
  evidence_kind='completed_supplemental_analysis',reason='原生Safety10k学习率/数据同预算配对分析更新',evidence=str(OUT)))

def watch_paths(ev):
 ev.add(R/'evaluation_specs',{x+'.json' for x in LABELS});ev.add(R/'owners/LR_STUDY',{'STATE.json'});ev.add(R/'owners/CAMPAIGN',{'STATE.json'})
 ev.add(R/'owners/NAVTEST_CADENCE',{'STATE.json'});ev.add(R,{'RUNS.json'})
 for a in ARMS:ev.add(R/'owners'/a,{'STATE.json'})
 for lab in LABELS:
  spec=read(R/'evaluation_specs'/(lab+'.json'),{})
  if not spec:continue
  out=Path(spec['output']);ev.add(out.parent,{out.name});ev.add(out,{'SUMMARY.json','identity.json'})
  for j in spec.get('jobs',[]):ev.add(Path(j['receipt'] if isinstance(j,dict) else j),{'RESULT.json'})

def run(hours):
 O.mkdir(parents=True,exist_ok=True)
 with (O/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),started_at=now(),script=str(Path(__file__).resolve()),
   script_sha256=sha(__file__),max_hours=hours,CPU_only=True,event_driven=True,output=str(OUT))
  atomic(O/'OWNER.json',owner);doc=read(OUT,initial())
  if doc.get('schema')!=SCHEMA:raise ValueError('Different study result schema')
  if doc['provenance']['script_sha256']!=sha(__file__):raise ValueError('Analyzer source changed since evidence publication')
  deadline=time.monotonic()+hours*3600;ev=Events()
  try:
   watch_paths(ev)
   while True:
    changed=analyze_once(doc)
    if changed:refresh()
    atomic(O/'STATE.json',dict(status=doc['status'],at=now(),progress=doc['progress'],output=str(OUT),owner=owner))
    if doc['status'].startswith(('COMPLETE','FINISHED')):break
    remaining=deadline-time.monotonic()
    if remaining<=0:
     doc.update(status='TIMED_OUT_WITH_MISSING_RESULTS',ended_at=now(),missing_results_not_imputed=True);atomic(OUT,doc);refresh();break
    watch_paths(ev)
    #1800s metadata fallback handles remoteNFS publications without journal polling.
    ev.wait(min(remaining,1800));watch_paths(ev)
  except Exception:
   doc.update(status='ANALYSIS_FAILED_REQUIRES_REVIEW',error=traceback.format_exc(),ended_at=now());atomic(OUT,doc);refresh();raise
  finally:
   ev.close();atomic(O/'STATE.json',dict(status=doc['status'],ended_at=now(),progress=doc.get('progress'),output=str(OUT),owner=owner))
 return doc['status']

def prepare(hours):
 d=dict(schema='LR_study_matched_analyzer_launch_plan_v1',status='READY_FOR_ROOT_REVIEW_NOT_STARTED',at=now(),script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
  output=str(OUT),arms=list(ARMS),global_gates=list(GATES),matched_contrasts=50,interactions=5,maximum_hours=hours,
  launch_argv=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python',str(Path(__file__).resolve()),'--launch','--max-hours',str(hours)],
  owner=str(O/'OWNER.json'),launch=str(O/'LAUNCH.json'),state=str(O/'STATE.json'),no_GPU_dispatch=True,no_journal_polling=True,
  refresh_scientific_result=False,all_actual_results_required_for_COMPLETE=True,missing_control='WAIT never substitute fixed10k',
  imports={str(A/'safety_evidence_reviewer.py'):sha(A/'safety_evidence_reviewer.py'),str(A/'navtest_cadence_owner_v2.py'):sha(A/'navtest_cadence_owner_v2.py')})
 atomic(PLAN,d);return d

def launch(hours):
 p=read(PLAN)
 if p['script_sha256']!=sha(__file__) or any(sha(k)!=v for k,v in p['imports'].items()):raise ValueError('Reviewed analysis source changed')
 O.mkdir(parents=True,exist_ok=True)
 with (O/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for old in (read(O/'OWNER.json',{}),read(O/'LAUNCH.json',{})):
   if not old:continue
   try:
    st=Path(f"/proc/{old['pid']}/stat").read_text().split()
    if int(st[21])==old['start_ticks'] and st[2] not in ('Z','X'):raise RuntimeError('Existing LR analyzer')
   except FileNotFoundError:pass
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run','--max-hours',str(hours)]
  with (O/'stdout.log').open('ab',buffering=0) as f:proc=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(pid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().split()[21]),started_at=now(),argv=cmd,script_sha256=sha(__file__),plan=str(PLAN));atomic(O/'LAUNCH.json',record)
  return record

def self_check():
 # Pure CPU stats signs/interaction, and missing controls create no point result.
 from unittest.mock import patch
 n=44*3;logs=np.repeat([str(x) for x in range(44)],3);tokens=[str(x) for x in range(n)];ids=dict(cohort='NAVTEST',tokens=tokens,count=n,token_sha256='t',data_sha256='d',population_sha256='p',seed=2101,numerics={},qualification=False,inference={},protocol=dict(sha256='p'))
 def toy(name,score):
  v=np.full((n,7),score,dtype=float)
  return dict(info=dict(identity=ids,label=name),values=v,tokens=tokens,logs=logs,hashes=['h']*n,provenance=dict(toy=True))
 a,b,c,d=[toy(x,y) for x,y in [('combo',.93),('low',.91),('data',.92),('control',.90)]]
 high=np.ones(n,bool);pair=contrast(a,b,high);assert abs(pair['delta_PDMS']-2)<1e-9 and abs(pair['CI95_PDMS'][0]-2)<1e-9
 x=interaction((a,b,c,d),high);assert abs(x['delta_PDMS'])<1e-9
 # Every same-u contrast explicitly names its controlu, notfixed10k.
 assert len(TASKS)==55 and len([v for v in TASKS.values() if len(v)==4])==5
 for u in GATES:
  for arm in ARMS:assert TASKS[f'{arm}_u{u:06}_minus_SAFETY_same_updates'][1]==label(CONTROL,u)
 doc=initial()
 with patch(__name__+'.info_once',side_effect=lambda lab:toy(lab,.91) if lab==label(CONTROL,10000) else None),patch(__name__+'.real_scientific_failure',return_value=None),patch(__name__+'.atomic'):
  analyze_once(doc)
 assert not doc['contrasts'] and not doc['factorial_interactions'] and len(doc['waiting'])==55 and doc['status']!='COMPLETE'
 result=dict(status='PASS',at=now(),script_sha256=sha(__file__),scope='CPUtoy44log component/pair/interaction signs; fixedparentcohort; all55 missing results never imputed; no model/GPU/real scene reads')
 atomic(A/'LR_STUDY_ANALYZER_CPU_CHECK_20261031.json',result);return result

def main():
 p=argparse.ArgumentParser(__doc__);g=p.add_mutually_exclusive_group(required=True)
 for a in ('prepare','self-check','launch','run','once'):g.add_argument('--'+a,action='store_true')
 p.add_argument('--max-hours',type=float,default=24);args=p.parse_args()
 if args.max_hours<=0:raise ValueError('Positive finite lifetime required')
 if args.prepare:out=prepare(args.max_hours)
 elif args.self_check:out=self_check()
 elif args.launch:out=launch(args.max_hours)
 elif args.run:out=run(args.max_hours)
 else:
  d=read(OUT,initial());analyze_once(d);out=d
 print(json.dumps(out,ensure_ascii=False,default=str))
if __name__=='__main__':main()
