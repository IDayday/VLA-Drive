"""Actual-complete CPU review, invoked asynchronously after pair evaluation.

No process signals/STOP files/dispatch; early-stop results are root decisions.
"""
import fcntl,json,re,threading,uuid
from pathlib import Path
import numpy as np
from rescue_pair_analyzer import value,contrast,train_value,METRICS
from safety_evidence_reviewer import atomic,sha,now
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis'
OUT=A/'SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json'
ARMS=('PPO_SAFETY_SUCCESS_MEMORY','PPO_SAFETY_FPO');GATES=(1000,5000,10000);CONTROL='PPO_SAFETY_EP'
CACHE={};LOCK=threading.RLock()

def read(p):return json.loads(Path(p).read_text())
def physical(s):return re.sub(r'_\d+_\d+$','',s)
def il_value(cohort):
 if cohort in CACHE:return CACHE[cohort]
 if cohort=='NAVTEST':
  from analyze_portfolio_10k import load_once
  p=load_once('IL70_NAVTEST');tokens=p['identity']['tokens']
  v=dict(label='IL70_NAVTEST',values=p['matrix'],tokens=tokens,logs=np.array([physical(p['rows'][t]['log']) for t in tokens]),
   protocol=p['identity']['protocol']['sha256'],hashes=[p['rows'][t]['feedback']['cache']['sha256'] for t in tokens],provenance=p['provenance'])
 else:
  spec=read(R/'IL_BASELINE_EVALUATION.json');path=Path(spec['training_bank']);b=read(path)
  tokens=read(R/'data/panels/TRAIN_DIAGNOSTIC_001024.json')['tokens']
  if set(tokens)!=set(b['records']):raise ValueError('IncompleteILTRAINpanel')
  v=dict(label='IL70_TRAIN1024',values=np.array([[b['records'][t]['metrics'][k] for k in METRICS] for t in tokens]),tokens=tokens,
   logs=np.array([physical(b['records'][t]['log']) for t in tokens]),protocol=b['inference']['protocol_sha256'],provenance=dict(bank=str(path),bank_sha256=sha(path)))
 CACHE[cohort]=v;return v

def matched(a,b):
 if a['tokens']!=b['tokens'] or a['protocol']!=b['protocol'] or not np.array_equal(a['logs'],b['logs']):raise ValueError('Exactmatchedcontextchanged')
 if 'hashes' in a and 'hashes' in b and a['hashes']!=b['hashes']:raise ValueError('Canonicalcachejoinchanged')

def early_stop_evidence(pair):
 high=pair['fixed_IL70_high_ge09'];conditions=dict(same_budget_PDMS_delta_below_minus1=pair['delta_PDMS'] < -1,
  same_budget_PDMS_CI_upper_negative=pair['CI95_PDMS'][1] < 0,
  fixed_IL_high_cohort_CI_upper_negative=high['CI95_PDMS'][1] < 0)
 return dict(status='ROOT_COOPERATIVE_EARLY_STOP_REVIEW' if all(conditions.values()) else 'CONTINUE_REGISTERED_FINITE_BUDGET_OR_ROOT_REVIEW',
  conditions=conditions,qualifies_for_root_review=all(conditions.values()),automatic_stop=False,
  definition='samebudgetSafetycontrol; fixedIL70score>=.9cohort, pairedCIupper<0; noextraeffectmargin guessed',
  scope='recommendationonly; rootownsSTOP_REQUESTED/nativecooperativeboundary; noSIGKILL/noalgorithmfutilityclaimfrompartialscore')

def annotate(a,b,high):
 matched(a,b);out=contrast(a,b);out['status']='COMPLETE';out['completed_at']=now()
 out['fixed_IL70_high_ge09']=contrast(a,b,high)
 out['provenance']=dict(target=a['provenance'],baseline=b['provenance'])
 out['canonical_targets']=dict(target_PDMS_ge92=out['target_PDMS']>=92,target_PDMS_ge94=out['target_PDMS']>=94,targets_are_not_claimed_achieved_beforecomplete=True)
 return out

def analyze_ready():
 with LOCK:
  with (A/'SAFETY_FRESH_PAIR_REVIEW.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   d=read(OUT) if OUT.exists() else dict(schema='world_fresh_IL70_Safety_success_FPO_matched_review_v1',status='PARTIAL',contrasts={},TRAIN_diagnostics={},early_stop_evidence={},completed_scores={},
    arms=list(ARMS),gates=list(GATES),final_registered_budget=10000,initialization='sameIL70,freshAdam/RNG42/update0; noRLwarmstart/nativefork',
    high_cohort_definition='fixedIL70canonicalscore>=.9; NAVTESTdiagnosticonly,nottrainingselection',
    estimator='scene-weighted physical-log paired bootstrap2000seed42; NAVTEST44logs; singletraining/evaluationseed',
    scope=dict(CPU_only=True,GPU_dispatch=0,process_signals=0,STOP_files=0,partial_scores_imputed=False,journal_reads=0,new_scores=0),
    final_target_zh='主目标PDMS>=92；94同时作为阈值显示，任何阶段分数都不承诺达到。')
   changed=False
   for cohort,key in [('NAVTEST','contrasts'),('TRAIN_DIAGNOSTIC','TRAIN_diagnostics')]:
    il=il_value(cohort);high=il['values'][:,0]>=.9
    for u in GATES:
     control=value(CONTROL,u,cohort);vals={a:value(a,u,cohort) for a in ARMS}
     for arm,v in vals.items():
      if v is None:continue
      if cohort=='NAVTEST':d['completed_scores'][v['label']]=dict(PDMS=float(v['values'][:,0].mean()*100),updates=u,
       ge92=bool(v['values'][:,0].mean()*100>=92),ge94=bool(v['values'][:,0].mean()*100>=94),provenance=v['provenance'])
      for basename,base in [('samebudgetSafetyEP',control),('IL70',il)]:
       name=f'{arm}_u{u:06}_minus_{basename}'
       if name in d[key] or base is None:continue
       d[key][name]=annotate(v,base,high);changed=True
       if cohort=='NAVTEST' and basename=='samebudgetSafetyEP' and u in (1000,5000):
        d['early_stop_evidence'][name]=early_stop_evidence(d[key][name])
     pair=f'SUCCESS_MEMORY_minus_FPO_u{u:06}'
     if pair not in d[key] and all(vals.values()):
      d[key][pair]=annotate(vals[ARMS[0]],vals[ARMS[1]],high);changed=True
   expected=15
   d['progress']=dict(complete_NAVTEST_contrasts=len(d['contrasts']),complete_TRAIN_contrasts=len(d['TRAIN_diagnostics']),expected_each=expected,
    complete_new_NAVTEST_models=len(d['completed_scores']),expected_new_models=6)
   d['status']='COMPLETE' if len(d['contrasts'])==expected and len(d['TRAIN_diagnostics'])==expected else 'PARTIAL_ACTUAL_COMPLETE_RESULTS_ONLY'
   d['updated_at']=now();d['source_sha256']=sha(__file__);atomic(OUT,d)
   if changed:
    atomic(R/'reporting/refresh_request.json',dict(id='Safetyfreshpair-'+uuid.uuid4().hex,scientific_result=False,
     evidence_kind='completed_supplemental_analysis',at=now(),evidence=str(OUT)))
   return d

if __name__=='__main__':print(json.dumps(analyze_ready(),ensure_ascii=False))
