"""Complete-only CPU matched zero/retention review of warmstarted Rescue pair."""
import json,re,sys,threading
from pathlib import Path
import numpy as np
from safety_evidence_reviewer import complete_input,load_metrics,atomic,sha,now
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');ARMS=('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM');METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
LOCK=threading.RLock();CACHE={};PATH=R/'analysis/RESCUE_PAIR_MATCHED_REVIEW.json'
def physical(x):return re.sub(r'_\d+_\d+$','',x)
def read(p):return json.loads(Path(p).read_text())
def train_value(arm,update):
 label=f'{arm}_u{update:06}_TRAIN1024';specpath=R/'evaluation_specs'/f'{label}.json'
 if not specpath.exists():return None
 spec=read(specpath);out=Path(spec['output']);summarypath=out/'SUMMARY.json';bankpath=Path(spec['training_bank'])
 if not summarypath.exists() or not bankpath.exists():return None
 s=read(summarypath);b=read(bankpath)
 if s.get('status')!='COMPLETE' or s.get('scenes')!=1024 or s.get('invalid_scenes',0)!=0:return None
 tokens=read(R/'data/panels/TRAIN_DIAGNOSTIC_001024.json')['tokens']
 if set(b['records'])!=set(tokens):raise ValueError('TRAIN panel must be full1024')
 return dict(label=label,values=np.array([[b['records'][t]['metrics'][k] for k in METRICS] for t in tokens]),tokens=tokens,logs=np.array([physical(b['records'][t]['log']) for t in tokens]),
  provenance=dict(summary=str(summarypath),summary_sha256=sha(summarypath),bank=str(bankpath),bank_sha256=sha(bankpath)),protocol=b['inference']['protocol_sha256'])
def value(arm,update,cohort):
 key=(arm,update,cohort)
 if key in CACHE:return CACHE[key]
 if cohort=='NAVTEST':
  info=complete_input(R,arm,update)
  if info is None:return None
  v=load_metrics(R,info);v['label']=info['label'];v['protocol']=info['identity']['protocol']['sha256']
 else:v=train_value(arm,update)
 if v is not None:CACHE[key]=v
 return v

def contrast(a,b,mask=None):
 if a['tokens']!=b['tokens'] or not np.array_equal(a['logs'],b['logs']) or a['protocol']!=b['protocol']:raise ValueError('Matched scene/context/protocol changed')
 mask=np.ones(len(a['tokens']),bool) if mask is None else mask;delta=a['values']-b['values'];names=list(dict.fromkeys(a['logs']));idx=np.array([names.index(x) for x in a['logs']]);n=np.bincount(idx[mask],minlength=len(names));sums=np.zeros((len(names),7));np.add.at(sums,idx[mask],delta[mask])
 draws=np.random.default_rng(42).integers(0,len(names),size=(2000,len(names)));den=n[draws].sum(1);boots=100*sums[draws].sum(1)/den[:,None];lo,hi=np.quantile(boots,[.025,.975],axis=0)
 old=b['values'][mask,0];new=a['values'][mask,0]
 return dict(target=a['label'],baseline=b['label'],scenes=int(mask.sum()),physical_log_clusters=len(names),bootstrap=2000,seed=42,
  target_PDMS=float(new.mean()*100),baseline_PDMS=float(old.mean()*100),delta_PDMS=float((new-old).mean()*100),CI95_PDMS=[float(lo[0]),float(hi[0])],
  components={k:dict(delta_pp=float(delta[mask,i].mean()*100),CI95_pp=[float(lo[i]),float(hi[i])]) for i,k in enumerate(METRICS)},
  zeros=dict(baseline=int((old<=1e-12).sum()),target=int((new<=1e-12).sum()),repaired=int(((old<=1e-12)&(new>1e-12)).sum()),new_zero=int(((old>1e-12)&(new<=1e-12)).sum()),
   high_to_zero=int(((old>=.9)&(new<=1e-12)).sum())),
  uncertainty='fixedseed physical-log cluster bootstrap,scene weighted; no across-training-seed claim')

def analyze_ready():
 with LOCK:
  d=read(PATH) if PATH.exists() else dict(schema='world_warmstarted_rescue_pair_matched_zero_retention_review_v1',status='PARTIAL',contrasts={},TRAIN_diagnostics={},
    baseline='SafetyEP10k fixed,current initialization knowledge10k +phase updates; not a freshIL70 same-budget FULL comparison',phase_gates=[1000,5000],high_cohort_definition='fixed Safety10k canonical score>=.9',
    TRAIN_not_heldout=True,Test_selection_used=True,NAVTEST_used_for_training_scene_selection=False,new_model_forwards=0,new_scores=0)
  for cohort,key in [('NAVTEST','contrasts'),('TRAIN_DIAGNOSTIC','TRAIN_diagnostics')]:
   base=value('PPO_SAFETY_EP',10000,cohort)
   if base is None:raise ValueError('Fixed complete Safety10k baseline missing')
   high=base['values'][:,0]>=.9
   for u in [1000,5000]:
    vals={a:value(a,u,cohort) for a in ARMS}
    for arm,v in vals.items():
     label=f'{arm}_phase{u}_minus_SAFETY10k'
     if v is None or label in d[key]:continue
     out=contrast(v,base);out['fixed_SAFETY10k_high_ge09']=contrast(v,base,high);out['provenance']=dict(target=v['provenance'],baseline=base['provenance'])
     if cohort=='TRAIN_DIAGNOSTIC':
      manifest=read(R/'data/rescue_retention/TRAIN_RESCUE_RETENTION_20261031/manifest.json');lookup={t:i for i,t in enumerate(v['tokens'])};out['original9_low_scene_diagnostics']={x['token']:dict(category=x['category'],baseline_score=x['canonical_SAFETY10k_score'],target_metrics=dict(zip(METRICS,v['values'][lookup[x['token']]].tolist())),repaired_zero=bool(x['canonical_SAFETY10k_score']<=1e-12 and v['values'][lookup[x['token']],0]>1e-12)) for x in manifest['low_scene_audit'] if x['canonical_SAFETY10k_score']<.95}
     if cohort=='TRAIN_DIAGNOSTIC':
      low=out['original9_low_scene_diagnostics'];zero=[x for x in low.values() if x['baseline_score']<=1e-12];pos=[x for x in low.values() if x['baseline_score']>1e-12]
      out['original_low_cohort_counts']=dict(original_zero=len(zero),zero_repaired=sum(x['repaired_zero'] for x in zero),zero_remaining=sum(x['target_metrics']['score']<=1e-12 for x in zero),
        original_positive_low=len(pos),positive_low_improved=sum(x['target_metrics']['score']>x['baseline_score']+1e-12 for x in pos),positive_low_degraded=sum(x['target_metrics']['score']<x['baseline_score']-1e-12 for x in pos))
     d[key][label]=out
    label=f'PPO_RESCUE_CFM_minus_REPLAY_phase{u}'
    if all(v is not None for v in vals.values()) and label not in d[key]:
     d[key][label]=contrast(vals[ARMS[1]],vals[ARMS[0]]);d[key][label]['fixed_SAFETY10k_high_ge09']=contrast(vals[ARMS[1]],vals[ARMS[0]],high)
  d['status']='COMPLETE' if len(d['contrasts'])==6 and len(d['TRAIN_diagnostics'])==6 else 'PARTIAL';d['updated_at']=now();atomic(PATH,d)
  return d
if __name__=='__main__':print(json.dumps(analyze_ready()))
