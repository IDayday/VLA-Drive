"""One finite CPU readout of already COMPLETE, stopped LR/data fork results.
Never dispatch, revive, score, mutate oldreview/controller files, or waitforlabels.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,sys
import numpy as np
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from safety_evidence_reviewer import complete_input,load_metrics,atomic,sha
from lr_study_matched_analyzer import bootstrap_delta,failure_changes,context_equal
from analyze_portfolio_10k import load_once
R=A.parent;OUT=A/'FROZEN_LR_EXISTING_EVIDENCE_REVIEW_20261031.json';ARMS=('PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS','PPO_SAFETY_DATA','PPO_SAFETY_LRDATA')
oldpath=A/'LR_STUDY_MATCHED_REVIEW_20261031.json';old=json.loads(oldpath.read_text());cache={}
def val(arm,u):
 key=(arm,u)
 if key in cache:return cache[key]
 info=complete_input(R,arm,u)
 if info is None:return None
 v=load_metrics(R,info);cache[key]=v;return v
parent=load_once('IL70_NAVTEST');tokens=parent['identity']['tokens'];high=parent['matrix'][:,0]>=.9
peak=val('PPO_SAFETY_EP',10000);missing=[];points=[];contrasts={}
for arm in ARMS:
 for u in (12000,14000):
  v=val(arm,u)
  if v is None:
   missing.append(dict(arm=arm,updates=u,status='NO_ACTUAL_COMPLETE_RESULT_DO_NOT_WAIT_OR_RESUME',not_imputed=True));continue
  if v['tokens']!=tokens or v['hashes']!=[parent['rows'][t]['feedback']['cache']['sha256'] for t in tokens] or v['info']['identity']['protocol']['sha256']!=parent['identity']['protocol']['sha256']:
   raise ValueError('ILcohort/token/protocol/cache join mismatch')
  row=dict(arm=arm,updates=u,additional_updates_since10k=u-10000,PDMS=v['info']['summary']['PDMS'],zero=int((v['values'][:,0]<=1e-12).sum()),
   means=v['info']['summary']['means'],provenance=v['provenance'],exceeds_Safety10k_peak=v['info']['summary']['PDMS']>peak['info']['summary']['PDMS'])
  il_delta=v['values']-parent['matrix'];il_metrics,nb=bootstrap_delta(il_delta,v['logs'],high)
  row['fixed_IL70_high_ge09']=dict(scenes=int(high.sum()),target_PDMS=float(v['values'][high,0].mean()*100),IL70_PDMS=float(parent['matrix'][high,0].mean()*100),
   delta_PDMS=il_metrics['score']['delta_pp'],CI95_PDMS=il_metrics['score']['CI95_pp'],metrics=il_metrics,
   target_below09=int((v['values'][high,0]<.9).sum()),target_zero=int((v['values'][high,0]<=1e-12).sum()),
   definition='fixed originalIL70 score>=.9; conditionaldiagnostic,notTRAINselection')
  points.append(row)
  for suffix,b in [('SAFETY_same_updates',val('PPO_SAFETY_EP',u)),('fixed_SAFETY10k',peak)]:
   if b is None:raise ValueError('ExpectedalreadycompleteSafetycontrolmissing')
   context_equal(v,b);key=f'{arm}_u{u:06}_minus_{suffix}';out=copy.deepcopy(old['contrasts'][key])
   if out['provenance']['target']['summary_seal']!=v['info']['summary']['sha256'] or out['provenance']['baseline']['summary_seal']!=b['info']['summary']['sha256']:
    raise ValueError('Previouslycomputedpair ancestrychanged')
   # Preserve preexisting whole-population bootstrap exactly. Only the newly
   # requested fixed-ILhighcohort is calculated here fromcompactcachedmetrics.
   m,n=bootstrap_delta(v['values']-b['values'],v['logs'],high)
   out.update(status='COMPLETE_EXISTING_RESULT_REUSED',target_PDMS=v['info']['summary']['PDMS'],baseline_PDMS=b['info']['summary']['PDMS'],
    overall_bootstrap_recomputed=False,
    fixed_IL70_high_ge09=dict(scenes=int(high.sum()),target_PDMS=float(v['values'][high,0].mean()*100),baseline_PDMS=float(b['values'][high,0].mean()*100),
      delta_PDMS=m['score']['delta_pp'],CI95_PDMS=m['score']['CI95_pp'],metrics=m,zeros=failure_changes(v,b,high),valid_bootstrap_draws=n,
      definition='fixed originalIL70score>=.9,comparetargetvsdeclaredbaselinewithincohort'),
    existing_pair_source=dict(path=str(oldpath),file_sha256=sha(oldpath),key=key),
    budget_interpretation='sameupdatesisolatesregisteredpost10k change; fixed10k ispeakretention comparison withdifferenttrainingbudget')
   contrasts[key]=out
for p in points:
 same=contrasts[f"{p['arm']}_u{p['updates']:06}_minus_SAFETY_same_updates"]
 p['vs_samebudget_Safety']=dict(delta_PDMS=same['delta_PDMS'],CI95_PDMS=same['CI95_PDMS'],NC=same['metrics']['no_at_fault_collisions'],DAC=same['metrics']['drivable_area_compliance'],TTC=same['metrics']['time_to_collision_within_bound'],EP=same['metrics']['ego_progress'])
 p['vs_Safety10k']=dict(delta_PDMS=contrasts[f"{p['arm']}_u{p['updates']:06}_minus_fixed_SAFETY10k"]['delta_PDMS'],CI95_PDMS=contrasts[f"{p['arm']}_u{p['updates']:06}_minus_fixed_SAFETY10k"]['CI95_PDMS'])
best=max(points,key=lambda p:p['PDMS'])
out=dict(schema='world_stopped_LR_data_completed_evidence_review_v1',status='COMPLETE_FINITE_EXISTING_POINTS_ONLY',created_at=datetime.now(timezone.utc).isoformat(),
 observed_points=points,missing_unobserved_points=missing,contrasts=contrasts,
 conclusions_zh=[
  '七个现成完整12/14k结果中没有超过Safety10k峰91.3498917370；本集合最高DATA14k91.0810526533。未完成LRDATA14k不补点、不等待标签、不恢复评测。',
  'LOW12k相对同预算Safety +.151588，CI[+.023512,+.279245]，是有限低步长保留收益证据；14k−.029542且CI跨0，未持续为14k总分优势；两点相对10k峰均明确下降。',
  'COS12/14k相对Safety均为小负差且CI跨0，没有cosine更优证据；已观测区间分别只走到cosinecontinuation的20%/40%，不能伪装完整10k schedule结论。',
  'DATA12/14k EP相对同预算Safety +.772879/+.757086，但NC和TTC均有明确负向CI；不能只按总分或EP认定80/10/10可安全迁移。',
  'LRDATA12k相对Safety +.180991但CI跨0，未超过10k峰；14k缺失使持续性与后期交互未知。',
  '原四臂停止原因是用户改焦点，不能改标算法失败。它们是同Safety10k exact-nativeAdam/RNG/cursor fork，samebudget对照可解释post10k干预；不是freshIL学习率从0开始的完整检验，也不是旧weights-onlyRescue。',
 ],
 fresh_schedule_design=dict(recommendation='OPTIONAL_SINGLE_FACTOR_LATER_NOT_REGISTERED_NOT_RESTART_OLD_ARMS',strength='limitedtransientLOW12kevidence,notnewbestornotcosineproof',
  priority='currentfreshSuccessMemory/FPOandP0gradientdiagnosisremainpriority; noextrapolationto92',
  if_authorized='sameIL70freshAdam/RNG42/update0,uniformfull/W3/SafetyA/IL70anchor unchanged; onegroup-relativeLRamplitude/schedulechangeonly; matchedcomplete1k/5k/10kSafetycontrol',
  avoid=['RL10kforkcalledfreshtraining','repeatoldstoppedGPUjobs','claimLOW12kgainpersistentat14k','combine80/10/10withoutNC/TTCguard','mixreference/teacherrefreshwithLRchange'],
  measured_future='bothgrouplrs/clock, actualPG-anchor gradientnorm+cos/update,canonicalPDMS+safety/EP+fixedILhighretention+repaired/newzero',
  effect_scale='LOW.2 action1e-5/Memory1e-6 onlyafterglobal10k inthishistory; freshwarmup+lowfrom0remainsuntested'),
 best_in_review=dict(arm=best['arm'],updates=best['updates'],PDMS=best['PDMS'],exceeds_historical_Safety10k_peak=best['exceeds_Safety10k_peak']),
 comparisons=dict(controls='samebudgetSafety12/14; fixedSafety10kpeakseparately',all_complete_scenes=12146,physical_logs=44,bootstrap=2000,seed=42,evaluation_seed=2101,
  inference='singleODE10shift5',protocol_sha256=peak['info']['identity']['protocol']['sha256'],fixed_IL_high_count=int(high.sum()),
  source_LR_review_status=old['status'],alltokens_logs_protocol_cachehashesmatched=True,uncertainty='singletrainingseed,conditionalILhighcohort,exploratorycomparisonsnotmultiplicitycorrected'),
 provenance=dict(prior_pair_review=str(oldpath),prior_pair_review_sha256=sha(oldpath),IL70=parent['provenance'],script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
  registration=str(R/'EXPERIMENT_LR_STUDY_REGISTRATION_20261031.json')),
 scope=dict(CPU_only=True,new_GPU_forwards=0,new_scores=0,oldwholepopulationbootstrap_reused=True,originalscene_population_rescans=0,training_resumed=False,evaluation_dispatched=False,stopped_source_or_review_mutated=False,git_stage=False,git_commit=False))
out['sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();atomic(OUT,out)
print(json.dumps(dict(path=str(OUT),status=out['status'],best=out['best_in_review'],ILhigh=int(high.sum()),points=len(points),contrasts=len(contrasts),missing=missing),ensure_ascii=False))
for k,v in contrasts.items():
 if k.endswith('SAFETY_same_updates'):
  h=v['fixed_IL70_high_ge09'];print(k,'ILhigh',h['delta_PDMS'],h['CI95_PDMS'])
