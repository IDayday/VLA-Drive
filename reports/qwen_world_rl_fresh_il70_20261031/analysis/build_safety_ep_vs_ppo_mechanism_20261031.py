"""CPU-only, cache-only matched SafetyEP/PPO mechanism audit. No scoring/model load."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import yaml
from functools import lru_cache

ROOT=Path('/mnt/project/qwen-world-rl-optimization-20261030')
OUT=ROOT/'analysis/SAFETY_EP_VS_PPO_MECHANISM_20261031.json'
KEYS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
GATES=(1000,5000,10000,20000);EPS=1e-12
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
physical=lambda x:re.sub(r'_\d+_\d+$','',str(x))
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(path):
 d=read(path);b=dict(d);saved=b.pop('sha256',None)
 if saved!=fp(b):raise ValueError('Changed seal '+str(path))
 return d
def atom(path,value):
 p=Path(path);q=p.with_name('.'+p.name+'.'+str(os.getpid())+'.tmp')
 with q.open('w')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def differences(a,b,p=''):
 if isinstance(a,dict)and isinstance(b,dict):
  return [row for k in sorted(a.keys()|b.keys())for row in differences(a.get(k),b.get(k),p+'.'+k)]
 return []if a==b else [dict(path=p,PPO_FULL=a,SafetyEP=b)]

@lru_cache(None)
def nav(arm,u):
 from safety_evidence_reviewer import complete_input
 info=complete_input(ROOT,arm,u)
 if info is None:return None
 label=info['label'];cache=ROOT/'analysis/evaluation_metric_cache';p=cache/(label+'.npz');m=cache/(label+'.json')
 if not p.exists()or not m.exists():return None
 meta=read(m);identity=info['identity'];summary=info['summary']
 assert meta['identity_sha256']==identity['sha256']and meta['summary_sha256']==summary['sha256']and sha(p)==meta['npz_sha256']
 with np.load(p,allow_pickle=False)as z:values=z['metrics'].copy();tokens=z['tokens'].tolist();logs=z['logs'].tolist();hashes=z['cache_hashes'].tolist()
 assert values.shape==(12146,7)and tokens==identity['tokens']and len(set(tokens))==12146
 assert np.max(abs(values.mean(0)-np.array([summary['means'][k]for k in KEYS])))<1e-10
 return dict(label=label,values=values,tokens=tokens,logs=np.array([physical(x)for x in logs]),hashes=hashes,identity=identity,
  provenance=dict(summary=str(info['summarypath']),summary_file_sha256=sha(info['summarypath']),identity_seal=identity['sha256'],cache=str(p),cache_sha256=meta['npz_sha256']),raw_progress=None)

@lru_cache(None)
def train(label):
 path=ROOT/'evaluation_specs'/(label+'.json')
 if not path.exists():return None
 spec=read(path);summary_path=Path(spec['output'])/'SUMMARY.json';bank_path=Path(spec['training_bank'])
 if not summary_path.exists()or not bank_path.exists():return None
 summary=sealed(summary_path);bank=sealed(bank_path);identity=sealed(Path(spec['output'])/'identity.json')
 if summary.get('status')!='COMPLETE'or summary.get('scenes')!=1024 or summary.get('invalid_scenes')!=0:return None
 tokens=read(ROOT/'data/panels/TRAIN_DIAGNOSTIC_001024.json')['tokens'];assert len(tokens)==1024 and set(tokens)==set(bank['records'])
 assert summary['evaluation_identity_sha256']==identity['sha256']==bank['evaluation_identity_sha256']
 assert identity['tokens']==tokens and summary['protocol_sha256']==bank['inference']['protocol_sha256']
 values=np.array([[bank['records'][t]['metrics'][k]for k in KEYS]for t in tokens]);logs=np.array([physical(bank['records'][t]['log'])for t in tokens])
 assert np.max(abs(values.mean(0)-np.array([summary['means'][k]for k in KEYS])))<1e-10
 return dict(label=label,values=values,tokens=tokens,logs=logs,hashes=None,identity=identity,
  provenance=dict(summary=str(summary_path),summary_file_sha256=sha(summary_path),bank=str(bank_path),bank_file_sha256=sha(bank_path),identity_seal=identity['sha256']),raw_progress=None)

def fixed_IL_nav():
 old=read(ROOT/'analysis/PPO_FULL_u001000_vs_IL70.json')['provenance'];sp=Path(old['IL_summary']['path']);ip=Path(old['IL_identity']['path']);s=sealed(sp);i=sealed(ip)
 cache=ROOT/'analysis/evaluation_metric_cache/IL70_NAVTEST.npz';m=read(cache.with_suffix('.json'))
 assert m['identity_sha256']==i['sha256']and m['summary_sha256']==s['sha256']and m['npz_sha256']==sha(cache)
 with np.load(cache,allow_pickle=False)as z:values=z['metrics'].copy();tokens=z['tokens'].tolist();logs=z['logs'].tolist();hashes=z['cache_hashes'].tolist()
 return dict(values=values,tokens=tokens,logs=np.array([physical(x)for x in logs]),hashes=hashes,provenance=dict(summary=str(sp),identity=str(ip),cache=str(cache),cache_sha256=m['npz_sha256']))

def join(a,b):
 assert a['tokens']==b['tokens']and np.array_equal(a['logs'],b['logs'])
 if a['hashes']is not None and b['hashes']is not None:assert a['hashes']==b['hashes']
 ia,ib=a.get('identity'),b.get('identity')
 if ia and ib:
  for key in ('cohort','tokens','count','seed','inference','numerics','parent_model_sha256','data_sha256','population_sha256'):
   assert ia.get(key)==ib.get(key),key
  assert ia['protocol']['sha256']==ib['protocol']['sha256']

def paired(a,b,mask=None):
 join(a,b);nall=len(a['tokens']);mask=np.ones(nall,bool)if mask is None else mask
 assert mask.shape==(nall,)and mask.any()
 names=list(dict.fromkeys(a['logs']));idx=np.array([names.index(x)for x in a['logs']]);groups=len(names)
 delta=a['values']-b['values'];counts=np.bincount(idx[mask],minlength=groups);sums=np.zeros((groups,7));np.add.at(sums,idx[mask],delta[mask])
 draws=np.random.default_rng(42).integers(0,groups,size=(2000,groups));den=counts[draws].sum(1);valid=den>0
 boots=100*sums[draws[valid]].sum(1)/den[valid,None];lo,hi=np.quantile(boots,[.025,.975],axis=0)
 target=a['values'][mask];base=b['values'][mask];oldzero=base[:,0]<=EPS;newzero=target[:,0]<=EPS
 metrics={k:dict(target_mean_percent=float(target[:,j].mean()*100),baseline_mean_percent=float(base[:,j].mean()*100),delta_pp=float(delta[mask,j].mean()*100),CI95_pp=[float(lo[j]),float(hi[j])])for j,k in enumerate(KEYS)}
 transitions={k:dict(new_failure=int(((base[:,j]>=1-EPS)&(target[:,j]<1-EPS)).sum()),repaired_failure=int(((base[:,j]<1-EPS)&(target[:,j]>=1-EPS)).sum()))for j,k in enumerate(KEYS)if j in (1,2,3)}
 score_delta=delta[mask,0]*100
 return dict(status='COMPLETE',target=a.get('label'),baseline=b.get('label'),scenes=int(mask.sum()),physical_log_clusters=groups,bootstrap=2000,seed=42,
  target_PDMS=metrics['score']['target_mean_percent'],baseline_PDMS=metrics['score']['baseline_mean_percent'],delta_PDMS=metrics['score']['delta_pp'],CI95_PDMS=metrics['score']['CI95_pp'],metrics=metrics,
  zeros=dict(baseline=int(oldzero.sum()),target=int(newzero.sum()),repaired=int((oldzero&~newzero).sum()),new_zero=int((~oldzero&newzero).sum()),baseline_ge09_to_zero=int(((base[:,0]>=.9)&newzero).sum())),
  safety_transitions=transitions,G_plus_PDMS=float(np.maximum(score_delta,0).sum()/len(score_delta)),G_minus_PDMS=float(np.minimum(score_delta,0).sum()/len(score_delta)),
  score_improved_scenes=int((score_delta>EPS).sum()),score_decreased_scenes=int((score_delta<-EPS).sum()),
  raw_progress=dict(status='UNAVAILABLE_IN_REUSED_COMPLETE_METRIC_CACHE_OR_TRAIN_BANK',mean=None,EP_is_not_raw_meters=True),
  uncertainty='same fixedseed,scene-weighted physical-log bootstrap; exploratory/unadjusted CIs; not across-training-seed uncertainty')

def score_accounting(a,b):
 x=b['values'];y=a['values'];n,d=x[:,1],x[:,2];dn,dd=y[:,1]-n,y[:,2]-d;out=np.zeros((len(x),5))
 for j,col,w in [(2,3,5/12),(3,4,5/12),(4,5,2/12)]:
  c=x[:,col];dc=y[:,col]-c
  out[:,0]+=w*dn*(d*c+.5*(dd*c+d*dc)+dd*dc/3)
  out[:,1]+=w*dd*(n*c+.5*(dn*c+n*dc)+dn*dc/3)
  out[:,j]+=w*dc*(n*d+.5*(dn*d+n*dd)+dn*dd/3)
 assert np.max(abs(out.sum(1)-(y[:,0]-x[:,0])))<1e-12
 return dict(method='exact5component Shapley,description notcausal',units='PDMSpoints',contributions=dict(zip(['NC','DAC','TTC','EP','Comfort'],(out.mean(0)*100).tolist())))

def ddc_population(v,safety10=None):
 m=v['values'];score=m[:,0];ddc=m[:,6];bad=ddc<1;formula=m[:,1]*m[:,2]*(5*m[:,4]+5*m[:,3]+2*m[:,5])/12
 result=dict(scenes=len(m),DDC_below1=int(bad.sum()),DDC_zero=int((ddc==0).sum()),DDC_half=int((ddc==.5).sum()),DDC_bad_with_positive_canonical_score=int((bad&(score>EPS)).sum()),
  observed_score_formula_max_error_without_DDC=float(abs(score-formula).max()),observed_score_formula_max_error_with_DDC=float(abs(score-formula*ddc).max()),
  actual_canonical_DDC_direct_score_weight=0,hypothetical_additional_DDC_multiplier_PDMS=float((score*ddc).mean()*100),hypothetical_score_change_PDMS=float((score*ddc-score).mean()*100),
  limitation='CurrentcanonicalDOESNOTmultiplyDDC; imaginedDDCpenaltychangesmetric andisnotcurrentPDMSloss,notreachablepolicygain. DDCisanindependentdrivingqualityreadout.')
 if safety10 is not None:
  assert v['tokens']==safety10['tokens'];high=safety10['values'][:,0]>=.9;delta=score-safety10['values'][:,0];negative=np.minimum(delta,0)
  result['fixed_Safety10k_high_ge09']=dict(scenes=int(high.sum()),DDC_bad=int((bad&high).sum()),new_DDC_bad=int(((safety10['values'][:,6]==1)&bad&high).sum()),
   negative_score_change_associated_with_target_DDCbad_contribution_to_high_PDMS=float(100*negative[high&bad].sum()/high.sum()),
   share_of_high_negative_score_change_on_target_DDCbad=float(negative[high&bad].sum()/negative[high].sum())if negative[high].sum()<0 else None,
   limitation='descriptivecooccurrence;DDCisnotascorefactor,cannotattributePDMSlossdirectlytoDDC')
 return result

def journal(path,target):
 start=target-63
 with path.open('rb')as f:
  lo=0;hi=path.stat().st_size
  while lo<hi:
   mid=(lo+hi)//2;f.seek(mid)
   if mid:f.readline()
   line=f.readline()
   if not line:hi=mid;continue
   row=json.loads(line)
   if row['updates']>=start:hi=mid
   else:lo=f.tell()
  f.seek(lo);rows=[]
  while True:
   line=f.readline()
   if not line:break
   row=json.loads(line)
   if row['updates']>target:break
   if row['updates']>=start:rows.append(row)
 assert [x['updates']for x in rows]==list(range(start,target+1)),(path,target)
 return rows

def journal_stats(runroot,u):
 rank_rows=[journal(runroot/f'metrics_rank{r}.jsonl',u)for r in range(8)];rows=[x for a in rank_rows for x in a]
 def mean(section,key):
  values=[x.get(section,{}).get(key)for x in rows];values=[v for v in values if isinstance(v,(int,float))]
  return float(np.mean(values))if values else None
 return dict(window=[u-63,u],rank_update_rows=len(rows),optimizer_updates=64,
  combined_DDP_gradnorm_rank0_mean=float(np.mean([x['gradient_norm']for x in rank_rows[0]])),
  gradclip1_trigger_fraction_rank0=float(np.mean([x['gradient_norm']>1 for x in rank_rows[0]])),
  metrics={k:mean('metrics',k)for k in ('policy_loss','kl_loss','ppo_blocked_fraction','positive_ppo_blocked_fraction','negative_ppo_blocked_fraction','clipfrac','adv_kept_frac','pg_logprob_weight_abs_mean')},
  exploration_all4inner_rows={k:mean('exploration',k)for k in ('canonical_random_mean','safe_fraction','zero_PG','positive_unsafe_with_safe_alternative','safety_differing')},
  scope='matched64update windows×8ranks; rollout evidence is reused fourinner,not512 independent freshgroups; combined gradient not per-component attribution')

def credit(data):
 metrics=data['metrics'];f=np.all(metrics[:,:,1:4]==1,axis=2);m=f.sum(1);mixed=(m>0)&(m<8);scores=metrics[:,:,0].astype(np.float32).astype(np.float64)
 assert len(metrics)%8==0
 offsets=scores-scores[:,:1];centered=offsets-offsets.mean(1,keepdims=True);std=scores.reshape(-1,64).std(1).repeat(8)
 appo=np.clip((centered/(std[:,None]+1e-8)).astype(np.float32),-5,5).astype(np.float64);asafety=np.clip(data['advantages'],-5,5)
 pzero=(offsets.std(1)<=1e-8);szero=np.all(data['A_raw']==0,1)
 replay=data['A_raw']/(data['normalization_std'][:,None]+1e-8)
 assert np.max(abs(replay-data['advantages']))<2e-6
 def describe(a,z):
  ep_proj=[];raw_proj=[]
  for i in range(len(a)):
   if m[i]>=2:
    ep=metrics[i,f[i],4];ep_proj.append(float(np.sum(a[i,f[i]]*(ep-ep.mean()))))
    raw=data['raw_progress'][i,f[i]]
    if np.isfinite(raw).all():raw_proj.append(float(np.sum(a[i,f[i]]*(raw-raw.mean()))))
  def signs(v):return dict(groups=len(v),positive=int((np.array(v)>1e-10).sum()),negative=int((np.array(v)<-1e-10).sum()),zero=int((abs(np.array(v))<=1e-10).sum()),mean=float(np.mean(v))if v else None)
  ddc_bad=metrics[:,:,6]<1;positive_bad=(a>EPS)&ddc_bad
  return dict(positive_unsafe_with_feasible_alternative=int(((a>EPS)&~f&mixed[:,None]).sum()),groups_with_positive_unsafe_in_mixed=int(np.any((a>EPS)&~f&mixed[:,None],1).sum()),
   positive_unsafe_allfail=int(((a>EPS)&~f&(m==0)[:,None]).sum()),feasible_negative_candidates_in_mixed=int(((a<-EPS)&f&mixed[:,None]).sum()),
   groups_with_feasible_negative_in_mixed=int(np.any((a<-EPS)&f&mixed[:,None],1).sum()),zero_PG_groups=int(z.sum()),
   mean_abs_credit=float(abs(a).mean()),mean_positive_feasible_credit=float(np.maximum(a,0)[f].mean()),
   within_feasible_EP_credit_projection=signs(ep_proj),within_feasible_raw_progress_credit_projection=signs(raw_proj),
   DDC_independent_quality=dict(DDC_bad_positive_A=int(positive_bad.sum()),strict3feasible_DDCbad_positive_A=int((positive_bad&f).sum()),strict3feasible_DDC0_positive_A=int(((a>EPS)&f&(metrics[:,:,6]==0)).sum()),strict3feasible_DDChalf_positive_A=int(((a>EPS)&f&(metrics[:,:,6]==.5)).sum()),
    strict3feasible_DDCbad_positive_EP_mean=float(metrics[:,:,4][positive_bad&f].mean())if(positive_bad&f).any()else None,strict3feasible_DDCbad_positive_canonical_score_mean=float(metrics[:,:,0][positive_bad&f].mean())if(positive_bad&f).any()else None,
    share_of_positive_feasible_credit_on_DDCbad=float(np.maximum(a,0)[f&ddc_bad].sum()/np.maximum(a,0)[f].sum())if np.maximum(a,0)[f].sum()>0 else None))
 dot=float((appo*asafety).sum());den=float(np.linalg.norm(appo)*np.linalg.norm(asafety))
 negative_ep=[]
 for i in range(len(appo)):
  if m[i]<2:continue
  ep=metrics[i,f[i],4];projection=float(np.sum(appo[i,f[i]]*(ep-ep.mean())))
  if projection<-1e-10:negative_ep.append(dict(comfort_varies=bool(np.ptp(metrics[i,f[i],5])>0),any_DDC_bad=bool((metrics[i,f[i],6]<1).any())))
 return dict(groups=len(metrics),candidates=int(metrics.shape[0]*8),collections_global8rank=len(metrics)//8,mixed_groups=int(mixed.sum()),
  original_PPO_counterfactual=describe(appo,pzero),Safety_actual=describe(asafety,szero),
  sign_changes=int(((appo>EPS)&(asafety<-EPS)|(appo<-EPS)&(asafety>EPS)).sum()),candidate_credit_cosine=dot/den if den else None,
  original_positive_unsafe_mixed_to_Safety_negative=int(((appo>EPS)&(asafety<-EPS)&~f&mixed[:,None]).sum()),
  PPO_active_Safety_zero_groups=int((~pzero&szero).sum()),PPO_zero_Safety_active_groups=int((pzero&~szero).sum()),
  Safety_saved_normalization_replay_max_error=float(np.max(abs(replay-data['advantages']))),
  original_negative_EP_projection_classification=dict(groups=len(negative_ep),feasible_Comfort_varies=sum(x['comfort_varies']for x in negative_ep),any_feasible_DDC_bad=sum(x['any_DDC_bad']for x in negative_ep),all_feasible_DDC1=sum(not x['any_DDC_bad']for x in negative_ep),
    interpretation='Allstrict3feasiblecandidateshaveNC=DAC=TTC1;EPnegativecreditcanreflectlegitimatecanonicalComforttradeoff,notnecessarilyerror.DDCisnotascorefactorhere.Scalarprojectionisnotparametergradient.'),
  limitations='same saved Safety candidates only; alternative credit is CPU counterfactual,not an originalPPO rollout or actualparametergradient; EP/rawprogress projections are scalar candidate-credit alignments,notgradients through model/logprob')

def main():
 registry=read(ROOT/'RUNS.json')['runs'];runmap={x['id']:x for x in registry};cfg={a:yaml.safe_load(Path(runmap[a]['config']).read_text())for a in ('PPO_FULL','PPO_SAFETY_EP')}
 diffs=differences(cfg['PPO_FULL'],cfg['PPO_SAFETY_EP']);assert {x['path']for x in diffs}=={'.research.arm','.research.structured_advantage'}
 ilnav=fixed_IL_nav();iltrain=train('IL70_TRAIN_001024');assert iltrain
 values={};unavailable=[];navrows=[];trainrows=[]
 for u in GATES:
  for arm in ('PPO_FULL','PPO_SAFETY_EP'):
   values[arm,u,'NAVTEST']=nav(arm,u);values[arm,u,'TRAIN_DIAGNOSTIC']=train(f'{arm}_u{u:06}_TRAIN1024')
  for cohort,out,il in [('NAVTEST',navrows,ilnav),('TRAIN_DIAGNOSTIC',trainrows,iltrain)]:
   a,b=values['PPO_SAFETY_EP',u,cohort],values['PPO_FULL',u,cohort]
   if a is None or b is None:unavailable.append(dict(updates=u,cohort=cohort,status='NO_COMPLETE_CACHED_PAIR_NOT_SCORED'));continue
   assert a['tokens']==il['tokens']and np.array_equal(a['logs'],il['logs'])
   if cohort=='NAVTEST':assert a['hashes']==il['hashes']
   mask=il['values'][:,0]>=.9;secondary=values['PPO_SAFETY_EP',10000,cohort]if u>=10000 else (nav('PPO_SAFETY_EP',10000)if cohort=='NAVTEST'else train('PPO_SAFETY_EP_u010000_TRAIN1024'))
   row=paired(a,b);row.update(updates=u,cohort=cohort,provenance=dict(target=a['provenance'],baseline=b['provenance']),fixed_IL70_high_ge09_primary=paired(a,b,mask),fixed_Safety10k_high_ge09_secondary=paired(a,b,secondary['values'][:,0]>=.9),score_accounting=score_accounting(a,b));out.append(row)
 all_windows=[];datasets=[]
 for p in sorted((ROOT/'analysis/SAFETY_EP_PROCESS_CACHE').glob('u*_rollouts.npz')):
  with np.load(p,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
  datasets.append(data);all_windows.append(dict(updates=int(p.name[1:7]),cache=str(p),cache_file_sha256=sha(p),**credit(data)))
 combined={key:np.concatenate([d[key]for d in datasets],axis=0)for key in datasets[0]};counterfactual=credit(combined)
 source_math=ROOT/'runtime/v1_42fa8eb8d358/starVLA/rl/simwam_recipe/source_advantages.py'
 journal_rows=[]
 for u in GATES:
  journal_rows.append(dict(updates=u,PPO_FULL=journal_stats(Path(runmap['PPO_FULL']['root']),u),SafetyEP=journal_stats(Path(runmap['PPO_SAFETY_EP']['root']),u)))
 process=read(ROOT/'analysis/SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json');probe_path=ROOT/'analysis/SAFETY_EP_NATIVE28K_GRADIENT_PROBE_20261031/SUMMARY.json'
 # A genuinely new compact26k extraction reads existing sealed scene receipts
 # once through the shared cache lock; it never invokes model/scorer code.
 extracted=[]
 cache26=ROOT/'analysis/evaluation_metric_cache/PPO_SAFETY_EP_u026000_NAVTEST.npz'
 if not cache26.exists():
  from safety_evidence_reviewer import complete_input,load_metrics
  info=complete_input(ROOT,'PPO_SAFETY_EP',26000)
  if info is not None:load_metrics(ROOT,info);extracted.append('PPO_SAFETY_EP_u026000_NAVTEST')
 ddc_rows=[];s10=values['PPO_SAFETY_EP',10000,'NAVTEST']
 for arm,updates in [('PPO_FULL',u)for u in GATES]+[('PPO_SAFETY_EP',u)for u in (*GATES,24000,26000)]:
  v=nav(arm,updates)
  if v is not None:ddc_rows.append(dict(arm=arm,updates=updates,**ddc_population(v,s10)))
 source_files={}
 for rel in ['starVLA/rl/simwam_recipe/source_advantages.py','starVLA/rl/simwam_recipe/collection.py','starVLA/rl/simwam_recipe/vendor/trainer_math.py','starVLA/rl/qwen_world_postcot/scope_runtime.py']:
  files={a:Path(runmap[a]['source'])/rel for a in cfg};source_files[rel]={a:dict(path=str(p),sha256=sha(p))for a,p in files.items()if p.exists()};source_files[rel]['byte_identical']=len(source_files[rel])==2 and len({n['sha256']for n in source_files[rel].values()})==1
 result=dict(schema='world_SafetyEP_vs_PPO_mechanism_v1',status='COMPLETE',created_at=datetime.now(timezone.utc).isoformat(),
  actions=dict(CPU_only=True,new_model_forwards=0,new_scores=0,optimizer_updates=0,signals=0,owner_changes=0,new_seeds=0,NAVTEST_training_selection=False),
  matched_scientific_comparison=dict(config_differences=diffs,all_other_yaml_scientific_fields_equal=True,parent_binding=process['parent_binding'],configs={a:dict(path=runmap[a]['config'],sha256=sha(runmap[a]['config']))for a in cfg},source_PPO_math=dict(path=str(source_math),sha256=sha(source_math)),frozen_source_file_comparison=source_files,
   unchanged=['same IL70 freshinitialization and fixedIL70 reference','canonicalRewardService/protocol','uniformfull103288','W3eta.1 steps7/8/9','action5e-5 memory5e-6,warmup100 thenconstant','4inner,Aclip5,PPOclip,beta50','actionLoRA+trainable scene/predictive memory,Qwenfrozen']),
  mathematical_change=dict(canonical_score_unchanged='NC*DAC*(5*EP+5*TTC+2*Comfort)/12',original='A=(R-mean_groupR)/(globalRAWreward_populationstd+eps),FP32rewards→FP64centering;groupstd mask',Safety='Araw=Asafe+withinfeasibleEP;anyfeasible:Asafe=f-meanf,allfail:meanc-c;globalpopulationstd of64Araw,exacttiesPGmask0',
   omissions='SafetyPG does not directly optimizeComfort/DDC credit; originalcanonicalfeedback staysintact; noadded safety scorerreward',coupled_scale='Changing RAWrewardstd to RAWAdvantageStd also changes PG scale relative to unchanged beta50; cannot isolate safetygate from scale without further controlled factor.'),
  NAVTEST_matched=navrows,TRAIN1024_matched=trainrows,unavailable=unavailable,
  same_Safety_rollout_credit_counterfactual=dict(aggregate=counterfactual,windows=all_windows,scope='reused10 bounded windows×128groups,notfull103k orwholetraining',original_PPO_rewards_cast='FP32 canonical score thenFP64offsetcentering;std per actual8rank64candidatecollection;sourcecap5',raw_progress='actual savedrawmeters available onlyin boundedcandidatecache; no fullNAV/TRAINrawmeter cache'),
  actual_training_journal_comparison=journal_rows,measured_28k_gradient_probe=dict(source=str(probe_path),summary=read(probe_path))if probe_path.exists()else dict(status='PENDING'),
  DDC_independent_quality_audit=dict(canonical_score_does_not_multiply_DDC=True,not_a_proposed_scorer_change=True,new_compact_extraction_from_existing_sealed_receipts=extracted,NAVTEST=ddc_rows,
   warning_zh='DDC为当前未加权驾驶质量读数；不能将额外乘DDC后的假想分数当现PDMS损失或可达提升，也不能据其修改固定scorer制造92。'),
  interpretation_zh=['不是给PPO再叠安全reward，而是改fresh候选的信用排序和尺度。','1/5/10/20k同预算与相同source配置比较为这一具体recipe提供证据；当前single trainingseed不提供跨seed算法保证。','同Safety候选CPU重算能隔离信用公式直接作用，但不能代表原PPO在同θ采样、真实梯度或策略终局因果。','固定IL70>=.9为primary，Safety10k>=.9仅secondary条件队列；后者选择自Safety峰值，不作无偏总性能或独立算法胜负结论。','候选A/EP/rawprogress投影不是参数梯度；具体PG/anchor方向只依据独立真实28k测量。','TRAIN1024为固定分层训练内诊断，162log而非NAVTEST44log，不作全量TRAIN均值或泛化代理。'],
  target=dict(PDMS_goal=92,attained=False,current_Safety_peak=91.34989173702405,peak_gap_to92=92-91.34989173702405,priority='preservehigh-quality skills while repairingnewzero;freshsameIL70 controlledP1/P2,notrestrictedtoexistingSafetycode',P1_P2_status='DEVELOPMENT_NOT_LAUNCHED_AT_THIS_ANALYSIS'),
  builder=dict(path=str(Path(__file__)),sha256=sha(__file__)))
 result['sha256']=fp(result);atom(OUT,result)
 print(json.dumps(dict(output=str(OUT),status=result['status'],NAVTEST=[{k:r[k]for k in ('updates','delta_PDMS','CI95_PDMS')}for r in navrows],counterfactual=counterfactual),ensure_ascii=False))

if __name__=='__main__':main()
