"""All Endpoint/Control first500 actual behavior; counterfactual credit CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,io,json,sys,time
from pathlib import Path
import numpy as np
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_safety_exploration.credit import canonical_components,raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.saturated_credit import decision as saturation_decision
from starVLA.rl.qwen_world_safety_exploration.protected_credit import decision_from_material as protected_decision
from starVLA.rl.qwen_world_safety_exploration.inversion_cancellation import cancel_inversions

torch.set_num_threads(1)
E=Path('/mnt/project/qwen-world-safety-exploration-20261031')
W=Path('/mnt/project/VLA-Drive-qwen-world-safety-exploration-20261031')
ARMS=('SAFETY_ENDPOINT_RAMP_AR1_G8','SAFETY_G8_CONTROL')
KEYS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
TARGET=E/'analysis/ENDPOINT_CREDIT_INTERACTION_500.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(a):
 a=np.ascontiguousarray(a);return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def stats(values):
 a=np.asarray(values,dtype=np.float64).reshape(-1)
 if not len(a):return dict(count=0)
 return dict(count=len(a),min=float(a.min()),mean=float(a.mean()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()),RMS=float(np.sqrt(np.mean(a*a))),mean_abs=float(np.mean(np.abs(a))),L1=float(np.abs(a).sum()),sum=float(a.sum()))
def jsonable(v):
 if torch.is_tensor(v):return v.detach().cpu().tolist()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 return v

def one(arm,rank,collection,identity):
 path=E/'runs'/arm/'behavior'/f'collection{collection:06d}'/f'rank{rank}.pt'
 rawbytes=path.read_bytes();digest=hashlib.sha256(rawbytes).hexdigest()
 x=torch.load(io.BytesIO(rawbytes),map_location='cpu',weights_only=False)
 version=x['behavior_version']
 if (version['run']!=identity or version['update']!=4*(collection-1) or x['group_size']!=8):raise ValueError('Run/update/G8 identity mismatch')
 records=x['records']
 if len(records)!=8 or any(r['status']!='OK' or r['token']!=x['token'] or r['log']!=x['log'] or r['protocol_sha256']!=x['protocol_sha256'] for r in records):raise ValueError('Scorer scene/log/protocol/status mismatch')
 material=canonical_components(records,8);rewards=material['canonical_rewards'];comp=material['components'];expected=raw_advantage_parts(comp)
 if not torch.equal(rewards,x['rewards']) or not torch.equal(rewards,x['canonical_rewards']):raise ValueError('Original canonical reward changed')
 s=x['structured_advantage']
 if not torch.equal(s['components'],comp):raise ValueError('Structured material differs from official records')
 for k,v in expected.items():
  if not torch.allclose(s[k],v,atol=1e-14,rtol=0):raise ValueError('Safety math mismatch '+k)
 metrics=np.asarray([[r['metrics'][k] for k in KEYS] for r in records],dtype=np.float64)
 requested=np.asarray([r['requested_trajectory'] for r in records])
 isendpoint=arm==ARMS[0]
 normalized_chain_endpoint=ActionCodec.denorm_odo(x['chain'][:,-1]).numpy()
 if isendpoint:
  scored=arr(x['endpoint_policy']['action'])
  mean=arr(x['endpoint_policy']['old_mean'])
  if scored.dtype!=np.float32 or not np.array_equal(scored,x['endpoint_policy']['unrounded_action'].numpy().astype(np.float32)):raise ValueError('Endpoint exact scored FP32 action/quantization mismatch')
  action_field='endpoint_policy.action; chain[-1] is UNSCORED clean ODE mean'
 else:scored=normalized_chain_endpoint;mean=None;action_field='ActionCodec.denorm_odo(chain[:,-1]) actual SDE endpoint'
 if requested.shape!=(8,8,3) or not np.array_equal(requested,scored):raise ValueError('Official labels not attached to actual scored action')
 if requested.dtype!=np.float32:raise ValueError('Scored trajectory not FP32')
 raw=arr(s['A_raw']).reshape(8);safe=arr(s['A_safe']).reshape(8);epA=arr(s['A_EP']).reshape(8)
 adv=arr(x['advantages']).reshape(8);mask=arr(x['adv_mask']).reshape(8);scale=float(x['normalization_std'])
 if not np.isfinite(scale) or scale<0:raise ValueError('Invalid normalization')
 norm_error=float(np.max(np.abs(adv-(raw/(scale+1e-8)).astype(np.float32))))
 if norm_error>2e-6 or not np.all((mask==0)|(mask==1)):raise ValueError('Normalized A/mask mismatch')
 pg0=bool(np.all(raw==0))
 if bool(np.all(mask==0))!=pg0:raise ValueError('PGmask/rawA0 mismatch')
 feasible=arr(expected['feasible']).reshape(8);cost=arr(expected['cost']).reshape(8)
 high=feasible&(metrics[:,5]==1)&(metrics[:,0]>=.9);neg=high&(raw<0)
 allsafe=bool(feasible.all());allfail=not bool(feasible.any());ep1=allsafe and bool(np.all(metrics[:,4]==1))
 cvar=bool(np.ptp(metrics[:,5])>0);allfull=bool(np.all(metrics[:,0]==1));costtie=allfail and bool(np.all(cost==cost[0]))
 if pg0:
  category='true_canonical_all1' if allfull else 'safe_EP1_Comfort_varied' if ep1 and cvar else 'safe_EP1_allComfort0' if ep1 and np.all(metrics[:,5]==0) else 'allfail_costtie' if costtie else 'safe_equal_EP_below1_Comfort_varied' if allsafe and cvar else 'safe_equal_EP_below1_canonical_tied' if allsafe else 'UNCLASSIFIED_PG0'
 else:category='active'
 sat=saturation_decision(rewards,comp)
 sat_raw=raw.copy()
 if sat['eligible']:sat_raw+=sat['delta'].numpy().reshape(8)
 quality=torch.tensor(metrics[:,[5,0]][None],dtype=torch.float64)
 protected=protected_decision(comp,quality,s['A_raw']);protected_raw=raw.copy();protected_raw[protected['eligible'].numpy().reshape(8)]=0
 inv=cancel_inversions(s['A_raw'],torch.tensor(metrics[:,0][None],dtype=torch.float64),comp,torch.tensor(metrics[:,5][None],dtype=torch.float64))
 inv_raw=inv['updated_rawA'].numpy().reshape(8)
 pair_rows=[];witness_count=0;witnesses=set()
 for i in np.flatnonzero(neg):
  js=np.flatnonzero((raw>0)&(metrics[:,0]<metrics[i,0]))
  witness_count+=int(len(js)>0)
  for j in js:
   dEP=metrics[j,4]-metrics[i,4];dC=metrics[j,5]-metrics[i,5];dR=metrics[j,0]-metrics[i,0]
   explained=bool(feasible[j] and dEP>0 and dC<0 and abs(dR-(5*dEP+2*dC)/12)<5e-15)
   if not explained:raise ValueError('Canonical credit inversion not EP/Comfort explained')
   witnesses.add(int(j));pair_rows.append(dict(high_candidate=int(i),lower_positive_candidate=int(j),high_score=float(metrics[i,0]),lower_score=float(metrics[j,0]),EP_gain_of_lower=float(dEP),Comfort_change_of_lower=float(dC),canonical_gap=float(-dR),high_rawA=float(raw[i]),lower_rawA=float(raw[j]),explained=True))
 counts=Counter(groups=1,candidates=8,strict_high=int(high.sum()),negative_high=int(neg.sum()),negative_high_groups=int(neg.any()),
  feasible_candidates=int(feasible.sum()),all_safe_groups=int(allsafe),all_fail_groups=int(allfail),mixed_groups=int(not allsafe and not allfail),
  PG0=int(pg0),canonical_allzero_groups=int(np.all(metrics[:,0]==0)),costtie_groups=int(costtie),source_cap5_candidates=int(np.sum(np.abs(adv)>5)),
  source_cap5_negative_high=int(np.sum(neg&(adv < -5))),positive_unsafe=int(np.sum((raw>0)&~feasible)),positive_unsafe_with_safe_alternative=int(np.sum((raw>0)&~feasible)) if feasible.any() else 0,
  saturation_eligible_groups=int(sat['eligible']),saturation_activated_candidates=8*int(sat['eligible']),
  saturation_new_positive_high_candidates=int(np.sum((sat_raw>0)&high&(raw==0))),saturation_new_negative_Comfort0_candidates=int(np.sum((sat_raw<0)&(metrics[:,5]==0)&(raw==0))),
  protected_groups=int(neg.any()),protected_candidates=int(neg.sum()),inversion_groups=int(inv['counts']['changed_groups']),inversion_changed_candidates=inv['counts']['changed_candidates'],
  negative_high_with_lower_canonical_positive=witness_count,inversion_original_pairs=len(pair_rows),inversion_distinct_lower_positive_candidates=len(witnesses),
  inversion_negative_high_cleared=inv['counts']['negative_high_cleared'],inversion_remaining_negative_high=inv['counts']['remaining_negative_high'],inversion_remaining_pairs=inv['counts']['remaining_qualifying_pairs'],inversion_new_PG0=inv['counts']['newly_allzero_groups'],
  action_bound_to_scored_endpoint=int(isendpoint),scored_action_differs_from_clean_chain_mean=int(not np.array_equal(scored,normalized_chain_endpoint)) if isendpoint else 0)
 counts['PG0_'+category]+=int(pg0)
 if pg0:counts['PG0_canonical_allzero']+=int(np.all(metrics[:,0]==0))
 row=dict(arm=arm,rank=rank,collection=collection,behavior_update=version['update'],token=x['token'],log=x['log'],run_identity=identity,
  behavior_path=str(path),behavior_sha256=digest,behavior_bytes=len(rawbytes),protocol_sha256=x['protocol_sha256'],metric_columns=list(KEYS),metrics=metrics.tolist(),
  scored_action_field=action_field,scored_action_hash=ah(scored),chain_clean_endpoint_hash=ah(normalized_chain_endpoint),
  original_rawA=raw.tolist(),A_safe=safe.tolist(),A_EP=epA.tolist(),original_normalized_A=adv.tolist(),original_mask=mask.tolist(),
  normalization_std=scale,global_canonical_reward_std=float(x['global_std']),raw_reward_std=float(x['raw_group_std']),normalized_A_max_error=norm_error,
  PG0_category=category,counts=dict(counts),
  counterfactuals=dict(saturated=dict(eligible=bool(sat['eligible']),status=sat['status'],raw_A=sat_raw.tolist()),
   protected=dict(raw_A=protected_raw.tolist()),inversion=dict(raw_A=inv_raw.tolist(),counts=inv['counts'],transfers=inv['pairs'],group_sum_roundoff=inv['max_group_sum_roundoff'])),
  canonical_inversion_pairs=pair_rows)
 return row


def worker(task):
 arm,rank,identity=task;rows=[];errors=[]
 for collection in range(1,126):
  try:rows.append(one(arm,rank,collection,identity))
  except Exception as ex:
   path=E/'runs'/arm/'behavior'/f'collection{collection:06d}'/f'rank{rank}.pt'
   errors.append(dict(arm=arm,rank=rank,collection=collection,path=str(path),exists=path.exists(),error=repr(ex),sha256=sha(path) if path.exists() else None))
 return dict(arm=arm,rank=rank,rows=rows,errors=errors,CUDA_initialized=torch.cuda.is_initialized())


def arm_summary(arm,rows):
 rows=sorted(rows,key=lambda r:(r['collection'],r['rank']));counts=Counter()
 for row in rows:counts.update(row['counts'])
 metrics=np.asarray([r['metrics'] for r in rows]);raw=np.asarray([r['original_rawA'] for r in rows]);adv=np.asarray([r['original_normalized_A'] for r in rows]);mask=np.asarray([r['original_mask'] for r in rows])
 high=(metrics[:,:,1:4]==1).all(2)&(metrics[:,:,5]==1)&(metrics[:,:,0]>=.9);neg=high&(raw<0)
 scales=[];cstd=[];maxnormerr=0.;maxcanonerr=0.;fallback=0;valid_collections=0
 for collection in range(1,126):
  selected=[r for r in rows if r['collection']==collection]
  if len(selected)!=8:continue
  selected.sort(key=lambda r:r['rank']);a=np.asarray([r['original_rawA'] for r in selected]);scale=float(a.std());saved=np.asarray([r['normalization_std'] for r in selected]);maxnormerr=max(maxnormerr,float(np.max(np.abs(saved-scale))))
  rewards=np.asarray([[np.float32(m[0]) for m in r['metrics']] for r in selected],dtype=np.float64)
  canonicalstd=float(rewards.std());maxcanonerr=max(maxcanonerr,max(abs(r['global_canonical_reward_std']-canonicalstd) for r in selected));scales.append(scale);cstd.append(canonicalstd);valid_collections+=1
  satraw=np.asarray([r['counterfactuals']['saturated']['raw_A'] for r in selected])
  satscale=scale if scale>0 else float(satraw.std());fallback+=int(scale==0)
  for r in selected:
   for variant in ('saturated','protected','inversion'):
    newraw=np.asarray(r['counterfactuals'][variant]['raw_A']);oldraw=np.asarray(r['original_rawA']);oldadv=np.asarray(r['original_normalized_A'],dtype=np.float32)
    newadv=oldadv.copy();changed=newraw!=oldraw;denom=satscale if variant=='saturated' else r['normalization_std']
    newadv[changed]=(newraw[changed]/(denom+1e-8)).astype(np.float32)
    if variant=='saturated' and scale==0:newadv=(newraw/(satscale+1e-8)).astype(np.float32)
    r['counterfactuals'][variant].update(normalized_A=newadv.tolist(),normalization_std=denom,
      mask='recomputed only for saturation to activate formerPG0; original mask retained for protected/inversion',
      PG0_after=bool(np.all(newadv==0)),raw_delta_score_alignment=float(np.sum((newraw-oldraw)*np.asarray(r['metrics'])[:,0])))
 scores=metrics[:,:,0];pairs=[p for r in rows for p in r['canonical_inversion_pairs']]
 variants={}
 for variant in ('saturated','protected','inversion'):
  newraw=np.asarray([r['counterfactuals'][variant]['raw_A'] for r in rows]);newadv=np.asarray([r['counterfactuals'][variant]['normalized_A'] for r in rows]);delta=newraw-raw
  variants[variant]=dict(changed_groups=int(np.any(newraw!=raw,axis=1).sum()),changed_candidates=int((newraw!=raw).sum()),
    negative_high_after=int(((newraw<0)&high).sum()),negative_high_removed=int((neg&~(newraw<0)).sum()),
    raw_delta_positive_mass=float(delta.clip(0).sum()),raw_delta_negative_mass=float((-delta.clip(max=0)).sum()),
    removed_negative_high_raw_mass=float(np.sum(delta[neg])),removed_negative_high_normalized_mass=float(np.sum((newadv-adv)[neg])),
    positive_high_normalized_mass_before=float(adv[(raw>0)&high].sum()),positive_high_normalized_mass_after=float(newadv[(newraw>0)&high].sum()),
    post_cap5_positive_high_mass=float(np.clip(newadv,-5,5)[(newraw>0)&high].sum()),
    post_cap5_negative_high_mass=float(-np.clip(newadv,-5,5)[(newraw<0)&high].sum()),
    raw_delta_score_alignment=float(np.sum(delta*scores)),alignment_is_not_gradient_or_predicted_gain=True,
    PG0_after=int(np.all(newadv==0,axis=1).sum()),source_cap5_candidates=int((np.abs(newadv)>5).sum()),
    max_group_sum_change=float(np.max(np.abs(newraw.sum(1)-raw.sum(1)))))
 out=dict(counts=dict(counts),rates=dict(strict_high_per_candidate=counts['strict_high']/counts['candidates'],negative_high_per_candidate=counts['negative_high']/counts['candidates'],
   negative_high_given_strict_high=counts['negative_high']/counts['strict_high'],PG0_per_group=counts['PG0']/counts['groups'],
   saturation_eligible_given_PG0=counts['saturation_eligible_groups']/counts['PG0'] if counts['PG0'] else 0.,
   negative_high_with_lower_positive_given_negative_high=counts['negative_high_with_lower_canonical_positive']/counts['negative_high'] if counts['negative_high'] else 0.,
   inversion_groups_per_group=counts['inversion_groups']/counts['groups']),
   PG0_breakdown={k[4:]:v for k,v in counts.items() if k.startswith('PG0_')},
   credit_scale=dict(independent_global_collections=valid_collections,normalization_Safety_rawA_std=stats(scales),canonical_FP32_reward_global_std=stats(cstd),
     max_saved_normalization_error=maxnormerr,max_saved_canonical_std_error=maxcanonerr,saturation_scale_zero_fallback_collections=fallback,
     raw_A=stats(raw),normalized_A=stats(adv),normalized_A_active_groups=stats(adv[np.any(raw!=0,axis=1)]),
     normalized_A_after_source_cap5=stats(np.clip(adv,-5,5)),strict_high_negative_raw=stats(raw[neg]),strict_high_negative_normalized=stats(adv[neg]),
     strict_high_negative_raw_mass=float(-raw[neg].sum()),strict_high_negative_normalized_mass=float(-adv[neg].sum()),
     strict_high_negative_postcap5_mass=float(-np.clip(adv,-5,5)[neg].sum()),
     negative_raw_mass_per_all_candidate=float(-raw[neg].sum()/raw.size),negative_normalized_mass_per_all_candidate=float(-adv[neg].sum()/adv.size),
     negative_raw_mass_per_strict_high=float(-raw[neg].sum()/high.sum()),negative_normalized_mass_per_strict_high=float(-adv[neg].sum()/high.sum())),
   inversion_quality=dict(pair_count=len(pairs),pairs_EP_Comfort_explained=sum(p['explained'] for p in pairs),
     canonical_gap_PDMS=stats([100*p['canonical_gap'] for p in pairs]),EP_gain_of_lower_score=stats([p['EP_gain_of_lower'] for p in pairs]),
     lower_score_Positive_Comfort_changes=stats([p['Comfort_change_of_lower'] for p in pairs])),
   counterfactual_credit_replays=variants,
   actual_candidate_mean_metrics={k:float(metrics[:,:,i].mean()) for i,k in enumerate(KEYS)},
   files_read=len(rows),bytes_read=sum(r['behavior_bytes'] for r in rows),unique_tokens=len({r['token'] for r in rows}),group_receipts=rows)
 return out


def enrich(result):
 ep=result['arms'][ARMS[0]];ct=result['arms'][ARMS[1]]
 categories=('true_canonical_all1','safe_EP1_Comfort_varied','safe_EP1_allComfort0','allfail_costtie','safe_equal_EP_below1_Comfort_varied','safe_equal_EP_below1_canonical_tied','UNCLASSIFIED_PG0')
 for arm,v in result['arms'].items():
  partition={k:sum(r['PG0_category']==k for r in v['group_receipts']) for k in categories}
  if sum(partition.values())!=v['counts']['PG0']:raise ValueError('PG0 partition is incomplete')
  v['PG0_partition']=partition
  v['global_all_rank_PG0_collections']=sum(all(r['PG0_category']!='active' for r in v['group_receipts'] if r['collection']==collection) for collection in range(1,126))
  v['PG0_scope']='Counts are local rank/scene groups; not entire global optimizer anchor-only steps'
  v['canonical_full_definition']='R=1 implies NC/DAC/TTC/EP/Comfort1 for this scorer; DDC not multiplied into canonical and reported separately'
  v['PG0_canonical_full_DDC_fail_candidates']=sum(sum(m[6]!=1 for m in r['metrics']) for r in v['group_receipts'] if r['PG0_category']=='true_canonical_all1')
  no_inversion=0;confirmed_higher_quality_witness=0;mixed_negative=0;allfeasible_negative=0
  for r in v['group_receipts']:
   m=np.asarray(r['metrics']);a=np.asarray(r['original_rawA']);safe=np.asarray(r['A_safe']);good=(m[:,1:4]==1).all(1)&(m[:,5]==1)&(m[:,0]>=.9)
   for i in np.flatnonzero(good&(a<0)):
    mixed_negative+=int(safe[i]>0);allfeasible_negative+=int(safe[i]==0)
    if not np.any((a>0)&(m[:,0]<m[i,0])):
     no_inversion+=1
     confirmed_higher_quality_witness+=int(np.any((a>0)&good&(m[:,0]>m[i,0])))
  v['negative_high_semantics']=dict(total=v['counts']['negative_high'],in_allfeasible=allfeasible_negative,in_mixed_with_nonnegative_safety_bonus=mixed_negative,
    with_lower_canonical_positive_witness=v['counts']['negative_high_with_lower_canonical_positive'],without_lower_canonical_positive_witness=no_inversion,
    without_inversion_with_actual_higher_strict_quality_positive_witness=confirmed_higher_quality_witness,
    interpretation='Relative downweighting versus higher-EP/higher-canonical safe/Comfort1 alternatives is not the same as a canonical order inversion; rawA sign is not parameter gradient')
  sat=v['counterfactual_credit_replays']['saturated'];before=sat['positive_high_normalized_mass_before']
  sat['positive_high_normalized_mass_increment']=sat['positive_high_normalized_mass_after']-before
  sat['positive_high_normalized_mass_increment_fraction']=sat['positive_high_normalized_mass_increment']/before
  inv=v['counterfactual_credit_replays']['inversion'];inv['negative_high_normalized_mass_removed_fraction']=inv['removed_negative_high_normalized_mass']/v['credit_scale']['strict_high_negative_normalized_mass']
  ordered=''.join(f"{r['collection']} {r['rank']} {r['behavior_sha256']}\n" for r in v['group_receipts'])
  v['all_actual_payload_hash_inventory_sha256']=hashlib.sha256(ordered.encode()).hexdigest()
 result['credit_scale_comparison']=dict(
   Endpoint_over_Control_mean_raw_normalization_std=ep['credit_scale']['normalization_Safety_rawA_std']['mean']/ct['credit_scale']['normalization_Safety_rawA_std']['mean'],
   Endpoint_over_Control_negative_normalized_mass_per_strict_high=ep['credit_scale']['negative_normalized_mass_per_strict_high']/ct['credit_scale']['negative_normalized_mass_per_strict_high'],
   Endpoint_over_Control_negative_raw_mass_per_strict_high=ep['credit_scale']['negative_raw_mass_per_strict_high']/ct['credit_scale']['negative_raw_mass_per_strict_high'],
   interpretation='Endpoint raw std lower; normalization fixes global RMS near1, not loss/parameter gradient. Zero groups remain in scale; active conditional RMS differs. No candidate reaches cap5, which says nothing about later PPO ratio clipping.')
 result['conclusions_zh']=[
  '完整2×1000组/2×8000候选，2000文件真实读取/hash、0异常，1000/1000同rank/collection场景。Endpoint全部1000组评分动作精确绑定endpoint_policy.action，全部不同于未评分的ODE mean。',
  'Endpoint135 PG0=109真正canonical全1+26全安全EP1但Comfort不同，0全failtie；Control60 PG0=1全1+57Comfort不同+2allfailtie。Endpoint更多PG0主要是成功满分，而非失败无信号。',
  'Saturated在Endpoint仅激活26/1000组（208候选，169高分新增正A），保留109满分PG0；control可激活57组。Endpoint正高分normalized信用质量只增加0.774%，control增加2.976%。这是冻结信用回放，非训练提升。',
  'Endpoint447/4756高分为负（9.399%），control220/2670（8.240%）。Endpoint仅64/447有较低canonical正A见证，control171/220；其余Endpoint383个都有实际更高canonical/更高EP的strict4正A替代。负high数量不能直接叫信用错误。',
  '真正可抵消的质量倒置Endpoint为25组/69对，control81组/340对；均为更高EP、Comfort1→0导致canonical下降，平均canonical差12.551/12.279分。Endpoint反序更少，但单对差距仍大。',
  'Inversion可在Endpoint清除34个负high，减少4.270normalized负信用质量（8.069%）；其余413仍负。Protected可删除全部447负high、移除52.918质量，却也删除383个没有canonical反序的相对负信用并允许组均值偏正。二者不能当同一规则。',
  'Endpoint原Safety world64 rawA std均值.313923，control.370154；两者global normalized RMS都约1，active-group RMS1.0752/1.0314，原cap5命中均0。Endpoint高分单位normalized负质量增加约21.50%，其raw单位质量反而略低；不可由此推断参数梯度更大。',
  '现有完整NAV500 Endpoint−control为−.537565分（已有44log CI[−.844736,−.242701]）；TRAIN候选更优并未保证部署更优。Endpoint jointSUM24条件密度、replay与control clamped W3 mean proxy不同，信用尺度不能替代这些机制的归因。']
 result['recommended_combination_review']=dict(status='EVIDENCE_FOR_FUTURE_SINGLE_FACTOR_DECISION_ONLY_NO_LAUNCH',
  first='Keep the already assigned Endpoint LR*.2 single-factor study clean, and read complete native1000 deployment evidence. These actual credit results do not support urgently stacking credit tricks to repair a supposed failed-PG0 crisis.',
  saturated=dict(priority='LOW_COMPLEMENTARY',target='Only26/1000 Endpoint groups;109 true-full groups intentionally unchanged',limitation='Small extra positive-high credit mass; no new failure-support repair; does not affect447 active high negatives'),
  inversion=dict(priority='LOW_PREVALENCE_BUT_PRECISE',target='25/1000 groups,64 involved negatives/69 clear canonical inversions; preserves raw group sum',limitation='Only8.07% Endpoint negative-high normalized mass removed, much less than control. Do not extrapolate to bulk high-retention problem'),
  protected=dict(priority='LARGER_INTERVENTION_REQUIRES_BIASED_OBJECTIVE_JUSTIFICATION',target='447/8000 candidates across161groups, all verifiedstrict4>=.9',limitation='383 lack observed canonical inversion; unconditional absolute-quality protection suppresses legitimate relative EP/canonical discrimination and changes groupmean. Needs its own complete results and independent Endpoint credit-only contrast, not presumed beneficial'),
  no_bundled_experiment=True,no_budget_or_live_configuration_changed=True,
  implementation_guard='Any later Endpoint+credit module must change finalized advantages only while retaining EndpointLoss conditionalSUM24, original sourceW3 anchor, likelihood/noise/RNG/optimizer and recovery. Existing source-PPO actor loss mixins cannot simply be stacked.')
 return result


def main():
 started=time.time();identities={};ident_sha={};seals={}
 for arm in ARMS:
  path=E/'runs'/arm/'identity.json';identities[arm]=json.loads(path.read_text());ident_sha[arm]=sha(path)
  cp=E/'runs'/arm/'checkpoints/update000500_inner0';seal=json.loads((cp/'COMPLETE').read_text())
  if seal['updates']!=500 or seal['phase']!='READY' or seal['world']!=8 or seal['next_inner']!=0:raise ValueError('Actual sealed500 boundary required')
  seals[arm]=dict(checkpoint=str(cp),seal=seal,COMPLETE_sha256=sha(cp/'COMPLETE'))
 results=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for future in as_completed([pool.submit(worker,(arm,rank,identities[arm]['sha256'])) for arm in ARMS for rank in range(8)]):
   r=future.result();results.append(r);print(r['arm'],r['rank'],'actual',len(r['rows']),'errors',len(r['errors']),flush=True)
 errors=[e for r in results for e in r['errors']];arms={arm:arm_summary(arm,[row for r in results if r['arm']==arm for row in r['rows']]) for arm in ARMS}
 complete=not errors and all(v['files_read']==1000 and v['credit_scale']['independent_global_collections']==125 and v['credit_scale']['max_saved_normalization_error']<1e-12 and v['credit_scale']['max_saved_canonical_std_error']<1e-12 for v in arms.values()) and not any(r['CUDA_initialized'] for r in results)
 one={ (r['collection'],r['rank']):r for r in arms[ARMS[0]]['group_receipts']};two={ (r['collection'],r['rank']):r for r in arms[ARMS[1]]['group_receipts']}
 same=sum(one[k]['token']==two[k]['token'] and one[k]['log']==two[k]['log'] for k in one.keys()&two.keys())
 result=dict(schema='world_complete_endpoint_control_credit_interaction_first500_v1',status='COMPLETE_ALL2000_ACTUAL_BEHAVIORS' if complete else 'INCOMPLETE_REQUIRES_REVIEW',created_unix=time.time(),elapsed_seconds=time.time()-started,
   population=dict(arms=2,updates_per_arm=500,collections_per_rank=125,ranks=8,expected_files=2000,actual_files=sum(v['files_read'] for v in arms.values()),expected_groups_per_arm=1000,expected_candidates_per_arm=8000,
    same_rank_collection_scene_join=dict(matched=same,total=1000),max_CPU_workers=4,actor_forwards=0,new_scores=0,optimizer_steps=0,random_sample=False),
   identities={a:dict(identity_sha256=identities[a]['sha256'],identity_file_sha256=ident_sha[a],research=identities[a]['recipe']['research'],seal=seals[a]) for a in ARMS},
   arms=arms,errors=errors,conclusions_zh=[],recommended_combination_review={},
   limits=['All first500 behavior payloads actually read+SHA256; every candidate label binds its actual submitted FP32 action. Endpoint chain[-1] is unscored ODE mean, never substituted for endpoint_policy.action.',
    'Same scene order does not mean same theta/roots/innovations: policy and RNG consumption diverge during training. Full measured encountered distributions, not a frozen-theta sampler causal comparison.',
    'Only frozen credit counterfactuals on already-scored actions; no new trajectory scores, policy logp, replay optimization or predicted NAV gains.',
    'Endpoint uses conditional jointSUM24 logp and sourceW3mean anchor; Control uses clamped dimension-mean W3 transition proxy. Equal normalizedA is not equal parameter gradient or anchor strength.',
    'More high-quality negativeA in absolute count cannot establish worse conflict. Rates per strict-high and weighted masses/gaps reported; A signs before cap/PPO are not parameter gradients.',
    'Do not stack an old Saturated/Protected/SourceLoss actor mixin onto Endpoint loss: future credit-only composition must keep EndpointLoss/conditional likelihood/anchor and native pending validators, independently registered.'],
   CPU_builder=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),math_source_files_sha256={p:sha(W/p) for p in ['starVLA/rl/qwen_world_safety_exploration/credit.py','starVLA/rl/qwen_world_safety_exploration/saturated_credit.py','starVLA/rl/qwen_world_safety_exploration/protected_credit.py','starVLA/rl/qwen_world_safety_exploration/inversion_cancellation.py','starVLA/model/modules/qwen_sourceflow/action_codec.py']}),
   live_modified=False,registered=False,committed=False)
 for arm,v in arms.items():
  c=v['counts'];result['conclusions_zh'].append(f"{arm}: strict高分{c['strict_high']}/8000，negativehigh{c['negative_high']}（占高分{100*v['rates']['negative_high_given_strict_high']:.3f}%）；PG0={c['PG0']}/1000，Saturated可激活{c['saturation_eligible_groups']}组，质量反序可抵消{c['inversion_groups']}组。")
 comparison=E/'analysis/PILOT_COMPARISONS/SAFETY_ENDPOINT_RAMP_AR1_G8_500_vs_SAFETY_G8_CONTROL_500.json'
 if comparison.exists():
  d=json.loads(comparison.read_text());result['deployment_comparison_existing']=dict(path=str(comparison),sha256=sha(comparison),status=d['status'],comparison=d['comparison'])
 result=enrich(result)
 TARGET.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(jsonable(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 for path in [TARGET,W/'reports/qwen_world_safety_exploration_20261031'/TARGET.name]:
  tmp=path.with_name(path.name+'.writing-'+str(os.getpid()));tmp.write_text(text);os.replace(tmp,path)
 builder=W/'reports/qwen_world_safety_exploration_20261031'/Path(__file__).name;builder.write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(status=result['status'],path=str(TARGET),counts={a:v['counts'] for a,v in arms.items()},rates={a:v['rates'] for a,v in arms.items()},PG0={a:v['PG0_breakdown'] for a,v in arms.items()},scale={a:{k:v['credit_scale'][k] for k in ['normalization_Safety_rawA_std','canonical_FP32_reward_global_std','strict_high_negative_raw_mass','strict_high_negative_normalized_mass','negative_normalized_mass_per_strict_high']} for a,v in arms.items()},counterfactuals={a:v['counterfactual_credit_replays'] for a,v in arms.items()},errors=errors),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
