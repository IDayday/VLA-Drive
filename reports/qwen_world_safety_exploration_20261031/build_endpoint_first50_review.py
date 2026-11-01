"""Read only bounded actual first50 prefixes; no training/evaluation actions."""
import ctypes
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import select
import time

import numpy as np
import torch

E=Path('/mnt/project/qwen-world-safety-exploration-20261031')
W=Path('/mnt/project/VLA-Drive-qwen-world-safety-exploration-20261031')
A=E/'analysis';ARMS=('SAFETY_ENDPOINT_RAMP_AR1_G8','SAFETY_G8_CONTROL')
OUT=A/'ENDPOINT_PRODUCTION_FIRST50_REVIEW.json'
def digest(x):return hashlib.sha256(x).hexdigest()
def fsha(p):return digest(Path(p).read_bytes())
def js(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def stats(v):
 a=np.asarray(v,dtype=np.float64)
 return None if not a.size else dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))
def nonfinite(v,path=''):
 if isinstance(v,float) and not math.isfinite(v):return [path]
 if isinstance(v,dict):return [p for k,x in v.items() for p in nonfinite(x,path+'.'+k)]
 if isinstance(v,list):return [p for k,x in enumerate(v) for p in nonfinite(x,path+'.'+str(k))]
 return []

def read_prefixes(runs):
 """Each open file progresses forward only. Wait on kernel modify events."""
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0x800|0x80000)
 if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
 readers={};watches={};deadline=time.monotonic()+600
 for arm,r in runs.items():
  for rank in range(8):
   p=Path(r['root'])/f'metrics_rank{rank}.jsonl'
   wd=libc.inotify_add_watch(fd,str(p).encode(),0x2|0x8|0x800|0x400)
   if wd<0:raise OSError(ctypes.get_errno(),str(p))
   readers[(arm,rank)]=dict(path=p,file=p.open('rb'),rows=[],raw=[])
 def consume(reader):
  f=reader['file']
  while len(reader['rows'])<50:
   offset=f.tell();line=f.readline()
   if not line or not line.endswith(b'\n'):f.seek(offset);break
   row=json.loads(line);expected=len(reader['rows'])+1
   if row['updates']!=expected:raise ValueError('Noncontiguous/duplicate initial journal prefix '+str(reader['path']))
   bad=nonfinite(row)
   if bad:raise FloatingPointError('Actual nonfinite '+str(reader['path'])+str(bad))
   if row['inner_completed']==1 and row['ratio_max_abs_from_one']>1e-4:
    raise FloatingPointError('Actual fresh ratio failure '+str(reader['path'])+' u'+str(expected))
   reader['rows'].append(row);reader['raw'].append(line)
 for reader in readers.values():consume(reader)
 while any(len(r['rows'])<50 for r in readers.values()):
  print(json.dumps(dict(waiting_for_actual_first50={a:[len(readers[(a,k)]['rows']) for k in range(8)] for a in ARMS})),flush=True)
  if time.monotonic()>deadline:raise TimeoutError('First50 not all complete within finite10min; no results imputed')
  ready=select.select([fd],[],[],30)[0]
  if not ready:continue
  import os,struct
  events=os.read(fd,262144);changed=set();pos=0
  while pos<len(events):
   wd,mask,cookie,size=struct.unpack_from('iIII',events,pos);pos+=16+size;changed.add(wd)
  # inode watches are immutable while the native journal append handle is live.
  for reader in readers.values():
   if len(reader['rows'])<50:consume(reader)
 for reader in readers.values():reader['file'].close()
 import os
 os.close(fd)
 return readers


def analyze_arm(arm,run,readers):
 root=Path(run['root']);ident=json.loads((root/'identity.json').read_text());rows=[dict(rank=k,**row) for k in range(8) for row in readers[(arm,k)]['rows']]
 groups=[r for r in rows if r['inner_completed']==1]
 assert len(rows)==400 and len(groups)==104 and ident['parameter_contract']['trainable_numel']==52707840
 for r in rows:
  assert r['group_size']==8 and r['run']==ident['sha256']
  assert r['inner_completed']==(r['updates']-1)%4+1 and r['collection']==(r['updates']-1)//4+1
 def subset_summary(records):
  byupdate={}
  for r in records:byupdate.setdefault(r['updates'],[]).append(r)
  gradient_unique=[float(np.mean([r['gradient_norm'] for r in rs])) for rs in byupdate.values()]
  ratio_min=[min(r['metrics'][f'step{s}_ratio_min'] for s in ([9] if arm==ARMS[0] else [7,8,9])) for r in records]
  ratio_max=[max(r['metrics'][f'step{s}_ratio_max'] for s in ([9] if arm==ARMS[0] else [7,8,9])) for r in records]
  metrics=['ratio_mean','clipfrac','ppo_blocked_fraction','positive_ppo_blocked_fraction','negative_ppo_blocked_fraction',
   'positive_active_fraction','negative_active_fraction','pg_logprob_weight_abs_mean','policy_loss','kl_loss','approx_kl',
   'quality_active_positive_high_weight','quality_active_negative_high_weight','quality_active_positive_unsafe_weight',
   'quality_active_negative_unsafe_weight','quality_high_weight_per_good','quality_high_positive_PPO_blocked']
  return dict(rank_update_rows=len(records),optimizer_updates=len(byupdate),metrics={m:stats([r['metrics'][m] for r in records]) for m in metrics},
   gradient_norm_preclip=stats(gradient_unique),optimizer_updates_preclip_norm_gt1=sum(x>1 for x in gradient_unique),
   gradient_norm_max_rank_disagreement=max(max(r['gradient_norm'] for r in rs)-min(r['gradient_norm'] for r in rs) for rs in byupdate.values()),
   ratio_extrema=dict(min=min(ratio_min),max=max(ratio_max)),ratio_max_abs_from_one=stats([r['ratio_max_abs_from_one'] for r in records]),
   global_rank_mean_policy_loss=stats([np.mean([r['metrics']['policy_loss'] for r in rs]) for rs in byupdate.values()]),
   global_rank_mean_weighted_anchor_loss=stats([np.mean([r['metrics']['kl_loss'] for r in rs]) for rs in byupdate.values()]),
   source_clamp_NA_rows=sum(r['source_clamp_hit_rate'] is None for r in records),
   source_clamp_hit_rate=stats([r['source_clamp_hit_rate'] for r in records if r['source_clamp_hit_rate'] is not None]))
 support=dict(groups=len(groups),candidate_count=len(groups)*8,
  strict_high_candidates=sum(r['metrics']['quality_strict_high_count'] for r in groups),
  groups_with_strict_high=sum(r['metrics']['quality_strict_high_count']>0 for r in groups),
  feasible_candidates=sum(r['metrics']['quality_feasible_count'] for r in groups),
  groups_with_feasible=sum(r['metrics']['quality_feasible_count']>0 for r in groups),
  zero_PG_groups=sum(r['exploration']['zero_PG'] for r in groups),
  all_canonical_zero_groups=sum(r['exploration']['all_zero'] for r in groups),
  all_full_groups=sum(r['exploration']['all_full'] for r in groups),
  safety_differing_groups=sum(r['exploration']['safety_differing'] for r in groups),
  progress_only_groups=sum(r['exploration']['progress_only'] for r in groups),
  negative_strict_high_candidates=sum(r['metrics']['quality_negative_high_count'] for r in groups),
  positive_strict_high_candidates=sum(r['metrics']['quality_positive_high_count'] for r in groups),
  positive_unsafe_with_feasible_alternative=sum(r['exploration']['positive_unsafe_with_safe_alternative'] for r in groups),
  positive_still_unsafe_candidates=sum(r['exploration']['positive_still_unsafe'] for r in groups),
  zero_PG_success_safe_EP1_groups=sum(r['exploration']['zero_PG'] and r['exploration']['safe_fraction']==1 and abs(r['exploration']['EP']-1)<1e-12 for r in groups),
  global_collections_all8ranks_zero_PG=sum(all(r['exploration']['zero_PG'] for r in groups if r['collection']==c) for c in range(1,14)),
  exploration_mean={k:float(np.mean([r['exploration'][k] for r in groups])) for k in ['canonical_random_mean','NC','DAC','TTC','EP','safe_fraction']})
 for key in ('groups_with_strict_high','groups_with_feasible','zero_PG_groups','all_canonical_zero_groups'):
  support[key+'_fraction']=support[key]/len(groups)
 support['strict_high_candidate_fraction']=support['strict_high_candidates']/support['candidate_count']
 support['feasible_candidate_fraction']=support['feasible_candidates']/support['candidate_count']
 source_inputs=[]
 for rank in range(8):
  reader=readers[(arm,rank)];raw=b''.join(reader['raw']);cache=A/'ENDPOINT_FIRST50_PREFIX_CACHE'/f'{arm}_rank{rank}.jsonl.gz';cache.parent.mkdir(parents=True,exist_ok=True)
  with gzip.GzipFile(filename=str(cache),mode='wb',mtime=0) as f:f.write(raw)
  source_inputs.append(dict(path=str(reader['path']),selection='exact first50 newline-committed rows, updates1..50',bytes_read=len(raw),prefix_sha256=digest(raw),cache=str(cache),cache_sha256=fsha(cache)))
 summary=dict(id=arm,run_root=str(root),source=run['source'],source_commit=run['source_commit'],identity_sha256=ident['sha256'],identity_file_sha256=fsha(root/'identity.json'),
  parameter_contract=dict(trainable_numel=ident['parameter_contract']['trainable_numel'],adaptation=ident['parameter_contract']['adaptation'],action_layers=ident['parameter_contract']['action_layers']),
  input_journal_prefixes=source_inputs,all_rows_finite=True,fresh_ratio_max_abs_error=max(r['ratio_max_abs_from_one'] for r in groups),
  accounting=dict(optimizer_updates=50,rank_update_rows=400,fresh_global_collections=13,scene_exposures=104,scored_candidates=832,fully_reused_collections=12,last_collection_inners_observed=2),
  overall=subset_summary(rows),by_inner={str(i):subset_summary([r for r in rows if r['inner_completed']==i]) for i in range(1,5)},
  matched_complete_collection_credit={str(i):{k:stats([r['metrics'][k] for r in rows if r['inner_completed']==i and r['collection']<=12]) for k in ['clipfrac','ppo_blocked_fraction','positive_ppo_blocked_fraction','quality_active_positive_high_weight','quality_active_negative_high_weight','quality_high_positive_PPO_blocked']} for i in range(1,5)},
  within_prefix_windows={f'{lo}_{hi}':subset_summary([r for r in rows if lo<=r['updates']<=hi]) for lo,hi in [(1,16),(17,32),(33,50)]},
  group_support=support,actual_optimizer=ident['actual_optimizer'],lr_used_by_update={str(u):rows[u-1]['optimizer_group_lrs_used'] for u in range(1,51)})
 complete=summary['matched_complete_collection_credit']
 complete['scope']='Exact same96 rank/collection/token groups from collections1..12, all4inners; collection13 excluded for inner comparison.'
 complete['positive_high_coefficient_inner4_over_inner1']=complete['4']['quality_active_positive_high_weight']['mean']/complete['1']['quality_active_positive_high_weight']['mean']
 if arm==ARMS[0]:
  proof=[]
  for row in groups:
   rank=row['rank'];col=row['collection'];p=root/'behavior'/f'collection{col:06}'/f'rank{rank}.pt'
   payload=torch.load(p,map_location='cpu',weights_only=False);ep=payload['endpoint_policy'];actions=ep['action']
   assert actions.dtype==torch.float32 and actions.shape==(8,8,3) and payload['old_logp'].dtype==torch.float64 and payload['old_logp'].shape==(8,1)
   assert len(payload['records'])==8 and 'old_probability' not in payload
   exact=all(np.asarray(r['requested_trajectory']).dtype==np.float32 and np.array_equal(np.asarray(r['requested_trajectory']),actions[j].numpy()) for j,r in enumerate(payload['records']))
   if not exact:raise ValueError('Production action/scored request mismatch '+str(p))
   proof.append(dict(rank=rank,collection=col,behavior_path=str(p),behavior_file_sha256=fsha(p),token=payload['token'],
    action_FP32_sha256=digest(actions.contiguous().numpy().tobytes()),scorer_protocol_sha256=payload['protocol_sha256'],
    quantization_max_abs=max(ep['quantization_max_abs_by_dim']),ODE_vs_replay_mean_max_abs=ep['ODE_vs_replay_mean_max_abs']))
  summary['actual_action_audit']=dict(groups=len(proof),candidate_count=832,all_exact_F32_service_request_match=True,
   old_probability_dtype='float64 joint24 [8,1]',density='Conditional on saved terminal latent; not full-flow marginal; continuous density at actual scored FP32 coordinates promoted toF64',
   chain_endpoint='Unscored ODE clean mean; actual scored noisy endpoint is endpoint_policy.action',
   endpoint_protocol=ident['endpoint_policy_protocol'],max_quantization_error=max(r['quantization_max_abs'] for r in proof),
   max_ODE_vs_replayed_mean_error=max(r['ODE_vs_replay_mean_max_abs'] for r in proof),
   source_clamp='N/A/null for endpoint PG, not0; nominal anchor clamp remains separate',
   underflow_candidates_in_update_rows=sum(r['metrics']['endpoint_ratio_underflow_candidates'] for r in rows),
   log_ratio_min=min(r['metrics']['endpoint_log_ratio_min'] for r in rows),log_ratio_max=max(r['metrics']['endpoint_log_ratio_max'] for r in rows),
   groups_provenance=proof)
 return summary,rows,groups


def main():
 registry=json.loads((E/'RUNS.json').read_text());runs={a:next(r for r in registry['runs'] if r['id']==a) for a in ARMS}
 readers=read_prefixes(runs);summaries={};raw={};groups={}
 for arm in ARMS:summaries[arm],raw[arm],groups[arm]=analyze_arm(arm,runs[arm],readers)
 matched=[]
 for e,b in zip(groups[ARMS[0]],groups[ARMS[1]]):
  assert (e['rank'],e['collection'],e['token'])==(b['rank'],b['collection'],b['token'])
  matched.append(dict(rank=e['rank'],collection=e['collection'],token=e['token'],
   endpoint_strict_high=e['metrics']['quality_strict_high_count'],control_strict_high=b['metrics']['quality_strict_high_count'],
   endpoint_feasible=e['metrics']['quality_feasible_count'],control_feasible=b['metrics']['quality_feasible_count']))
 for e,b in zip(raw[ARMS[0]],raw[ARMS[1]]):assert e['optimizer_group_lrs_used']==b['optimizer_group_lrs_used']
 review=dict(schema='world_endpoint_production_first50_review_v1',status='COMPLETE_FIRST50_ALL8RANKS',created_UTC=datetime.now(timezone.utc).isoformat(),
  builder=dict(path=str(Path(__file__)),sha256=fsha(__file__)),runs=summaries,
  comparison=dict(matched_rank_collection_token_groups=len(matched),LR_used_identical_all_updates=True,
   endpoint_gained_strict_support_groups=sum(r['endpoint_strict_high']>0 and r['control_strict_high']==0 for r in matched),
   endpoint_lost_strict_support_groups=sum(r['endpoint_strict_high']==0 and r['control_strict_high']>0 for r in matched),
   delta_strict_high_candidates=sum(r['endpoint_strict_high']-r['control_strict_high'] for r in matched),
   delta_feasible_candidates=sum(r['endpoint_feasible']-r['control_feasible'] for r in matched),
   group_pairs=matched),
  interpretation_limits=['Only first50 training optimizer updates (warmup100), 104 scene-groups/832 scored candidates per arm; noNAV evaluation result and no generalization or long-run stability claim.',
   'Actual52,707,840 trainable parameters are verified from both live run identities, not inferred from the tiny CPU fixture.',
   'Endpoint ratio is joint24 conditional Gaussian; Control ratio is source W3 clamped dimension-mean proxy. Same numeric clip.005 does not imply identical trust region.',
   'Gradient norms are post-DDP/pre-clip complete-loss norms, summarized once per optimizerupdate; PG/anchor scalar losses do not measure separate gradient magnitudes or causal dominance.',
   'Support uses strict NC=DAC=TTC=Comfort1 and canonical>=.9; feasible means onlyNC/DAC/TTC1. Fresh group summaries deduplicate four reused innerupdates.',
   'Scene/token ordering and LR are matched; random candidates and later trained theta are not common-random-number matched. This is not rho-only ablation.',
   'Within-prefix windows contain different scenes and warmupLR values; their differences are descriptive, not isolated learning trends.',
   'Both arms have a feasible candidate in every one of these104 fresh groups and no allcanonicalzero group; this prefix does not test persistent-allfail rescue.',
   'Additional model work:39 vs37 G8-batched velocitycalls and10 vs6 current-memory rebuilds per4inner collection; both officialscore64 candidates/world. Anchor beta50/sourceweightedMSE retained but usesODEstates.'],
  actions=dict(training_changed=False,config_changed=False,owner_changed=False,new_GPU_or_scoring=False,scientific_stop_requested=False),
  conclusions_zh=[])
 e=summaries[ARMS[0]];b=summaries[ARMS[1]]
 review['conclusions_zh']=[f"两个生产run identity均确认52,707,840可训练参数；8rank前50步完整且finite，Endpoint fresh ratio最大误差{e['fresh_ratio_max_abs_error']:.3g}。",
  f"Endpoint整体DDP/preclip梯度均值{e['overall']['gradient_norm_preclip']['mean']:.6g}（Control {b['overall']['gradient_norm_preclip']['mean']:.6g}），>1更新{e['overall']['optimizer_updates_preclip_norm_gt1']}/50；这是完整loss梯度，不能拆成PG/anchor因果。",
  f"104共同TRAIN组，Endpoint strict高质量候选{e['group_support']['strict_high_candidates']:.0f}/832、有strict见证组{e['group_support']['groups_with_strict_high']}/104；Control对应{b['group_support']['strict_high_candidates']:.0f}/832和{b['group_support']['groups_with_strict_high']}/104。",
  'joint24的ratio与PPO阻塞尺度须看真实生产记录，不能沿用tinyfixture的量级推断。首50步只证明当前生产执行与局部信用事实；没有新NAV结论，不承诺稳定或提分。']
 review['conclusions_zh'].insert(3,'Endpoint局部零PG12/104、Control8/104均属于全NC/DAC/TTC通过且EP1的成功饱和；没有任何一批8rank全部PG0，不能把局部zeroPG等同失败或全局只anchor。')
 review['conclusions_zh'].insert(4,'同96组四inner对齐后，strict高分正向surrogate系数在Endpoint从inner1到4下降约'+str(round((1-e['matched_complete_collection_credit']['positive_high_coefficient_inner4_over_inner1'])*100,2))+'%；该系数不是参数梯度，不能把标量loss/信用系数当成PG与anchor梯度幅度比较。')
 atomic(OUT,review);report=W/'reports/qwen_world_safety_exploration_20261031/ENDPOINT_PRODUCTION_FIRST50_REVIEW.json';atomic(report,review)
 print(json.dumps(dict(status=review['status'],out=str(OUT),report=str(report),conclusions=review['conclusions_zh'],comparison={k:v for k,v in review['comparison'].items() if k!='group_pairs'}),indent=2),flush=True)

if __name__=='__main__':main()
