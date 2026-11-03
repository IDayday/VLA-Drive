#!/usr/bin/env python3
"""Finite read-only CPU audit: all8 journals, native0/800 trainables+Adam only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from datetime import datetime,timezone
import hashlib,json,math,pathlib,sys,time
import numpy as np
import torch
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103');S=R/'runtime/formal_v1'
sys.path.insert(0,str(S));torch.set_num_threads(1)
from starVLA.rl.d44_native_rl.recovery import Boundary
ARMS=('D0','D5','D3_FPO','D3_AWM');N=103288

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def stat(v):
 v=[float(x)for x in v if x is not None];a=np.asarray(v,dtype=np.float64)
 if not len(a):return dict(n=0)
 finite=bool(np.isfinite(a).all())
 return dict(n=len(a),finite=finite,mean=float(a.mean())if finite else None,min=float(a.min())if finite else None,
  p10=float(np.quantile(a,.1))if finite else None,median=float(np.median(a))if finite else None,
  p90=float(np.quantile(a,.9))if finite else None,max=float(a.max())if finite else None)
def role(n):
 if n.startswith('action_expert.readouts.'):return'post_readout'
 if '.time_films.'in n:return'film_lora'
 if n.endswith('action_position'):return'action_position'
 if '.action_input.'in n or '.action_output.'in n:return'action_io'
 return'text_lora'
def part(n):return 'lora_B'if n.endswith('lora_B')else 'lora_A'if n.endswith('lora_A')else'non_LoRA_trainable'

def read_journals(root,identity):
 allrows=[];evidence=[];errors=[]
 for rank in range(8):
  p=root/f'metrics_rank{rank}.jsonl';b=p.read_bytes();lines=b.splitlines(keepends=True)
  if lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
  rows=[json.loads(x)for x in lines if x.strip()]
  for expected,row in enumerate(rows,1):
   if row['updates']!=expected or row['run']!=identity['sha256']:errors.append(f'rank{rank} wrong update/run at{expected}')
   if row['collection']!=(expected+1)//2 or row['inner_completed']!=(expected-1)%2+1:errors.append(f'rank{rank} wrong M2 ancestry at{expected}')
  allrows.append(rows);evidence.append(dict(rank=rank,path=str(p),snapshot_rows=len(rows),snapshot_bytes=sum(map(len,lines)),
   complete_lines_sha256=hashlib.sha256(b''.join(lines)).hexdigest()))
 common=min(map(len,allrows));complete=common//2*2
 if complete<800:errors.append('Not yet800 all8 updates')
 return allrows,evidence,errors,complete

def window(rows,start,end):
 rr=[v[start-1:end]for v in rows];updates=list(range(start,end+1));collections=list(range((start+1)//2,(end+1)//2+1))
 flat=[x for rank in rr for x in rank];local_losskeys=sorted(set(k for x in flat for k in x['losses']))
 # All rank local loss means are equally weighted; no padding before one epoch.
 per_update=[]
 for u in updates:
  batch=[rank[u-1]for rank in rows]
  per_update.append(dict(updates=u,mean_gradient_norm=float(np.mean([x['gradient_norm']for x in batch])),
   min_gradient_norm=min(x['gradient_norm']for x in batch),max_gradient_norm=max(x['gradient_norm']for x in batch),
   mean_reference_KL_per_coordinate=float(np.mean([x['losses']['reference_KL_per_coordinate']for x in batch])),
   beta=batch[0]['beta'],lr=batch[0]['lr'],
   mean_native_gaussian_clip_fraction=float(np.mean([x['losses']['clip_fraction']for x in batch])),
   mean_actual_fpo_clip_fraction=float(np.mean([x['losses'].get('fpo_clip_fraction',0)for x in batch]))if'fpo_clip_fraction'in batch[0]['losses']else None))
 # Exploration/std are already all-reduced globals copied to each rank/inner.
 globalrows=[rows[0][2*c-2]for c in collections]
 gcount=sum(x['exploration']['raw_groups']for x in globalrows)
 eff=sum(round(x['exploration']['effective_PG_group_fraction']*x['exploration']['raw_groups'])for x in globalrows)
 allzero=sum(x['exploration']['all_zero_groups']for x in globalrows);allone=sum(x['exploration']['all_one_groups']for x in globalrows)
 roles=sorted(set(k for x in flat for k in x['role_gradient_norms']))
 clocks={k:stat([x[k]for x in flat])for k in('gradient_norm','global_advantage_std','update_seconds','peak_GPU_bytes','beta')}
 loss={k:stat([x['losses'].get(k)for x in flat])for k in local_losskeys}
 inner={}
 for i in(1,2):
  values=[x for x in flat if x['inner_completed']==i]
  inner[str(i)]=dict(gradient_norm=stat([x['gradient_norm']for x in values]),losses={k:stat([x['losses'].get(k)for x in values])for k in local_losskeys},
   role_gradients_post_clip={k:stat([x['role_gradient_norms'].get(k)for x in values])for k in roles})
 timing=[]
 for c in collections:
  local=[]
  for rank in rows:
   one,two=rank[2*c-2],rank[2*c-1]
   local.append(dict(generation=one['collection_seconds_generation'],scoring=one['collection_seconds_scoring'],
    optimization=one['update_seconds']+two['update_seconds']))
  timing.append(dict(collection=c,generation_worker_seconds_sum=sum(v['generation']for v in local),
   scoring_worker_seconds_sum=sum(v['scoring']for v in local),
   model_update_worker_seconds_sum=sum(v['optimization']for v in local),
   critical_rank_reported_components_seconds=max(sum(v.values())for v in local)))
 first_at=min(rank[start-1]['at']for rank in rows);last_at=max(rank[end-1]['at']for rank in rows)
 wall=last_at-first_at;exposed=len(collections)*64
 rank_timing=[dict(rank=i,reported_generation_seconds=sum(rank[2*c-2]['collection_seconds_generation']for c in collections),
  reported_scoring_seconds=sum(rank[2*c-2]['collection_seconds_scoring']for c in collections),
  optimization_seconds=sum(x['update_seconds']for x in rank[start-1:end]))for i,rank in enumerate(rows)]
 consensus=max(x['max_gradient_norm']-x['min_gradient_norm']for x in per_update)
 fresh=[x['losses']['fresh_ratio_error']for x in flat if x['inner_completed']==1]
 finite=all(math.isfinite(v)for x in flat for v in [x['gradient_norm'],*x['lr'],*x['losses'].values()]if isinstance(v,(int,float)))
 return dict(update_start=start,update_end=end,collections=len(collections),collection_start=collections[0],collection_end=collections[-1],
  actual_valid_scene_exposures_in_window=exposed,scene_exposures_end=rows[0][end-1]['scene_exposures'],
  epochs_end=rows[0][end-1]['scene_epochs'],actual_candidate_exposures=exposed*16,all8_rank_rows=len(flat),finite_losses_and_gradients=finite,
  all8_max_gradient_norm_disagreement=consensus,first_inner_fresh_ratio_error=stat(fresh),gradient_clipped_updates_fraction=float(np.mean([x['gradient_norm']>1for x in rows[0][start-1:end]])),
  scalar_summaries=clocks,local_losses_equal_rank_weight=loss,by_inner=inner,
  role_gradients_post_clip={k:stat([x['role_gradient_norms'].get(k)for x in flat])for k in roles},
  LR_by_optimizer_group_index={str(j):stat([x['lr'][j]for x in flat])for j in range(len(flat[0]['lr']))},
  exploration_global_once_per_collection=dict(groups=gcount,effective_raw_A_nonzero_groups=eff,effective_raw_A_fraction=eff/gcount,
   raw_A_allzero_groups=gcount-eff,canonical_allzero_groups=allzero,canonical_allone_groups=allone,
   feasible_strict_NC_DAC_TTC_candidate_fraction=stat([x['exploration']['feasible_candidate_fraction']for x in globalrows]),
   sampled_candidate_score_mean=stat([x['exploration']['random_candidate_mean']for x in globalrows]),
   sampled_best_of_G16_score_mean=stat([x['exploration']['oracle_mean']for x in globalrows])),
  time=dict(timestamp_first_optimizer_end=first_at,timestamp_last_optimizer_end=last_at,
   elapsed_between_these_commits_seconds=wall,approx_valid_scenes_per_wall_second=exposed/wall if wall else None,
   timing_excludes_first_collection_before_first_timestamp=True,reported_per_rank=rank_timing,
   collections=timing,critical_rank_reported_components_seconds=stat([x['critical_rank_reported_components_seconds']for x in timing]),
   note='Collection generation/scoring repeats in both inner rows and is counted once. Recorded generation ends before FPO/AWM CFM-material construction; checkpoint/I/O/wait time is not all in update_seconds. Wall timestamps include contention and publication overhead.'),
  per_update=per_update)

def checkpoint_material(root,u):
 p=root/'checkpoints'/f'update{u:06}_inner0';co=load(p/'COMPLETE');files=load(p/'files.json');i=load(p/'identity.json')
 errors=[]
 if co['world']!=8 or co['updates']!=u or co['next_inner']!=0 or sha(p/'files.json')!=co['files_sha256']:errors.append('seal/counters')
 for n,h in files.items():
  if sha(p/n)!=h:errors.append('hash '+n)
 common=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True);summaries=[]
 for rank in range(8):
  v=torch.load(p/f'rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
  summary=Boundary.validate_boundary(v['state'],v['rollout'],i,common['optimizer'],common['scheduler']);summaries.append(summary)
 if summaries!=load(p/'transaction.json')['ranks']:errors.append('actual boundary vs transaction')
 Boundary.validate_ranks(summaries)
 return common,dict(checkpoint=str(p),complete=co,files_sha256=sha(p/'files.json'),full_inventory_actual_hashes_checked=len(files),
  status='PASS'if not errors else 'REVIEW_REQUIRED',errors=errors,scheduler=common['scheduler'],all8_boundary_summary=summaries,
  optimizer_state_entries=len(common['optimizer']['state']),identity_sha256=i['sha256'])

def parameter_change(root):
 before,s0=checkpoint_material(root,0);after,s8=checkpoint_material(root,800)
 if list(before['adapters'])!=list(after['adapters']):raise ValueError('Trainable closure changed')
 bn=before['adapters'];an=after['adapters'];names={}
 for n in an:names.setdefault(role(n),[]).append(n)
 role_groups={g['name']:g for g in after['optimizer']['param_groups']};by={};moments={};eps_total={}
 for label in list(names)+['ALL']:
  nn=list(an)if label=='ALL'else names[label];by[label]={}
  for subtype in('all','lora_A','lora_B','non_LoRA_trainable'):
   subset=nn if subtype=='all'else[n for n in nn if part(n)==subtype]
   count=changed=0;d2=b2=t2=0.;maxd=0.
   for n in subset:
    x=an[n].double();b=bn[n].double();delta=x-b
    count+=x.numel();changed+=int(torch.count_nonzero(delta));d2+=float(delta.square().sum());b2+=float(b.square().sum());t2+=float(x.square().sum());maxd=max(maxd,float(delta.abs().max()))
   if count:by[label][subtype]=dict(numel=count,baseline_l2=math.sqrt(b2),at800_l2=math.sqrt(t2),delta_l2=math.sqrt(d2),
    delta_RMS=math.sqrt(d2/count),max_abs_delta=maxd,changed_fraction=changed/count,relative_delta_l2=math.sqrt(d2/b2)if b2 else None)
 for ro,nn in names.items():
  group=role_groups[ro]
  if len(nn)!=len(group['params']):raise ValueError('Optimizer mapping length '+ro)
  aggregate={}
  for n,pid in zip(nn,group['params']):
   moment=after['optimizer']['state'].get(pid)
   if not moment:continue
   m=moment['exp_avg'];v=moment['exp_avg_sq']
   if m.shape!=an[n].shape or v.shape!=an[n].shape:raise ValueError('Optimizer mapping shape '+n)
   step=int(moment['step']);b1,b2=group['betas'];eps=group['eps']
   mh=m.double()/(1-b1**step);vh=(v.double()/(1-b2**step)).sqrt();den=vh+eps
   adaptive=group['lr']*mh/den;decay=group['lr']*group['weight_decay']*an[n].double()
   for subtype in ('all',part(n)):
    a=aggregate.setdefault(subtype,dict(numel=0,step_values=set(),sqrt_vhat_le_eps=0,sqrt_vhat_le_10eps=0,sqrt_vhat_zero=0,
      sqrt_vhat_sum=0.,epsilon_attenuation_sum=0.,adaptive_step_squared_sum=0.,decoupled_weight_decay_step_squared_sum=0.,finite=True))
    a['numel']+=m.numel();a['step_values'].add(step);a['sqrt_vhat_le_eps']+=int((vh<=eps).sum());a['sqrt_vhat_le_10eps']+=int((vh<=10*eps).sum());a['sqrt_vhat_zero']+=int((vh==0).sum())
    a['sqrt_vhat_sum']+=float(vh.sum());a['epsilon_attenuation_sum']+=float((vh/den).sum());a['adaptive_step_squared_sum']+=float(adaptive.square().sum());a['decoupled_weight_decay_step_squared_sum']+=float(decay.square().sum())
    a['finite']=a['finite']and bool(torch.isfinite(mh).all()and torch.isfinite(vh).all()and torch.isfinite(adaptive).all())
  for subtype,a in aggregate.items():
   total=a['numel'];a['step_values']=sorted(a['step_values']);a['fraction_sqrt_vhat_le_eps']=a.pop('sqrt_vhat_le_eps')/total;a['fraction_sqrt_vhat_le_10eps']=a.pop('sqrt_vhat_le_10eps')/total
   a['fraction_sqrt_vhat_zero']=a.pop('sqrt_vhat_zero')/total;a['mean_sqrt_vhat']=a.pop('sqrt_vhat_sum')/total;a['mean_epsilon_attenuation']=a.pop('epsilon_attenuation_sum')/total
   a['estimated_next_adaptive_step_RMS_at_saved_LR']=math.sqrt(a.pop('adaptive_step_squared_sum')/total)
   a['decoupled_weight_decay_step_RMS_at_saved_theta_LR']=math.sqrt(a.pop('decoupled_weight_decay_step_squared_sum')/total)
  moments[ro]=dict(eps=group['eps'],lr=group['lr'],initial_lr=group['initial_lr'],betas=group['betas'],subsets=aggregate)
 return dict(status='PASS'if s0['status']==s8['status']=='PASS'else'REVIEW_REQUIRED',native0=s0,native800=s8,
  theta_changes=by,actual_Adam_moments_at800=moments,
  units='Raw trainable parameter coordinates, not action meters/policy KL. LoRA factorization means equal parameter distance does not imply equal function distance.',
  epsilon_analysis='Using saved biased moments m,v and step800: mhat=m/(1-b1^800), sqrt_vhat=sqrt(v/(1-b2^800)); attenuation=sqrt_vhat/(sqrt_vhat+eps). This quantifies numerical scale breaking, not causal proof or an optimizer-change recommendation. Saved LR gives illustrative next-step magnitude with current moments, not a replay of actual past updates.')

def main():
 start=time.time();campaign=load(R/'CAMPAIGN.json');arms={}
 for arm in ARMS:
  run=campaign['runs'][arm];root=pathlib.Path(run['output']);identity=load(root/'identity.json')
  rows,evidence,errors,complete=read_journals(root,identity)
  first=window(rows,1,800);last=window(rows,complete-199,complete)
  p=parameter_change(root)
  arms[arm]=dict(identity_sha256=identity['sha256'],source=str(S),source_files_fingerprint=fp(identity['source_files']),
   parent_model_sha256=identity['parent_model_sha256'],configuration=identity['recipe'],trainable_numel=identity['scope']['trainable_numel'],
   journal_evidence=evidence,journal_errors=errors,all8_common_committed_complete_update=complete,
   equal_exposure_first800=first,recent100_complete_collections=last,native_parameter_and_Adam=p,
   objective_semantics=('actual clipped native conditional joint Gaussian PG on W3'if arm in('D0','D5')else
    'actual clipped exp(oldCFMenergy-newCFMenergy) surrogate; unprefixed Gaussian clip/ratio diagnostic only, not used PG'if arm=='D3_FPO'else
    'actual positive-A-weighted CFMenergy plus coefficient1 clean-x reference MSE; Gaussian clip/ratio diagnostic only; adaptive Gaussian KL also retained'),
   comparison_limitations='D5 changes scope/rank/multiple role LRs; AWM also changes LR and reference objective. Recent100 windows are at distinct exposures and not matched candidate/parameter states.')
  print(arm,'complete',complete,'deltaL2',p['theta_changes']['ALL']['all']['delta_l2'],flush=True)
 ratios={}
 d0=arms['D0']
 for a,v in arms.items():
  ratios[a]=dict(gradient_mean_first800_relative_D0=v['equal_exposure_first800']['scalar_summaries']['gradient_norm']['mean']/d0['equal_exposure_first800']['scalar_summaries']['gradient_norm']['mean'],
   total_parameter_delta_L2_relative_D0=v['native_parameter_and_Adam']['theta_changes']['ALL']['all']['delta_l2']/d0['native_parameter_and_Adam']['theta_changes']['ALL']['all']['delta_l2'],
   text_LoRA_B_delta_L2_relative_D0=v['native_parameter_and_Adam']['theta_changes']['text_lora']['lora_B']['delta_l2']/d0['native_parameter_and_Adam']['theta_changes']['text_lora']['lora_B']['delta_l2'])
 output=dict(schema='d46_early_optimization_health_v1',status='COMPLETE_BOUNDED_CPU_AUDIT',
  generated_utc=datetime.now(timezone.utc).isoformat(),started_unix=start,finished_unix=time.time(),builder=dict(path=__file__,file_sha256=sha(__file__)),
  matched_window=dict(optimizer_updates=800,global_collections=400,world=8,global_scenes_per_collection=64,candidates_per_scene=16,
   inner_updates=2,valid_scene_exposures=25600,epochs=25600/N,physical_candidates=409600,
   note='Same exposure/update clock, same actual D46 IL deployment parent; sampler material/aux RNG and D5 scope are not identical across methods.'),
  source_audit={p:sha(S/p)for p in ('tools/d44_native_rl/train.py','starVLA/rl/d44_native_rl/actor.py','starVLA/rl/d44_native_rl/fpo.py','starVLA/rl/d44_native_rl/expanded_scope.py','starVLA/rl/d44_native_rl/adaptive_kl.py')},
  arms=arms,cross_arm_scale_comparison=ratios,
  evidence_boundaries_zh=[
   '全部统计来自四条自有正式run的八rank日志，以及真实封存native0/u800；无模型forward/GPU/新评分，未改训练或调度。',
   'gradient_norm为clip_grad_norm_(max1)前总范数；role_gradient_norms为裁剪后组范数。D5较大总梯度不能直接与裁剪后role数值比较。',
   'effectivePG、candidate可行率和allzero/allone已在采样时all_reduce到64组，并复制到八rank/两个inner；统计每collection只取一次，不乘8或2。有效PG是rawA非0，不等于最终有益参数方向。',
   'FPO真正更新看fpo_*；unprefixed ratio/clip是Gaussian核偏移诊断。AWM没有ratio clipping目标，pg内含positive energy与clean-x reference MSE。',
   '旧reference beta/KL与AWM coefficient1 reference MSE不同。标量不能确定各项参数梯度强弱，本次没有额外反向分解。',
   '梯度整体缩小不等价Adam更新同比缩小；实读theta差与moments、eps衰减。参数距离和采样候选分均不是单B1部署成绩，不能据此判最终胜负。',
   '最近100个完整collection是各自全八rank共同journal前缀，当前曝光不同；可能尚未全部落到最后原生checkpoint，若故障只从严格封存CP恢复。',
   'generation/scoring数字为原日志已记录阶段；CFM材料准备、存盘与共享资源等待不全覆盖在该分项。吞吐以真实墙钟与报告口径并列解释。'])
 output['sha256']=fp(output);p=R/'analysis/EARLY_OPTIMIZATION_HEALTH_20261103.json';tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
 print(json.dumps(dict(path=str(p),file_sha256=sha(p),ratios=ratios)),flush=True)
if __name__=='__main__':main()
