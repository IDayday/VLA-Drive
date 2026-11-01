"""Read-only bounded9.5–10.5k/11.5–12k process and native metadata audit.

optimizer.pt uses mmap; only scheduler/group/scalar-step/tensor-shape metadata
are inspected. No adapter or moment array is traversed/hashed or copied.
"""
import copy,gc,hashlib,json,math,os,pickle,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch,yaml
from starVLA.rl.qwen_world_research_data.data import SceneStream
from starVLA.rl.qwen_world_safety_exploration.curriculum import OnlinePriorityStream
from starVLA.rl.qwen_world_safety_exploration.standardized_recipe import multiplier
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');H=E/'improvement_training_20261101';OUT=H/'analysis';OUT.mkdir(parents=True,exist_ok=True);REPORT=Path(__file__).parent
RANGES={'pre10k':(9501,10000),'post10k':(10001,10500),'late12k':(11501,12000)}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def digest_small(x):return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def numeric_summary(values):return dict(n=len(values),mean=float(np.mean(values)),min=min(values),max=max(values))if values else None
specs=[]
for study in('standard_training_20261101','revival_training_20261101'):
 R=E/study;registry=read(R/'RUNS.json')
 for run in registry['runs']:specs.append((R,registry,run))

def audit(spec):
 R,registry,run=spec;root=Path(run['root']);cfg=yaml.safe_load(Path(run['config']).read_text());ident=read(root/'identity.json');rows={k:[]for k in RANGES};sources=[];errors=[];boundary=[]
 for rank in range(8):
  p=root/f'metrics_rank{rank}.jsonl';hh={k:hashlib.sha256()for k in RANGES};cc=defaultdict(int)
  with p.open('rb')as f:
   for index,line in enumerate(f,1):
    if index>12000:break
    ks=[k for k,(a,b)in RANGES.items()if a<=index<=b]
    if not ks:continue
    v=json.loads(line)
    if v['updates']!=index:errors.append(dict(rank=rank,line=index,updates=v['updates']));continue
    for k in ks:rows[k].append((rank,v));hh[k].update(line);cc[k]+=1
    if index in(9999,10000,10001,10002,11999,12000):boundary.append(dict(rank=rank,updates=index,collection=v['collection'],inner=v['inner_completed'],invocation=v['invocation'],behavior_update=v['behavior']['update'],token=v['token'],lr_used=v['optimizer_group_lrs_used'],lr_next=v['optimizer_group_lrs'],ratio_deviation=v['ratio_max_abs_from_one']))
  sources.append(dict(path=str(p),windows={k:dict(rows=cc[k],selected_bytes_sha256=hh[k].hexdigest())for k in RANGES}))
 windows=[]
 for k,rs in rows.items():
  gs=[v for _,v in rs if v['inner_completed']==1];ex=[v['exploration']for v in gs];globalgroups=defaultdict(list)
  for rank,v in rs:globalgroups[v['updates']].append(v)
  maxlrerr=max((abs(v['optimizer_group_lrs_used']['action_lora']-cfg['learning_rate']*multiplier(v['updates']-1))for _,v in rs),default=None)
  memerr=max((abs(v['optimizer_group_lrs_used']['scene_predictive_memory']-cfg['memory_learning_rate']*multiplier(v['updates']-1))for _,v in rs),default=None)
  if maxlrerr is not None and maxlrerr>1e-15:errors.append(dict(window=k,LR_error=maxlrerr))
  if memerr is not None and memerr>1e-15:errors.append(dict(window=k,memory_LR_error=memerr))
  metrics={name:numeric_summary([v['metrics'][name]for _,v in rs if name in v['metrics']])for name in('ppo_blocked_fraction','positive_ppo_blocked_fraction','negative_ppo_blocked_fraction','clipfrac','pg_logprob_weight_abs_mean','kl_loss','policy_loss','quality_active_positive_high_weight','quality_active_negative_high_weight')}
  zerors=[v for _,v in rs if v['exploration']['zero_PG']]
  windows.append(dict(name=k,range=list(RANGES[k]),rank_update_rows=len(rs),expected_rows=4000,groups=len(gs),expected_groups=1000,complete=len(rs)==4000 and len(gs)==1000,group_counts={name:sum(bool(v.get(name,False))for v in ex)for name in('all_full','all_zero','tied','zero_PG','safety_differing','progress_only')},global_all8rank_PG0_updates=sum(len(vs)==8 and all(v['exploration']['zero_PG']for v in vs)for vs in globalgroups.values()),local_PG0_anchor_nonzero_rank_updates=sum(v['metrics']['kl_loss']>0 for v in zerors),max_action_LR_formula_error=maxlrerr,max_memory_LR_formula_error=memerr,total_gradient_norm=numeric_summary([v['gradient_norm']for _,v in rs]),actual_LR_used_action=numeric_summary([v['optimizer_group_lrs_used']['action_lora']for _,v in rs]),actual_LR_used_memory=numeric_summary([v['optimizer_group_lrs_used']['scene_predictive_memory']for _,v in rs]),global_canonical_reward_std=numeric_summary([v['global_reward_std']for v in ex]),scalar_loss_and_surrogate_metrics=metrics,scalar_losses_are_not_component_gradient_norms=True))
 checkpoints=[];firstpost={rank:v for rank,v in rows['post10k']if v['updates']==10001};actual_recipe=ident['recipe'];cp_identities=[]
 for u in(10000,12000):
  cp=root/f'checkpoints/update{u:06}_inner0'
  if not(cp/'COMPLETE').exists():checkpoints.append(dict(updates=u,available=False));continue
  seal=read(cp/'COMPLETE');files=read(cp/'files.json');tx=read(cp/'transaction.json');ci=read(cp/'identity.json');cp_identities.append(ci)
  if sha(cp/'files.json')!=seal['files_sha256']:raise ValueError('Checkpoint fileinventory seal mismatch')
  common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True);opt=common['optimizer'];sched=copy.deepcopy(common['scheduler']);steps=[float(v['step'])for v in opt['state'].values()];group=[{k:v for k,v in g.items()if k!='params'}|dict(parameter_tensors=len(g['params']))for g in opt['param_groups']];names=list(common['adapters']);numel=sum(v.numel()for v in common['adapters'].values());ddtypes=sorted({str(v.dtype)for v in common['adapters'].values()});momenttypes=sorted({str(t.dtype)for st in opt['state'].values()for n,t in st.items()if n in('exp_avg','exp_avg_sq')});expected=[cfg['learning_rate']*multiplier(u),cfg['memory_learning_rate']*multiplier(u)];expectedNames=set(ident['parameter_contract']['trainable_names'])
  if set(names)!=expectedNames or numel!=52707840:errors.append(dict(checkpoint=u,closure_mismatch=True))
  if any(s!=u for s in steps)or len(steps)!=len(names):errors.append(dict(checkpoint=u,Adam_step_mismatch=True))
  if sched['last_epoch']!=u or sched['_step_count']!=u+1 or any(abs(g['lr']-e)>1e-15 for g,e in zip(group,expected)):errors.append(dict(checkpoint=u,scheduler_mismatch=True))
  own=[]
  for rank in range(8):
   p=cp/f'rank{rank}.pt';r=torch.load(p,map_location='cpu',weights_only=False,mmap=True);st=r['state'];txr=tx['ranks'][rank]
   for name in('updates','collections','scene_cursor','next_inner','phase'):
    if st[name]!=txr[name]:errors.append(dict(checkpoint=u,rank=rank,state_transaction_mismatch=name))
   c=st.get('curriculum');nextToken=None;nextMatch=None
   if u==10000:
    base=SceneStream(registry['manifest'],8)
    if cfg['research']['data_policy']=='online_priority':
     stream=OnlinePriorityStream(base,cfg['research']['curriculum']);stream.bind(copy.deepcopy(st));nextToken,_=stream.at(st['scene_cursor'],rank)
    else:nextToken,_=base.at(st['scene_cursor'],rank)
    nextMatch=rank in firstpost and firstpost[rank]['token']==nextToken
    if not nextMatch:errors.append(dict(checkpoint=u,rank=rank,next_scene_mismatch=True))
   rng={name:digest_small(v)for name,v in r['rng'].items()}
   own.append(dict(rank=rank,state={k:v for k,v in st.items()if k not in('curriculum',) and not k.startswith('seconds_')},rng_small_state_sha256=rng,rollout_absent_READY=r['rollout']is None,expected_next_token_from_restored_cursor=nextToken,next_token_matches_actual10001=nextMatch,curriculum_summary=None if c is None else dict(committed_collections=c['committed_collections'],uniform_draws=c['uniform_draws'],priority_exposures=c['stats']['priority_exposures'],bank_size=len(c['bank']),private_RNG_sha256=digest_small(c['rng_state']))))
   del r
  checkpoints.append(dict(updates=u,available=True,path=str(cp),COMPLETE_file_sha256=sha(cp/'COMPLETE'),identity_sha256=ci['sha256'],same_original_run_identity=ci==ident,config_recipe_sha256=fp(ci['recipe']),execution_identity_sha256=tx.get('execution_identity',{}).get('sha256'),migration_receipt_sha256=tx.get('migration_receipt_sha256'),scheduler=sched,optimizer_groups=group,Adam_state_tensors=len(steps),Adam_step_min=min(steps),Adam_step_max=max(steps),moment_dtypes=momenttypes,trainable_tensors=len(names),trainable_numel=numel,adapter_dtypes=ddtypes,any_reference_tensor_saved_as_trainable=any(n.startswith('ref_')for n in names),ranks=own,optimizer_payload_hash_from_sealed_files_json=files['optimizer.pt'],optimizer_payload_not_rehashed_or_traversed=True,storage_mmap_metadata_only=True))
  del common,opt;gc.collect()
 invocations=[]
 for uuid in sorted({v['invocation']for _,v in rows['post10k']}):
  for rank in range(8):
   p=root/'invocations'/uuid/f'rank{rank}.json'
   if p.exists():
    v=read(p);invocations.append(dict(rank=rank,invocation=uuid,start_update=v['start_update'],start_collection=v['start_collection'],start_cursor=v['start_cursor'],resume_checkpoint=v['resume_checkpoint'],resume_files_sha256=v['resume_files_sha256'],end_update=v.get('end_update'),status=v['status'],source=ref(p)))
 def sourceRef(path):return dict(path=str(path),sha256=sha(path))
 # Every borrowed source is immutable; line reading never uses an execution host.
 return dict(arm=run['id'],study=str(R),errors=errors,window_sources=sources,windows=windows,boundary_rows=boundary,native_checkpoints=checkpoints,post10k_invocations=invocations,recipe_audit=dict(current_config_file_sha256=sha(run['config']),registered_config_sha256=run['config_sha256'],resolved_recipe_file_sha256=sha(root/'resolved_recipe.yaml'),recipe_matches_checkpoint=actual_recipe==cfg,max_steps=cfg['max_steps'],save_every=cfg['save_every_optimizer_updates'],evaluate_updates=cfg['evaluate_updates'],parent_model_sha256=ident.get('IL_parent_model_sha256',ident.get('parent_model_sha256')),conditions_identity=ident.get('projected_condition_identity'),stream_identity=ident.get('data_stream_identity'),source=run['source'],execution_source=run.get('execution_source'),migration_receipt=run.get('execution_migration_receipt'),fresh_or_fork_fields={k:cfg['research'].get(k)for k in('warmstart','native_fork','success_memory','preference_auxiliary','rescue_auxiliary','data_policy')},LR_horizon=cfg['research'].get('standardized_training'),teacher_reference='fixed original IL70, no newteacher at10k'),full_identity_10k_equals12k=len(cp_identities)==2 and cp_identities[0]==cp_identities[1])
def ref(p):return dict(path=str(p),sha256=sha(p))
with ThreadPoolExecutor(max_workers=4)as pool:arms=list(pool.map(audit,specs))
code=E/'standard_training_20261101/runtime/standard_v1'
sourcefiles=['tools/qwen_world_safety_exploration/train.py','starVLA/rl/simwam_recipe/checkpoint.py','starVLA/rl/simwam_recipe/lora_adapter.py','starVLA/rl/qwen_world_postcot/scope_actor.py','starVLA/rl/qwen_world_postcot/scope_conditions.py','starVLA/rl/qwen_world_research_data/data.py','starVLA/rl/simwam_recipe/vendor/samplers.py']
result=dict(schema='world_common10k_regression_configuration_process_audit_v1',status='COMPLETE_BOUNDED_NATIVE_METADATA_AND_ALL8RANK_LIGHT_JOURNAL_WINDOWS',created_unix=time.time(),arms=arms,total_errors=sum(len(r['errors'])for r in arms),conclusions_zh=['未发现共同10k配置/原生恢复断点：实际LR按50k clock连续、Adam steps不重置、两组/52.7M完整，scene_cursor20000→24000。','--stop-after-update只控制合作保存退出；max_steps50k与调度周期不随10k预算续期改变。故预算扩大本身不重warmup，也不突然升LR。','每次invocation确会重建actor/conditions，但从同IL70构造冻结reference后原位restore全部current trainables/Adam/RNG，reference不换成当前RL模型。Qwen/reference cache仅一scene，currentMemory每概率调用重建。','完整人口seed42 randperm，10k处cursor仅20000/103288，未跨epoch；Active保留自己的bank/RNG/uniform cursor，无10k清空。不能把共同下降直接归为新难场景块。','共同配置机制仍值得实验：50k慢cos使10k nextLR=.92714peak、12k=.89266peak、20k=.70541peak；4inner复用、每次只8different scenes均未改。这是持续高步长/更新分布假说，不是已证实根因。','标量kl_loss/policy_loss与总gradient_norm不能分离anchor和PG梯度；F/G这两个窗口无实际组件gradientnorm记录，不能把loss比例冒充梯度主导。'],repair_recommendations=[dict(kind='no_specific_code_fix_supported',text='本次无证据支持修改resume/reference/cursor代码；若后续数据指出具体不连续，应先保留证据再修复，不以虚构bug解释曲线。'),dict(kind='authorized_new_recipe_test',text='优先已独立实现P0早衰减20k；P1不同sceneB32也只在独立source下对照；F/G frozen配置保持。'),dict(kind='future_low_cost_observability',text='下一新source可记录pre-optimizer两组LR、Adam step range、resumeCP/SHA、reference identity、stream epoch、smallRNG digest；组件gradientprobe只在预注册少数点做，不能追补假数据。')],limits=['mmap optimizer只读元数据/step标量/shape/dtype，不遍历或hash633MB模型与moment数组；完整tensor-bit恢复依据实际nativevalidator/旧CPUrecovery资格，非本审计重新证明。','所有window为指定连续9501..10500与11501..12000全8rank范围，不外推全训练率。','两份COMPLETE间RNG变动正常，不等于逐位证明实际进程每步随机数调用；nexttoken从保存state CPU重建并与10001真journal逐rank对齐。','No newgradients/forwards/scoring/bootstrap or GPU; no liveowner/source/config edit.'],source_inspection=[ref(code/p)for p in sourcefiles],scheduler_next_samples={str(u):dict(factor=multiplier(u),action=5e-5*multiplier(u),memory=5e-6*multiplier(u))for u in(9999,10000,10001,10500,11500,12000,20000)},existing_curve_audit=ref(E/'analysis/LONG_RUN_PHASE_GAIN_RETENTION_AUDIT_20261101.json'),scope=dict(GPU_actions=0,new_scores=0,new_forwards=0,model_images_loaded=0,optimizer_tensor_payloads_traversed=0,live_mutations=0))
result['sha256']=fp(result)
for p in(OUT/'COMMON_10K_REGRESSION_CONFIG_AUDIT.json',REPORT/'COMMON_10K_REGRESSION_CONFIG_AUDIT.json'):p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(OUT/'COMMON_10K_REGRESSION_CONFIG_AUDIT.json'),errors=result['total_errors'],arms=[dict(arm=a['arm'],identity_same=a['full_identity_10k_equals12k'],windows=[dict(name=w['name'],rows=w['rank_update_rows'],PG0=w['group_counts']['zero_PG'],gradient_norm_mean=w['total_gradient_norm']['mean'],rewardStd_mean=w['global_canonical_reward_std']['mean'],PPO_blocked=w['scalar_loss_and_surrogate_metrics']['ppo_blocked_fraction']['mean'])for w in a['windows']])for a in arms]),indent=2))
