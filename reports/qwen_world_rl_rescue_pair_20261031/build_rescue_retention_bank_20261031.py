"""Finite TRAIN-only positive teacher bank; never modifies any live bank/run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,hashlib,json,math,pickle,random,re,sys,datetime
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
W=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030');sys.path.insert(0,str(W))
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.contracts import seed_for
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';D=R/'data/rescue_retention/TRAIN_RESCUE_RETENTION_20261031'
METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
STRICT=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')
LOW_SCORE=.95;MIN_EP=.5

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def digest(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((str(x.dtype),x.shape)).encode()+x.tobytes()).hexdigest()
def seal(x):return dict(x,sha256=fp(x))
def write(p,x):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def physical(log):return re.sub(r'_\d+_\d+$','',log)
def seed_prior(token):
 seed=seed_for('native_qwen_sourceflow_ODE',2101,token);return torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)[0].numpy(),seed

def codec(normalized,requested):
 req=np.asarray(requested,np.float64)
 if normalized is None:
  norm=ActionCodec.norm_odo(torch.from_numpy(req.astype(np.float32))).numpy();origin='REQUESTED_ONLY_EXPLICIT_NUMERIC_INVERSE_NOT_ORIGINAL_LATENT'
 else:norm=np.asarray(normalized,np.float32);origin='ORIGINAL_NORMALIZED_ENDPOINT'
 dec=ActionCodec.denorm_odo(torch.from_numpy(norm)).numpy().astype(np.float64);error=np.abs(dec-req).max(0)
 envelope=(8*np.finfo(np.float32).eps*(np.abs(req)+np.array([66.74+1.57+1,42+19.68+1,3.53+1.67+1]))).max(0)
 if not np.isfinite(norm).all() or not np.isfinite(req).all() or np.any(error>envelope):raise ValueError('Positive teacher codec mismatch')
 return norm,req,dict(origin=origin,status='EXACT_NATIVE_FP32_DECODER_MATCH' if np.max(error)==0 else 'WITHIN_EXPLICIT_FP32_NUMERIC_ENVELOPE',max_abs_by_dimension=error.tolist(),bound_by_dimension=envelope.tolist(),units=['m','m','rad'],bound_is_not_metric_invariance_or_policy_replay=True)


def main():
 if (D/'manifest.json').exists():raise FileExistsError('Immutable independent bank already exists')
 D.mkdir(parents=True,exist_ok=True)
 current_manifest=read(R/'data/NAVTRAIN/manifest.json');pop=read(current_manifest['source_population_path']);full=read(R/'data/manifests/FULL_103288.json');fulltokens=full['tokens'];fullset=set(fulltokens);test=set(pop['navtest_tokens']);testlogs={physical(pop['token_logs'][t]) for t in test}
 with (R/'data/NAVTRAIN/records.pkl').open('rb') as f:observations=pickle.load(f)
 targetbank=read(R/'banks/PPO_SAFETY_EP_u010000_TRAIN1024.json');targeteval=read(R/'evaluation_specs/PPO_SAFETY_EP_u010000_TRAIN1024.json');targetout=Path(targeteval['output']);targetidentity=read(targetout/'identity.json')
 base={t:r['metrics'] for t,r in targetbank['records'].items()};probe=read(A/'FIXED_THETA_ODE_SDE_ZERO_PROBE.json');lowtokens15=read(R/'data/probes/ZERO_ODE8_SDE8_FIXED_THETA_20261031/PPO_SAFETY_EP.json')['tokens']
 candidates=collections.defaultdict(list);protocol=targetbank['inference']['protocol_sha256'];source_artifacts={}
 def candidate(token,record,norm,prior,source,source_scene):
  m=record['metrics'];log=record['log']
  if record['status']!='OK' or record['protocol_sha256']!=protocol or log!=observations[token]['log']:raise ValueError('Teacher scoring scene/protocol mismatch')
  if not all(np.isfinite(m[k]) and 0<=m[k]<=1 for k in METRICS):raise ValueError('Invalid teacher score')
  n,q,roundtrip=codec(norm,record['requested_trajectory'])
  source=dict(source,source_scene=str(source_scene),source_scene_sha256=sha(source_scene),protocol_sha256=protocol,cache_sha256=record['cache']['sha256'],prior_sha256=digest(prior))
  candidates[token].append(dict(token=token,log=log,metrics=m,raw_progress_meters=record['raw_progress_meters'],clean_normalized=n,requested_trajectory=q,source_prior_noise=prior,source=source,codec_roundtrip=roundtrip))
 for filename,level,includeODE in [('FIXED_THETA_ODE_SDE_ZERO_PROBE.json',.1,True),('FIXED_THETA_NOISE003_ZERO_PROBE.json',.03,False)]:
  collection=read(A/filename)
  if collection['status']!='COMPLETE' or collection['errors']:raise ValueError('Incomplete scored probe')
  source_artifacts[str(A/filename)]=sha(A/filename)
  for job in collection['jobs']:
   out=Path(job['output']);identity=read(out/'identity.json')
   for token in lowtokens15:
    path=out/'scenes'/f'{token}.json';row=read(path);arrpath=out/row['arrays']['path']
    with np.load(arrpath,allow_pickle=False) as z:
     for mode in (['ODE8','SDE8'] if includeODE else ['SDE8']):
      for k,rec in enumerate(row[mode]):
       src=dict(arm=job['arm'],checkpoint=identity['checkpoint'],updates=identity['updates'],policy_kind='RL',source_kind='ACTUAL_SCORED_POLICY_ENDPOINT',
        sampler='ODE' if mode=='ODE8' else 'rigorous_SDE',noise_level=0. if mode=='ODE8' else level,steps=10,shift=5.,stochastic_steps=[] if mode=='ODE8' else [7,8,9],candidate=k,
        probe_identity_sha256=identity['sha256'],arrays=str(arrpath),arrays_sha256=row['arrays']['sha256'],normalized_array_key='ODE_chain' if mode=='ODE8' else 'SDE_chain',normalized_endpoint_index=-1,
        prior_key='prior',prior_candidate=k,innovations_key='innovations' if mode=='SDE8' else None,innovations_candidate=k if mode=='SDE8' else None,
        innovation_sha256=digest(z['innovations'][:,k]) if mode=='SDE8' else None)
       candidate(token,rec,z['ODE_chain' if mode=='ODE8' else 'SDE_chain'][k,-1],z['prior'][k],src,path)
 # Existing independently scored reconstructed proposals: do not call them policy rollouts.
 registry=read(R/'data/counterfactuals/FINAL_SDE_STEP_20261031/registry.json');cfout=R/'counterfactual_evaluations/FINAL_SDE_STEP_20261031/groups'
 with np.load(R/'data/counterfactuals/FINAL_SDE_STEP_20261031/candidates.npz',allow_pickle=False) as z:
  for i,g in enumerate(registry['groups']):
   path=cfout/f"{g['arm']}__{g['token']}.json";row=read(path)
   with np.load(g['original_arrays'],allow_pickle=False) as original:
    for mode,key in [('mu9','mu9_records'),('det9','det9_records')]:
     for k,rec in enumerate(row[key]):
      src=dict(arm=g['arm'],checkpoint=g['checkpoint'],updates=g['updates'],policy_kind='RL',source_kind='SCORED_RECONSTRUCTED_COUNTERFACTUAL_NOT_POLICY_ROLLOUT',
       sampler=mode+'_conditional_original_SDE01_prefix',noise_level=.1,steps=10,shift=5.,surviving_innovation_steps=[7,8],candidate=k,
       registry=str(R/'data/counterfactuals/FINAL_SDE_STEP_20261031/registry.json'),registry_sha256=sha(R/'data/counterfactuals/FINAL_SDE_STEP_20261031/registry.json'),reconstruction_rounding=registry['rounding'],
       arrays=str(R/'data/counterfactuals/FINAL_SDE_STEP_20261031/candidates.npz'),arrays_sha256=registry['artifact']['sha256'],normalized_array_key=mode+'_normalized',normalized_group=i,
       prior_arrays=g['original_arrays'],prior_arrays_sha256=g['original_arrays_sha256'],prior_key='prior',prior_candidate=k,innovation_key='innovations',innovation_candidate=k,
       surviving_innovations_sha256=digest(original['innovations'][:2,k]))
      candidate(g['token'],rec,z[mode+'_normalized'][i,k],original['prior'][k],src,path)
 # Include actual canonical single-ODE endpoints of pinned policies, including IL70.
 canonical_sources=[('IL70',Path('/mnt/project/qwen-world-postcot-20261025/rl_optimization_20261030/evaluations/IL70_TRAIN_001024'))]
 canonical_sources.extend((j['arm'],Path(j['output'])) for j in probe['jobs'])
 for arm,out in canonical_sources:
  identity=read(out/'identity.json')
  for token in lowtokens15:
   if arm=='IL70':path=out/'scenes'/f'{token}.json';row=read(path)
   else:
    outer=read(out/'scenes'/f'{token}.json');cache=outer['canonical_single_ODE'];canonicalout=Path(read(out.parent/(arm+'.json'))['canonical_ODE_cache']['output']);path=canonicalout/'scenes'/f'{token}.json';row=read(path)
   prior,seed=seed_prior(token)
   src=dict(arm=arm,checkpoint=(identity.get('checkpoint') if arm!='IL70' else str(R/'IL_PARENT')),updates=identity.get('updates',70000),policy_kind='IL' if arm=='IL70' else 'RL',
     source_kind='ACTUAL_CANONICAL_SINGLE_ODE',sampler='ODE',noise_level=0.,steps=10,shift=5.,seed=2101,generator_seed=seed,prior_provenance='recreated original CPU Generator namespace native_qwen_sourceflow_ODE, not a new model sample',stochastic_steps=[])
   candidate(token,row['feedback'],row.get('normalized_action'),prior,src,path)
 # A fixed bank has one nonconflicting teacher per scene.
 low=[];unresolved=[];low_audit=[]
 for token in lowtokens15:
  score=base[token]['score'];xs=candidates[token];strict=[x for x in xs if all(x['metrics'][k]==1 for k in STRICT)]
  valid=[x for x in strict if x['metrics']['ego_progress']>=MIN_EP and x['metrics']['score']>score+1e-6]
  item=dict(token=token,canonical_SAFETY10k_score=score,category='ZERO' if score<=1e-12 else 'LOW_POSITIVE' if score<LOW_SCORE else 'HIGH_NOT_LOW_POOL',candidate_count=len(xs),strict4_success_candidates=len(strict),eligible_improving_teacher_candidates=len(valid))
  if score<LOW_SCORE:
   if not valid:unresolved.append(token);item['status']='UNRESOLVED_NO_VERIFIED_IMPROVING_TEACHER'
   else:
    bestEP=max(x['metrics']['ego_progress'] for x in valid);near=[x for x in valid if x['metrics']['ego_progress']>=bestEP-1e-8];bestprogress=max(x['raw_progress_meters'] for x in near);near=[x for x in near if x['raw_progress_meters']>=bestprogress-.001]
    near.sort(key=lambda x:(x['source']['source_kind'].startswith('SCORED_RECONSTRUCTED'),x['source']['sampler']!='ODE',fp(x['source'])))
    chosen=near[0];chosen['bank_kind']='low';chosen['baseline_score']=score;low.append(chosen);item.update(status='RESCUED_VERIFIED_TEACHER',teacher_EP=chosen['metrics']['ego_progress'],teacher_score=chosen['metrics']['score'],teacher_source=chosen['source'])
  else:item['status']='NOT_A_LOW_SCENE'
  low_audit.append(item)
 lowset={x['token'] for x in low}
 # High-retention eligibility is the actual Safety10k canonical single endpoint.
 history=[read(R/'banks'/f) for f in ['IL70_TRAIN_001024.json','PPO_SAFETY_EP_u001000_TRAIN1024.json','PPO_SAFETY_EP_u005000_TRAIN1024.json']]
 attrs={};high_candidates=[]
 for token,m in base.items():
  if m['score']<.95 or not all(m[k]==1 for k in STRICT) or token in lowset:continue
  o=observations[token];speed=float(np.hypot(*o['proprio'][:2]));speedbin='stationary_lt0.5' if speed<.5 else 'low_0.5_to3' if speed<3 else 'medium_3_to10' if speed<10 else 'high_ge10'
  oldscores=[h['records'][token]['metrics']['score'] for h in history];attr=dict(physical_log=physical(o['log']),current_speed_mps=speed,speed_bin=speedbin,navigation=o['navigation'],command=o['proprio'][4:8],historical_scores=oldscores,previously_below08=min(oldscores)<.8,previously_zero=min(oldscores)<=1e-12)
  attrs[token]=attr;high_candidates.append(token)
 rankkey=lambda t:(not attrs[t]['previously_below08'],hashlib.sha256(('42:'+t).encode()).hexdigest())
 bylog=collections.defaultdict(list);bystratum=collections.defaultdict(list)
 for t in high_candidates:
  bylog[attrs[t]['physical_log']].append(t);bystratum[(attrs[t]['physical_log'],attrs[t]['speed_bin'],attrs[t]['navigation'])].append(t)
 selected=[]
 def add(t):
  if t not in selected and len(selected)<512:selected.append(t)
 for log,ts in sorted(bylog.items()):add(sorted(ts,key=rankkey)[0])
 for k,ts in sorted(bystratum.items(),key=lambda kv:(len(kv[1]),hashlib.sha256(str(kv[0]).encode()).hexdigest())):add(sorted(ts,key=rankkey)[0])
 for t in sorted(high_candidates,key=rankkey):add(t)
 high=[]
 for token in selected:
  path=targetout/'scenes'/f'{token}.json';row=read(path)
  if row['feedback']['metrics']!=base[token]:raise ValueError('Safety10k bank/scene metrics mismatch')
  prior,seed=seed_prior(token);n,q,rt=codec(row.get('normalized_action'),row['requested_trajectory'])
  high.append(dict(token=token,log=row['log'],bank_kind='high',metrics=base[token],raw_progress_meters=row['feedback']['raw_progress_meters'],clean_normalized=n,requested_trajectory=q,source_prior_noise=prior,codec_roundtrip=rt,retention_strata=attrs[token],baseline_score=base[token]['score'],
    source=dict(arm='PPO_SAFETY_EP',checkpoint=targeteval['rl_checkpoint'],updates=10000,policy_kind='RL',source_kind='ACTUAL_CANONICAL_SINGLE_ODE',sampler='ODE',steps=10,shift=5.,noise_level=0.,seed=2101,generator_seed=seed,
     prior_provenance='recreated original CPU Generator namespace native_qwen_sourceflow_ODE',source_scene=str(path),source_scene_sha256=sha(path),protocol_sha256=protocol,cache_sha256=row['feedback']['cache']['sha256'],prior_sha256=digest(prior))))
 teachers=low+high;assert len({x['token'] for x in teachers})==len(teachers)
 if len(low)==0:raise ValueError('No legal low pool; do not synthesize zero teachers')
 for x in teachers:
  if x['token'] not in fullset or x['token'] in test or physical(x['log']) in testlogs:raise ValueError('Teacher TRAIN/test population overlap')
  if not all(x['metrics'][k]==1 for k in STRICT):raise ValueError('Teacher not strict4 verified')
 # Write compact teacher payload; no old logp, rewards normalization or on-policy A.
 arrays=dict(clean_normalized=np.stack([x['clean_normalized'] for x in teachers]).astype(np.float32),requested_trajectory=np.stack([x['requested_trajectory'] for x in teachers]).astype(np.float64),
  source_prior_noise=np.stack([x['source_prior_noise'] for x in teachers]).astype(np.float32),metrics=np.array([[x['metrics'][k] for k in METRICS] for x in teachers],np.float64),metric_keys=np.array(METRICS),tokens=np.array([x['token'] for x in teachers]),logs=np.array([x['log'] for x in teachers]),bank_kind=np.array([x['bank_kind'] for x in teachers]))
 np.savez_compressed(D/'teachers.npz',**arrays);records=[]
 for index,x in enumerate(teachers):
  reqhash=digest(x['requested_trajectory']);payload=dict(token=x['token'],bank_kind=x['bank_kind'],source=x['source'],requested_trajectory_sha256=reqhash)
  rec=dict(teacher_id=fp(payload),token=x['token'],log=x['log'],bank_kind=x['bank_kind'],normalized_index=index,metrics=x['metrics'],raw_progress_meters=x['raw_progress_meters'],protocol_sha256=protocol,cache_sha256=x['source']['cache_sha256'],
    requested_trajectory_sha256=reqhash,normalized_sha256=digest(x['clean_normalized']),source=x['source'],source_scene=x['source']['source_scene'],source_scene_sha256=x['source']['source_scene_sha256'],codec_roundtrip=x['codec_roundtrip'],
    baseline_SAFETY10k_score=x['baseline_score'],metric_verified=True,human_optimal=False,offline_only=True,on_policy_GRPO_eligible=False,advantages_status='NOT_APPLICABLE_OFFLINE_TEACHER_NO_PG_A')
  if 'retention_strata' in x:rec['retention_strata']=x['retention_strata']
  records.append(rec)
 with (D/'teachers.jsonl').open('w') as f:
  for row in records:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
 # Static10000-exposure census, exact80/10/10 and eight distinct per fresh batch.
 pools={'full':list(fulltokens),'low':[x['token'] for x in low],'high':[x['token'] for x in high]};orders={};positions=collections.Counter();epochs=collections.Counter()
 def draw(pool,used):
  while True:
   if pool not in orders or positions[pool]>=len(orders[pool]):
    order=list(pools[pool]);random.Random(42+100003*epochs[pool]+{'full':0,'low':1,'high':2}[pool]).shuffle(order);orders[pool]=order;positions[pool]=0;epochs[pool]+=1
   token=orders[pool][positions[pool]];positions[pool]+=1
   if token not in used:return token
 ordered=[];tags=[]
 for cycle in range(250):
  patterns=[(6,1,1)]*3+[(7,1,0),(7,0,1)];random.Random(42+cycle).shuffle(patterns)
  for batch,counts in enumerate(patterns):
   kinds=[k for k,n in zip(['full','low','high'],counts) for _ in range(n)];random.Random(420000+cycle*5+batch).shuffle(kinds);used=set()
   for kind in kinds:
    t=draw(kind,used);used.add(t);ordered.append(t);tags.append(kind)
 assert len(ordered)==10000 and collections.Counter(tags)=={'full':8000,'low':1000,'high':1000} and all(len(set(ordered[i:i+8]))==8 for i in range(0,10000,8))
 census=seal(dict(schema='world_TRAIN_rescue_retention_static_census_v1',seed=42,world=8,four_inner_updates=4,pilot_optimizer_updates=5000,total_exposures=10000,ordered_tokens=ordered,ordered_pools=tags,
  counts=dict(collections.Counter(tags)),full_population_count=103288,full_population_manifest=str(R/'data/manifests/FULL_103288.json'),full_population_file_sha256=sha(R/'data/manifests/FULL_103288.json'),
  teacher_npz_sha256=sha(D/'teachers.npz'),teacher_jsonl_sha256=sha(D/'teachers.jsonl'),every_batch8distinct=True,selection='fixed40-scene cycles:three6/1/1 batches,one7/1/0,one7/0/1; seed42 permutations; full pool includes all103288, only within-batch duplicates skipped',used_NAVTEST=False))
 write(D/'census.json',census)
 stats=dict(probe_scenes=15,canonical_SAFETY10k_zero=sum(base[t]['score']<=1e-12 for t in lowtokens15),canonical_positive_low=sum(0<base[t]['score']<LOW_SCORE for t in lowtokens15),canonical_high_not_low=sum(base[t]['score']>=LOW_SCORE for t in lowtokens15),
  low_verified_teachers=len(low),low_unique_scenes=len(low),unresolved=unresolved,high_eligible=len(high_candidates),high_selected=len(high),high_unique_physical_logs=len({attrs[t]['physical_log'] for t in selected}),high_eligible_physical_logs=len(bylog),
  high_selected_joint_strata=len({(attrs[t]['physical_log'],attrs[t]['speed_bin'],attrs[t]['navigation']) for t in selected}),high_eligible_joint_strata=len(bystratum),
  high_previously_below08_selected=sum(attrs[t]['previously_below08'] for t in selected),high_previously_zero_selected=sum(attrs[t]['previously_zero'] for t in selected),
  high_by_speed=dict(collections.Counter(attrs[t]['speed_bin'] for t in selected)),high_by_navigation=dict(collections.Counter(attrs[t]['navigation'] for t in selected)),
  low_teacher_source_modes=dict(collections.Counter(x['source']['sampler'] for x in low)),low_teacher_EP_min=min(x['metrics']['ego_progress'] for x in low),codec_max_abs_by_dimension=np.max([x['codec_roundtrip']['max_abs_by_dimension'] for x in teachers],0).tolist())
 manifest=seal(dict(schema='world_TRAIN_rescue_retention_positive_teacher_manifest_v1',status='READY_OFFLINE_VERIFIED_TEACHERS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),count=len(teachers),low_tokens=pools['low'],high_tokens=pools['high'],
  protocol_sha256=protocol,metric_keys=list(METRICS),artifacts={f:dict(path=str(D/f),sha256=sha(D/f)) for f in ['teachers.npz','teachers.jsonl','census.json']},
  full_population_count=103288,full_population_manifest=str(R/'data/manifests/FULL_103288.json'),full_population_manifest_sha256=sha(R/'data/manifests/FULL_103288.json'),data_manifest_sha256=sha(R/'data/NAVTRAIN/manifest.json'),
  gates=dict(strict1=list(STRICT),low_baseline_policy='PPO_SAFETY_EP10k canonical B1 ODE seed2101',low_baseline_score_below=LOW_SCORE,low_teacher_EP_min=MIN_EP,low_teacher_score_must_exceed_baseline=True,high_score_min=.95),
  teacher_selection=dict(low='all scored ODE/.1/.03/mu9/det9 plus pinned canonical IL70/threeRL ODE; highestEP, highestrawprogress with1mm tie, prefer actual policy before reconstructed if tied; one teacher per scene',
   high='Safety10k B1 only,one per physical log then joint speed/navigation strata then earlier<.8 recovery priority; up to512',
   teacher_id_fingerprint='sha256 JSON sort_keys=True,separators comma/colon,ensure_ascii=False,allow_nan=False over {token,bank_kind,source,requested_trajectory_sha256}',
   array_digest='sha256(str((str(dtype),shape)).encode()+contiguous.tobytes()); requested NPZ dtypefloat64'),
  statistics=stats,low_scene_audit=low_audit,warmstart=dict(reference='IL70 frozen',current='SafetyEP10k current trainable LoRA+scene/predictive memory',kind='explicit weights-only newphase, freshAdam/RNG42/update0; not native continuation'),
  initial_sampling=dict(full=.8,low=.1,high=.1,static_census=str(D/'census.json'),phase_budget_optimizer_updates=5000,every_batch8distinct=True),
  offline_only=True,on_policy_GRPO_eligible=False,metric_verified_not_human_optimal=True,NAVTEST_tokens_or_logs_included=False,old_DEV_is_TRAIN_diagnostic=True,
  source_artifacts=source_artifacts,actor_inputs='unchanged current images/ego/navigation only; future PDM cache metrics are training labels, never actor inputs',
  source_builder=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),new_model_forwards=0,new_scores=0,CUDA_initialized=torch.cuda.is_initialized())))
 write(D/'manifest.json',manifest);write(A/'RESCUE_RETENTION_BANK_20261031.json',dict(schema='rescue_retention_bank_build_summary_v1',status='COMPLETE',manifest=str(D/'manifest.json'),manifest_file_sha256=sha(D/'manifest.json'),statistics=stats,initial_census_counts=census['counts'],NAVTEST_used=False,live_v1_modified=False))
 print(json.dumps(dict(manifest=str(D/'manifest.json'),manifest_file_sha256=sha(D/'manifest.json'),statistics=stats),ensure_ascii=False))

if __name__=='__main__':main()
