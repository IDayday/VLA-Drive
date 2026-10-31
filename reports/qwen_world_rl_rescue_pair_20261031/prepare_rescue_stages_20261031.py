"""Append-only standard SceneStream manifests, preserving original bank/census."""
import argparse,collections,hashlib,json,os,random,sys
from pathlib import Path
W=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030');sys.path.insert(0,str(W))
from starVLA.rl.qwen_world_research_data.manifest import atomic_json,sealed,read_sealed,file_sha
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');D=R/'data/rescue_retention/TRAIN_RESCUE_RETENTION_20261031'
def generate():
 bank=json.loads((D/'manifest.json').read_text());original=json.loads((D/'census.json').read_text());full=read_sealed(R/'data/manifests/FULL_103288.json')
 registry=R/'data/rescue_retention/TRAIN_RESCUE_RETENTION_STAGES_20261031';manifest=R/'data/manifests/RESCUE_80_10_10_STATIC_103288.json'
 if manifest.exists() or registry.exists():raise FileExistsError('Never overwrite a sealed rescue stream/stage registry')
 registry.mkdir(parents=True);body={k:v for k,v in full.items() if k!='sha256'}
 body.update(name='RESCUE_80_10_10_STATIC_103288',mode='adaptive',stage_exposures=10000,stage_registry=str(registry),
   sampling_policy='static80full/10low/10high; adaptive is only the existing staged-reader storage mode, no feedback changes',
   teacher_manifest=str(D/'manifest.json'),teacher_manifest_file_sha256=file_sha(D/'manifest.json'),initial_budget_updates=5000,presealed_stages=10,presealed_capacity_is_not_budget_commitment=True)
 atomic_json(manifest,sealed(body));streamsha=file_sha(manifest)
 pools={'full':full['tokens'],'low':bank['low_tokens'],'high':bank['high_tokens']};orders={};pos=collections.Counter();epoch=collections.Counter();allstats=[]
 def draw(pool,used):
  while True:
   if pool not in orders or pos[pool]>=len(orders[pool]):
    order=list(pools[pool]);random.Random(42+100003*epoch[pool]+{'full':0,'low':1,'high':2}[pool]).shuffle(order);orders[pool]=order;pos[pool]=0;epoch[pool]+=1
   token=orders[pool][pos[pool]];pos[pool]+=1
   if token not in used:return token
 roles={t:'low' for t in pools['low']};roles.update({t:'high' for t in pools['high']})
 for stage in range(10):
  tokens=[];tags=[]
  for cycle in range(stage*250,(stage+1)*250):
   patterns=[(6,1,1)]*3+[(7,1,0),(7,0,1)];random.Random(42+cycle).shuffle(patterns)
   for batch,counts in enumerate(patterns):
    kinds=[k for k,n in zip(['full','low','high'],counts) for _ in range(n)];random.Random(420000+cycle*5+batch).shuffle(kinds);used=set()
    for kind in kinds:
     t=draw(kind,used);used.add(t);tokens.append(t);tags.append(kind)
  if stage==0:assert tokens==original['ordered_tokens'] and tags==original['ordered_pools']
  assert len(tokens)==10000 and all(len(set(tokens[i:i+8]))==8 for i in range(0,10000,8))
  stats=dict(stage_index=stage,selected_pool_counts=dict(collections.Counter(tags)),aux_role_token_counts=dict(collections.Counter(roles.get(t,'none') for t in tokens)),unique_scenes=len(set(tokens)),
    unique_by_selected_pool={k:len({t for t,tag in zip(tokens,tags) if tag==k}) for k in pools})
  value=sealed(dict(schema='world_rl_research_stage_census_v1',stream_manifest_sha256=streamsha,stage_index=stage,start_cursor=stage*10000,end_cursor=(stage+1)*10000,
    ordered_tokens=tokens,ordered_pools=tags,count=10000,seed=42,world_size=8,mode='rescue_retention_static80_10_10',teacher_manifest_file_sha256=file_sha(D/'manifest.json'),
    statistics=stats,source_original_stage0=str(D/'census.json'),original_stage0_file_sha256=file_sha(D/'census.json'),every_batch8distinct=True,full_eligible_count=103288,NAVTEST_used_for_training=False))
  path=registry/f'stage{stage:06}.json';atomic_json(path,value);allstats.append(dict(stats,path=str(path),file_sha256=file_sha(path),stage_sha256=value['sha256']))
 receipt=dict(schema='rescue_static_stages_generation_receipt_v1',status='COMPLETE_PRESEALED_NOT_TRAINING_COMMITMENT',generator=str(Path(__file__).resolve()),generator_sha256=file_sha(__file__),
   manifest=str(manifest),manifest_file_sha256=streamsha,stage_registry=str(registry),stages=allstats,original_bank_file_sha256=file_sha(D/'manifest.json'),original_stage0_file_sha256=file_sha(D/'census.json'),
   original_stage0_order_and_tags_preserved=True,initial_budget_updates=5000,eligible_population_not_full_traversal=True,presealed_capacity_optimizer_updates=50000,NAVTEST_used=False)
 atomic_json(registry/'GENERATION.json',receipt);atomic_json(R/'analysis/RESCUE_STAGE_GENERATION_20261031.json',receipt)
 print(json.dumps(dict(manifest=str(manifest),manifest_sha256=streamsha,stage_registry=str(registry),stages=10)))
if __name__=='__main__':generate()
