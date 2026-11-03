"""Read-only historical formal logs; no GPU, scoring, or controller changes."""
import hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
O=Path('/mnt/project/d46-priority-rl-20261103/analysis/D_THROUGHPUT_ORIGINAL_BASELINES_20261103.json')
E=Path('/mnt/project/qwen-world-abcd-rl-best-20261031');D=Path('/mnt/project/d44-native-rl-20261103')
inputs={}
def load(p):
 p=Path(p);data=p.read_bytes();inputs[str(p)]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data));return json.loads(data)
def readrows(p):
 p=Path(p);data=p.read_bytes();inputs[str(p)]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data));return [json.loads(x)for x in data.splitlines()]
def stats(values):
 return dict(n=len(values),mean=statistics.fmean(values),median=statistics.median(values),minimum=min(values),maximum=max(values))
def jobinfo(root):
 jobs=[]
 for job in sorted(Path(root).iterdir()):
  p=job/'RESULT.json'
  if not p.exists():continue
  r=load(p)
  if r.get('returncode')!=0:continue
  jobs.append(dict(job=str(job),host=r['host'],start_unix=r['start_unix'],end_unix=r['end_unix'],returncode=r['returncode'],lease_slot=r.get('lease_slot'),shared_headroom_MiB=r.get('shared_headroom_MiB'),source=r['cwd'],physical_gpu_uuids=r.get('gpu_uuids')))
 return jobs
out=[]
for name,root,kind,jroot in [('ABCD_D_JAT_ACTION_IL12K',E/'runs/ARCH_D_JAT_ACTION','seconds_update',E/'jobs/ARCH_D_JAT_ACTION'),('D44_FIRST_NATIVE_D0_IL44K',D/'runs/D0_B64G16_M2_v1','update_seconds',D/'jobs/D0')]:
 ident=load(root/'identity.json');cfg=ident['recipe'];rows=[readrows(root/f'metrics_rank{r}.jsonl')for r in range(8)];maps=[{x['updates']:x for x in rr}for rr in rows];win=[[m[u]for u in range(102,202)]for m in maps]
 assert all(len({x['invocation']for x in w})==1 for w in win)
 assert len({w[0]['invocation']for w in win})==1
 invs={}
 for p in sorted((root/'invocations').glob('*/rank0.json')):
  inv=load(p);iid=inv.get('invocation',inv.get('id'));invs[iid]=inv
 for r in range(8):
  assert [x['updates']for x in rows[r]]==list(range(1,len(rows[r])+1))
 if name.startswith('ABCD'):
  B,G,M=8,cfg['grpo']['sample']['group_size'],cfg['grpo']['train']['num_inner_epochs'];scope=ident['parameter_contract'];parent=ident['IL_parent'];scene_key='raw_scene_exposures';query_key='new_candidates';parent_update=12000
 else:
  B,G,M=cfg['global_scenes'],cfg['group'],cfg['inner_updates'];scope=ident['scope'];parent=ident['parent'];scene_key='scene_exposures';query_key='PG_candidates';parent_update=44000
 jobs=jobinfo(jroot);all_inv=[]
 for iid,inv in invs.items():
  u0,u1=inv['start_update'],inv['end_update'];assert all(m[u]['invocation']==iid for m in maps for u in range(u0+1,u1+1))
  start=inv.get('start_unix',inv.get('started'));job=next((j for j in jobs if j['start_unix']<=start<=j['end_unix']),None)
  count=u1-u0;wall=inv['worker_wall_seconds'];delta_s=maps[0][u1][scene_key]-(maps[0][u0][scene_key]if u0 else 0);delta_q=maps[0][u1][query_key]-(maps[0][u0][query_key]if u0 else 0)
  all_inv.append(dict(invocation=iid,start_update=u0,end_update=u1,optimizer_updates=count,worker_wall_seconds=wall,end_to_end_seconds_per_Adam=wall/count,
   wall_definition='one actual uninterrupted invocation; excludes time between invocations and model preparation; includes collection, scorer waits, behavior persistence, optimizer updates and native saves',
   recorded_preparation_seconds=inv.get('preparation_seconds'),global_new_scene_exposures=delta_s,global_new_PG_candidates=delta_q,
   scene_exposures_per_second=delta_s/wall,new_PG_candidates_per_second=delta_q/wall,job=job,
   rank0_generation_worker_seconds=inv.get('seconds_generation',inv.get('generation_worker_seconds')),
   rank0_scoring_wait_worker_seconds=inv.get('seconds_scoring',inv.get('scoring_wait_worker_seconds')),
   generation_accounting_note='legacy invocation seconds_generation versus native invocation generation_worker_seconds have different timer boundaries; do not compare them as isolated GPU kernels'))
 before=maps[0][101];end=maps[0][201];window=dict(update_first=102,update_last=201,Adam_updates=100,invocation=win[0][0]['invocation'],all8_ranks_same_invocation=True,
   pure_update_seconds_rank0=stats([x[kind]for x in win[0]]),pure_update_seconds_all_rank_rows=stats([x[kind]for w in win for x in w]),
   mean_of_max_rank_update_seconds=statistics.fmean(max(w[j][kind]for w in win)for j in range(100)),
   per_rank_mean_update_seconds=[statistics.fmean(x[kind]for x in w)for w in win],
   per_inner_rank0={str(i):stats([x[kind]for x in win[0]if x['inner_completed']==i])for i in range(1,M+1)},
   global_scene_counter_delta=end[scene_key]-before[scene_key],global_PG_query_counter_delta=end[query_key]-before[query_key],
   timing_scope='forward/current prefix, backward/DDP, grad check/clip, Adam and scheduler, then cuda.synchronize; excludes fresh collection/scorer and checkpoint saves')
 if 'at'in end:
  assert before['invocation']==end['invocation'];wall=end['at']-before['at'];window.update(end_to_end_seconds=wall,end_to_end_seconds_per_Adam=wall/100,
    at_anchor_update=101,at_first=before['at'],at_last=end['at'],end_to_end_definition='same-invocation rank0 at[201]-at[101], exactly 100 committed optimizer intervals; includes collections/score waits/native behavior and checkpoint I/O within interval, no pause',
    global_scene_exposures_per_second=window['global_scene_counter_delta']/wall,global_new_PG_candidates_per_second=window['global_PG_query_counter_delta']/wall)
 else:window.update(end_to_end_seconds=None,end_to_end_seconds_per_Adam=None,end_to_end_definition='No per-row at was recorded; do not synthesize this window wall by adding rank timers. Use separately documented full invocation wall only.')
 winjob=next(x['job']for x in all_inv if x['invocation']==window['invocation'])
 item=dict(id=name,status='ACTUAL_FORMAL_HISTORY_NOT_QUALIFICATION',root=str(root),identity_sha256=ident['sha256'],parent_IL_updates=parent_update,parent_model_sha256=ident.get('parent_model_sha256',ident.get('SFT_sha256')),
   configuration=dict(world=8,global_scenes_per_collection=B,local_scenes_per_rank=B//8,group=G,inner_Adam_updates_per_collection=M,
    new_PG_candidates_per_collection=B*G,amortized_new_scenes_per_Adam=B/M,amortized_new_PG_candidates_per_Adam=B*G/M,
    current_scene_candidate_microbatch=G,different_scenes_accumulated_sequentially=True,trainable_numel=scope['trainable_numel'],trainable_tensor_count=len(scope['trainable_names']),
    text_LoRA_rank=cfg.get('text_rank',cfg.get('grpo',{}).get('lora',{}).get('r')),action_shape=[8,3],sampler='K10 shift5 rigorous eta.1 W3[7,8,9], untruncated innovations',
    learning_rate_peak=cfg['learning_rate'],clip=cfg.get('clip',cfg.get('grpo',{}).get('train',{}).get('ppo_clip_range')),checkpoint_execution_policy=cfg.get('checkpoint_policy','parent native adapter source activation checkpointing default True; no all_off override in original runner'),
    credit='canonical centered / global64 centered std'if name.startswith('ABCD')else'SafetyEP credit / global1024 raw-credit population std',manifest=str(ident.get('data_stream_identity',ident.get('data',{})))),
   stable_100_Adam_window=window,window_actual_job=winjob,invocations=all_inv,last_retained_metrics_update=len(rows[0]),
   cohost_context=dict(recorded_slot=winjob['lease_slot'],actual_8_GPU_UUIDs=winjob['physical_gpu_uuids'],exclusive_compute_claim=False,continuous_cohost_timeline_available=False,
    note='Shared-headroom/slot records prove admission mode, not absence or exact number of cohosts for all updates; no host probe performed. Historical capacity snapshots below are timestamped context only.'))
 out.append(item)
# Exact historical snapshots only; neither is a contemporary GPU inspection.
cap=load(E/'resources/ABCD_LAUNCH_CAPACITY.json');inv=cap['hosts']['training-vlawm-zt4']['inventory']
capold=dict(path=str(E/'resources/ABCD_LAUNCH_CAPACITY.json'),at_unix=inv['at_unix'],GPU_name=inv['gpus'][0]['name'],free_MiB_range=[min(x['free_MiB']for x in inv['gpus']),max(x['free_MiB']for x in inv['gpus'])],visible_owner_count=[x['visible_owner_count']for x in inv['gpus']],note='Snapshot before original D initial launch; cannot certify whole training window exclusive.')
cap=load(D/'operations/CAPACITY_CURRENT.json');wm=next(x for x in cap['rows']if x['host']=='training-vlawm-zt4');owners=wm['result']['owners']
capnative=dict(path=str(D/'operations/CAPACITY_CURRENT.json'),at_unix=cap['at'],actual_visible_entries=sorted({x.get('entry')for x in owners}),actual_output_paths=sorted({x.get('roles',{}).get('--output','')for x in owners}),
    note='Snapshot before native D0 attempt008; records S0_FAST_PACK4 plus S6 qualification processes. It is not an uninterrupted cohost record for u102..201.')
old=out[0];native=out[1];oldwall=next(x['end_to_end_seconds_per_Adam']for x in old['invocations']if x['start_update']==0);newwall=native['stable_100_Adam_window']['end_to_end_seconds_per_Adam']
result=dict(schema='D_throughput_actual_formal_original_baselines_v1',status='COMPLETE_READ_ONLY_ACTUAL_HISTORY',created_UTC=datetime.now(timezone.utc).isoformat(),baselines=out,historical_capacity_context=[capold,capnative],
 comparison=dict(common='same D native Qwen backbone family, same r16 text LoRA 6422528 parameters/224 tensors, world8, FP32 states/params/probability and BF16 neural forward, K10shift5W3; same physical WM4 eight A800 UUIDs',
  different='IL12k vs IL44k parent; B/G/M, credit, clip, LR/schedule, checkpoint recomputation and reference/conditioning implementations, time and cohost load differ. This is observed throughput, not a matched execution speedup ablation.',
  earliest_pure_update_seconds=old['stable_100_Adam_window']['pure_update_seconds_rank0']['mean'],D44_pure_update_seconds=native['stable_100_Adam_window']['pure_update_seconds_rank0']['mean'],
  earliest_end_to_end_seconds_per_Adam_first_actual_invocation=oldwall,D44_end_to_end_seconds_per_Adam_stable_same_invocation_window=newwall,
  earliest_amortized_fresh_scene_rate=2/oldwall,D44_amortized_fresh_scene_rate=32/newwall,earliest_amortized_fresh_query_rate=16/oldwall,D44_amortized_fresh_query_rate=512/newwall,
  D44_vs_original_new_scene_throughput_ratio=(32/newwall)/(2/oldwall),D44_vs_original_new_PG_query_throughput_ratio=(512/newwall)/(16/oldwall),
  rate_limit='Rates amortize fresh collection over M optimizer updates; not policy forwards/sec, unique token coverage/sec, or exclusive GPU FLOPs. Original rate uses full actual 1000-update invocation because it has no per-row at; native rate uses its continuous 100-update window.'),
 guardrails=['No qualification timings used as formal throughput','No timestamps across pause/resume combined','No new GPU/scoring/host probe/worker signal/controller edit','Original ABCD run terminated; retained logs do not authorize checkpoint recovery','Current D46 live route speeds/evaluation status are outside this baseline-only artifact'],input_files=inputs,
 builder=dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
O.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n');print(json.dumps({'output':str(O),'sha256':hashlib.sha256(O.read_bytes()).hexdigest(),'comparison':result['comparison']},indent=2))
