"""Finite read-only journal/source/resource analysis. No torch, GPU model, or live mutation."""
import collections,datetime,hashlib,json,math,pathlib,statistics
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103')
W=pathlib.Path('/mnt/project/VLA-Drive-d46-priority-rl-20261103')
SPECS=[('D0','D46_D0_WORLD16_v2',16,'world16_science_v3'),('D3_FPO','D46_D3_FPO_WORLD16_v1',16,'world16_fpo_science_v1'),('D2','D46_D2',8,'formal_v1')]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def seal(x):x['sha256']=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();return x
def stats(v):
 v=[float(x)for x in v if x is not None and math.isfinite(float(x))]
 if not v:return None
 s=sorted(v)
 def pct(q):
  a=(len(s)-1)*q;i=int(a);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(a-i)
 return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v),p10=pct(.1),p90=pct(.9))
def tail(p,limit=2097152):
 with p.open('rb')as f:
  f.seek(0,2);size=f.tell();off=max(0,size-limit);f.seek(off);b=f.read()
 if off:b=b.split(b'\n',1)[1]
 lines=b.splitlines();rows=[]
 for i,l in enumerate(lines):
  try:rows.append(json.loads(l))
  except json.JSONDecodeError:
   if i!=len(lines)-1:raise
 return rows,dict(path=str(p),size_bytes=size,read_offset=off,bytes_read=len(b),read_segment_sha256=hashlib.sha256(b).hexdigest(),rows=len(rows))
all_windows={}
for arm,name,world,source in SPECS:
 root=R/'runs'/name;identity=json.loads((root/'identity.json').read_text());cfg=identity['recipe'];assert cfg['inner_updates']==2 and cfg['group']==16 and cfg['global_scenes']==64
 indexed=[];inputs=[]
 for rank in range(world):
  rows,receipt=tail(root/f'metrics_rank{rank}.jsonl');by={}
  for row in rows:
   u=row['updates']
   if u in by:raise ValueError(f'duplicate update {arm} {rank} {u}')
   by[u]=row
  indexed.append(by);inputs.append(receipt)
 common=set.intersection(*(set(b)for b in indexed));ends=sorted(u for u in common if u%2==0 and all(indexed[r][u]['inner_completed']==2 for r in range(world)))
 end=ends[-1];invocations=[indexed[r][end]['invocation']for r in range(world)]
 candidates=[u for u in ends if u<=end and u-2 in common and u-1 in common and all(indexed[r][k]['invocation']==invocations[r]for r in range(world)for k in(u-2,u-1,u))]
 ends=candidates[-40:];assert len(ends)>=30,(arm,len(ends));assert ends==list(range(ends[0],ends[-1]+1,2))
 cp=[]
 for p in(root/'checkpoints').glob('*/COMPLETE'):
  d=json.loads(p.read_text());u=d.get('updates')
  if u is not None:cp.append(dict(updates=u,path=str(p),file_sha256=sha(p)))
 checkpoint_updates={x['updates']for x in cp};timings=[]
 for u in ends:
  per=[]
  for rank,b in enumerate(indexed):
   a,c,d=b[u-2],b[u-1],b[u]
   assert c['collection']==d['collection'] and c['inner_completed']==1 and d['collection']==a['collection']+1
   wall=d['at']-a['at'];gen=d['collection_seconds_generation'];scoring=d['collection_seconds_scoring'];upd=c['update_seconds']+d['update_seconds']
   per.append(dict(rank=rank,wall_seconds=wall,generation_seconds=gen,base_scoring_seconds=scoring,optimizer_updates_seconds=upd,
                   unexplained_residual_seconds=wall-gen-scoring-upd,model_work_delta={k:d['model_work'][k]-a['model_work'][k]for k in d['model_work']}))
  timings.append(dict(end_update=u,collection=indexed[0][u]['collection'],checkpoint_before_interval=u-2 in checkpoint_updates,
    rank0=per[0],rank_wall_spread_seconds=max(x['wall_seconds']for x in per)-min(x['wall_seconds']for x in per),
    max_rank_accounted_seconds=max(x['generation_seconds']+x['base_scoring_seconds']+x['optimizer_updates_seconds']for x in per),
    per_rank=per))
 steady=[x for x in timings if not x['checkpoint_before_interval']]
 fields=('wall_seconds','generation_seconds','base_scoring_seconds','optimizer_updates_seconds','unexplained_residual_seconds')
 start=ends[0]-2
 counters={k:stats((b[ends[-1]]['model_work'][k]-b[start]['model_work'][k])/len(ends)for b in indexed)for k in indexed[0][end]['model_work']}
 total_wall=sum(t['rank0']['wall_seconds']for t in steady)
 parts={k:sum(t['rank0'][k]for t in steady)/total_wall for k in fields if k!='wall_seconds'}
 diag={k:stats(indexed[0][u]['exploration'].get(k)for u in ends)for k in ['feasible_candidate_fraction','effective_PG_group_fraction','all_one_groups','all_zero_groups']}
 losses={k:stats(indexed[r][u]['losses'].get(k)for r in range(world)for u in ends)for k in ['clip_fraction','fpo_clip_fraction','ratio_mean','fpo_ratio_mean','reference_KL_joint','pg','auxiliary_loss']}
 loss_keys=[k for k in indexed[0][end]['losses']]
 # Do not read or hash tensor behavior/checkpoint payloads.
 all_windows[arm]=dict(status='COMPLETE',run_root=str(root),world=world,source=str(R/'runtime'/source),identity_sha256=identity['sha256'],identity_file_sha256=sha(root/'identity.json'),
   scientific_recipe={k:cfg.get(k)for k in ['arm','global_scenes','group','inner_updates','mc_samples','probability_layout','save_every']},
   window=dict(start_completed_update=start,end_completed_update=ends[-1],complete_collections=len(ends),optimizer_updates=2*len(ends),
     first_collection=timings[0]['collection'],last_collection=timings[-1]['collection'],end_scene_exposures=indexed[0][end]['scene_exposures'],
     start_wall_unix=indexed[0][start]['at'],end_wall_unix=indexed[0][end]['at'],common_prefix_end=end,invocation_by_rank=invocations),
   all_intervals_rank0={k:stats(t['rank0'][k]for t in timings)for k in fields},steady_intervals_rank0={k:stats(t['rank0'][k]for t in steady)for k in fields},
   steady_complete_collections=len(steady),global_scenes_per_second=64*len(steady)/total_wall,steady_rank0_time_fractions=parts,
   base_scoring_zero_cost_impossible_upper_speedup=1/(1-parts['base_scoring_seconds']),
   optimizer_section_half_cost_conditional_speedup=1/(1-.5*parts['optimizer_updates_seconds']),
   current_generation_section_half_cost_conditional_speedup=1/(1-.5*parts['generation_seconds']),
   all_rank_per_collection_reported_timing={k:stats(t['per_rank'][r][k]for t in timings for r in range(world))for k in fields},
   rank_wall_spread_seconds=stats(t['rank_wall_spread_seconds']for t in timings),
   per_rank_per_collection_model_work_delta=counters,
   global_query_deltas={k:indexed[0][ends[-1]][k]-indexed[0][start][k]for k in ['raw_scene_exposures','scene_exposures','PG_candidates','extra_B1_queries','adaptive_G32_queries']},
   exploration_per_collection_not_rank_multiplied=diag,loss_diagnostics_all_rank_second_inner=losses,loss_keys=loss_keys,
   checkpoint_inventory_metadata=cp,journal_read_receipts=inputs,collection_timing_ledger=timings,
   caveat='Rank0 wall end-to-end includes control/I/O/DDP waiting. Per-rank clocks are not subtracted across hosts. Generation is base G16+W3 current/reference only; excludes FPO old MC energy, D2 nominal/rescue, finalization and behavior serialization. Residual is not a causal CPU-only timer. Counters omit direct denoise_x0 MC/aux calls; activation-checkpoint backward replay is not counted.')
rawp=R/'analysis/THROUGHPUT_RESOURCE_SAMPLING_20261103.json';raw=json.loads(rawp.read_text());nodes={}
for key,v in raw['native_node_samples'].items():
 samples=v['samples'];first=samples[0];last=samples[-1];elapsed=last['at_unix']-first['at_unix'];hz=v['clock_ticks_per_second'];cpu=[]
 a={r['pid']:r for r in first['known_rank_processes']};b={r['pid']:r for r in last['known_rank_processes']}
 for pid,x in a.items():
  y=b.get(pid)
  if y and x.get('start_ticks')==y.get('start_ticks')and all(k in y for k in ('utime_ticks','stime_ticks')):
   cpu.append(dict(pid=pid,start_ticks=x['start_ticks'],cpu_cores=((y['utime_ticks']+y['stime_ticks'])-(x['utime_ticks']+x['stime_ticks']))/hz/elapsed))
 gpus=[g for s in samples for g in s['gpus']]
 nodes[key]=dict(host=v['host'],sample_seconds=elapsed,n_samples=len(samples),exact_rank_process_count=len(v['exact_known_ranks']),
  utilization_pct=stats(g['utilization_pct']for g in gpus),power_W=stats(g['power_W']for g in gpus),
  free_MiB=stats(g['free_MiB']for g in gpus),used_MiB=stats(g['used_MiB']for g in gpus),
  known_rank_cpu_cores=stats(x['cpu_cores']for x in cpu),cpu_by_exact_rank=cpu,
  caveat='GPU device-wide instantaneous NVML samples, not per-process compute time. CPU excludes PDM scoring subprocesses. TSH3/4 may contain known protected IL evaluation; no foreign command/environment emitted.')
paths=['starVLA/model/modules/qwen_world_compare/qwen_flow_adapter.py','starVLA/rl/qwen_world_compare/policy_adapter.py','starVLA/rl/d44_native_rl/actor.py','starVLA/rl/d44_native_rl/batch_runtime.py','starVLA/rl/d44_native_rl/conditions.py','starVLA/rl/qwen_world_compare/runtime.py','starVLA/rl/simwam_recipe/reward_readouts.py','tools/d44_native_rl/train.py']
sources={s:[dict(path=p,file_sha256=sha(R/'runtime'/s/p))for p in paths]for s in {x[3]for x in SPECS}}
result=dict(schema='d46_actual_throughput_bottlenecks_v1',status='COMPLETE',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 analysis_scope='All ranks, last40 complete same-invocation M2 collections each of D0 world16/FPO world16/D2 world8; finite60s live resource sampling only. No model forward, score, profiler, signals, locks or live mutations.',
 windows=all_windows,one_minute_resource_summary=nodes,resource_raw=dict(path=str(rawp),file_sha256=sha(rawp)),
 benchmark_candidate_GPUs=raw['qualification_candidate_GPU_probes'],source_hotspots=[
 dict(priority='P0',name='Repeated CUDA-to-host prefix selection decisions',path=paths[0],lines=[28,75],
 evidence='28 Qwen layers: 56 K/V +valid,next_positions,reasoning =59 _select calls. Repeated indices cause59 bool(torch.all(...)) +59 int(indices[0]) +G16 names int =134 host scalar conversions per ordinary index_select; not a timed measurement. For G1 names one =>119. _prepare constructs candidate_to_scene on CUDA.',
 optimization='Carry a validated CPU scene-index/one-scene-broadcast descriptor, reuse its integer/boolean once across all K/V; do not repeatedly inspect CUDA metadata. Preserve expand view and candidate ordering, and strict invalid-index validation.',
 risk='Must preserve autograd, repeated-scene stride0 packing eligibility, exact prefix names and bounds; no proposal/loss change.'),
 dict(priority='P1',name='Redundant current W3 collection recomputation and frozen reference prefix',path=paths[2],lines=[51,77],
 evidence='Sequential policy: per scene10 sampled denoise +3 current W3 verification +3 frozen-reference W3 +2x3 current update calls. Four current and one reference prefix builds per scene. Existing sampler already saves actual W3 means; current verify3 and its prefix consume compute solely to recheck fresh sampler identity.',
 optimization='Share the no-grad current prefix between collection sampling and its W3 verification while the policy is unchanged. The frozen reference prefix is already built only once per fresh scene, so there is no demonstrated repeated-reference-prefix saving in this window. Consider moving sampler-vs-recompute check to bounded qualification while retaining actual old means and fresh ratio validation; this removes a check, so requires explicit execution-equivalence review rather than silent hotfix.',
 static_forward_call_bounds='Removing all3 current verification calls is3/22=13.6% of D0 denoise forward invocation count, not13.6% wall: prefixes, batch sizes, backward recompute and sync differ.'),
 dict(priority='P1',name='FPO MC4 old-energy and MC4 backward are real extra transformer work',path=paths[3],lines=[90,102],
 evidence='Per scene4 no-grad old-energy denoise after timed generation/scoring; each of2 optimizer updates adds4 current MC denoise +backprop beyond3 W3 anchor denoise. Extra12 forward invocations per scene (34 vs22 including reference), absent from velocity-call counters.',
 optimization='Evaluate exact MC batching using shared prefix/scene packing, and prefix reuse within scope. Reducing MC or dropping W3 anchor changes the method and must be a separate scientific experiment; cannot claim free speedup.'),
 dict(priority='P2',name='Serial scene scoring and D2 uninstrumented rescue queries',path=paths[-2],lines=[27,36],
 evidence='RewardService.score is submit(...).result(); trainer processes scene generation then synchronous score sequentially. D2after.5E adds B1 and conditional G16 expansion after base generation/scoring timers.',
 optimization='Use existing async submit to overlap bounded candidate CPU scoring with next scene GPU generation while preserving query IDs/order, RNG and exact frozen policy version. Keep pending query ledger/native save consistent.',
 caveat='Base score timer is small forD0/FPO; eliminating it entirely gives only the reported impossible Amdahl ceiling. D2 true total scoring includes uninstrumented extra queries, so base timer cannot bound all scoring cost.'),
 dict(priority='P2',name='Frozen visual preparation is already bounded-LRU cached',path=paths[4],lines=[8,22],
 evidence='Per-scene frozen prefill counter and seconds include observation decode/processor/encode_current; current/referential text C/P prefix is recomputed intentionally for trainable current text. No evidence image preprocessing dominates this window.',
 optimization='Retain correct frozen vision cache; do not cache trainable current C/P across Adam updates. Add separate condition/old-energy/extra-score timers before deciding larger pipeline rewrite.')],
 observed_cpu_sync_interpretation='Near-one busy CPU core per exact rank together with modest device-wide GPU utilization is consistent with Python launches/synchronization and small kernels, not proof of percentage attributable to DPrefix. No CUDA profiler was attached; GPU hardware peak/headroom is not a measured attainable training speedup.',
 conditional_speedup_warning='Amdahl figures remove/halve only explicitly timed rank0 sections and assume all else fixed. They are scenario bounds, not measured/guaranteed gains. They exclude startup/full evaluation and are not algorithm score improvements.',
 source_files=sources,source_script=dict(path=str(pathlib.Path(__file__)),file_sha256=sha(__file__)),
 conclusions_zh=[])
for a,w in all_windows.items():
 t=w['steady_intervals_rank0'];result['conclusions_zh'].append(f"{a}: 最近{w['window']['complete_collections']}个完整collection，稳态整轮中位{t['wall_seconds']['median']:.3f}s；基准G16生成{t['generation_seconds']['median']:.3f}s、已计基础评分{t['base_scoring_seconds']['median']:.3f}s、两次Adam合计{t['optimizer_updates_seconds']['median']:.3f}s，残余{t['unexplained_residual_seconds']['median']:.3f}s。分项不可当独立因果归因。")
result['conclusions_zh']+=['优先做不改数学的CUDA索引主机同步消除与prefix/候选打包小基准；FPO额外MC4前反向不能靠把计时字段改名消除。','D0/FPO基础评分等待占比有限；D2额外B1/自适应评分没有纳入旧评分字段，须先拆计时再决定异步化收益。','WM2GPU0/1和RL2GPU1/7采样时均无compute owner、util0、约79GiB（81kMiB）空闲，只有只读候选证据，实际任务仍须UUID/lease/授权/headroom重新准入。']
out=R/'analysis/THROUGHPUT_BOTTLENECKS_20261103.json';out.write_text(json.dumps(seal(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(out),'file_sha256':sha(out),'windows':{a:{'updates':w['window'],'wall':w['steady_intervals_rank0'],'fractions':w['steady_rank0_time_fractions'],'calls':w['per_rank_per_collection_model_work_delta']}for a,w in all_windows.items()}},ensure_ascii=False))
