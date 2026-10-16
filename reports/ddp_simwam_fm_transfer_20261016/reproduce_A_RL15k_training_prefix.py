import json,pathlib,time
import torch
from scripts.analysis.simwam_recipe.summarize_training import read_journal,summarize_window
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.simwam_recipe.contracts import file_sha,POPULATION
from starVLA.rl.flow_grpo.transactions import atomic_json
root=pathlib.Path('/mnt/project/fm-transfer-20261016');run=root/'runs/A_DDP_TIME_R_SOURCE_seed42_v1';end=15000
output=root/'analysis/A_DDP_TIME_RL_update015000_training_prefix.json'
if output.exists():raise FileExistsError(output)
identity=json.loads((run/'identity.json').read_text());ranks=[read_journal(run/f'metrics_rank{rank}.jsonl') for rank in range(8)]
if min(map(len,ranks))<end:raise ValueError('Missing fixed milestone rows')
for rows in ranks:
 validate_rows(rows,len(rows))
 for row in rows[:end]:
  if row['behavior']['run']!=identity['sha256'] or row['collection']!=(row['updates']-1)//4+1 or row['inner_completed']!=(row['updates']-1)%4+1:raise ValueError('Changed four-inner identity')
checkpoint=run/'checkpoints/update015000_inner0';seal=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
if file_sha(checkpoint/'files.json')!=seal['files_sha256'] or seal['updates']!=end or seal['phase']!='READY':raise ValueError('Wrong sealed boundary')
states=[]
for rank in range(8):
 p=checkpoint/f'rank{rank}.pt'
 if file_sha(p)!=files[p.name]:raise ValueError('Changed rank checkpoint')
 state=torch.load(p,map_location='cpu',weights_only=False)['state']
 if state['updates']!=end or state['collections']!=3750 or state['raw_scene_exposures']!=30000 or state['new_candidates']!=240000 or state['next_inner']!=0:raise ValueError('Budget/state mismatch')
 states.append(state)
progress=json.loads((root/'analysis/A_DDP_TIME_RL_update015000_progress_snapshot.json').read_text())
if progress['updates']!=end or progress['invocation_start_update']!=0:raise ValueError('Wrong exact boundary cost snapshot')
cost_snapshot=json.loads((root/'analysis/A_DDP_TIME_RL_update015000_invocation_cost_snapshot.json').read_text())
if cost_snapshot['progress']!=progress or cost_snapshot['updates']!=end:raise ValueError('Different cost snapshot boundary')
phases=['rollout','old_probability','update','scoring_wait','scoring_cpu'];wall=progress['wall_seconds_this_invocation']
fresh=[row for rank in ranks for row in rank[:end] if row['inner_completed']==1];population=json.loads(pathlib.Path(POPULATION).read_text())
result=dict(schema='fm_transfer_fixed_RL_prefix_v2',status='COMPLETE_15K_TRAINING_PREFIX_NOT_END_OF_RUN',
 captured_unix=time.time(),analysis_script_sha256=file_sha(__file__),shared_analyzer_sha256=file_sha('scripts/analysis/simwam_recipe/summarize_training.py'),
 run_identity=identity['sha256'],checkpoint=str(checkpoint),checkpoint_files_sha256=seal['files_sha256'],
 observed_updates_by_rank=[len(x) for x in ranks],cumulative=summarize_window(ranks,0,end),
 windows=[summarize_window(ranks,s,s+500) for s in range(0,end,500)],
 unique_scene_count=len({x['token'] for x in fresh}),unique_log_count=len({population['token_logs'][x['token']] for x in fresh}),
 exact_boundary_progress_sha256=file_sha(root/'analysis/A_DDP_TIME_RL_update015000_progress_snapshot.json'),
 cost=dict(rank0_training_loop_wall_seconds=wall,
 rank0_preparation_seconds=cost_snapshot['invocation_records'][0]['preparation_seconds'],
 all_rank_preparation_seconds=[x['preparation_seconds'] for x in cost_snapshot['invocation_records']],
 all_rank_cumulative_save_seconds_through_milestone=[x['seconds_save'] for x in cost_snapshot['invocation_records']],
 exact_save_cost_snapshot_sha256=file_sha(root/'analysis/A_DDP_TIME_RL_update015000_invocation_cost_snapshot.json'),updates_per_rank0_wall_second=end/wall,
 cumulative_rank_wall_seconds_by_phase={p:sum(x['seconds_'+p] for x in states) for p in phases},
 rank0_wall_seconds_by_phase={p:states[0]['seconds_'+p] for p in phases},
 note='Overlapping rank wall times, not exclusive GPU-hours; scoring_cpu is scorer elapsed, not CPU core seconds. Rank0 loop wall starts after actor/DDP preparation, includes checkpoint publication, and is not complete process wall. Preparation and exact save-through-milestone wall are separate; no exit cost exists while training continues.'),
 limits=['Fixed difficult TRAIN population, not DEV/Navtest or the full TRAIN mean.',
 'G8 stochastic policy scores and finite candidate maxima are not single-ODE deployment.',
 'A fresh collection is counted once across four inner updates.',
 'Within-run window changes include scene mix; infer efficacy from matched ODE evaluation.'])
atomic_json(output,result);print(json.dumps(dict(output=str(output),cumulative=result['cumulative'],cost=result['cost'],unique_scenes=result['unique_scene_count'],unique_logs=result['unique_log_count'])))
