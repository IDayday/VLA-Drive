"""Aggregate the actual eight-rank journals without counting inner reuse as samples.

Read-only: this script never refreshes a chain, changes training, or interprets
stochastic rollout scores as single-ODE deployment performance.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def read_journal(path):
    content=Path(path).read_text()
    # A concurrently written final, unterminated line is not a committed row.
    lines=content.splitlines()
    if content and not content.endswith('\n'):lines=lines[:-1]
    rows=[json.loads(line) for line in lines]
    if [r['updates'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Journal has gaps or duplicate optimizer updates: '+str(path))
    return rows


def summarize_window(ranks, start, end):
    updates=[row for rows in ranks for row in rows[start:end]]
    # All ranks, only the first inner row of fully completed fresh collections.
    fresh=[row for row in updates if row['inner_completed']==1]
    ex=[r['exploration'] for r in fresh]
    mean=lambda values:float(np.mean(values)) if values else None
    count=lambda key:sum(int(e[key]) for e in ex)
    groups=len(fresh)
    collections=(end-start)//4
    by_collection={}
    for row in fresh:by_collection.setdefault(row['collection'],[]).append(row)
    if len(by_collection)!=collections or any(len(v)!=8 for v in by_collection.values()):
        raise ValueError('Incomplete global fresh rollout in window')
    all_zero=sum(all(row['exploration']['zero_PG'] for row in rows) for rows in by_collection.values())
    inner={}
    for i in range(1,5):
        rows=[r for r in updates if r['inner_completed']==i]
        inner[str(i)]=dict(ratio_max_abs_from_one=max(r['ratio_max_abs_from_one'] for r in rows),
            mean_PPO_clip_fraction=mean([r['metrics']['clipfrac'] for r in rows]),
            policy_loss_rank_mean=mean([r['metrics']['policy_loss'] for r in rows]),
            beta50_anchor_loss_rank_mean=mean([r['metrics']['kl_loss'] for r in rows]),
            mean_gradient_norm=mean([r['gradient_norm'] for r in rows]))
    safe_best=[e['finite_best_safe'] for e in ex if e['finite_best_safe'] is not None]
    return dict(first_optimizer_update=start+1,last_optimizer_update=end,
        actual_global_optimizer_updates=end-start,fresh_global_rollouts=collections,
        scene_exposures=groups,new_candidates=groups*8,
        global_RL_updates=(collections-all_zero)*4,global_anchor_only_updates=all_zero*4,
        local_PG_groups=groups-count('zero_PG'),local_PG_group_fraction=1-count('zero_PG')/groups,
        full_score_groups=count('all_full'),zero_score_groups=count('all_zero'),
        other_tied_groups=count('tied')-count('all_full')-count('all_zero'),
        canonical_tied_groups=count('tied'),zero_final_PG_groups=count('zero_PG'),
        progress_only_groups=count('progress_only'),safety_differing_groups=count('safety_differing'),
        stochastic_policy_mean={k:mean([e[k] for e in ex]) for k in
            ['canonical_random_mean','train_reward_mean','NC','DAC','TTC','EP','safe_fraction']},
        finite_best_safe=dict(mean_among_groups_with_safe_candidate=mean(safe_best),defined_groups=len(safe_best),
            interpretation='Finite G8 candidate opportunity, never deployment performance'),
        positive_candidate_counts={k:count(k) for k in ['positive_safe','positive_still_unsafe','positive_unsafe_with_safe_alternative']},
        global_raw_reward_std=dict(mean=mean([e['global_reward_std'] for e in ex]),
            min=min(e['global_reward_std'] for e in ex),max=max(e['global_reward_std'] for e in ex)),
        first_inner=inner['1'],inner_updates=inner,
        mean_update_seconds_per_rank=mean([r['seconds_update'] for r in updates]),
        source_relative_clamp_hit_rate=mean([r['source_clamp_hit_rate'] for r in updates]))


def checkpoint_cost(run):
    progress=json.loads((run/'progress.json').read_text())
    ck=Path(progress['checkpoint']);seal=json.loads((ck/'files.json').read_text())
    complete=json.loads((ck/'COMPLETE').read_text())
    if file_sha(ck/'files.json')!=complete['files_sha256']:raise ValueError('Checkpoint seal mismatch')
    states=[]
    for rank in range(8):
        path=ck/f'rank{rank}.pt'
        if file_sha(path)!=seal[path.name]:raise ValueError('Changed rank recovery state')
        states.append(torch.load(path,map_location='cpu',weights_only=False)['state'])
    if len({s['updates'] for s in states})!=1:raise ValueError('Unmatched rank checkpoint')
    phases=['rollout','old_probability','update','scoring_wait','scoring_cpu']
    sums={phase:sum(s['seconds_'+phase] for s in states) for phase in phases}
    wall=progress['wall_seconds_this_invocation'];done=states[0]['updates']
    return dict(checkpoint=str(ck),checkpoint_updates=done,
        cumulative_rank_wall_seconds_by_phase=sums,
        max_rank_wall_seconds_by_phase={phase:max(s['seconds_'+phase] for s in states) for phase in phases},
        all_rank_phase_seconds={phase:[s['seconds_'+phase] for s in states] for phase in phases},
        invocation_wall_seconds=wall,observed_optimizer_updates_per_second=done/wall,
        linear_remaining_training_hours=(20000-done)*wall/done/3600,
        allocated_eight_GPU_wall_hours=8*wall/3600,
        note='Phase timers are overlapping rank wall times, not exclusive CPU/GPU compute. scoring_cpu is scorer elapsed, not CPU-core time. GPU-wall includes sharing; ETA assumes current throughput and excludes evaluation. Invocation ETA valid for these uninterrupted formal runs only.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--window-updates',type=int,default=500);a=p.parse_args()
    if a.window_updates<=0 or a.window_updates%4:raise ValueError('Window must contain whole four-inner collections')
    run=Path(a.run);identity=json.loads((run/'identity.json').read_text())
    if identity['world']!=8:raise ValueError('Formal source recipe is eight-rank')
    ranks=[read_journal(run/f'metrics_rank{rank}.jsonl') for rank in range(8)]
    end=min(len(rows) for rows in ranks)//4*4
    if not end:raise ValueError('No complete fresh global rollout yet')
    for rank,rows in enumerate(ranks):
        for row in rows[:end]:
            if row['behavior']['run']!=identity['sha256']:raise ValueError('Mixed run identity')
            if row['collection']!=(row['updates']-1)//4+1 or row['inner_completed']!=(row['updates']-1)%4+1:
                raise ValueError('Broken four-inner contract')
    windows=[summarize_window(ranks,start,min(start+a.window_updates,end)) for start in range(0,end,a.window_updates)]
    complete=all((run/f'result_rank{rank}.json').exists() and json.loads((run/f'result_rank{rank}.json').read_text())['status']=='COMPLETE' for rank in range(8))
    result=dict(status='TRAINING_COMPLETE' if complete else 'INTERIM_TRAINING_ONLY',
        captured_utc=datetime.now(timezone.utc).isoformat(),run_identity=identity['sha256'],variant=identity['recipe']['variant'],
        matched_completed_updates=end,observed_updates_by_rank=[len(rows) for rows in ranks],
        cumulative=summarize_window(ranks,0,end),windows=windows,cost=checkpoint_cost(run),
        first_behavior_adapter_sha256=[rows[0]['behavior']['adapter_sha256'] for rows in ranks],
        source_files=identity['source_files'],
        limits=['Stochastic G8 scores are not original single-ODE deployment scores.',
                'The fixed hard scene list changes training population; do not compare this mean with full-TRAIN or DEV.',
                'A repeated group across four inner updates contributes only one fresh sample group.',
                'Global update count uses all ranks, not rank0 local PG activity.',
                'Window changes also include scene mix; matched fixed-population ODE evaluation determines efficacy.'])
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],variant=result['variant'],updates=end,
        scene_exposures=result['cumulative']['scene_exposures'],PG_group_fraction=result['cumulative']['local_PG_group_fraction'],
        remaining_hours=result['cost']['linear_remaining_training_hours'],output=str(output))))


if __name__=='__main__':main()
