"""Summarize a sealed 5k/10k/15k RL boundary without reading live costs as history.

This extracts the already executed B-5k prefix analysis into one parameterized
entry point. It reads journals, native states and exact boundary cost captures;
it never generates trajectories, scores candidates or updates a policy. The
existing complete-run summary remains the entry point for the 20k endpoint.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from scripts.analysis.simwam_recipe import summarize_training
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha
from starVLA.rl.simwam_recipe.invocation import validate_rows


def read(path):
    return Path(path).read_text()


def load(path):
    return json.loads(read(path))


def summarize(run, end, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior evidence; select a new output')
    if end not in (5000, 10000, 15000):
        raise ValueError('Only registered intermediate RL boundaries are supported')
    identity = load(run/'identity.json')
    recipe = identity['recipe']
    variant = recipe['supervised_transfer']['variant']
    if (identity['purpose'] != 'formal' or identity['world'] != 8
            or variant not in ('A_DDP_TIME', 'B_SOURCE_ACTION_DIT')
            or recipe['max_steps'] != 20000
            or recipe['batch_size'] != 1
            or recipe['gradient_accumulation_steps'] != 1
            or recipe['grpo']['sample']['group_size'] != 8
            or recipe['grpo']['train']['num_inner_epochs'] != 4):
        raise ValueError('This summary requires the registered eight-rank G8/inner4 recipe')

    root = run.parent.parent
    stem = f'{variant}_RL_update{end:06d}'
    checkpoint = run/f'checkpoints/update{end:06d}_inner0'
    seal, files = load(checkpoint/'COMPLETE'), load(checkpoint/'files.json')
    if (file_sha(checkpoint/'files.json') != seal['files_sha256']
            or seal['updates'] != end or seal['phase'] != 'READY'
            or seal['world'] != 8 or seal['next_inner'] != 0):
        raise ValueError('Wrong or unsealed intermediate boundary')
    expected = dict(updates=end, collections=end//4,
                    raw_scene_exposures=end*2, new_candidates=end*16,
                    next_inner=0)
    states = []
    for rank in range(8):
        path = checkpoint/f'rank{rank}.pt'
        if file_sha(path) != files[path.name]:
            raise ValueError('Changed rank recovery state')
        state = torch.load(path, map_location='cpu', weights_only=False)['state']
        if any(state[key] != value for key, value in expected.items()):
            raise ValueError('Rank budget/state mismatch')
        states.append(state)

    progress_path = root/'analysis'/f'{stem}_progress_snapshot.json'
    cost_path = root/'analysis'/f'{stem}_invocation_cost_snapshot.json'
    progress, cost_snapshot = load(progress_path), load(cost_path)
    if (any(progress[key] != value for key, value in expected.items())
            or progress['checkpoint'] != str(checkpoint)
            or progress['invocation_start_update'] != 0
            or cost_snapshot['progress'] != progress
            or cost_snapshot['updates'] != end):
        raise ValueError('Missing exact uninterrupted-invocation boundary cost capture')
    records = cost_snapshot['invocation_records']
    if len(records) != 8 or [r['rank'] for r in records] != list(range(8)):
        raise ValueError('Incomplete or reordered rank cost records')
    for record in records:
        if (record['run'] != identity['sha256']
                or record['invocation'] != progress['invocation']
                or record['start_update'] != 0
                or record['start_collection'] != 0
                or record['start_cursor'] != 0
                or record['last_legal_checkpoint'] != str(checkpoint)):
            raise ValueError('Mixed invocation or checkpoint cost identity')
    wall = progress['wall_seconds_this_invocation']
    if wall <= 0:
        raise ValueError('Missing positive measured boundary wall time')

    ranks = [summarize_training.read_journal(run/f'metrics_rank{rank}.jsonl')
             for rank in range(8)]
    if min(map(len, ranks)) < end:
        raise ValueError('Missing fixed milestone rows')
    for rows in ranks:
        validate_rows(rows, len(rows))
        for row in rows[:end]:
            if (row['behavior']['run'] != identity['sha256']
                    or row['collection'] != (row['updates']-1)//4+1
                    or row['inner_completed'] != (row['updates']-1)%4+1):
                raise ValueError('Changed four-inner behavior identity')
    fresh = [row for rank in ranks for row in rank[:end]
             if row['inner_completed'] == 1]
    population = load(POPULATION)
    phases = ['rollout', 'old_probability', 'update', 'scoring_wait', 'scoring_cpu']
    result = dict(
        schema='fm_transfer_fixed_RL_prefix_v2',
        status=f'COMPLETE_{end//1000}K_TRAINING_PREFIX_NOT_END_OF_RUN',
        captured_unix=time.time(), analysis_script_sha256=file_sha(__file__),
        shared_analyzer_sha256=file_sha(summarize_training.__file__),
        run_identity=identity['sha256'], checkpoint=str(checkpoint),
        checkpoint_files_sha256=seal['files_sha256'],
        observed_updates_by_rank=[len(x) for x in ranks],
        cumulative=summarize_training.summarize_window(ranks, 0, end),
        windows=[summarize_training.summarize_window(ranks, start, start+500)
                 for start in range(0, end, 500)],
        unique_scene_count=len({row['token'] for row in fresh}),
        unique_log_count=len({population['token_logs'][row['token']] for row in fresh}),
        exact_boundary_progress_sha256=file_sha(progress_path),
        cost=dict(
            rank0_training_loop_wall_seconds=wall,
            rank0_preparation_seconds=records[0]['preparation_seconds'],
            all_rank_preparation_seconds=[r['preparation_seconds'] for r in records],
            all_rank_cumulative_save_seconds_through_milestone=[r['seconds_save'] for r in records],
            exact_save_cost_snapshot_sha256=file_sha(cost_path),
            updates_per_rank0_wall_second=end/wall,
            cumulative_rank_wall_seconds_by_phase={
                phase: sum(s['seconds_'+phase] for s in states) for phase in phases},
            rank0_wall_seconds_by_phase={phase: states[0]['seconds_'+phase] for phase in phases},
            note='Overlapping rank wall times, not exclusive GPU-hours; scoring_cpu is scorer elapsed, not CPU core seconds. Rank0 loop wall starts after actor/DDP preparation, includes checkpoint publication, and is not complete process wall. Preparation and exact save-through-milestone wall are separate; no exit cost exists while training continues.'),
        limits=[
            'Fixed difficult TRAIN population, not DEV/Navtest or the full TRAIN mean.',
            'G8 stochastic policy scores and finite candidate maxima are not single-ODE deployment.',
            'A fresh collection is counted once across four inner updates.',
            'Within-run window changes include scene mix; infer efficacy from matched ODE evaluation.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--updates', required=True, type=int, choices=(5000, 10000, 15000))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.run, args.updates, args.output)
    print(json.dumps(dict(status=result['status'], output=str(args.output),
                          updates=result['cumulative']['actual_global_optimizer_updates'],
                          candidates=result['cumulative']['new_candidates'],
                          unique_scenes=result['unique_scene_count'])))


if __name__ == '__main__':
    main()
