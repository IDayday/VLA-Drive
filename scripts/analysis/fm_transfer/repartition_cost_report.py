"""Recover inherited scoring costs without repeating inference or scoring."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from scripts.analysis.simwam_recipe.cost_ledger import evaluation_cost
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def aggregate_attempts(current, inherited):
    rows = list(inherited) + list(current)
    identities = [r['invocation'] for r in rows]
    if not rows or len(set(identities)) != len(identities):
        raise ValueError('Missing or duplicate physical scoring invocation')
    if any(r['status'] == 'RUNNING' or 'ended_utc' not in r for r in rows):
        raise ValueError('Scoring invocation remains open')
    starts = [datetime.fromisoformat(r['started_utc']).timestamp() for r in rows]
    ends = [datetime.fromisoformat(r['ended_utc']).timestamp() for r in rows]
    return dict(
        physical_scoring_calls=sum(r['physical_scoring_calls'] for r in rows),
        inherited_scoring_calls=sum(r['physical_scoring_calls'] for r in inherited),
        new_execution_scoring_calls=sum(r['physical_scoring_calls'] for r in current),
        cumulative_CPU_worker_scoring_seconds=sum(r['scoring_worker_seconds'] for r in rows),
        cumulative_invocation_wall_seconds=sum(r['wall_seconds'] for r in rows),
        first_start_to_last_end_seconds=max(ends)-min(starts),
        closed_incomplete_attempts=sum(r['status'] != 'ASSIGNED_COMPLETE' for r in rows),
        incomplete_attempt_wall_seconds=sum(r['wall_seconds'] for r in rows if r['status'] != 'ASSIGNED_COMPLETE'),
        uncommitted_inflight_call_cost='UNKNOWN_NOT_ZERO' if any(r['status'] != 'ASSIGNED_COMPLETE' for r in rows) else None,
        note='CPU worker sum and invocation sum are distinct from elapsed wall; shared GPU accounting is separate.')


def report(evaluation, rescored):
    evaluation, rescored = Path(evaluation), Path(rescored)
    identity = json.loads((evaluation/'manifest.json').read_text())
    migration = json.loads((evaluation/'REPARTITION_COMPLETE.json').read_text())
    score_migration = json.loads((rescored/'REPARTITION_COMPLETE.json').read_text())
    for row in migration['inherited_invocations']:
        original = Path(row['path'])
        relative = original.relative_to(Path(migration['old_evaluation']))
        if file_sha(original) != row['sha256'] or file_sha(evaluation/relative) != row['sha256']:
            raise ValueError('Original inference cost ancestry changed')
    parent = Path(score_migration['parent_score_bank'])
    if file_sha(parent/'manifest.json') != score_migration['parent_manifest_sha256']:
        raise ValueError('Original scorer identity changed')
    inherited = score_migration['inherited_attempts']
    actual = [json.loads(p.read_text()) for p in (parent/'attempts').glob('*.json')]
    by_id = lambda rows: {r['invocation']: r for r in rows}
    if len(by_id(actual)) != len(actual) or by_id(actual) != by_id(inherited):
        raise ValueError('Original scorer attempt ancestry changed')
    current = [json.loads(p.read_text()) for p in (rescored/'attempts').glob('*.json')]
    ranks = [json.loads((evaluation/f'rank{i}.json').read_text()) for i in range(identity['shards'])]
    if any(r['status'] != 'ASSIGNED_COMPLETE' for r in ranks):
        raise ValueError('GPU shards have not completed')
    inference = evaluation_cost(evaluation, identity, ranks)
    if not inference['complete_cost_accounting'] or inference['new_scenes'] != identity['scenes']:
        raise ValueError('Inference counter does not cover the full committed population')
    cost = aggregate_attempts(current, inherited)
    if sum(r['new'] for r in current+inherited) != identity['scenes']:
        raise ValueError('Scoring attempts do not match unique committed scene count')
    if cost['inherited_scoring_calls'] != score_migration['inherited_physical_scoring_calls']:
        raise ValueError('Inherited scorer counter changed')
    return dict(schema='Navtest_repartition_complete_cost_v1', evaluation=str(evaluation),
        rescored=str(rescored), scenes=identity['scenes'], modes=identity['modes'],
        inference=inference, v1_0_scoring=cost,
        execution_migration_seconds=migration['wall_seconds'],
        migration_reused_scenes=migration['reused_scenes'],
        additional_inference_for_report=0, additional_scoring_for_report=0,
        note='Historical partial scorer attempts include deliberate boundary drain; their work is retained. '
             'Incomplete calls may have unobserved counters. Metadata migration is not GPU inference. '
             'This supplements the earlier per-new-bank derived scorer cost without changing any score.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--rescored', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError('Preserve previous cost summaries')
    result = report(args.evaluation, args.rescored)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), scenes=result['scenes'],
        physical_scoring_calls=result['v1_0_scoring']['physical_scoring_calls'])))


if __name__ == '__main__': main()
