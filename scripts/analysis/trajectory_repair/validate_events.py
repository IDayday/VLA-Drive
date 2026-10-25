"""Offline real-TRAIN event export qualification against sealed canonical records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.simwam_recipe.contracts import CACHE_ROOTS
from starVLA.rl.trajectory_repair.feedback import read_feedback
from starVLA.rl.trajectory_repair import official_events


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bank-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out, root = Path(args.output), Path(args.bank_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = json.loads((root / 'index_TRAIN.json').read_text())
    body = dict(index)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed sealed TRAIN index')
    buckets = {k: {} for k in ('NC', 'TTC', 'DAC', 'SAFE')}
    for name in sorted(index['TRAIN']):
        path = Path(name).resolve()
        if path.parent != root / 'TRAIN':
            raise ValueError('Only the registered TRAIN population may supply labels')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['status'] != 'COMPLETE' or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Unsealed candidate payload')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if (value['split'] != 'TRAIN' or value['checkpoint_id'] != index['base_model_sha256'] or
                value['protocol_sha256'] != index['protocol_sha256']):
            raise ValueError('Candidate-bank scientific identity mismatch')
        for row in value['reference_records'] + value['candidate_records']:
            if row['split'] != 'TRAIN' or row['scene_token'] != value['scene_token']:
                raise ValueError('Candidate split/scene mismatch')
            m = row['feedback']['metrics']
            flags = dict(NC=m['no_at_fault_collisions'] < 1, TTC=m['time_to_collision_within_bound'] < 1,
                         DAC=m['drivable_area_compliance'] < 1)
            flags['SAFE'] = not any(flags.values())
            for kind, applies in flags.items():
                if applies and len(buckets[kind]) < 2:
                    buckets[kind].setdefault(value['scene_token'], row)
        if all(len(values) == 2 for values in buckets.values()):
            break
    if not all(len(values) == 2 for values in buckets.values()):
        raise ValueError('Qualification needs two distinct TRAIN scenes for each event/control stratum')
    unique = {}
    for rows in buckets.values():
        for row in rows.values():
            unique[(row['scene_token'], row['candidate_id'])] = row
    protocol = v1_reward.protocol_identity(Path.cwd())
    v1_reward.initialize(dict(repo=str(Path.cwd()), maps_root='/mnt/navsim/maps', cache_roots=CACHE_ROOTS,
                              protocol_sha256=protocol['sha256']))
    records = []
    maximum_metric_error = maximum_state_error = 0.
    count_events = defaultdict(int)
    for row in unique.values():
        original = row['feedback']
        if original['protocol_sha256'] != protocol['sha256']:
            raise ValueError('Canonical scorer identity changed since original saved feedback')
        request = np.asarray(original['requested_trajectory'])
        newer = official_events.score_scene_with_events((row['scene_token'], row['log_id'], [request]))[0]
        parsed = read_feedback(newer)
        metric_error = max(abs(newer['metrics'][k] - original['metrics'][k]) for k in original['metrics'])
        state_error = float(np.max(np.abs(np.asarray(newer['simulated_states']) - original['simulated_states'])))
        if metric_error > 2e-12 or state_error > 1e-8:
            atomic_torch(out / 'failed_comparison.pt', dict(original=original, with_events=newer))
            raise ValueError('Event export changed canonical score or simulated execution')
        if not np.array_equal(newer['non_drivable_body_by_time'], original['non_drivable_body_by_time']):
            raise ValueError('Event export changed official road mask')
        for key in ('official_first_at_fault_collision_seconds', 'official_first_ttc_infraction_seconds'):
            if newer[key] != original[key]:
                raise ValueError('Event export changed first-event time')
        maximum_metric_error, maximum_state_error = max(maximum_metric_error, metric_error), max(maximum_state_error, state_error)
        for key in ('full_collision_mask', 'full_ttc_rule_mask'):
            count_events[key] += int(parsed.labels[key].sum())
        records.append(dict(scene_token=row['scene_token'], log_id=row['log_id'], split='TRAIN',
                            base_checkpoint_sha256=index['base_model_sha256'], feedback=newer))
    atomic_torch(out / 'private_process_labels.pt', records)
    summary = dict(schema='trajectory_repair_official_event_export_qualification_v1', status='PASS',
                   checkpoint_sha256=index['base_model_sha256'], canonical_protocol_sha256=protocol['sha256'],
                   analysis_source_sha256=file_sha(__file__), exporter_source_sha256=file_sha(official_events.__file__),
                   private_labels_sha256=file_sha(out / 'private_process_labels.pt'),
                   fixed_selected_TRAIN_strata={k: len(v) for k, v in buckets.items()},
                   candidate_records=len(records), distinct_scenes=len({k[0] for k in unique}),
                   maximum_metric_abs_error=maximum_metric_error, maximum_simulated_state_abs_error=maximum_state_error,
                   event_time_indices_recorded=dict(count_events), simulated_proposals_including_reference=2 * len(records),
                   new_candidate_scoring_calls=len(records), new_actor_forwards=0, new_GPU_tasks=0, optimizer_updates=0,
                   wall_seconds=time.monotonic() - started,
                   limitations='Targeted TRAIN exporter qualification, not a fitted diagnosis/repair model or deployment result. '
                   'Full masks mean official accepted rule events, not physical contact duration; edit directions and geometric margins remain absent.')
    atomic_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
