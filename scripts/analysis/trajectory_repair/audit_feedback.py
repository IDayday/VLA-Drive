"""Read-only TRAIN-bank label coverage audit; no actor/scorer/GPU work."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.trajectory_repair.feedback import SAFETY, candidate_evidence
from starVLA.rl.trajectory_repair import feedback as feedback_contract


def audit(root, limit=None, useful_gain=.005):
    root = Path(root)
    index = json.loads((root / 'index_TRAIN.json').read_text())
    body = dict(index)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed common-bank index')
    # This index intentionally contains both populations; only TRAIN is read.
    entries = index['TRAIN']
    for split in ('TRAIN', 'DEV_SELECT'):
        if any(Path(name).resolve().parent != (root / split).resolve() or Path(name).suffix != '.pt'
               for name in index[split]):
            raise ValueError('Index path escapes its declared sealed bank population')
    tokens = [Path(name).stem for name in entries]
    dev_tokens = [Path(name).stem for name in index['DEV_SELECT']]
    if len(tokens) != len(set(tokens)) or set(tokens) & set(dev_tokens):
        raise ValueError('Invalid TRAIN membership or leaked development scenes')
    tokens = sorted(tokens)
    if not tokens:
        raise ValueError('An empty bank cannot establish TRAIN label coverage')
    if limit is not None:
        if limit <= 0:
            raise ValueError('Limit must be positive')
        tokens = tokens[:limit]
    counts, availability, sources = Counter(), Counter(), []
    count_logits = Counter()
    started = time.monotonic()
    for token in tokens:
        path = root / 'TRAIN' / (token + '.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        digest = file_sha(path)
        if (seal['status'] != 'COMPLETE' or seal['payload_sha256'] != digest or
                seal['split'] != 'TRAIN' or seal['protocol_sha256'] != index['protocol_sha256']):
            raise ValueError('Unsealed or wrong-split input bank')
        # Trusted local artifacts only; the hash is checked before unpickling.
        value = torch.load(path, map_location='cpu', weights_only=False)
        if (value['split'] != 'TRAIN' or value['scene_token'] != token or
                value['protocol_sha256'] != index['protocol_sha256'] or
                value['checkpoint_id'] != index['base_model_sha256'] or
                seal['checkpoint_id'] != index['base_model_sha256']):
            raise ValueError('Bank scientific identity mismatch')
        candidates = value['reference_records'] + value['candidate_records']
        for row in candidates:
            if (row['split'] != 'TRAIN' or row['scene_token'] != token or
                    row['log_id'] != value['log_id'] or row['status'] != 'OK' or
                    row['checkpoint_id'] != value['checkpoint_id']):
                raise ValueError('Candidate does not belong to the registered TRAIN scene')
            f = row['feedback']
            if (abs(row['raw_reward'] - f['metrics']['score']) > 1e-10 or
                    f['protocol_sha256'] != row['scorer_config_hash']):
                raise ValueError('Candidate reward/scorer identity mismatch')
        evidence = candidate_evidence([r['feedback'] for r in candidates], useful_gain)
        scene_events = Counter()
        for row, parsed in zip(candidates, evidence['parsed']):
            if not np.array_equal(np.asarray(row['planned_trajectory']), parsed.requested_local_poses):
                raise ValueError('Recorded plan differs from scored request')
            availability.update(k for k, present in parsed.available.items() if present)
            counts['candidates'] += 1
            counts['fully_safe_candidates'] += all(parsed.official_metrics[k] == 1 for k in SAFETY)
            counts['zero_score_candidates'] += parsed.official_metrics['score'] == 0
            counts['full_score_candidates'] += parsed.official_metrics['score'] == 1
            if parsed.available['road_body_mask']:
                counts['future_body_outside_frames'] += int(parsed.labels['road_body_outside_mask'][1:].sum())
            for key in ('first_at_fault_collision_seconds', 'first_ttc_rule_infraction_seconds'):
                counts[key + '_present_event'] += parsed.labels[key] is not None
                scene_events[key] += parsed.labels[key] is not None
        counts['scenes'] += 1
        counts['reference_candidate_records'] += len(value['reference_records'])
        counts['independent_pool_candidate_records'] += len(value['candidate_records'])
        counts['with_safe_higher_scoring_alternative'] += int(evidence['has_safe_higher_scoring_alternative'].sum())
        counts['with_safe_higher_scoring_higher_EP_alternative'] += int(evidence['has_safe_higher_scoring_higher_EP_alternative'].sum())
        counts['scenes_with_safe_candidate'] += bool(evidence['feasible'].any())
        for key in ('first_at_fault_collision_seconds', 'first_ttc_rule_infraction_seconds'):
            counts['scenes_with_' + key] += scene_events[key] > 0
        count_logits[value['log_id']] += 1
        sources.append(dict(payload_sha256=digest, seal_sha256=file_sha(path.with_suffix('.json'))))
    names = evidence['parsed'][0].available if tokens else {}
    return dict(schema='trajectory_repair_existing_TRAIN_feedback_audit_v1',
                analysis_source_files_sha256=dict(audit=file_sha(__file__), feedback=file_sha(feedback_contract.__file__)),
                base_checkpoint_sha256=index['base_model_sha256'], source_protocol_sha256=index['protocol_sha256'],
                source_bank_index_sha256=file_sha(root / 'index_TRAIN.json'),
                consumed_payloads_sha256=fingerprint(sources), split='TRAIN', distinct_logs=len(count_logits),
                useful_gain_raw_score=useful_gain, counts=dict(counts),
                label_availability_counts={k: availability[k] for k in names},
                no_actor_updates=True, new_actor_forwards=0, new_scorer_calls=0, no_GPU_work=True,
                audit_wall_seconds=time.monotonic() - started,
                interpretation='Coverage only on the previous fixed diagnostic TRAIN population, not full TRAIN. '
                'Alternatives are finite-pool opportunities, not local repair labels, causal directions, '
                'oracle deployment gains or evidence that slowing down is wrong. '
                'Unavailable full masks, margins and edit directions remain unavailable; no zero filling.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bank-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--useful-gain', type=float, default=.005)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError('Preserve previous coverage audit; choose a new output')
    result = audit(args.bank_root, args.limit, args.useful_gain)
    result['sha256'] = fingerprint(result)
    atomic_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
