"""Recover scored local edits from sealed historical TRAIN records, read-only.

Does not restart the paused search study, run a model, or score a new proposal.
Joint edit parameters are counterfactual labels, not causal PDMS gradients.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.qwen_flow_exploration.feedback.proposal import structured_proposal
from starVLA.rl.trajectory_repair.feedback import SAFETY, read_feedback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--train-search-root', required=True)
    parser.add_argument('--bank-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, bank, out = Path(args.train_search_root).resolve(), Path(args.bank_root).resolve(), Path(args.output)
    if source.name != 'TRAIN':
        raise ValueError('Only explicit TRAIN search records may supply repair pairs')
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((bank / 'index_TRAIN.json').read_text())
    body = dict(index)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed TRAIN source-bank index')
    membership = {Path(p).stem for p in index['TRAIN']}
    paths = sorted((source / 'scenes').glob('*.json'))
    if {p.stem for p in paths} != membership:
        raise ValueError('Historical TRAIN search population differs from sealed bank')
    counts, pairs, positive_scenes, safety_repair_scenes, source_hashes = Counter(), [], set(), set(), []
    started = time.monotonic()
    maximum_error = 0.
    historical_scores = 0
    for path in paths:
        record = json.loads(path.read_text())
        body = dict(record)
        if body.pop('sha256') != fingerprint(body) or record['status'] != 'COMPLETE' or record['split'] != 'TRAIN':
            raise ValueError('Unsealed or wrong-split search record')
        token = record['scene_token']
        if (token != path.stem or record['base_model_sha256'] != index['base_model_sha256'] or
                record['common_protocol_sha256'] != index['protocol_sha256'] or
                record['common_bank_file_sha256'] != file_sha(bank / 'TRAIN' / (token + '.pt'))):
            raise ValueError('Historical edit data/checkpoint identity mismatch')
        historical_scores += record['result']['costs']['actual_new_candidate_scores']
        candidates = {r['candidate_id']: r for r in record['result']['candidates']}
        seen = {}
        for row in candidates.values():
            if not row.get('external_proposal'):
                continue
            counts['edit_attempts'] += 1
            if row.get('raw_reward') is None:
                counts['unscored_invalid_attempts'] += 1
                continue
            parent = candidates[row['parent_candidate_id']]
            before, after = read_feedback(parent['feedback']), read_feedback(row['feedback'])
            diagnostic = row['proposal_validation']
            reconstructed, _ = structured_proposal(parent['planned_trajectory'], diagnostic['initial_velocity_xy'], row['parameters'])
            target = np.asarray(row['planned_trajectory'], dtype=np.float32)
            error = float(np.max(np.abs(reconstructed - target)))
            maximum_error = max(maximum_error, error)
            if error > 1e-5:
                raise ValueError('Recorded edit parameters do not reproduce the actually scored requested trajectory')
            gain = after.official_metrics['score'] - before.official_metrics['score']
            safe_after = all(after.official_metrics[k] == 1 for k in SAFETY)
            safe_before = all(before.official_metrics[k] == 1 for k in SAFETY)
            qualified = safe_after and gain >= .005
            category = 'safe_repair' if qualified and not safe_before else 'safe_quality_improvement' if qualified else 'nonqualifying_edit'
            counts[category] += 1
            if qualified:
                positive_scenes.add(token)
            if category == 'safe_repair':
                safety_repair_scenes.add(token)
            key = fingerprint(target.tolist())
            duplicate_of = seen.get(key)
            pair_id = fingerprint(dict(source=record['sha256'], candidate=row['candidate_id']))
            if duplicate_of is not None:
                counts['duplicate_requested_targets'] += 1
            else:
                seen[key] = pair_id
                counts['unique_scored_edits'] += 1
                if qualified:
                    counts['unique_qualified_edits'] += 1
            pairs.append(dict(pair_id=pair_id, split='TRAIN', scene_token=token, log_id=record['log_id'],
                              source_record_sha256=record['sha256'], base_checkpoint_sha256=index['base_model_sha256'],
                              parent_candidate_id=row['parent_candidate_id'], candidate_id=row['candidate_id'],
                              original_requested_trajectory=before.requested_local_poses,
                              edited_requested_trajectory=after.requested_local_poses,
                              edit_parameters=row['parameters'], parameter_names=diagnostic['parameter_names'],
                              before_metrics=before.official_metrics, after_metrics=after.official_metrics,
                              raw_score_gain=gain, qualified=qualified, category=category, duplicate_of_pair=duplicate_of,
                              on_policy_logprob=None, source_file=str(path)))
        source_hashes.append(file_sha(path))
    atomic_torch(out / 'private_repair_pairs.pt', pairs)
    result = dict(schema='trajectory_repair_recovered_TRAIN_edits_v1', status='COMPLETE_READ_ONLY',
                  scene_population=len(paths), qualified_scene_coverage=len(positive_scenes),
                  fully_safe_repair_scene_coverage=len(safety_repair_scenes), counts=dict(counts),
                  useful_gain_raw_score=.005, maximum_edit_reconstruction_abs_error=maximum_error,
                  base_checkpoint_sha256=index['base_model_sha256'], scorer_protocols_unchanged=True,
                  source_files_aggregate_sha256=fingerprint(source_hashes),
                  private_pairs_sha256=file_sha(out / 'private_repair_pairs.pt'), analysis_source_sha256=file_sha(__file__),
                  reused_historical_new_candidate_scores=historical_scores,
                  new_model_forwards=0, new_scorer_calls=0, new_GPU_tasks=0, optimizer_updates=0,
                  wall_seconds=time.monotonic() - started,
                  interpretation='Recovered observed joint local edits and scored outcomes, including failures/duplicates. '
                  'Not a learned repair policy, not unique causal attribution, not a policy gradient or Navtest gain. '
                  'Historical data collection cost is retained; duplicates must not become extra supervision mass.')
    atomic_json(out / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
