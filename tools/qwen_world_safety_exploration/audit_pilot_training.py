"""Read every registered pilot's eight-rank metric prefix, without CUDA or mutation.

Counts at first inner are collection observations, not four duplicated rollouts.
Surrogate weights and preclip gradient norms do not establish causal interference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_run(run, updates):
    if updates <= 0 or updates % 4:
        raise ValueError('A complete four-inner collection boundary is required')
    root = Path(run['root']); counts = Counter(); sums = Counter()
    by_inner = defaultdict(Counter); values = defaultdict(list)
    rank_sources = []; scene_order = {}; traces = {}; zero_pg_ranks = Counter(); curriculum = Counter()
    for rank in range(8):
        path = root / f'metrics_rank{rank}.jsonl'
        prefix = hashlib.sha256(); seen = []; ordered = []
        with path.open('rb') as stream:
            for line in stream:
                row = json.loads(line)
                update = row['updates']
                if update > updates:
                    break
                if update != len(seen) + 1:
                    raise ValueError(f'Missing or duplicate native update: {path}:{update}')
                if row['group_size'] != run['group_size']:
                    raise ValueError('Registered candidate count changed')
                collection = (update - 1) // 4 + 1; inner = (update - 1) % 4 + 1
                if row['collection'] != collection or row['inner_completed'] != inner:
                    raise ValueError('Collection/inner ordering differs from native optimizer updates')
                prefix.update(line); seen.append(update)
                m = row['metrics']; e = row['exploration']; bucket = by_inner[inner]
                bucket['rank_updates'] += 1
                for key in ('clipfrac', 'ppo_blocked_fraction', 'quality_active_positive_high_weight',
                            'quality_active_negative_high_weight', 'quality_active_positive_unsafe_weight'):
                    bucket[key] += m[key]
                values['preclip_gradient_norm'].append(row['gradient_norm'])
                if row['source_clamp_hit_rate'] is None:
                    counts['source_clamp_not_applicable_rank_updates'] += 1
                else:
                    values['source_clamp_hit_rate'].append(row['source_clamp_hit_rate'])
                values['ratio_max_abs_from_one'].append(row['ratio_max_abs_from_one'])
                for key, value in row.get('optimizer_group_lrs_used', {'action_lora': row['lr']}).items():
                    values['actual_used_lr_' + key].append(value)
                if inner != 1:
                    continue
                ordered.append(row['token']); counts['groups'] += 1
                traces[(rank, collection)] = dict(updates=update, rank=rank, token=row['token'],
                    behavior_hash=row['behavior']['adapter_sha256'], chain_hash=row['chain_identity']['chain'],
                    credit_activated=any(value > 0 for key, value in m.items() if key.endswith('_activated_groups')))
                counts['candidates'] += row['group_size']
                for key in ('all_zero', 'tied', 'safety_differing', 'progress_only'):
                    counts[key + '_groups'] += bool(e[key])
                # The journal's zero_PG tests the saved mask. Inversion deliberately
                # retains that mask even when cancellation removes all actual A.
                mask_zero = bool(e['zero_PG'])
                true_zero = bool(m.get('inversion_credit_post_raw_PG0_groups', mask_zero))
                counts['journal_adv_mask_zero_groups'] += mask_zero
                counts['true_zero_advantage_groups'] += true_zero
                counts['nonzero_mask_but_zero_advantage_groups'] += not mask_zero and true_zero
                counts['all_zero_and_zero_advantage_groups'] += bool(e['all_zero'] and true_zero)
                counts['positive_unsafe_with_safe_alternative_candidates'] += e['positive_unsafe_with_safe_alternative']
                zero_pg_ranks[collection] += true_zero
                for key in ('quality_strict_high_count', 'quality_feasible_count',
                            'quality_positive_high_count', 'quality_negative_high_count'):
                    counts[key] += int(m[key])
                counts['groups_with_strict_high'] += bool(m['quality_has_strict_high'])
                for key in ('canonical_random_mean', 'NC', 'DAC', 'TTC', 'EP', 'safe_fraction'):
                    sums[key] += e[key]
                for key, value in m.items():
                    if key.startswith(('dense_dac_', 'saturated_credit_', 'protected_credit_', 'inversion_credit_')):
                        if key.endswith(('_groups', '_candidates', '_transfers', '_negative_high', '_qualifying_pairs')):
                            counts[key] += value
        if len(seen) != updates:
            raise ValueError(f'Incomplete native metric prefix: {path}: {len(seen)} != {updates}')
        rank_sources.append(dict(path=str(path), prefix_updates=updates, prefix_sha256=prefix.hexdigest()))
        scene_order[rank] = ordered
        result_path = root / f'result_rank{rank}.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('updates') == updates:
                for key, value in result.get('curriculum', {}).get('stats', {}).items():
                    curriculum[key] += value
    counts['globally_all_eight_ranks_zero_PG_collections'] = sum(v == 8 for v in zero_pg_ranks.values())
    all_tokens = [token for rank in scene_order.values() for token in rank]
    counts['unique_scene_tokens'] = len(set(all_tokens))
    counts['repeated_scene_exposures'] = len(all_tokens) - len(set(all_tokens))
    result = dict(arm=run['id'], updates=updates, group_size=run['group_size'],
        counts=dict(counts), mean_first_inner_exploration={k: v / counts['groups'] for k, v in sums.items()},
        by_inner_mean={str(i): {k: v / b['rank_updates'] for k, v in b.items() if k != 'rank_updates'}
                       for i, b in by_inner.items()},
        update_statistics={k: dict(mean=sum(v)/len(v), minimum=min(v), maximum=max(v),
                                  above_one_count=sum(x > 1 for x in v)) for k, v in values.items()},
        curriculum_exact_cap_result_stats=dict(curriculum), sources=rank_sources,
        scene_order_sha256=digest(scene_order), metric_rows=updates * 8,
        counts_are_local_rank_observations_not_global_update_counts=True,
        strict_high='NC=DAC=TTC=Comfort=1 and canonical score>=.9',
        noncausal_limits=['Surrogate coefficients are not parameter-gradient norms',
            'Different changing policies can visit different actions even under identical scene order',
            'No claim that all frozen-Qwen representations are sufficient',
            'No new rollout, scorer call, optimizer update, GPU dispatch or source mutation'])
    return result, scene_order, traces


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--registry', help='Explicit earlier registry snapshot, e.g. before admitting a later pilot')
    parser.add_argument('--output', required=True); args = parser.parse_args()
    root = Path(args.root); registry_path = Path(args.registry) if args.registry else root / 'RUNS.json'
    raw = registry_path.read_bytes(); registry = json.loads(raw); arms = {}; orders = {}; traces = {}
    for run in registry['runs']:
        # An independently registered later pilot is not silently treated as complete.
        if not Path(run['root']).resolve().is_relative_to(root.resolve() / 'runs'):
            raise ValueError('Registry contains a foreign run root')
        arms[run['id']], orders[run['id']], traces[run['id']] = audit_run(run, args.updates)
    control = orders['SAFETY_G8_CONTROL']
    for arm, order in orders.items():
        matched = sum(a == b for rank in range(8) for a, b in zip(order[rank], control[rank]))
        arms[arm]['same_rank_collection_scene_as_control'] = dict(matched=matched, total=args.updates * 2)
        rows = traces[arm]; activated = [r for r in rows.values() if r['credit_activated']]
        if activated:
            reference = traces['SAFETY_G8_CONTROL']
            first = min(r['updates'] for r in activated)
            diff = lambda key: min(((r['updates'], r['rank']) for k, r in rows.items()
                                    if r[key] != reference[k][key]), default=None)
            arms[arm]['activation_and_divergence'] = dict(first_activated_update=first,
                first_behavior_parameter_hash_difference=diff('behavior_hash'),
                first_chain_hash_difference=diff('chain_hash'),
                first_activation_records=[{k: r[k] for k in ('updates', 'rank', 'token')}
                                          for r in activated if r['updates'] == first],
                exact_hashes_do_not_quantify_functional_or_gradient_interference=True)
    result = dict(schema='world_full_pilot_native_metric_census_v1', status='COMPLETE',
        created_unix=time.time(), updates_per_arm=args.updates, arms=arms,
        registry_path=str(registry_path), registry_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        metric_rows=sum(a['metric_rows'] for a in arms.values()),
        groups=sum(a['counts']['groups'] for a in arms.values()),
        candidate_observations=sum(a['counts']['candidates'] for a in arms.values()),
        incomplete_or_sampled_population=False)
    result['sha256'] = digest(result)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2)); tmp.replace(output)
    print(json.dumps({k: result[k] for k in ('status', 'metric_rows', 'groups', 'candidate_observations')}))


if __name__ == '__main__':
    main()
