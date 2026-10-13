"""Audit equal-budget candidate opportunities separately from policy scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import candidate_summary, log_bootstrap_difference


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--banks', nargs='+', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    pop = json.loads(Path(a.population).read_text())
    methods, identities, costs = {}, {}, {}
    for name in a.banks:
        root = Path(name)
        bank = json.loads((root/'manifest.json').read_text())
        if bank['stage'] != 'D3' or bank['population_identity'] != pop['identity_sha256'] or bank['logical_budget_per_scene'] != 64:
            raise ValueError('Require matching locked D3 64-candidate banks')
        method = bank['method']
        if method in methods:
            raise ValueError('Multiple implementations of one method need separate comparisons')
        rows = []
        for token in bank['tokens']:
            path = root/'scenes'/token[:2]/(token+'.pt')
            seal = json.loads(path.with_suffix('.json').read_text())
            if seal['identity_sha256'] != bank['sha256'] or seal['payload_sha256'] != file_sha(path):
                raise ValueError('Incomplete search; no missing scene denominator exclusions')
            row = torch.load(path, map_location='cpu', weights_only=True)
            candidates = row['candidates']
            refs = [c for c in candidates if c['is_reference']]
            if len(refs) != 4 or len(candidates)+len(row['invalid_attempts']) != 64:
                raise ValueError('Candidate and failed-proposal accounting does not sum to the registered budget')
            metrics = [c['feedback']['metrics'] for c in candidates]
            reward = np.array([m['score'] for m in metrics])
            c = np.array([[1-m[k] for k in SAFETY_METRICS] for m in metrics])
            q_cost = np.mean([[1-r['feedback']['metrics'][k] for k in SAFETY_METRICS] for r in refs], axis=0)
            r_ref = float(np.mean([r['feedback']['metrics']['score'] for r in refs]))
            fixed_pass = [all(r['feedback']['metrics'][k] == 1 for k in SAFETY_METRICS) for r in refs]
            reference_safe_scores = [r['feedback']['metrics']['score'] for r,passed in zip(refs,fixed_pass) if passed]
            already_found = bool(reference_safe_scores and max(reference_safe_scores)-r_ref >= .005)
            group = 'stable_safe' if all(fixed_pass) else ('intermittent_failure' if any(fixed_pass) else 'stable_failure')
            feasible = (c == 0).all(1)
            gain = float(reward[feasible].max()-r_ref) if feasible.any() else None
            requests = np.stack([np.asarray(x['requested_trajectory']) for x in candidates])
            unique_requests = {fingerprint(np.asarray(x, dtype=float).tolist()) for x in requests}
            summary = candidate_summary(requests, metrics)
            rows.append(dict(token=token, log=row['log_name'], subset='random' if token in pop['diagnostic_random_tokens'] else 'targeted',
                             reference_group=group, reference_reward=r_ref, F_feasible=bool(feasible.any()), G_safe=gain,
                             safe_gain_nonnegative=max(0., gain) if gain is not None else 0.,
                             discovery_001=bool(gain is not None and gain >= .001),
                             discovery_005=bool(gain is not None and gain >= .005),
                             discovery_005_already_in_shared_reference=already_found,
                             newly_discovered_scene_005=bool(gain is not None and gain >= .005 and not already_found),
                             complete_repair_from_reference_failure=bool(not all(fixed_pass) and feasible.any()),
                             partial_cost_improvement_only=bool(not feasible.any() and np.any(np.all(c <= q_cost+1e-12, axis=1) & np.any(c < q_cost-1e-12, axis=1))),
                             invalid_proposals=len(row['invalid_attempts']), unique_scored_requests=len(unique_requests),
                             **summary))
        methods[method], identities[method] = rows, bank
        status = [json.loads(f.read_text()) for f in root.glob('rank*.json')]
        if any(x['status'] != 'ASSIGNED_POPULATION_COMPLETE' for x in status):
            raise ValueError('Search workers have not completed their assigned populations')
        costs[method] = {k: sum(x[k] for x in status) for k in ('actual_new_scoring_calls', 'velocity_forward_calls', 'GPU_generation_seconds', 'logical_proposals', 'duplicate_proposals', 'invalid_physical_proposals')}
        costs[method]['reference_logical_scoring_cost'] = 4*len(rows)
        costs[method]['new_plus_charged_common_reference_calls'] = costs[method]['actual_new_scoring_calls']+4*len(rows)
        costs[method]['all_worker_wall_seconds_sum'] = sum(x['wall_seconds'] for x in status)
        costs[method]['discoveries_005_per_100_new_scores'] = 100*sum(r['discovery_005'] for r in rows)/max(1, costs[method]['actual_new_scoring_calls'])
        costs[method]['discoveries_005_per_100_calls_including_common_reference'] = 100*sum(r['discovery_005'] for r in rows)/costs[method]['new_plus_charged_common_reference_calls']
        costs[method]['newly_discovered_scenes_005_per_100_new_scores'] = 100*sum(r['newly_discovered_scene_005'] for r in rows)/max(1,costs[method]['actual_new_scoring_calls'])
    first = next(iter(identities.values()))
    if any(x['tokens'] != first['tokens'] or x['protocol_sha256'] != first['protocol_sha256'] for x in identities.values()):
        raise ValueError('Search methods differ in scenes or scoring protocol')
    tables = {}
    for method, rows in methods.items():
        table = {}
        for subset in ('random', 'targeted'):
            for group in ('all', 'stable_safe', 'intermittent_failure', 'stable_failure', 'progress_only'):
                selected = [r for r in rows if r['subset'] == subset and (group == 'all' or r['reference_group'] == group or (group == 'progress_only' and r['progress_only_effective']))]
                if not selected:
                    continue
                valid_gains = [r['G_safe'] for r in selected if r['G_safe'] is not None]
                table[subset+'/'+group] = dict(scenes=len(selected),
                    **{k: float(np.mean([r[k] for r in selected])) for k in ('F_feasible', 'safe_gain_nonnegative', 'discovery_001', 'discovery_005', 'partial_cost_improvement_only')},
                    G_safe_mean_on_feasible=float(np.mean(valid_gains)) if valid_gains else None,
                    invalid_proposals=sum(r['invalid_proposals'] for r in selected),
                    fully_repaired_scenes=sum(r['complete_repair_from_reference_failure'] for r in selected))
        tables[method] = table
    paired = {}
    if 'ODE' in methods:
        original = {r['token']: r for r in methods['ODE']}
        for method, rows in methods.items():
            if method == 'ODE':
                continue
            paired[method] = {subset: {key: log_bootstrap_difference(
                [float(r[key])-float(original[r['token']][key]) for r in rows if r['subset'] == subset],
                [r['log'] for r in rows if r['subset'] == subset])
                for key in ('safe_gain_nonnegative', 'F_feasible', 'discovery_005')} for subset in ('random', 'targeted')}
    report = dict(schema='flow_safe_D3_finite_opportunities_v1', candidate_scores_are_not_policy_scores=True,
                  units='raw_0_to_1', identities={k:v['sha256'] for k,v in identities.items()},
                  tables=tables, costs=costs, paired_exploration_minus_ODE=paired,
                  missing_methods=sorted({'ODE','SDE','LATENT','BEHAVIOR'}-set(methods)),
                  no_search_failure_proves_infeasibility=True)
    report['sha256'] = fingerprint(report)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    atomic_json(out/'per_scene.json', methods)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
