"""Matched sampler means and finite safe opportunities, never policy best-of-N."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    by_run, identities = {}, {}
    for name in a.runs:
        root = Path(name)
        manifest = json.loads((root/'manifest.json').read_text())
        identities[root.name] = manifest
        rows = []
        for token in manifest['tokens']:
            path = root/'scenes'/token[:2]/(token+'.pt')
            seal = json.loads(path.with_suffix('.json').read_text())
            if seal['identity_sha256'] != manifest['sha256'] or seal['payload_sha256'] != file_sha(path):
                raise ValueError('Incomplete/changed diagnostic: no silent denominator exclusion')
            row = torch.load(path, map_location='cpu', weights_only=True)
            fixed = row['fixed_SFT_reference_feedback']
            record = dict(token=token, log=row['log_name'], subset=row['diagnostic_subpopulation'],
                          fixed_SFT_deployment_mean=float(np.mean([f['metrics']['score'] for f in fixed])))
            for sampler, payload in row['samples'].items():
                metrics = [f['metrics'] for f in payload['feedback']]
                reward = np.array([m['score'] for m in metrics])
                safe = np.array([all(m[k] == 1 for k in SAFETY_METRICS) for m in metrics])
                if len(reward) != 16:
                    raise ValueError('Both samplers need the same sixteen candidate budget')
                values = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
                values.update(joint_safe_fraction=float(safe.mean()), feasible=bool(safe.any()),
                              finite_safe_best=float(reward[safe].max()) if safe.any() else None,
                              any_seed_zero=bool(np.any(reward == 0)), all_seeds_zero=bool(np.all(reward == 0)))
                record[sampler] = values
            rows.append(record)
        by_run[root.name] = rows
    first = next(iter(identities.values()))
    if any(v['tokens'] != first['tokens'] or v['paired_root_seeds'] != first['paired_root_seeds'] or v['protocol_sha256'] != first['protocol_sha256'] for v in identities.values()):
        raise ValueError('Unmatched diagnostics cannot be compared')
    summaries = {}
    for name, all_rows in by_run.items():
        summaries[name] = {}
        for subset in ('random', 'targeted', 'combined_descriptive'):
            rows = [r for r in all_rows if subset == 'combined_descriptive' or r['subset'] == subset]
            samplers = {}
            for sampler in ('ode', 'legacy_sde'):
                metrics = ('score', 'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
                           'ego_progress', 'comfort', 'driving_direction_compliance', 'joint_safe_fraction')
                # Exact metric names are inherited from the locked adapter.
                metrics = [k for k in metrics if k in rows[0][sampler]]
                means = {k: float(np.mean([r[sampler][k] for r in rows])) for k in metrics}
                feasible = [r for r in rows if r[sampler]['feasible']]
                means.update(feasible_scenes=len(feasible), scenes=len(rows),
                             finite_safe_best_mean_on_feasible=float(np.mean([r[sampler]['finite_safe_best'] for r in feasible])) if feasible else None,
                             safe_opportunity_gain_on_feasible=float(np.mean([r[sampler]['finite_safe_best']-r['fixed_SFT_deployment_mean'] for r in feasible])) if feasible else None,
                             net_mean_against_fixed_SFT_deployment=float(np.mean([r[sampler]['score']-r['fixed_SFT_deployment_mean'] for r in rows])))
                samplers[sampler] = means
            difference = log_bootstrap_difference([r['legacy_sde']['score']-r['ode']['score'] for r in rows], [r['log'] for r in rows])
            both = [r for r in rows if r['ode']['feasible'] and r['legacy_sde']['feasible']]
            summaries[name][subset] = dict(samplers=samplers, SDE_mean_minus_ODE_mean=difference,
                                          SDE_safe_best_minus_ODE_safe_best_on_jointly_feasible=float(np.mean([r['legacy_sde']['finite_safe_best']-r['ode']['finite_safe_best'] for r in both])) if both else None,
                                          jointly_feasible_scenes=len(both))
    paired = {}
    sft_names = [k for k, v in identities.items() if v['behavior_sha256'] == v['SFT_sha256']]
    if len(sft_names) == 1:
        baseline = {r['token']: r for r in by_run[sft_names[0]]}
        for name, rows in by_run.items():
            if name == sft_names[0]:
                continue
            paired[name] = {subset: {sampler: log_bootstrap_difference(
                [r[sampler]['score']-baseline[r['token']][sampler]['score'] for r in rows if r['subset'] == subset],
                [r['log'] for r in rows if r['subset'] == subset]) for sampler in ('ode', 'legacy_sde')}
                for subset in ('random', 'targeted')}
    report = dict(schema='flow_safe_D1_summary_v1', status='COMPLETE_FOR_LISTED_CHECKPOINTS',
                  run_identities={k: v['sha256'] for k, v in identities.items()}, summaries=summaries,
                  RL_minus_matched_SFT=paired, score_units='raw_0_to_1',
                  caveats=['FP32 common roots isolate samplers; deployment uses original BF16 roots',
                           'Targeted and combined sets are not population-unbiased estimates',
                           'Finite safe best is candidate opportunity, not deployable policy performance'])
    report['sha256'] = fingerprint(report)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    atomic_json(out/'per_scene.json', by_run)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
