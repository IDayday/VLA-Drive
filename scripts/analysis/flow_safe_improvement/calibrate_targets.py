"""Lock one raw-score temperature using the preregistered TRAIN-only rule."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.analysis.flow_safe_improvement.prepare_target_bank import unpack
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.candidate_bank import deduplicate
from starVLA.rl.safe_flow_improvement.constrained_weights import solve_weights
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, SAFETY_METRICS, file_sha, fingerprint


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.bank)
    bank = json.loads((root/'manifest.json').read_text())
    pop = json.loads(Path(a.population).read_text())
    if bank['stage'] != 'CALIBRATION' or bank['method'] != 'ODE' or bank['tokens'] != pop['calibration_tokens']:
        raise ValueError('Use exactly the predeclared 256 TRAIN ODE calibration scenes')
    PopulationContract(*[frozenset(pop[k]) for k in ('train_tokens', 'dev_tokens', 'navtest_tokens')], pop['token_logs']).require_training(bank['tokens'], 'eta calibration')
    rows, payloads = [], {}
    for token in bank['tokens']:
        path = root/'scenes'/token[:2]/(token+'.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != bank['sha256'] or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Incomplete or changed calibration bank')
        row = torch.load(path, weights_only=True, map_location='cpu')
        if row['invalid_attempts']:
            raise ValueError('Do not silently drop failed calibration proposals')
        selected, qref, empirical, _ = deduplicate(unpack(row))
        rows.append((token, np.array([c.score for c in selected]),
                     np.array([[1-c.metrics[k] for k in SAFETY_METRICS] for c in selected]), qref, empirical))
        payloads[token] = seal['payload_sha256']
    details, summaries = [], []
    for eta in (.005, .02, .05):
        records = []
        for token, reward, costs, qref, empirical in rows:
            for safe in (False, True):
                solved = solve_weights(reward, costs, qref, empirical, eta=eta, safe=safe)
                d = solved.diagnostics
                records.append(dict(token=token, safe=safe, **d,
                                    excess_KL=d['kl_to_original_u']+float(np.log(d['prior_mass_on_allowed_support']))))
        median = float(np.median([r['excess_KL'] for r in records]))
        summaries.append(dict(eta=eta, median_excess_KL=median, distance_to_registered_target=abs(median-.1),
                              median_ESS=float(np.median([r['ess_weights'] for r in records])),
                              max_primal_residual=max(r['primal_residual'] for r in records),
                              mean_alpha=float(np.mean([r['alpha'] for r in records])),
                              mean_gain_before=float(np.mean([r['gain_before_mixing'] for r in records])),
                              mean_gain_after=float(np.mean([r['gain_after_mixing'] for r in records]))))
        details.extend(records)
    selected = min(summaries, key=lambda x: (x['distance_to_registered_target'], -x['median_ESS'], -x['eta']))
    receipt = dict(schema='flow_safe_common_eta_calibration_v1', population_identity=pop['identity_sha256'],
                   bank_identity=bank['sha256'], candidate_payloads_sha256=fingerprint(payloads),
                   source_sha256=file_sha(__file__), scenes=len(rows), selected_eta=selected['eta'],
                   selection_rule='pooled_reward_and_safe_targets_closest_median_excess_KL_to_0.1_then_highest_ESS',
                   summaries=summaries, dev_or_navtest_used=False, alpha_max=.5, delta_useful_raw=.005)
    receipt['sha256'] = fingerprint(receipt)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Calibration is immutable; use a new identity for corrections')
    out.mkdir(parents=True)
    atomic_json(out/'per_scene_diagnostics.json', details)
    atomic_json(out/'registration.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
