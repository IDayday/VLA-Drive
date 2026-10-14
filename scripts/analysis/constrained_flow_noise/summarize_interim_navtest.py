"""Compare complete matched Navtest observations, with scene-weighted log CIs."""
import argparse
from pathlib import Path
import time

import numpy as np

from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from starVLA.rl.constrained_flow_noise.contracts import fingerprint, file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from .analyze_ode import SAFETY
from .interim_navtest import SCHEMA
from .lock_final import read, sealed


def summarize(root, population, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Immutable complete comparison already exists')
    lock = sealed(root/'observation_lock.json')
    if lock['schema'] != SCHEMA:
        raise ValueError('Not the user-requested matched interim observation')
    reports, banks, costs = {}, {}, {}
    for name in ('SFT', 'C_SDE', 'N_FLOW_NOISE'):
        model = root/name
        report = sealed(model/'report.json')
        manifest, bank, digest = load_scores(model/'scores', population, 'evaluation', [2101])
        if (report['scenes'] != 12146 or report['logs'] != 136
                or report['behavior_sha256'] != lock['models'][name]['weight_sha256']
                or report['score_payloads_sha256'] != digest
                or report['observation_context']['lock_sha256'] != lock['sha256']
                or manifest['protocol_sha256'] != lock['v1_protocol_sha256']
                or set(bank) != set(population['navtest_tokens'])):
            raise ValueError('Incomplete or unmatched full Navtest bank')
        cost = read(model/'cost.json')
        shards = cost['inference_shards']
        if len(shards) != 8 or any(x['status'] != 'ASSIGNED_POPULATION_COMPLETE' for x in shards):
            raise ValueError('Not all eight inference shards completed')
        score_status = cost['scoring']
        if (score_status['status'] != 'COMPLETE' or score_status['errors']
                or score_status['new_scenes']+score_status['reused_scenes'] != 12146):
            raise ValueError('Infrastructure failures cannot be excluded')
        launch = read(model/'inference_launch.json')
        last_shard_write = max(p.stat().st_mtime for p in (model/'samples').glob('shard*.json'))
        costs[name] = dict(**cost,
                          reserved_GPU_hours_upper=8*(last_shard_write-launch['started_unix'])/3600,
                          GPU_cost_definition='Eight reserved physical devices from torchrun launch through last complete shard write; includes CPU/model loading, not exclusive GPU compute and excludes post-inference analysis',
                          shared_training_not_stopped=True)
        reports[name], banks[name] = report, bank
    tokens = population['navtest_tokens']
    logs = [population['token_logs'][t] for t in tokens]
    c, n = banks['C_SDE'], banks['N_FLOW_NOISE']
    paired = {}
    for key in c[tokens[0]]:
        delta = np.array([n[t][key][0]-c[t][key][0] for t in tokens])
        paired[key] = log_bootstrap_difference(delta, logs)
    safety = {}
    for key in (*SAFETY, 'joint'):
        fields = SAFETY if key == 'joint' else (key,)
        cp = np.array([all(c[t][k][0] == 1 for k in fields) for t in tokens])
        npass = np.array([all(n[t][k][0] == 1 for k in fields) for t in tokens])
        safety[key] = dict(C_violations=int((~cp).sum()), N_violations=int((~npass).sum()),
                           C_pass_N_fail=int((cp & ~npass).sum()), C_fail_N_pass=int((~cp & npass).sum()),
                           violation_difference=log_bootstrap_difference((~npass).astype(float)-(~cp).astype(float), logs))
    result = dict(schema='cfn_complete_interim_navtest_comparison_v1',
                  status='COMPLETE_USER_REQUESTED_INTERIM_NAVTEST', observation_lock_sha256=lock['sha256'],
                  population_identity=population['identity_sha256'], scenes=12146, logs=136,
                  training_seed=42, inference_seed=2101, RL_raw_exposure_per_variant=25398,
                  formal_budget_fraction=.125, formal_training_complete=False,
                  models={name:dict(report_sha256=r['sha256'], metrics=r['metrics'], safety=r['safety'],
                                    zeros=r['zeros'], motion=r['motion'], arrival=r['arrival'],
                                    meets_componentwise_safety_rule=r['meets_componentwise_safety_rule'])
                          for name,r in reports.items()},
                  N_minus_C=dict(metrics=paired, safety=safety), costs=costs,
                  inference_protocol='Original raw observation / single candidate / 10-step ODE / no learned noise / canonical v1.1',
                  uncertainty='Scene-weighted log-cluster bootstrap; no training or inference seed uncertainty',
                  checkpoint_selection_used_Navtest=False, historically_observed_Navtest_not_fully_blind=True,
                  source_sha256=file_sha(__file__))
    result['sha256'] = fingerprint(result)
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('root', 'population', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    require_dispatch()
    root = Path(args.root)
    while not all((root/m/'report.json').is_file() for m in ('SFT','C_SDE','N_FLOW_NOISE')):
        if not args.wait:
            raise ValueError('All three complete Navtest reports are required')
        failures = [p for m in ('SFT','C_SDE','N_FLOW_NOISE') for p in (root/m).glob('failure_*.json')]
        if failures:
            raise RuntimeError('Evaluation failure needs explicit recovery: '+str(failures))
        require_dispatch()
        time.sleep(30)
    result = summarize(root, read(args.population), args.output)
    print({m:100*r['metrics']['score']['current'] for m,r in result['models'].items()}, flush=True)


if __name__ == '__main__':
    main()
