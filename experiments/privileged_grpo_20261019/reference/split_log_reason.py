"""Group-disjoint 90/10 split with primary-reason stratification; CPU only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'mining_v3r2'
DEST = ROOT / 'splits/log_reason_90_10_v1'
PRIORITY = ['NC', 'DAC', 'TTC', 'DDC', 'EP', 'CF']


def group(log):
    source, start, end = log.rsplit('_', 2)
    assert start.isdigit() and end.isdigit() and int(start) < int(end), log
    return source


def primary(row):
    return next(k for k in PRIORITY if row['base_score']['metrics'][k] < 1)


def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not (DEST / 'ASSIGNMENTS.json').exists(), 'Use a new version; do not overwrite split.'
    population = json.loads((SOURCE / 'POPULATION.json').read_text())
    pairs = [json.loads(line) for line in (SOURCE / 'evaluation/TRAINING_PAIRS.jsonl').open()]
    acceptance = json.loads((SOURCE / 'FINAL_ACCEPTANCE.json').read_text())
    assert digest(SOURCE / 'evaluation/TRAINING_PAIRS.jsonl') == acceptance['export_sha256']['TRAINING_PAIRS.jsonl']
    token_log = {r['token']: r['log'] for r in population}
    assert len(token_log) == len(population) == acceptance['scored']
    groups = sorted({group(r['log']) for r in population})
    index = {g: i for i, g in enumerate(groups)}
    totals = Counter(primary(r) for r in pairs)
    target_n = math.floor(len(pairs) * .1 + .5)
    targets = {k: math.floor(n * target_n / len(pairs)) for k, n in totals.items()}
    rank = sorted(totals, key=lambda k: (-(totals[k] * target_n / len(pairs) - targets[k]), PRIORITY.index(k)))
    for k in rank[:target_n - sum(targets.values())]:
        targets[k] += 1
    features = defaultdict(lambda: np.zeros(len(groups)))
    for row in population:
        features['population'][index[group(row['log'])]] += 1
    for log in sorted(set(token_log.values())):
        features['logs'][index[group(log)]] += 1
    for row in pairs:
        i = index[group(row['log'])]
        features['primary_' + primary(row)][i] += 1
        for k in PRIORITY:
            features['metric_' + k][i] += int(row['base_score']['metrics'][k] < 1)
    # Absolute deviations of overlapping metrics and population size are secondary.
    soft = [(name, float(features[name].sum()) * .1, 1 / max(1, float(features[name].sum()) * .1))
            for name in ['metric_' + k for k in PRIORITY] + ['population', 'logs']]
    n = len(groups); width = n + len(soft)
    A, lo, hi = [], [], []
    def constraint(values, lower, upper):
        A.append(values); lo.append(lower); hi.append(upper)
    for k, target in targets.items():
        a = np.zeros(width); a[:n] = features['primary_' + k]
        constraint(a, target, target)
    a = np.zeros(width); a[:n] = features['population']
    constraint(a, math.ceil(len(population) * .095), math.floor(len(population) * .105))
    objective = np.zeros(width)
    for j, (name, target, weight) in enumerate(soft):
        a = np.zeros(width); a[:n] = features[name]; a[n+j] = -1
        constraint(a, -np.inf, target)
        a = np.zeros(width); a[:n] = -features[name]; a[n+j] = -1
        constraint(a, -np.inf, -target)
        objective[n+j] = weight
    result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(len(soft))],
                  bounds=Bounds(np.zeros(width), np.r_[np.ones(n), np.full(len(soft), np.inf)]),
                  constraints=LinearConstraint(np.asarray(A), np.asarray(lo), np.asarray(hi)),
                  options={'time_limit': 45, 'mip_rel_gap': .005})
    assert result.x is not None, result.message
    selected = result.x[:n] > .5
    for k, target in targets.items():
        assert int(features['primary_' + k] @ selected) == target
    assert .095 <= features['population'] @ selected / len(population) <= .105
    group_side = {g: 'test' if selected[i] else 'train' for i, g in enumerate(groups)}
    log_side = {log: group_side[group(log)] for log in sorted(set(token_log.values()))}
    tokens = {side: sorted(t for t, log in token_log.items() if log_side[log] == side) for side in ['train', 'test']}
    assignments = {'identity': acceptance['identity'], 'group_rule': "log.rsplit('_', 2)[0]", 'groups': group_side, 'logs': log_side}
    dump(DEST / 'ASSIGNMENTS.json', assignments)
    for side in tokens:
        (DEST / side).mkdir()
        dump(DEST / side / 'TOKENS.json', tokens[side])
        dump(DEST / side / 'LOGS.json', sorted(log for log in log_side if log_side[log] == side))
    # Preserve bytes and all metadata, including GT targets that do not improve PDMS.
    export_stats = {}
    for name in ['TRAINING_PAIRS.jsonl', 'ALL_GT.jsonl']:
        seen = set(); counts = Counter(); source_hash = hashlib.sha256()
        handles = {s: (DEST / s / name).open('xb') for s in tokens}
        try:
            with (SOURCE / 'evaluation' / name).open('rb') as f:
                for line in f:
                    source_hash.update(line); row = json.loads(line)
                    token = row['token']; log = row['log']
                    assert token not in seen and token_log[token] == log
                    assert row['identity'] == acceptance['identity']
                    seen.add(token); side = log_side[log]; counts[side] += 1
                    handles[side].write(line)
        finally:
            for f in handles.values(): f.close()
        expected = set(token_log) if name == 'ALL_GT.jsonl' else {r['token'] for r in pairs}
        assert seen == expected
        assert source_hash.hexdigest() == acceptance['export_sha256'][name]
        export_stats[name] = {'counts': dict(counts), 'source_sha256': source_hash.hexdigest(), 'sha256': {s: digest(DEST / s / name) for s in tokens}}
    summaries = {}
    for side in tokens:
        chosen = [r for r in pairs if log_side[r['log']] == side]
        summaries[side] = {
            'failure_pairs': len(chosen), 'all_gt': len(tokens[side]),
            'primary_reasons': dict(Counter(primary(r) for r in chosen)),
            'overlapping_metrics': {k: sum(r['base_score']['metrics'][k] < 1 for r in chosen) for k in PRIORITY},
            'logs': sum(s == side for s in log_side.values()),
            'failure_logs': len({r['log'] for r in chosen}),
            'source_drive_groups': sum(s == side for s in group_side.values()),
            'gt_quality': {k: sum(r['guidance']['eligibility'][k] for r in chosen) for k in ['gt_safety_pass', 'gt_pdms_improves', 'gt_verified_repair']},
        }
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'identity': acceptance['identity'],
        'status': 'EXPORTED_PENDING_INDEPENDENT_VALIDATION',
        'test_primary_targets': targets, 'partitions': summaries,
        'solver': {'status': int(result.status), 'message': result.message, 'objective': float(result.fun), 'mip_gap': float(result.mip_gap), 'time_limit_s': 45, 'note': 'Feasible integer primary targets checked; secondary optimum not required or claimed.'},
        'exports': export_stats,
        'limitations': ['Source drive group inferred from verified log name structure; no same-source windows cross partitions.', 'No spatial deduplication between different source drives.', 'Test is held out from future expert training, not from Base training.', 'No separate development set; repeated test-based selection invalidates final-test interpretation.', 'Rare primary NC and DDC have only two test cases each.'],
        'script_sha256': digest(Path(__file__)),
    }
    dump(DEST / 'SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
