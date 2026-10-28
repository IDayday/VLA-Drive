"""Read-only aligned online TRAIN-loss / full Navtest checkpoint analysis."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from starVLA.rl.qwen_world_postcot.contracts import fingerprint

WINDOWS = (100, 500, 1000)
PRIMARY_WINDOW = 500
COMPONENTS = ('x', 'dino', 'future', 'object', 'post')


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), pearson=None, spearman=None)
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isclose(r, pearsonr(x, y).statistic, atol=1e-12):
        raise ValueError('Independent Pearson arithmetic differs')
    return dict(n=len(x), pearson=r, spearman=float(spearmanr(x, y).statistic))


def read_rank(path, last, selected):
    rows = {}; digest = hashlib.sha256(); expected = 1; byte_count = 0
    with Path(path).open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row['updates'] > last:
                break
            if row['updates'] != expected or row['cursor'] != expected * 32:
                raise ValueError('Duplicate/gap/exposure mismatch in effective journal: ' + str(path))
            expected += 1; digest.update(line); byte_count += len(line)
            if row['updates'] in selected:
                if (row['phase'] != 'joint' or row['optimizer_global_steps'] != row['updates']
                        or row['optimizer_skipped_steps'] != 0):
                    raise ValueError('Invalid joint optimizer boundary')
                values = np.array([row['losses'][k] for k in COMPONENTS], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite loss')
                rows[row['updates']] = dict(values=values, bank=row['candidate_bank_hash_used'])
    if expected != last + 1 or set(rows) != selected:
        raise ValueError('Incomplete fixed log prefix')
    return rows, dict(path=str(path), through_update=last, bytes=byte_count, prefix_sha256=digest.hexdigest())


def load_scores(plan):
    rows = []
    for item in sorted(plan, key=lambda x: x['update']):
        summary = json.loads(Path(item['summary']).read_text())
        complete = json.loads(Path(item['complete']).read_text())
        reg = json.loads(Path(item['registration']).read_text())
        body = dict(summary); saved = body.pop('sha256')
        reg_body = dict(reg); reg_sha = reg_body.pop('sha256')
        if (fingerprint(body) != saved or fingerprint(reg_body) != reg_sha
                or complete['registration_sha256'] != reg_sha
                or not complete['all_owned_workers_exited'] or complete['status'] != 'COMPLETE'
                or summary['status'] != 'COMPLETE' or summary['scenes'] != 12146
                or summary['scene_denominator'] != 12146 or summary['invalid_scenes'] != 0
                or summary['cohort'] != 'NAVTEST' or summary['seed'] != 2101
                or reg['updates'] != item['update'] or reg['prediction_type'] != 'x0'
                or reg['inference'] != 'single_candidate_cleanx_K10_shift5'
                or abs(complete['PDMS'] - summary['PDMS']) > 1e-10):
            raise ValueError('Unsealed/incomplete/misbound Test observation')
        rows.append(dict(update=item['update'], PDMS=summary['PDMS'], summary_sha256=saved,
                         registration_sha256=reg_sha, protocol=summary['protocol_sha256'],
                         population=summary['token_sha256']))
    if len({r['update'] for r in rows}) != len(rows) or len({(r['protocol'], r['population']) for r in rows}) != 1:
        raise ValueError('Repeated checkpoint or changed Test protocol/population')
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--scores', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run = Path(args.run); out = Path(args.output)
    if out.exists():
        raise FileExistsError('Keep previous analysis immutable')
    scores = load_scores(json.loads(Path(args.scores).read_text()))
    steps = [r['update'] for r in scores]; last = max(steps)
    identity = json.loads((run/'IDENTITY.json').read_text()); cfg = identity['config']
    weights = np.array([cfg['loss_weights'][k] for k in COMPONENTS], dtype=np.float64)
    if min(steps) - max(WINDOWS) < max(cfg['interface_updates'], cfg['aux_warmup_updates']):
        raise ValueError('This comparison requires identical fully-ramped joint loss weights')
    selected = {u for step in steps for u in range(step-max(WINDOWS)+1, step+1)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        ranks = list(pool.map(lambda rank: read_rank(run/f'metrics_rank{rank}.jsonl', last, selected), range(8)))
    # These values were already all-reduced by train_il. Ranks are replicas,
    # not eight independent observations or an extra loss sum.
    reference = ranks[0][0]
    for rows, _ in ranks[1:]:
        if any(not np.array_equal(rows[u]['values'], reference[u]['values']) or rows[u]['bank'] != reference[u]['bank'] for u in selected):
            raise ValueError('Global loss replicas differ across ranks')
    by_window = {}; statistics = {}
    for width in WINDOWS:
        table = []
        for score in scores:
            values = [reference[u] for u in range(score['update']-width+1, score['update']+1)]
            components = np.stack([v['values'] for v in values]).mean(axis=0)
            row = dict(score, window=width, total=float(components @ weights),
                       **dict(zip(COMPONENTS, map(float, components))),
                       bank_counts=dict(Counter(v['bank'] for v in values)))
            table.append(row)
        pdms = [r['PDMS'] for r in table]
        stats = {}
        for key in ('total', *COMPONENTS):
            losses = [r[key] for r in table]
            dl, dp = np.diff(losses), np.diff(pdms)
            stats[key] = dict(levels=correlation(losses, pdms), adjacent_changes=correlation(dl, dp),
                              inverse_direction_transitions=int(np.sum(dl*dp < 0)), nonzero_transitions=int(np.sum(dl*dp != 0)))
        by_window[str(width)] = table; statistics[str(width)] = stats
    result = dict(schema='world_IL_online_train_loss_vs_navtest_v1', primary_window=PRIMARY_WINDOW,
        score_count=len(scores), loss_definition='10*x + dino + 0.5*future + 0.1*object + 0.2*post',
        x_definition='normalized clean-x L1, not velocity FM MSE', weights=dict(zip(COMPONENTS, map(float, weights))),
        loss_population='online TRAIN batches ending at saved update; moving weights/noise/data, not fixed-checkpoint Test loss',
        score_population='full Navtest12146 single-candidate shift5 ODE, seed2101',
        ranks_already_globally_reduced=True, checked_global_loss_replica_ranks=8,
        log_prefix_receipts=[v[1] for v in ranks], identity_sha256=hashlib.sha256((run/'IDENTITY.json').read_bytes()).hexdigest(),
        points_by_window=by_window, correlations=statistics,
        interpretation='negative correlation means lower numerical loss accompanies higher PDMS; descriptive dependent checkpoints, not causal or independent replicates')
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    fields = ['update','PDMS','total',*COMPONENTS]
    with (out/'checkpoint_losses.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in by_window[str(PRIMARY_WINDOW)])
    print(json.dumps(dict(primary_table=by_window[str(PRIMARY_WINDOW)], correlations=statistics), indent=2), flush=True)


if __name__ == '__main__':
    main()
