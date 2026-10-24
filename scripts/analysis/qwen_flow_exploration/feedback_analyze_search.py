"""Read-only, paired discovery analysis; this never fits or updates a policy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.qwen_flow_exploration.feedback.process import SAFETY
from starVLA.rl.qwen_flow_exploration.feedback.search import METHODS
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


def sealed(path):
    value = json.loads(path.read_text())
    body = dict(value)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed sealed evidence: '+str(path))
    return value


def safe(row):
    return all(row['native_subscores'][key] == 1. for key in SAFETY)


def bootstrap(values, logs):
    return log_bootstrap_difference(values, logs, repeats=5000, seed=42)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--bank-index', required=True)
    parser.add_argument('--search-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    index = json.loads(Path(args.bank_index).read_text())
    if index['protocol_sha256'] != protocol['sha256'] or index['base_model_sha256'] != protocol['base_model_sha256']:
        raise ValueError('Other common diagnostic population')
    root, output = Path(args.search_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(schema='native_feedback_discovery_analysis_v1', protocol_sha256=protocol['sha256'],
        base_model_sha256=protocol['base_model_sha256'], bank_index_sha256=file_sha(args.bank_index),
        discovery_only_not_policy_deployment=True, method_order=list(METHODS), splits={})
    hashes = {}
    for split in ('TRAIN', 'DEV_SELECT'):
        paths = index.get('splits', index)[split]
        paths = [Path(row['path'] if isinstance(row, dict) else row) for row in paths]
        banks = {}
        for path in paths:
            bank = torch.load(path, map_location='cpu', weights_only=False)
            token = bank['scene_token']
            references = bank['reference_records']
            assert len(references) == 5 and bank['split'] == split
            scores = [row['raw_reward'] for row in references]
            if all(row['raw_reward'] >= .9 and safe(row) for row in references):
                stratum = 'stable_high'
            elif all(score <= 1e-6 for score in scores):
                stratum = 'stable_zero'
            elif np.mean(scores) < .6:
                stratum = 'low_nonzero'
            else:
                stratum = 'other_or_unstable'
            banks[token] = dict(log_id=bank['log_id'], reference_mean=float(np.mean(scores)),
                               base0_reward=bank['candidate_records'][0]['raw_reward'], stratum=stratum)
        tokens = sorted(banks)
        logs = [banks[token]['log_id'] for token in tokens]
        result = dict(scenes=len(tokens), logs=len(set(logs)), methods={}, paired={}, teacher_common_support={})
        all_rows = {}
        for method in METHODS:
            rows, curves, teacher_rows = {}, {k: [] for k in (8, 16, 24, 32)}, []
            folder = root/method/split
            if {p.stem for p in (folder/'scenes').glob('*.json')} != set(tokens):
                raise ValueError('Incomplete or additional search scenes: '+str(folder))
            manifest = {}
            for token in tokens:
                path = folder/'scenes'/f'{token}.json'
                record = sealed(path)
                if (record['status'] != 'COMPLETE' or record['scene_token'] != token or record['split'] != split or
                    record['common_protocol_sha256'] != protocol['sha256'] or
                    record['base_model_sha256'] != protocol['base_model_sha256'] or record['result']['method'] != method):
                    raise ValueError('Search identity differs')
                candidates = record['result']['candidates']
                if len(candidates) != 32:
                    raise ValueError('Invalid attempts disappeared from budget')
                manifest[token] = record['sha256']
                base = banks[token]
                teacher = record['teacher_candidates']
                if split != 'TRAIN' and teacher:
                    raise ValueError('Development teachers must be absent')
                valid = [row for row in candidates if row.get('raw_reward') is not None]
                feasible = [row for row in valid if safe(row)]
                scores = [row['raw_reward'] for row in valid]
                best_safe = max((row['raw_reward'] for row in feasible), default=None)
                best_external = max((row['raw_reward'] for row in feasible if row['external_proposal']), default=None)
                first8_best = max((row['raw_reward'] for row in candidates[:8] if safe(row)), default=None)
                rows[token] = dict(scene_token=token, log_id=base['log_id'], stratum=base['stratum'],
                    oracle=max(scores), feasible=bool(feasible), safe_oracle=best_safe,
                    safe_gain_vs_refmean=(best_safe-base['reference_mean']) if best_safe is not None else None,
                    safe_useful_vs_refmean=best_safe is not None and best_safe >= base['reference_mean']+.005,
                    safe_useful_vs_base0=best_safe is not None and best_safe >= base['base0_reward']+.005,
                    feasible_fraction=len(feasible)/32, scored=len(valid),
                    external_useful_over_first8=(best_external is not None and
                        (first8_best is None or best_external >= first8_best+.005)),
                    teacher_count=len(teacher), external_teacher_count=sum(row['external_proposal'] for row in teacher),
                    teacher_best=max((row['raw_reward'] for row in teacher), default=None),
                    teacher_mean=float(np.mean([row['raw_reward'] for row in teacher])) if teacher else None,
                    novelty_proposals=sum(row['novelty_proposals'] for row in record['result']['rounds']),
                    invalid=record['result']['costs']['invalid_attempts'],
                    duplicates=record['result']['costs']['duplicate_attempts'])
                for k in curves:
                    prefix = candidates[:k]
                    v = [row for row in prefix if row.get('raw_reward') is not None]
                    f = [row for row in v if safe(row)]
                    curves[k].append(dict(oracle=max(row['raw_reward'] for row in v), feasible=bool(f),
                        safe_useful_vs_refmean=any(row['raw_reward'] >= base['reference_mean']+.005 for row in f)))
            hashes[split+'/'+method] = fingerprint(manifest)
            table = list(rows.values())
            method_summary = dict(
                oracle_PDMS=100*float(np.mean([r['oracle'] for r in table])),
                feasible_scenes=sum(r['feasible'] for r in table),
                useful_safe_vs_refmean_scenes=sum(r['safe_useful_vs_refmean'] for r in table),
                useful_safe_vs_base0_scenes=sum(r['safe_useful_vs_base0'] for r in table),
                mean_feasible_candidate_fraction=float(np.mean([r['feasible_fraction'] for r in table])),
                external_useful_over_shared8_scenes=sum(r['external_useful_over_first8'] for r in table),
                teacher_scenes=sum(r['teacher_count'] > 0 for r in table),
                teacher_candidates=sum(r['teacher_count'] for r in table),
                external_teacher_scenes=sum(r['external_teacher_count'] > 0 for r in table),
                external_teacher_candidates=sum(r['external_teacher_count'] for r in table),
                novelty_proposals=sum(r['novelty_proposals'] for r in table),
                invalid_attempts=sum(r['invalid'] for r in table), duplicate_attempts=sum(r['duplicates'] for r in table),
                strata={name: dict(scenes=sum(r['stratum']==name for r in table),
                    feasible=sum(r['feasible'] and r['stratum']==name for r in table),
                    useful_safe=sum(r['safe_useful_vs_refmean'] and r['stratum']==name for r in table))
                    for name in sorted({r['stratum'] for r in table})},
                curves={str(k): dict(oracle_PDMS=100*float(np.mean([r['oracle'] for r in values])),
                    feasible_scenes=sum(r['feasible'] for r in values),
                    useful_safe_vs_refmean_scenes=sum(r['safe_useful_vs_refmean'] for r in values))
                    for k, values in curves.items()},
                costs=json.loads((folder/'SUMMARY_rank000.json').read_text()))
            result['methods'][method] = method_summary
            all_rows[method] = rows
            with (output/f'{split}_{method}_scene_summary.csv').open('w') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        for left, right in zip(METHODS[1:], METHODS[:-1]):
            a,b=all_rows[left],all_rows[right]
            common=[token for token in tokens if a[token]['teacher_count'] and b[token]['teacher_count']]
            result['paired'][left+'-'+right]=dict(
                oracle_difference_raw=bootstrap([a[t]['oracle']-b[t]['oracle'] for t in tokens],logs),
                feasible_discovery_rate_difference=bootstrap([int(a[t]['feasible'])-int(b[t]['feasible']) for t in tokens],logs),
                safe_useful_discovery_rate_difference=bootstrap([int(a[t]['safe_useful_vs_refmean'])-int(b[t]['safe_useful_vs_refmean']) for t in tokens],logs),
                strictly_higher_oracle_scenes=sum(a[t]['oracle']>b[t]['oracle']+1e-12 for t in tokens),
                strictly_lower_oracle_scenes=sum(a[t]['oracle']<b[t]['oracle']-1e-12 for t in tokens),
                new_safe_scenes=sum(a[t]['feasible'] and not b[t]['feasible'] for t in tokens),
                lost_safe_scenes=sum(b[t]['feasible'] and not a[t]['feasible'] for t in tokens))
            result['teacher_common_support'][left+'-'+right]=dict(scenes=len(common),
                mean_best_teacher_difference_raw=float(np.mean([a[t]['teacher_best']-b[t]['teacher_best'] for t in common])) if common else None)
        summary['splits'][split]=result
    summary['search_scene_manifest_sha256']=hashes
    summary['statistical_scope']='Finite K32 discovery; natural diagnostic samples; scene-weighted log-cluster bootstrap; no training-seed uncertainty; no deployment claim'
    summary['sha256']=fingerprint(summary)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({split:{method:{k:v for k,v in s.items() if k not in ('costs','curves','strata')} for method,s in result['methods'].items()} for split,result in summary['splits'].items()},indent=2))


if __name__ == '__main__':
    main()
