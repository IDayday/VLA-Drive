"""Collect completed repair diagnostics; no geometry, model or scorer rerun."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registries', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cpu, started = time.process_time(), time.monotonic()
    inputs, reports, table = {}, [], []
    for registry in a.registries:
        for job in read(registry)['jobs']:
            root = Path(job['output'])
            complete, identity = read(root / 'COMPLETE.json'), read(root / 'identity.json')
            if complete['scenes'] != 1696 or complete['identity'] != identity['identity']:
                raise ValueError('Incomplete or mismatched behavior evaluation')
            dist = job['distribution']
            if dist not in inputs:
                files = sorted((root / 'records').glob('*.json'))
                if len(files) != 1696:
                    raise ValueError('Full existing per-scene records required')
                with ThreadPoolExecutor(max_workers=8) as pool:
                    rows = list(pool.map(read, files))
                if len({r['token'] for r in rows}) != 1696:
                    raise ValueError('Duplicate scene')
                inputs[dist] = {}
                for group in complete['groups']:
                    subset = [r for r in rows if r['groups'][group]]
                    inputs[dist][group] = {
                        'scenes': len(subset),
                        'q0_FDE_m': sum(r['q0_FDE_m'] for r in subset) / len(subset) if subset else None,
                    }
            for group, stats in complete['groups'].items():
                if stats['scenes'] != inputs[dist][group]['scenes']:
                    raise ValueError('Changed fixed input grouping')
                plan = stats.get('planning', {})
                table.append({
                    'run_id': job['id'], 'distribution': dist, 'kind': job['kind'],
                    'seed': job['seed'], 'group': group, 'scenes': stats['scenes'],
                    'q0_ADE_m': stats.get('q0_ADE_m'),
                    'q0_FDE_m': inputs[dist][group]['q0_FDE_m'],
                    'final_ADE_m': stats.get('final_ADE_m'), 'final_FDE_m': stats.get('final_FDE_m'),
                    'input_PDMS': plan.get('PDMS_baseline_percent'),
                    'final_PDMS': plan.get('PDMS_first_percent'),
                    'delta_PDMS_pp': plan.get('delta_percentage_points'),
                    'mean_change_m': stats.get('mean_change_m'), 'max_change_m': stats.get('max_change_m'),
                    'new_known_constraint_violation_fraction': stats.get('new_known_constraint_violation'),
                    'endpoint_below_1m_transition_fraction': stats.get('new_stop'),
                })
            reports.append({'job': job, 'identity': identity, 'complete': complete,
                            'complete_sha256': digest(root / 'COMPLETE.json')})
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output / 'SUMMARY.json', {
        'registered_sources': {str(x): digest(x) for x in a.registries},
        'reports': reports, 'input_groups': inputs,
        'no_new_inference_scoring_geometry_or_thresholds': True,
        'scope': 'Development geometry/behavior strata overlap; not additional independent populations. KEEP is scoped local acceptability; new_stop was only an endpoint displacement proxy.',
        'CPU_process_seconds': time.process_time() - cpu,
        'elapsed_seconds': time.monotonic() - started, 'tool_sha256': digest(__file__),
    })
    with (a.output / 'GROUPS.csv').open('w') as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'completed_evaluations': len(reports), 'group_rows': len(table),
                      'seconds': time.monotonic() - started}))


if __name__ == '__main__':
    main()
