"""Verify published small artifacts offline; never load models or launch jobs."""
from pathlib import Path
import csv
import hashlib
import json
import math


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    checks = 0
    for record in manifest['files']:
        data = (root / record['archive_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256'], record
        assert len(data) == record['bytes'], record
        checks += 1
    for record in manifest['prior_archives_reverified']:
        data = (repo / record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256'], record
        assert len(data) == record['bytes'], record
        checks += 1

    def close(a, b):
        assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9), (a, b)

    base_rows = list(csv.DictReader((root / 'BASE_NAVTEST.csv').open()))
    assert len(base_rows) == 7
    for row in base_rows:
        result = json.loads((root / row['source']).read_text())
        step = int(row['optimizer_step'])
        receipt = json.loads((root / f'evidence/base/step_{step:06d}_CHECKPOINT_RECEIPT.json').read_text())
        reference = json.loads((root / f'evidence/base/step_{step:06d}_REFERENCE_EQUIVALENCE.json').read_text())
        assert reference['passed'] and reference['identical_named_components_and_total']
        assert result['status'] == 'COMPLETE'
        assert result['scenes'] == result['valid_scores'] == 12146
        assert result['logs'] == len(result['per_log']) == 136
        assert sum(x['scenes'] for x in result['per_log'].values()) == 12146
        weighted = sum(x['PDMS_01'] * x['scenes'] for x in result['per_log'].values()) / 12146
        close(weighted, result['PDMS_01'])
        close(row['PDMS_percent'], 100 * result['PDMS_01'])
        close(row['zero_count'], result['zero_fraction'] * 12146)
        assert row['checkpoint_hash'] == result['checkpoint']['checksum'] == receipt['checksum']
        assert step == receipt['optimizer_step'] == result['checkpoint']['optimizer_step']
        assert row['dataset_hash'] == result['dataset_hash']
        assert row['score_protocol_hash'] == result['score_protocol_hash'] == reference['protocol_hash']
    assert len({r['dataset_hash'] for r in base_rows}) == 1
    assert len({r['score_protocol_hash'] for r in base_rows}) == 1

    prior = root.parent / 'iqe_results_summary_20261014'
    original = list(csv.DictReader((prior / 'RESULTS.csv').open()))
    for row in original:
        row['source'] = '../iqe_results_summary_20261014/' + row['source']
    for row in csv.DictReader((prior / 'GEOMETRIC_FINAL_RESULTS.csv').open()):
        row['family'] = 'geometric_completed'
        row['source'] = '../iqe_results_summary_20261014/' + row['source']
        original.append(row)
    rows = list(csv.DictReader((root / 'CURRENT_RESULTS.csv').open()))
    assert rows == original and len(rows) == 29
    for row in rows:
        assert (root / row['source']).exists()
        if row['regret_pp']:
            close(float(row['PDMS_percent']) - float(row['base_PDMS_percent']), row['gain_pp'])
            close(float(row['Oracle_PDMS_percent']) - float(row['PDMS_percent']), row['regret_pp'])
    assert (root / 'EXPERT_SPECIALIZATION.csv').read_bytes() == (prior / 'EXPERT_SPECIALIZATION.csv').read_bytes()
    assert (root / 'EXPERT_TRAINING_DATA.csv').read_bytes() == (prior / 'evidence/TRAINING_DATA_SUMMARY.csv').read_bytes()

    temporal = json.loads((root / 'evidence/temporal/COMPLETE.json').read_text())
    assert temporal['status'] == 'COMPLETE'
    assert temporal['original_bank_score_max_abs_error'] == 0.0
    workers = [json.loads((root / f'evidence/temporal/worker_{i}.json').read_text()) for i in range(4)]
    assert all(w['status'] == 'COMPLETE' and not w['failed'] for w in workers)
    assert sum(w['done'] for w in workers) == temporal['official_candidate_calls'] == 747
    status = json.loads((root / 'STATUS.json').read_text())
    assert status['scorers']['Scorer_navtest'] == 'NOT_EVALUATED'
    assert not status['resource_and_deployment']['new_jobs_started']
    print(json.dumps({'status': 'PASS', 'archive_hashes_checked': checks,
                      'base_curve_rows': len(base_rows), 'current_result_rows': len(rows),
                      'model_tests_rerun': False, 'model_or_metric_cache_loaded': False}))


if __name__ == '__main__':
    main()
