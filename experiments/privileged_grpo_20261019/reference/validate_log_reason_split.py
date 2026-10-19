"""Independent export validation against the immutable accepted JSONL sources."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / 'splits/log_reason_90_10_v1'
SOURCE = ROOT / 'mining_v3r2'


def main():
    summary = json.loads((SPLIT / 'SUMMARY.json').read_text())
    acceptance = json.loads((SOURCE / 'FINAL_ACCEPTANCE.json').read_text())
    assignments = json.loads((SPLIT / 'ASSIGNMENTS.json').read_text())
    population = json.loads((SOURCE / 'POPULATION.json').read_text())
    expected = {r['token']: r['log'] for r in population}
    stats = {}
    for name in ['TRAINING_PAIRS.jsonl', 'ALL_GT.jsonl']:
        source_hash = hashlib.sha256(); line_hashes = {}
        with (SOURCE / 'evaluation' / name).open('rb') as stream:
            for line in stream:
                source_hash.update(line); token = json.loads(line)['token']
                assert token not in line_hashes
                line_hashes[token] = hashlib.sha256(line).digest()
        assert source_hash.hexdigest() == acceptance['export_sha256'][name]
        sets = {}; report = {}
        for side in ['train', 'test']:
            tokens = set(); logs = set(); groups = set(); failures = set(); reasons = Counter()
            h = hashlib.sha256()
            with (SPLIT / side / name).open('rb') as stream:
                for line in stream:
                    h.update(line); r = json.loads(line); token = r['token']; log = r['log']
                    assert token not in tokens and expected[token] == log
                    assert hashlib.sha256(line).digest() == line_hashes[token]
                    assert r['identity'] == acceptance['identity']
                    group = log.rsplit('_', 2)[0]
                    assert assignments['logs'][log] == assignments['groups'][group] == side
                    tokens.add(token); logs.add(log); groups.add(group)
                    if name == 'TRAINING_PAIRS.jsonl':
                        reasons[next(k for k in ['NC', 'DAC', 'TTC', 'DDC', 'EP', 'CF'] if r['base_score']['metrics'][k] < 1)] += 1
                    elif r['base_failure']:
                        failures.add(token)
            assert h.hexdigest() == summary['exports'][name]['sha256'][side]
            if name == 'ALL_GT.jsonl':
                assert tokens == set(json.loads((SPLIT / side / 'TOKENS.json').read_text()))
                assert logs == set(json.loads((SPLIT / side / 'LOGS.json').read_text()))
                assert failures == stats['TRAINING_PAIRS.jsonl'][side]['tokens']
            else:
                assert dict(reasons) == summary['partitions'][side]['primary_reasons']
                if side == 'test': assert dict(reasons) == summary['test_primary_targets']
            sets[side] = {'tokens': tokens, 'logs': logs, 'groups': groups}
            report[side] = {'count': len(tokens), 'logs': len(logs), 'source_groups': len(groups)}
        for key in ['tokens', 'logs', 'groups']:
            assert not (sets['train'][key] & sets['test'][key]), (name, key)
        assert sets['train']['tokens'] | sets['test']['tokens'] == set(line_hashes)
        if name == 'ALL_GT.jsonl': assert set(line_hashes) == set(expected)
        stats[name] = sets
        summary['exports'][name]['validation_counts'] = report
    result = {
        'status': 'ACCEPTED', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'identity': acceptance['identity'], 'token_overlap': 0, 'log_overlap': 0,
        'source_drive_group_overlap': 0, 'missing_or_duplicated_tokens': 0,
        'modified_jsonl_rows': 0, 'all_gt_failures_match_pair_tokens': True,
        'source_and_export_sha256_verified': True,
        'primary_reason_targets_matched': True,
        'failure_counts': {s: len(stats['TRAINING_PAIRS.jsonl'][s]['tokens']) for s in ['train', 'test']},
        'all_gt_counts': {s: len(stats['ALL_GT.jsonl'][s]['tokens']) for s in ['train', 'test']},
    }
    with (SPLIT / 'VALIDATION.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
