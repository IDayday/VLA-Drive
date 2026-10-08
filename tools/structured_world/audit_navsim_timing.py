"""Audit the whole native training population without opening sensor files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ('current-root', 'log-root', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.current_root/'index.json').read_text())
    groups = {}
    for row in rows: groups.setdefault(row['log'], []).append(row['token'])
    def log_times(item):
        log, tokens = item
        with (args.log_root/(log+'.pkl')).open('rb') as stream: frames = pickle.load(stream)
        lookup = {f['token']: i for i, f in enumerate(frames)}
        errors, ineligible = [], []
        for token in tokens:
            start = lookup[token]
            if start+8 >= len(frames): ineligible.append({'token': token, 'reason': 'missing_native_future'}); continue
            actual = np.asarray([f['timestamp'] for f in frames[start+1:start+9]], dtype=np.int64)
            delta = (actual-frames[start]['timestamp'])/1e6-.5*np.arange(1, 9)
            errors.append(np.abs(delta))
            if (np.abs(delta) > .06).any(): ineligible.append({'token': token, 'reason': 'scene_label_time_mismatch', 'delta_s': delta.tolist()})
        return errors, ineligible
    results = list(ThreadPoolExecutor(max_workers=8).map(log_times, groups.items()))
    errors = np.asarray([row for result in results for row in result[0]])
    mismatches = [row for result in results for row in result[1]]
    report = {'population': len(rows), 'future_points': int(errors.size), 'mismatches': mismatches,
        'absolute_time_error_s_quantiles': dict(zip(['median', 'p90', 'p99', 'max'], np.quantile(errors, [.5, .9, .99, 1.]).tolist())),
        'scope': 'whole original NAVSIM training population; no label/development score selection'}
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps({k: v if k != 'mismatches' else len(v) for k,v in report.items()}), flush=True)


if __name__ == '__main__': main()
