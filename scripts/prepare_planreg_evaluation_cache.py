#!/usr/bin/env python3
"""Repackage immutable PlanReg banks for the validated per-log PDM evaluator.

No inference, candidate selection, or metric computation takes place here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--banks', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-scenes', type=int, default=12146)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    payloads = []
    predictions = {}
    for path in args.banks:
        with np.load(path, allow_pickle=False) as source:
            data = {name: source[name] for name in source.files}
        tokens = data['tokens'].astype(str)
        if data['proposals'].shape != (len(tokens), 64, 8, 3):
            raise ValueError('Invalid proposal shape')
        if not np.isfinite(data['proposals']).all() or not np.isfinite(data['predicted_log_pdm']).all():
            raise ValueError('Nonfinite model prediction')
        if not np.array_equal(data['selected_indices'], data['predicted_log_pdm'].argmax(axis=1)):
            raise ValueError('Stored selections differ from original scorer argmax')
        for i, token in enumerate(tokens):
            if token in predictions:
                raise ValueError(f'Duplicate scene {token}')
            predictions[token] = {'proposals': data['proposals'][i],
                                  'predicted_scores': data['predicted_log_pdm'][i]}
        payloads.append(data)
    if len(predictions) != args.expected_scenes:
        raise ValueError(f'Expected {args.expected_scenes} scenes, found {len(predictions)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        pickle.dump(predictions, stream, protocol=pickle.HIGHEST_PROTOCOL)
    merged_path = None
    if len(payloads) > 1:
        count = len(payloads[0]['tokens'])
        static = {'component_names', 'official_component_names'}
        arrays = {}
        for name in payloads[0]:
            if name in static:
                if any(not np.array_equal(row[name], payloads[0][name]) for row in payloads):
                    raise ValueError(f'Static field differs: {name}')
                arrays[name] = payloads[0][name]
            else:
                if not all(row[name].ndim and len(row[name]) == len(row['tokens']) for row in payloads):
                    raise ValueError(f'Unexpected non-scene field: {name}')
                arrays[name] = np.concatenate([row[name] for row in payloads], axis=0)
        merged_path = args.output.with_name('candidate_bank.npz')
        with merged_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
    manifest = {'schema_version': 1, 'scene_count': len(predictions), 'candidate_count': 64,
                'source_banks': {str(p.resolve()): sha256(p) for p in args.banks},
                'output': str(args.output.resolve()), 'output_sha256': sha256(args.output),
                'selected_indices_preserved': True, 'proposal_coordinates_unchanged': True}
    if merged_path:
        manifest.update(candidate_bank=str(merged_path), candidate_bank_sha256=sha256(merged_path))
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
