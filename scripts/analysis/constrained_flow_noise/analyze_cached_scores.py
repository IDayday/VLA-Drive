"""Prefetch immutable score bytes; execute the unchanged original analyzer.

This changes file IO only. Every original seal/identity/denominator check and
all metric, motion and bootstrap calculations still execute. No actor/scorer
or training input cache is involved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import time

import torch

from scripts.analysis.flow_safe_improvement import evaluate_policy as score_reader
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_ode import analyze
from .lock_final import read


@contextmanager
def cached_score_bytes(roots, population, workers=16):
    started = time.monotonic()
    paths = []
    for root in dict.fromkeys(os.path.abspath(x) for x in roots):
        root = Path(root)
        if not (root/'COMPLETE').is_file():
            raise ValueError('Only immutable complete score banks can be prefetched')
        manifest = read(root/'manifest.json')
        sample = manifest['sample_manifest']
        tokens = sample.get('explicit_tokens') if sample['cohort'] == 'diagnostic_tokens' else population[sample['cohort']]
        paths.append(root/'manifest.json')
        for token in tokens:
            path = root/'scenes'/token[:2]/(token+'.pt')
            paths.extend((path, path.with_suffix('.json')))
    def fetch(path):
        payload = path.read_bytes()
        return str(path), payload, hashlib.sha256(payload).hexdigest()
    cache, hashes = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, payload, digest in pool.map(fetch, paths):
            cache[name], hashes[name] = payload, digest
    evidence = dict(files=len(cache), bytes=sum(map(len,cache.values())), workers=workers,
                    prefetch_seconds=time.monotonic()-started,
                    immutable_file_snapshot_sha256=fingerprint(hashes))
    native_load, native_sha, native_read = torch.load, score_reader.file_sha, Path.read_text
    def load(path, *args, **kwargs):
        key = os.path.abspath(path) if isinstance(path,(str,os.PathLike)) else None
        return native_load(io.BytesIO(cache[key]) if key in cache else path, *args, **kwargs)
    def sha(path):
        key = os.path.abspath(path)
        return hashes[key] if key in hashes else native_sha(path)
    def read_text(path, *args, **kwargs):
        key = os.path.abspath(path)
        if key not in cache:
            return native_read(path,*args,**kwargs)
        encoding = kwargs.get('encoding') or (args[0] if args else None) or 'utf-8'
        errors = kwargs.get('errors') or (args[1] if len(args)>1 else None) or 'strict'
        return cache[key].decode(encoding,errors)
    try:
        torch.load, score_reader.file_sha, Path.read_text = load, sha, read_text
        yield evidence
    finally:
        torch.load, score_reader.file_sha, Path.read_text = native_load, native_sha, native_read


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('scores','baseline-scores','population','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--compare-existing', help='Require exact complete original report parity')
    parser.add_argument('--workers',type=int,default=16)
    args = parser.parse_args()
    require_dispatch()
    if not 1<=args.workers<=32:
        raise ValueError('Bounded prefetch concurrency required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Analysis snapshots are immutable')
    output.mkdir(parents=True)
    population = read(args.population)
    started = time.monotonic()
    with cached_score_bytes([args.scores,args.baseline_scores],population,args.workers) as evidence:
        core_started = time.monotonic()
        result = analyze(args.scores,args.baseline_scores,population)
        evidence['original_analyzer_seconds'] = time.monotonic()-core_started
    evidence.update(wall_seconds=time.monotonic()-started, source_sha256=file_sha(__file__),
                    original_analyzer_unchanged=True, report_sha256=result['sha256'])
    if args.compare_existing:
        original = read(args.compare_existing)
        if result != original:
            atomic_json(output/'parity_failure.json',dict(equal=False,
                        different_keys=[k for k in set(result)|set(original) if result.get(k)!=original.get(k)]))
            raise ValueError('Full report differs from the original completed analyzer')
        evidence.update(exact_original_report_match=True, original_report_sha256=original['sha256'])
    atomic_json(output/'report.json',result)
    atomic_json(output/'cost.json',evidence)
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    main()
