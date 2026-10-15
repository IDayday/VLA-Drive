"""Byte-identical temporary mirror of completed versioned repair labels.

Only storage location changes. Native training continues to validate every
record, tensor, identity and checksum. Canonical data remain on shared storage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha
from starVLA.model.modules.decision_local.minimal_repair_contract import SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-labels', type=Path, nargs='+', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    resources = check_host()
    started, cpu_started = time.monotonic(), time.process_time()
    target = a.destination.resolve()
    if not str(target).startswith('/dev/shm/decision-local-repair-labels-') or target.exists():
        raise ValueError('A new task-specific RAM directory is required')
    if subprocess.check_output(['stat','-f','-c','%T',str(target.parent)], text=True).strip() != 'tmpfs':
        raise ValueError('Container disk is not an eligible cache destination')
    if not str(a.receipt.resolve()).startswith('/mnt/project/') or a.receipt.exists():
        raise ValueError('A new durable shared receipt is required')
    if a.max_seconds <= 0 or not 1 <= a.workers <= 16:
        raise ValueError('Finite bounded staging is required')
    files, sources, total = [], [], 0
    for source in a.source_labels:
        source = source.resolve()
        if not str(source).startswith('/mnt/project/minimal-repair-artifacts/'):
            raise ValueError('Only current-task versioned labels are eligible')
        meta, complete = read(source/'identity.json'), read(source/'COMPLETE.json')
        if meta['schema'] != SCHEMA or complete['identity'] != meta['identity'] or (source/'CORRECTNESS_HOLD.json').exists():
            raise ValueError('Incomplete, foreign or held labels cannot be mirrored')
        if any(source.name == x.name for x in sources):
            raise ValueError('Duplicate source name')
        tokens = [x['token'] for x in meta['index']]
        if len(tokens) != len(set(tokens)) or complete['scenes'] != len(tokens):
            raise ValueError('Population mismatch')
        for token in tokens:
            if not token or any(c not in '0123456789abcdef' for c in token):
                raise ValueError('Unexpected token path')
            record, data = source/'records'/(token+'.json'), source/'scenes'/(token+'.npz')
            total += record.stat().st_size + data.stat().st_size
            files.append((source, token, meta['identity']))
        sources.append(source)
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
    if total > 16*1024**3 or min(available, shutil.disk_usage(target.parent).free) < 2*total+64*1024**3:
        raise ValueError('Insufficient RAM headroom for the bounded label set')
    target.mkdir(mode=0o750)
    for source in sources:
        for sub in ('records','scenes'):
            (target/source.name/sub).mkdir(parents=True)
        shutil.copyfile(source/'identity.json',target/source.name/'identity.json')
    def copy_row(item):
        source, token, ident = item
        if time.monotonic()-started >= a.max_seconds:
            raise TimeoutError('Finite staging budget reached; no COMPLETE published')
        record_bytes = (source/'records'/(token+'.json')).read_bytes()
        record = json.loads(record_bytes)
        if record['identity'] != ident:
            raise ValueError('Foreign record')
        blob = (source/'scenes'/(token+'.npz')).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record['sha256']:
            raise ValueError('Changed source labels')
        dst = target/source.name
        (dst/'scenes'/(token+'.npz')).write_bytes(blob)
        (dst/'records'/(token+'.json')).write_bytes(record_bytes)
        if digest(dst/'scenes'/(token+'.npz')) != record['sha256'] or (dst/'records'/(token+'.json')).read_bytes() != record_bytes:
            raise ValueError('RAM copy differs')
        return len(blob)+len(record_bytes)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        copied = sum(pool.map(copy_row, files))
    if copied != total:
        raise ValueError('Incomplete copied bytes')
    for source in sources:
        dst = target/source.name
        if digest(source/'identity.json') != digest(dst/'identity.json'):
            raise ValueError('Metadata copy differs')
        shutil.copyfile(source/'COMPLETE.json',dst/'COMPLETE.json')
    atomic_json(a.receipt, dict(complete=True, source_sha=source_sha(ROOT), tool_sha256=digest(__file__),
        resources=resources, canonical_sources=[str(x) for x in sources],
        temporary_RAM_labels=[str(target/x.name) for x in sources],
        identities={x.name:read(x/'identity.json')['identity'] for x in sources},
        rows=len(files), bytes=total, seconds=time.monotonic()-started, CPU_process_seconds=time.process_time()-cpu_started,
        all_data_and_record_bytes_exact=True, new_container_disk_bytes=0,
        training_model_source_or_scientific_schema_changed=False,
        lifetime='Temporary RAM only; canonical data and receipts remain on shared /mnt/project; discard after readers exit'))


if __name__ == '__main__':
    main()
