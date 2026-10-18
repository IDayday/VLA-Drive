"""Bounded read-only JSON metadata census; never open source images/models/logs."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import time

ROOT = Path('/mnt/project/st-world-policy-20261016/audits/history_data/full_causal_population_v1/train')
MAX_BYTES = 1280 << 20
MAX_SECONDS = 120
RATE = 32 << 20


def main():
    started = time.monotonic()
    io_before = Path('/proc/self/io').read_text()
    shards = sorted((ROOT / 'history_shards').iterdir())
    if not 1 <= len(shards) <= 4096:
        raise ValueError('Unbounded shard count')
    rows = []
    tokens, logs = set(), set()
    decoder = json.JSONDecoder()
    total = 0
    for ordinal, p in enumerate(shards):
        before = p.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                not re.fullmatch(r'[0-9]{6}_[0-9a-f]{16}\.jsonseq', p.name) or
                int(p.name[:6]) != ordinal or not 0 < before.st_size <= 4 << 20):
            raise ValueError('Noncontiguous, unsafe, oversized or partial shard')
        total += before.st_size
        if total > MAX_BYTES or time.monotonic() - started > MAX_SECONDS:
            raise ValueError('Read/time cap')
        # Bound aggregate metadata reads without changing global page-cache state.
        delay = total / RATE - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        with p.open('rb') as stream:
            raw = stream.read((4 << 20) + 1)
        after = p.stat()
        version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if version(before) != version(after) or len(raw) != before.st_size:
            raise ValueError('Shard changed during census')
        data = raw.decode('utf-8')
        pos, count, log = 0, 0, None
        while pos < len(data):
            record, end = decoder.raw_decode(data, pos)
            token = record['token']
            if record.get('schema') != 'st_causal_history_v1' or token in tokens:
                raise ValueError('Unknown record or duplicate scene')
            if log is not None and log != record['log']:
                raise ValueError('Cross-log shard')
            log = record['log']
            if hashlib.sha256(log.encode()).hexdigest()[:16] != p.stem.split('_')[1]:
                raise ValueError('Log filename mismatch')
            tokens.add(token)
            logs.add(log)
            count += 1
            pos = end
            if data[pos:pos + 1] != '\n':
                raise ValueError('Record boundary mismatch')
            pos += 1
        if not 1 <= count <= 128:
            raise ValueError('Record count per shard')
        rows.append({'name': p.name, 'bytes': len(raw), 'scenes': count,
                     'sha256': hashlib.sha256(raw).hexdigest(),
                     'mtime_ns': before.st_mtime_ns})
    print(json.dumps({'schema': 'st_partial_history_metadata_census_v1',
        'status': 'CENSUS_COMPLETE_NOT_INPUT_QUALIFICATION', 'root': str(ROOT),
        'shards': len(rows), 'scenes': len(tokens), 'logs': len(logs),
        'expected_scenes': 101592, 'missing_scene_count': 101592 - len(tokens),
        'metadata_bytes_read': total, 'rows': rows,
        'source_images_logs_models_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'max_seconds': MAX_SECONDS, 'max_bytes': MAX_BYTES,
        'io_before': io_before, 'io_after': Path('/proc/self/io').read_text()}, sort_keys=True))


if __name__ == '__main__':
    main()
