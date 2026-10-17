"""Metadata-only index for existing ST teacher requests and COMPLETE caches.

Accepts the repository's sorted, indent=2 atomic_json format, not arbitrary JSON
streams. Each row is decoded by stdlib json; no images or tensor files are
opened. The original semantic identities and raw file hashes remain authoritative.
The sidecar is execution metadata, not a new teacher cache or training approval.
"""
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import sys
import uuid
import zlib

from .st_world_history import file_sha256
from .st_world_future_provenance import LogCoverage, validate_ledger
from .st_world_schema import TeacherFeatureContract, identity_hash, require_hash
from .st_world_targets import atomic_json, file_lock, payload_key, validate_target_manifest

MiB = 1 << 20
GiB = 1 << 30
SCHEMA = 'st_teacher_metadata_index_v1'
ENCODING = 'canonical_json_zlib_level1_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decoded_bytes(value):
    """Bound decoded JSON containers; returned/caller-retained rows are not a cache."""
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(decoded_bytes(k)+decoded_bytes(v) for k,v in value.items())
    elif isinstance(value, list):
        total += sum(decoded_bytes(v) for v in value)
    return total


def pack(text):
    # Only JSON metadata bytes; never image/teacher tensors or lossy quantization.
    return zlib.compress(text.encode(), level=1)


def unpack(body):
    if not isinstance(body, bytes) or len(body) > 64*1024:
        raise ValueError('Indexed metadata encoding/size mismatch')
    decoder = zlib.decompressobj()
    plain = decoder.decompress(body, 64*1024+1)
    if len(plain) > 64*1024 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Indexed metadata decompression exceeds bounded row or is incomplete')
    return plain.decode('utf-8')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _loads(text):
    value = json.loads(text, object_pairs_hook=_pairs)
    canonical(value)  # Reject NaN/Infinity, which stdlib decoding otherwise permits.
    return value


def version(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def checked_sha(path):
    stamp = version(path)
    digest = file_sha256(path)
    if version(path) != stamp:
        raise ValueError('Metadata changed while hashing')
    return digest, stamp


def _atomic_rows(path, arrays, consume, *, max_row_bytes=64*1024):
    """Read only the exact atomic_json layout; retain at most one bounded row.

    This intentionally rejects compact/reformatted external files instead of
    silently switching to a full json.load. It never rewrites source metadata.
    """
    header, counts, digest = {}, {}, hashlib.sha256()
    digest.update(b'{')
    first, previous, total_header = True, None, 0
    with Path(path).open('r', encoding='utf-8', newline='') as stream:
        def line():
            value = stream.readline(max_row_bytes + 1)
            if len(value.encode()) > max_row_bytes:
                raise ValueError('Metadata line exceeds bounded row budget')
            return value

        if line() != '{\n':
            raise ValueError('Metadata requires sorted indent=2 atomic_json format')
        while True:
            text = line()
            if text == '}\n':
                break
            if not text.startswith('  "') or ': ' not in text:
                raise ValueError('Invalid atomic_json top-level field')
            key_text, value_text = text[2:].split(': ', 1)
            key = _loads(key_text)
            if not isinstance(key, str) or (previous is not None and key <= previous):
                raise ValueError('Metadata top-level keys must be unique and sorted')
            previous = key
            include = key != 'identity'
            if include:
                digest.update(('' if first else ',').encode())
                digest.update((canonical(key) + ':').encode())
                first = False
            if key in arrays and value_text == '[\n':
                if include:
                    digest.update(b'[')
                count = 0
                while True:
                    row_text = line()
                    if row_text in ('  ],\n', '  ]\n'):
                        comma = row_text == '  ],\n'
                        break
                    if not row_text.startswith('    {'):
                        raise ValueError('Indexed arrays must contain atomic_json object rows')
                    pieces, size = [row_text], len(row_text.encode())
                    while pieces[-1] not in ('    },\n', '    }\n'):
                        next_line = line()
                        if not next_line:
                            raise ValueError('Truncated metadata row')
                        pieces.append(next_line)
                        size += len(next_line.encode())
                        if size > max_row_bytes:
                            raise ValueError('Metadata row exceeds bounded row budget')
                    body = ''.join(pieces).rstrip()
                    row_comma = body.endswith(',')
                    row = _loads(body[:-1] if row_comma else body)
                    encoded = canonical(row)
                    if include:
                        digest.update((',' if count else '').encode())
                        digest.update(encoded.encode())
                    consume(key, count, row, encoded)
                    count += 1
                    if not row_comma:
                        closing = line()
                        if closing not in ('  ],\n', '  ]\n'):
                            raise ValueError('Missing metadata array closing delimiter')
                        comma = closing == '  ],\n'
                        break
                counts[key] = count
                if include:
                    digest.update(b']')
            else:
                pieces, size = [value_text], len(value_text.encode())
                while True:
                    candidate = ''.join(pieces).rstrip()
                    comma = candidate.endswith(',')
                    try:
                        value = _loads(candidate[:-1] if comma else candidate)
                        break
                    except json.JSONDecodeError:
                        next_line = line()
                        if not next_line:
                            raise ValueError('Truncated metadata field')
                        pieces.append(next_line)
                        size += len(next_line.encode())
                        if size > max_row_bytes:
                            raise ValueError('Metadata field exceeds bounded header budget')
                if key in arrays:
                    if value != []:
                        raise ValueError('Indexed arrays require atomic_json row layout')
                    counts[key] = 0
                else:
                    total_header += len(canonical(value).encode())
                    if total_header > max_row_bytes:
                        raise ValueError('Metadata header exceeds bounded header budget')
                    header[key] = value
                if include:
                    digest.update(canonical(value).encode())
            if not comma:
                if line() != '}\n':
                    raise ValueError('Invalid metadata closing delimiter')
                break
        if stream.read(1):
            raise ValueError('Trailing metadata content')
    digest.update(b'}')
    return header, counts, digest.hexdigest()


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux qualification host.


def _limit(value, name, upper):
    if type(value) is not int or not 0 < value <= upper:
        raise ValueError(f'{name} must be a positive integer <= {upper}')


def build_index(output, *, requests, expected_request_identity, contract,
                cache_root=None, expected_cache_identity=None, max_scenes,
                max_disk_bytes=2*GiB, max_builder_rss_bytes=4*GiB):
    """Stream one full declared manifest, never subset it or change row order.

    max_scenes is an upper bound, not sampling. Missing/truncated inputs fail.
    Interrupted temporary DBs remain evidence; only COMPLETE admits readers.
    """
    _limit(max_scenes, 'max_scenes', 2_000_000)
    _limit(max_disk_bytes, 'max_disk_bytes', 2*GiB)
    _limit(max_builder_rss_bytes, 'max_builder_rss_bytes', 4*GiB)
    require_hash(expected_request_identity, 'request identity')
    contract = contract.validate()
    requests, output = Path(requests).resolve(), Path(output).resolve()
    paths = {'requests': requests}
    if cache_root is not None:
        cache_root = Path(cache_root).resolve()
        require_hash(expected_cache_identity, 'cache identity')
        if requests != cache_root / 'requests.json':
            raise ValueError('Cache adapter must index its own exact request file')
        paths.update(identity=cache_root/'identity.json', complete=cache_root/'COMPLETE.json')
    elif expected_cache_identity is not None:
        raise ValueError('Cache identity without a cache root')
    dependencies = {k: {'path': str(p), 'sha256': checked_sha(p)[0]} for k, p in paths.items()}
    if _rss() > max_builder_rss_bytes:
        raise ValueError('Builder RSS already exceeds registered budget')
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output/'BUILD.lock'):
        if (output/'COMPLETE.json').exists():
            receipt_path = output/'COMPLETE.json'
            if receipt_path.stat().st_size > MiB:
                raise ValueError('Oversized existing metadata receipt')
            prior = _loads(receipt_path.read_text())
            with MetadataIndex(output, expected_request_identity=expected_request_identity,
                               expected_index_identity=prior['identity']) as existing:
                ledger = validate_ledger(existing.receipt['manifest_header'])
                expected_dependencies = dict(dependencies)
                if ledger is not None: expected_dependencies['source_logs_ledger'] = ledger
                if (existing.receipt['dependencies'] != expected_dependencies or
                        existing.receipt['contract'] != contract.to_dict() or
                        existing.receipt['cache_identity'] != expected_cache_identity or
                        existing.receipt['scene_count'] > max_scenes or
                        sum(p.stat().st_size for p in output.glob('*') if p.is_file()) > max_disk_bytes):
                    raise ValueError('Immutable metadata index conflict')
                return existing.receipt
        temporary = output / ('metadata.' + uuid.uuid4().hex + '.partial.sqlite')
        db = sqlite3.connect(temporary)
        db.execute('PRAGMA page_size=4096')
        # Leave >=1/2 of the disk budget for rollback journal and publication metadata.
        db.execute(f'PRAGMA max_page_count={max_disk_bytes//(2*4096)}')
        db.execute('PRAGMA cache_size=-8192')
        db.execute('PRAGMA mmap_size=0')
        db.execute('PRAGMA temp_store=MEMORY')
        db.executescript('''
          CREATE TABLE scenes(pos INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, body BLOB NOT NULL, sha TEXT NOT NULL);
          CREATE TABLE records(pos INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, body BLOB NOT NULL, sha TEXT NOT NULL);
          CREATE TABLE expected(key TEXT PRIMARY KEY, body BLOB NOT NULL, sha TEXT NOT NULL) WITHOUT ROWID;
          CREATE TABLE receipts(shard INTEGER PRIMARY KEY, sha TEXT NOT NULL);
        ''')
        checks = 0
        log_coverage = LogCoverage()
        def budget():
            nonlocal checks
            checks += 1
            if checks % 128 == 0:
                db.commit()
                if _rss() > max_builder_rss_bytes:
                    raise ValueError('Builder exceeded registered RSS budget')
                if sum(p.stat().st_size for p in output.glob('*') if p.is_file()) > max_disk_bytes:
                    raise ValueError('Metadata directory exceeds registered disk budget, including orphan evidence')

        def consume(key, pos, row, encoded):
            budget()
            sha = hashlib.sha256(encoded.encode()).hexdigest()
            if key == 'scenes':
                if pos >= max_scenes:
                    raise ValueError('Request population exceeds max_scenes; no partial publication')
                db.execute('INSERT INTO scenes VALUES (?,?,?,?)', (pos, row['token'], pack(encoded), sha))
            elif key == 'records':
                require_hash(row.get('key'), 'record key')
                db.execute('INSERT INTO records VALUES (?,?,?,?)', (pos, row['key'], pack(encoded), sha))
            elif key == 'shards':
                if type(row.get('shard')) is not int or row['shard'] != pos:
                    raise ValueError('Shard receipts must retain complete contiguous order')
                db.execute('INSERT INTO receipts VALUES (?,?)',
                           (pos, require_hash(row.get('receipt_sha256'), 'shard receipt hash')))
        try:
            manifest, counts, digest = _atomic_rows(requests, {'scenes'}, consume)
            if manifest.get('identity') != expected_request_identity or digest != expected_request_identity:
                raise ValueError('Original request semantic identity mismatch')
            for name in ('source_student_identity', 'source_index_sha256', 'partition_sha256'):
                require_hash(manifest.get(name), name)
            ledger = validate_ledger(manifest)
            if ledger is not None:
                dependencies['source_logs_ledger'] = ledger
                paths['source_logs_ledger'] = Path(ledger['path'])
            if tuple(manifest['view_names']) != contract.view_names:
                raise ValueError('Teacher/request camera contract differs')
            # Reuse the original per-scene validator without ever materializing all scenes.
            single = dict(manifest, scenes=[])
            single['identity'] = identity_hash({k:v for k,v in single.items() if k != 'identity'})
            validate_target_manifest(single)
            for (encoded,) in db.execute('SELECT body FROM scenes ORDER BY pos'):
                scene = _loads(unpack(encoded))
                single['scenes'] = [scene]
                single['identity'] = identity_hash({k:v for k,v in single.items() if k != 'identity'})
                validate_target_manifest(single)
                if ledger is not None: log_coverage.add(scene['token'], scene['log'])
                for frame in scene['frames']:
                    for ref in frame['views']:
                        if ref is not None:
                            key = payload_key(ref, contract)
                            record = canonical({'key': key, 'image': ref})
                            db.execute('INSERT OR IGNORE INTO expected VALUES (?,?,?)',
                                       (key, pack(record), hashlib.sha256(record.encode()).hexdigest()))
                budget()
            if ledger is not None:
                validate_ledger(manifest, coverage=log_coverage.records())
            metadata, complete = None, None
            if cache_root is not None:
                metadata, record_counts, digest = _atomic_rows(paths['identity'], {'records'}, consume)
                complete, receipt_counts, _ = _atomic_rows(paths['complete'], {'shards'}, consume)
                if (metadata.get('schema') != 'st_teacher_cache_v1' or digest != expected_cache_identity or
                        metadata.get('identity') != expected_cache_identity or complete.get('identity') != expected_cache_identity or
                        complete.get('status') != 'COMPLETE' or metadata['request_manifest_identity'] != expected_request_identity or
                        complete.get('identity_sha256') != dependencies['identity']['sha256'] or
                        complete.get('requests_sha256') != dependencies['requests']['sha256'] or
                        TeacherFeatureContract.from_dict(metadata['contract']).identity != contract.identity):
                    raise ValueError('Original COMPLETE teacher cache/hash/contract mismatch')
                n = record_counts.get('records', -1)
                ips = metadata['images_per_shard']
                if (type(ips) is not int or not 1 <= ips <= 1024 or
                        counts.get('scenes') != complete['scenes'] or n != complete['unique_images'] or
                        receipt_counts.get('shards') != metadata['shards'] or metadata['shards'] != (n+ips-1)//ips):
                    raise ValueError('Original COMPLETE coverage/count mismatch')
                # Exact first-reference semantics and sorted content keys, not only set membership.
                expected_iter = db.execute('SELECT key,body FROM expected ORDER BY key')
                actual_iter = db.execute('SELECT key,body FROM records ORDER BY pos')
                from itertools import zip_longest
                if any(a != b for a,b in zip_longest(expected_iter, actual_iter)):
                    raise ValueError('Teacher records differ from exact original request coverage/order')
                db.execute('DROP TABLE expected')
            else:
                # Requests have no source shard offsets. Their canonical record
                # order is sorted content key; retain the existing SQL key index
                # directly instead of duplicating the entire expected-record table.
                db.execute('DROP TABLE records')
                db.execute('ALTER TABLE expected RENAME TO records')
            db.commit()
            record_count = db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            db.close()
            for k, path in paths.items():
                if file_sha256(path) != dependencies[k]['sha256']:
                    raise ValueError('Source metadata changed during build')
            if _rss() > max_builder_rss_bytes:
                raise ValueError('Builder exceeded registered RSS budget')
            total_bytes = sum(p.stat().st_size for p in output.glob('*') if p.is_file())
            if total_bytes > max_disk_bytes:
                raise ValueError('Metadata directory exceeds registered disk budget')
            final = output/'metadata.sqlite'
            if final.exists():
                # An unpublished DB from the checkpoint/receipt window is retained, never trusted.
                final.rename(output/('metadata.orphan.'+uuid.uuid4().hex+'.sqlite'))
            os.replace(temporary, final)
            with final.open('rb') as f:
                os.fsync(f.fileno())
            receipt = {'schema': SCHEMA, 'status': 'COMPLETE', 'encoding': ENCODING,
                'builder_source_sha256': file_sha256(__file__),
                'dependencies': dependencies, 'request_identity': expected_request_identity,
                'cache_identity': expected_cache_identity, 'cache_root': str(cache_root) if cache_root else None,
                'contract': contract.to_dict(), 'manifest_header': manifest, 'cache_header': metadata,
                'complete_header': complete, 'scene_count': counts['scenes'], 'record_count': record_count,
                'database_sha256': file_sha256(final), 'database_bytes': final.stat().st_size,
                'max_disk_bytes': max_disk_bytes, 'max_builder_rss_bytes': max_builder_rss_bytes,
                'builder_peak_rss_bytes': _rss(), 'tensor_files_opened': 0, 'image_files_opened': 0,
                'tensor_disk_bytes': 0, 'coverage': 'exact_original_manifest', 'training_admission': False}
            receipt['identity'] = identity_hash(receipt)
            if len(canonical(receipt).encode()) > 64*1024 or decoded_bytes(receipt) > MiB:
                raise ValueError('Index receipt exceeds bounded reader header budget')
            if sum(p.stat().st_size for p in output.glob('*') if p.is_file()) + len(json.dumps(receipt,sort_keys=True,indent=2).encode()) + 1 > max_disk_bytes:
                raise ValueError('Metadata receipt would exceed registered disk budget')
            atomic_json(output/'COMPLETE.json', receipt, immutable=True)
            return receipt
        except BaseException:
            db.close()
            raise  # Keep partial DB as evidence; no misleading COMPLETE.


class MetadataIndex:
    """Read-only SQLite, bounded page cache, no population-sized Python containers."""
    def __init__(self, root, *, expected_request_identity, expected_index_identity,
                 max_metadata_cache_bytes=16*MiB):
        _limit(max_metadata_cache_bytes, 'max_metadata_cache_bytes', 256*MiB)
        if max_metadata_cache_bytes < 16*MiB:
            raise ValueError('Metadata budget must cover headers plus decoded rows and SQLite pages (>=16 MiB)')
        require_hash(expected_request_identity, 'request identity')
        require_hash(expected_index_identity, 'index identity')
        self.root = Path(root).resolve()
        receipt_path, path = self.root/'COMPLETE.json', self.root/'metadata.sqlite'
        if receipt_path.stat().st_size > MiB:
            raise ValueError('Oversized metadata receipt')
        receipt_digest, receipt_stamp = checked_sha(receipt_path)
        self.receipt = _loads(receipt_path.read_text())
        d = self.receipt
        database_digest, database_stamp = checked_sha(path)
        if (d.get('schema') != SCHEMA or d.get('status') != 'COMPLETE' or d.get('encoding') != ENCODING or
                d['identity'] != identity_hash({k:v for k,v in d.items() if k != 'identity'}) or
                d['request_identity'] != expected_request_identity or
                d['identity'] != expected_index_identity or decoded_bytes(d) > MiB or
                d['database_sha256'] != database_digest or d['database_bytes'] != database_stamp[2]):
            raise ValueError('Metadata index incomplete/identity/checksum mismatch')
        if checked_sha(receipt_path) != (receipt_digest, receipt_stamp):
            raise ValueError('Metadata receipt changed while loading')
        self._versions = {path: database_stamp, receipt_path: receipt_stamp}
        for dep in d['dependencies'].values():
            p = Path(dep['path'])
            digest, stamp = checked_sha(p)
            if digest != dep['sha256']:
                raise ValueError('Original metadata changed after indexing')
            self._versions[p] = stamp
        ledger = validate_ledger(d['manifest_header'])
        if ledger is not None and d['dependencies'].get('source_logs_ledger') != ledger:
            raise ValueError('Index does not bind verified raw-log ledger dependency')
        self.db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True, cached_statements=16)
        self.max_metadata_cache_bytes = max_metadata_cache_bytes
        # Half the declared allowance remains for bounded decoded rows/headers/statements.
        self.db.execute(f'PRAGMA cache_size=-{max_metadata_cache_bytes//2048}')
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('PRAGMA temp_store=MEMORY')
        self.manifest = dict(d['manifest_header'], scenes=Rows(self, 'scenes', d['scene_count']))
        self.records = Rows(self, 'records', d['record_count'])
        self.scenes = Lookup(self, 'scenes', 'token', d['scene_count'])
        self.locations = Lookup(self, 'records', 'key', d['record_count'], location=True) if d['cache_header'] else None
        self.receipt_hashes = Lookup(self, 'receipts', 'shard', (d['cache_header'] or {}).get('shards', 0))

    def check(self):
        for path, stamp in self._versions.items():
            if version(path) != stamp:
                raise ValueError('Immutable metadata file version changed: '+str(path))

    def row(self, table, column, key):
        self.check()
        row = self.db.execute(f'SELECT body,sha FROM {table} WHERE {column}=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return self.decode_row(row)

    def decode_row(self, row):
        plain = unpack(row[0])
        if hashlib.sha256(plain.encode()).hexdigest() != row[1]:
            raise ValueError('Indexed metadata record corruption')
        decoded = _loads(plain)
        if decoded_bytes(decoded) > MiB:
            raise ValueError('Decoded metadata row exceeds RAM budget')
        return decoded

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Rows(Sequence):
    def __init__(self, index, table, count):
        self.index, self.table, self.count = index, table, count
        self.order = 'key' if table == 'records' and index.receipt['cache_header'] is None else 'pos'

    def __len__(self):
        return self.count

    def __iter__(self):
        self.index.check()
        for row in self.index.db.execute(f'SELECT body,sha FROM {self.table} ORDER BY {self.order}'):
            self.index.check()
            yield self.index.decode_row(row)

    def __getitem__(self, pos):
        if isinstance(pos, slice):
            r = range(*pos.indices(self.count))
            if len(r) > 1024:
                raise ValueError('Unbounded metadata slicing is forbidden')
            result, size = [], 0
            for i in r:
                row = self[i]
                size += decoded_bytes(row)
                if size > 2*MiB:
                    raise ValueError('Metadata slice exceeds bounded decoded output budget')
                result.append(row)
            return result
        if pos < 0:
            pos += self.count
        if not 0 <= pos < self.count:
            raise IndexError(pos)
        if self.order == 'key':
            # Diagnostic ordinal access only; provider uses scenes[token], and
            # full verification uses the streaming O(N) iterator above.
            self.index.check()
            key = self.index.db.execute('SELECT key FROM records ORDER BY key LIMIT 1 OFFSET ?', (pos,)).fetchone()[0]
            return self.index.row('records', 'key', key)
        return self.index.row(self.table, 'pos', pos)


class Lookup(Mapping):
    def __init__(self, index, table, column, count, *, location=False):
        self.index, self.table, self.column, self.count, self.location = index, table, column, count, location

    def __len__(self):
        return self.count

    def __iter__(self):
        self.index.check()
        for (key,) in self.index.db.execute(f'SELECT {self.column} FROM {self.table} ORDER BY {self.column}'):
            yield key

    def __getitem__(self, key):
        if not self.location and self.table != 'receipts':
            return self.index.row(self.table, self.column, key)
        self.index.check()
        field = 'pos' if self.location else 'sha'
        row = self.index.db.execute(f'SELECT {field} FROM {self.table} WHERE {self.column}=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        if self.location:
            return divmod(row[0], self.index.receipt['cache_header']['images_per_shard'])
        return row[0]
