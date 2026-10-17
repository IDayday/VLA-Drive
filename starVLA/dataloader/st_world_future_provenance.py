"""Small immutable raw-log ledger for new future-only request preparation.

Validates metadata, not raw logs/images. Builders verify raw bytes before issuing
this ledger. Existing request files with inline source_logs are unchanged.
"""
import hashlib
import json
from pathlib import Path

from .st_world_history import file_sha256
from .st_world_schema import identity_hash, require_hash

LEDGER_SCHEMA = 'st_raw_log_provenance_v1'
REF_SCHEMA = 'st_raw_log_provenance_ref_v1'
MAX_LEDGER_BYTES = 4 << 20
FIELDS = ('source_student_identity', 'source_index_sha256', 'partition_sha256',
          'registration_identity', 'selection_identity')


class LogCoverage:
    """Bounded one digest/count per registered log; retain original token order."""
    def __init__(self):
        self.entries = {}

    def add(self, token, log):
        if not isinstance(token, str) or not token or not isinstance(log, str) or not log:
            raise ValueError('Invalid scene/log coverage')
        if log not in self.entries:
            if len(self.entries) >= 20000:
                raise ValueError('Raw-log coverage exceeds metadata bound')
            self.entries[log] = [hashlib.sha256(b'['), 0]
        digest, count = self.entries[log]
        digest.update(((',' if count else '') + json.dumps(token)).encode())
        self.entries[log][1] += 1

    def records(self):
        result = []
        for log, (digest, count) in self.entries.items():
            final = digest.copy(); final.update(b']')
            result.append({'log': log, 'scenes': count, 'tokens_sha256': final.hexdigest()})
        return result


def validate_ledger(manifest, *, coverage=None):
    """Return verified file dependency, or None for the unmodified legacy path."""
    ref = manifest.get('source_logs_ledger')
    if ref is None:
        return None
    if 'source_logs' in manifest:
        raise ValueError('Cannot mix inline and ledger raw-log provenance')
    if (not isinstance(ref, dict) or set(ref) != {'schema','ledger_schema','path','sha256','identity'} or
            ref['schema'] != REF_SCHEMA or ref['ledger_schema'] != LEDGER_SCHEMA):
        raise ValueError('Unknown raw-log ledger reference schema')
    path = Path(ref['path'])
    if not path.is_absolute() or not 0 < path.stat().st_size <= MAX_LEDGER_BYTES:
        raise ValueError('Raw-log ledger path/size outside metadata bound')
    before = path.stat()
    actual = file_sha256(path)
    if actual != require_hash(ref['sha256'], 'raw-log ledger SHA'):
        raise ValueError('Raw-log ledger checksum mismatch')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError('Duplicate ledger JSON key')
            result[key] = value
        return result
    body = json.loads(path.read_text(), object_pairs_hook=pairs)
    after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (
            after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Raw-log ledger changed during verification')
    identity = require_hash(ref['identity'], 'raw-log ledger identity')
    if (body.get('schema') != LEDGER_SCHEMA or body.get('identity') != identity or
            identity_hash({k:v for k,v in body.items() if k != 'identity'}) != identity):
        raise ValueError('Raw-log ledger semantic identity mismatch')
    if body.get('split') != manifest['split'] or body['split'] not in ('train','dev'):
        raise ValueError('Raw-log ledger role mismatch')
    for field in FIELDS:
        if require_hash(body.get(field), field) != require_hash(manifest.get(field), field):
            raise ValueError('Raw-log ledger source/registration mismatch: '+field)
    rows = body.get('logs')
    if not isinstance(rows, list) or not 0 < len(rows) <= 20000:
        raise ValueError('Raw-log ledger requires registered logs')
    summaries, seen = [], set()
    for row in rows:
        log = row.get('log')
        if not isinstance(log,str) or not log or Path(log).name != log or log in seen:
            raise ValueError('Duplicate/invalid registered raw log')
        seen.add(log)
        if (not Path(row['path']).is_absolute() or Path(row['path']).name != log+'.pkl' or
                type(row.get('bytes')) is not int or row['bytes'] <= 0 or
                type(row.get('scenes')) is not int or row['scenes'] <= 0 or
                type(row.get('mtime_ns')) is not int):
            raise ValueError('Invalid raw-log provenance record')
        require_hash(row.get('sha256'), 'original raw-log SHA')
        require_hash(row.get('tokens_sha256'), 'original ordered log tokens')
        summaries.append({k:row[k] for k in ('log','scenes','tokens_sha256')})
    if coverage is not None and summaries != coverage:
        raise ValueError('Raw-log ledger does not cover exact ordered scene/log population')
    return {'path':str(path), 'sha256':actual}
