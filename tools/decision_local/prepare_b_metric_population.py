"""Prepare/assemble B-only indices around the unchanged official cache builder.

No model, label approximation, scoring rule, or training population is changed.
Existing verified full-reference caches are referenced without copying. The
remaining scenes go through the pinned native builder in a separate directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha

ROLES = ('B_predictor', 'B_refiner')
PROTOCOL_KEYS = ('code_sha', 'navsim_python_tree_sha256',
                 'nuplan_python_tree_sha256', 'numerical_versions')


def role_rows(source_audit):
    audit = read(source_audit)
    result = {}
    seen = set()
    for role in ROLES:
        path = source_audit.parent / (role + '.json')
        rows = read(path)
        if digest(path) != audit['index_sha256'][role]:
            raise ValueError('Changed registered role index: ' + role)
        if len(rows) != audit['role_populations'][role]['scenes']:
            raise ValueError('Incomplete role population')
        for row in rows:
            if (row['token'] in seen or
                    audit['roles'][row['log'].rsplit('_', 2)[0]] != role):
                raise ValueError('Duplicate scene or incorrect recording role')
            seen.add(row['token'])
        result[role] = rows
    return audit, result


def load_bank(root):
    status = read(root / 'status.json')
    if status['status'] != 'complete' or status['failed']:
        raise ValueError('Only complete native metric banks may be reused')
    meta = read(root / 'identity.json')
    with (root / 'scenes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        if row['status'] != 'ok' or row['token'] in lookup:
            raise ValueError('Invalid or duplicate metric record')
        lookup[row['token']] = row
    index = read(root / 'cache_index.json')
    if (len(rows) != status['scenes'] or len(index) != len(rows) or
            len({r['token'] for r in index}) != len(index) or
            any(any(lookup.get(r['token'], {}).get(k) != r[k]
                    for k in ('token', 'log', 'cache_path')) for r in index)):
        raise ValueError('Metric bank population mismatch')
    return meta, lookup


def verify_files(rows):
    def verify(row):
        if digest(row['cache_path']) != row['sha256']:
            raise ValueError('Changed official metric cache: ' + row['token'])
        return Path(row['cache_path']).stat().st_size
    with ThreadPoolExecutor(8) as pool:
        return sum(pool.map(verify, rows))


def prepare(args):
    audit, roles = role_rows(args.source_audit)
    proof = read(args.recording_proof)
    if not proof['population_ready'] or proof['source_audit_identity'] != audit['identity']:
        raise ValueError('Recording separation is not ready')
    meta, bank = load_bank(args.reuse_bank)
    population = [row for role in ROLES for row in roles[role]]
    reused, missing = [], []
    for row in population:
        if row['token'] not in bank:
            missing.append(row)
        else:
            cached = bank[row['token']]
            if cached['log'] != row['log']:
                raise ValueError('Reused token/log mismatch')
            reused.append(cached)
    reused_bytes = verify_files(reused)
    declaration = dict(schema='natural_b_metric_population_v1',
        purpose='offline official scoring and independent target execution only; never a model input',
        source_audit=str(args.source_audit), source_audit_identity=audit['identity'],
        source_audit_sha256=digest(args.source_audit),
        recording_proof=str(args.recording_proof), recording_proof_sha256=digest(args.recording_proof),
        roles={role: dict(scenes=len(rows), index_sha256=audit['index_sha256'][role])
               for role, rows in roles.items()}, population_identity=identity(population),
        reuse_bank=str(args.reuse_bank), reuse_identity_sha256=digest(args.reuse_bank / 'identity.json'),
        protocol={k: meta[k] for k in PROTOCOL_KEYS}, reused_scenes=len(reused),
        reused_bytes=reused_bytes, missing_scenes=len(missing), scenes=len(population),
        source_sha=source_sha(ROOT), tool_sha256=digest(__file__), original_caches_modified=False)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'population.json', population)
    atomic_json(args.output / 'missing_index.json', missing)
    atomic_json(args.output / 'reused_records.json', reused)
    declaration.update(population_sha256=digest(args.output / 'population.json'),
        missing_index_sha256=digest(args.output / 'missing_index.json'),
        reused_records_sha256=digest(args.output / 'reused_records.json'))
    atomic_json(args.output / 'POPULATION.json', dict(identity=identity(declaration), **declaration))
    print({k: declaration[k] for k in ('scenes', 'reused_scenes', 'missing_scenes', 'reused_bytes')})


def assemble(args):
    registration = read(args.output / 'POPULATION.json')
    if registration['identity'] != identity({k: v for k, v in registration.items() if k != 'identity'}):
        raise ValueError('Changed preparation identity')
    for name, field in [('population.json', 'population_sha256'),
                        ('missing_index.json', 'missing_index_sha256'),
                        ('reused_records.json', 'reused_records_sha256')]:
        if digest(args.output / name) != registration[field]:
            raise ValueError('Changed prepared population')
    audit_path = Path(registration['source_audit'])
    if digest(audit_path) != registration['source_audit_sha256']:
        raise ValueError('Changed role source')
    _, roles = role_rows(audit_path)
    old_root = Path(registration['reuse_bank'])
    if digest(old_root / 'identity.json') != registration['reuse_identity_sha256']:
        raise ValueError('Changed reused native identity')
    old_meta, _ = load_bank(old_root)
    new_meta, fresh = load_bank(args.new_bank)
    if any(new_meta[k] != registration['protocol'][k] for k in PROTOCOL_KEYS):
        raise ValueError('Native builder/evaluator environment changed')
    missing = read(args.output / 'missing_index.json')
    if (set(fresh) != {r['token'] for r in missing} or
            any(fresh[r['token']]['log'] != r['log'] for r in missing) or
            new_meta['index_sha256'] != registration['missing_index_sha256']):
        raise ValueError('New cache bank differs from the exact missing population')
    reused = read(args.output / 'reused_records.json')
    for row in reused:
        log = row['log']
        if log in new_meta['raw_log_sha256'] and old_meta['raw_log_sha256'][log] != new_meta['raw_log_sha256'][log]:
            raise ValueError('Raw log revision changed across banks')
    lookup = {r['token']: r for r in reused}
    lookup.update(fresh)
    population = read(args.output / 'population.json')
    if set(lookup) != {r['token'] for r in population}:
        raise ValueError('Missing/extra B scenes')
    total_bytes = verify_files([lookup[r['token']] for r in population])
    for role, rows in {**roles, 'B_all': population}.items():
        path = args.output / (role + '_metric_index.json')
        value = [{k: lookup[row['token']][k] for k in ('token', 'log', 'cache_path')} for row in rows]
        if path.exists() and read(path) != value:
            raise ValueError('Preserve previously assembled index')
        atomic_json(path, value)
    atomic_json(args.output / 'COMPLETE.json', dict(identity=registration['identity'], scenes=len(population),
        role_scenes={k: len(v) for k, v in roles.items()}, verified_bytes=total_bytes,
        new_bank=str(args.new_bank), new_bank_identity_sha256=digest(args.new_bank / 'identity.json'),
        index_sha256={k: digest(args.output / (k + '_metric_index.json')) for k in (*ROLES, 'B_all')},
        q0_scored=False, original_caches_modified=False, resources=check_host()))
    print({'scenes': len(population), 'verified_bytes': total_bytes, 'q0_scored': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'assemble'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path)
    parser.add_argument('--recording-proof', type=Path)
    parser.add_argument('--reuse-bank', type=Path)
    parser.add_argument('--new-bank', type=Path)
    args = parser.parse_args()
    check_host()
    required = ('source_audit', 'recording_proof', 'reuse_bank') if args.phase == 'prepare' else ('new_bank',)
    if any(getattr(args, name) is None for name in required):
        parser.error('Missing required inputs for ' + args.phase)
    started = time.monotonic()
    (prepare if args.phase == 'prepare' else assemble)(args)
    print({'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
