#!/usr/bin/env python3
"""Read-only, bounded small-metadata inventory; not a probe registration or launcher."""
import time
START = time.monotonic()
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat

HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/project/st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017')
SOURCE = '581399eb309677c20fe7464791b4f18685e947a6'
READ_CAP = 64 << 20
_cache = {}
_read_bytes = 0
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Metadata budget exhausted')))
signal.setitimer(signal.ITIMER_REAL, 120)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read(path, expected=None):
    global _read_bytes
    path = Path(path)
    if path.suffix not in ('.json', '.py'):
        raise ValueError('Only explicit small source/JSON metadata permitted')
    if path not in _cache:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 2 << 20:
                raise ValueError('Not bounded regular metadata')
            if _read_bytes + before.st_size > READ_CAP:
                raise ValueError('Explicit metadata read cap exhausted')
            raw = f.read((2 << 20) + 1)
            after = os.fstat(f.fileno())
        version = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if len(raw) != before.st_size or version(before) != version(after) or version(after) != version(path.stat()):
            raise ValueError('Metadata changed during read')
        _read_bytes += len(raw)
        _cache[path] = raw
    raw = _cache[path]
    if expected is not None and sha(raw) != expected:
        raise ValueError('Pinned metadata changed: ' + str(path))
    return raw, dict(path=str(path), sha256=sha(raw), bytes=len(raw))


def doc(path, expected=None):
    raw, pin = read(path, expected)
    return json.loads(raw), pin


def sealed(value, key='identity'):
    if value[key] != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError('Semantic identity mismatch')
    return value


def required(scenes, tokens):
    payload = tokens * 2048 * 4 + tokens
    result = dict(scenes=scenes, fp32_world_bytes_per_scene=tokens * 2048 * 4,
                  bool_mask_bytes_per_scene=tokens,
                  output_bytes=scenes * (payload + 8192), staging_bytes=payload + (4 << 20),
                  metadata_bytes=(scenes + 10) * 65536)
    result['total_reserved_bytes'] = sum(result[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes'))
    return result


def main():
    protocol, protocol_pin = doc(RUNTIME / 'reports/st_world_policy/20261017/spatial_probe/PROTOCOL.json',
                               '7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e')
    sealed(protocol)
    populations = {}
    logs = {}
    for split, count in (('train', 64), ('dev', 16)):
        ref = protocol['populations'][split]
        data, pin = doc(ref['path'], ref['sha256'])
        assert data['split'] == split and len(data['scenes']) == count
        assert len({r['token'] for r in data['scenes']}) == count
        logs[split] = {r['log'] for r in data['scenes']}
        populations[split] = dict(pin=pin, scenes=count, logs=len(logs[split]))
    assert not logs['train'] & logs['dev']
    labels, label_pin = doc(protocol['labels']['path'], protocol['labels']['sha256'])
    assert labels['label_schema'] == protocol['label_schema']
    for split in populations:
        assert len(labels['roles'][split]) == populations[split]['scenes']

    probe_raw, probe_pin = read(RUNTIME / 'tools/st_world_policy/spatial_probe.py')
    tree = ast.parse(probe_raw)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'cache_requirements')
    # Evaluate only this pure arithmetic function, without importing torch or the runtime.
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-pure-byte-formula>', 'exec'), {}, scope)
    for n in (144, 288):
        for count in (64, 16):
            estimate = required(count, n)
            assert scope['cache_requirements'](count, n, 2048) == {
                k: estimate[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes')}
    readout_raw, readout_pin = read(RUNTIME / 'starVLA/model/modules/st_world/sparse_readout.py')

    arms = []
    for number in range(4):
        short, arm = 'SP' + str(number), 'S-P' + str(number)
        registration = BASE / f'formal/spatial/registrations/{short}_fulltrain_seed42_official_5k_v1/REGISTRATION.json'
        reg, reg_pin = doc(registration)
        cfg, cfg_pin = doc(reg['artifacts']['config']['path'], reg['artifacts']['config']['sha256'])
        env, env_pin = doc(registration.parent / 'ENVIRONMENT.json')
        export, export_pin = doc(BASE / f'formal/evaluation/jobs/{short}_step005000_dev1696_seed42_v1/export/IDENTITY.json')
        sealed(export)
        cp = sealed(export['checkpoint'], 'sha256')
        assert cp['completed'] == 5000 and cp['scope'] == 'FORMAL' and cp['weights_usable_for_formal_init'] is True
        assert cfg['space_time']['cell'] == arm and cfg['data']['history_root'] is None
        assert cp['run'] == str(Path(cfg['runtime']['output_root']) / reg['run_id'])
        root = Path(cp['run']) / 'checkpoints' / cp['tag']
        receipt, receipt_pin = doc(root / 'COMPLETE.json', cp['native_receipt_sha256'])
        sealed(receipt, 'receipt_identity')
        assert receipt['receipt_identity'] == cp['receipt_identity']
        assert receipt['state']['completed'] == 5000 and receipt['state']['identity'] == cp['run_identity']
        assert receipt['state']['tag'] == cp['tag']
        assert digest({k: v['sha256'] for k, v in receipt['files'].items()}) == cp['native_checkpoint_sha256']
        file_stats = []
        for name, expected in receipt['files'].items():
            assert Path(name).name == name
            path = root / name
            s = path.lstat()  # STAT ONLY. Never open checkpoint data.
            assert stat.S_ISREG(s.st_mode) and s.st_size == expected['bytes']
            file_stats.append(dict(name=name, bytes=s.st_size, sha256_from_receipt=expected['sha256'],
                                   file_version=[s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]))
        for name, expected in cp['deployment_source_contract'].items():
            source_raw, _ = read(RUNTIME / name)
            source_tree = ast.parse(source_raw)
            if name.endswith('st_world_model.py'):
                cls = next(x for x in source_tree.body if isinstance(x, ast.ClassDef) and x.name == 'SpaceTimeStudent')
                cls.body = [x for x in cls.body if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) or x.name != 'forward']
            assert sha(ast.dump(source_tree, include_attributes=False).encode()) == expected, name
        tokens = cfg['space_time']['world_queries_per_view'] * 3
        assert tokens == (288 if number == 3 else 144)
        plans = {split: required(count, tokens) for split, count in (('train', 64), ('dev', 16))}
        arms.append(dict(arm=arm, registration=reg_pin, config=cfg_pin, environment=env_pin,
            export_identity=export_pin, checkpoint=cp, native_receipt=receipt_pin,
            native_state=receipt['state'], native_files_stat_only=file_stats,
            native_bytes=sum(x['bytes'] for x in file_stats), checkpoint_payload_read=False,
            full_payload_hash_revalidated=False, deployment_source_AST_compatible=True,
            memory_tokens=tokens, hidden_dim=2048, dtype='float32', requirements=plans,
            total_W_reservation_bytes=sum(x['total_reserved_bytes'] for x in plans.values()),
            cache_root=str(BASE / f'caches/learned_w_probe_{short}_5000_v1'),
            cache_root_exists=(BASE / f'caches/learned_w_probe_{short}_5000_v1').exists(),
            proposed_job_root=str(BASE / f'audits/spatial_probe_actual_{short}_5000_v1')))
    job = BASE / 'audits/spatial_probe_actual_SP0_5000_v1'
    failed, failed_pin = doc(job / 'stages/train_extract/RESULT.json')
    fit_stage, fit_stage_pin = doc(job / 'stages/fit/RESULT.json')
    fit, fit_pin = doc(job / 'fit/RESULT.json')
    fit_manifest, fit_manifest_pin = doc(job / 'fit/MANIFEST.json')
    fit_reg, fit_reg_pin = doc(job / 'controls_581399eb_v2_metadata_only/FIT_REGISTRATION.json')
    assert failed['status'] == 'STOPPED_NO_AUTO_RETRY'
    assert failed['resources']['limit_reason'] == 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
    assert fit_stage['status'] == fit['status'] == 'COMPLETE' and fit['completed'] == 300
    assert fit_reg['protocol'] == {k: protocol_pin[k] for k in ('path', 'sha256')}
    total = sum(a['total_W_reservation_bytes'] for a in arms)
    inventory = dict(schema='st_matched_spatial_probe_small_metadata_handoff_v1',
        status='METADATA_LOCATED_NOT_REGISTERED_NOT_LAUNCHED', captured_at_unix=time.time(),
        runtime=dict(path=str(RUNTIME), source_sha=SOURCE, probe=probe_pin, readout=readout_pin),
        protocol=protocol_pin, protocol_identity=protocol['identity'], recipe=protocol['recipe'],
        populations=populations, labels=label_pin, label_payload_read=False,
        arms=arms, storage=dict(frozen_W_cap_bytes=1 << 30,
            existing_SP0_formula_reservation_bytes=arms[0]['total_W_reservation_bytes'],
            incremental_SP1_SP2_SP3_formula_reservation_bytes=total-arms[0]['total_W_reservation_bytes'],
            all_four_formula_reservation_bytes=total, unallocated_after_all_four_bytes=(1 << 30)-total,
            fit_output_per_arm_bytes=64 << 20, fit_output_not_in_W_formula=True,
            no_storage_claim_or_ledger_write=True),
        original_SP0=dict(failed_extract=failed_pin, failed_status=failed['status'],
            failed_reason=failed['resources']['limit_reason'],
            cleanup=failed['resources']['process_cleanup'], fit_stage=fit_stage_pin,
            fit_result=fit_pin, fit_manifest=fit_manifest_pin, fit_registration=fit_reg_pin,
            completed_fit_updates=300, parameter_count=fit_manifest['parameter_count'],
            historical_failure_preserved=True), dispatch=False, tensor_reads=0,
        write_scope='this isolated handoff report directory only')
    assert total < 1 << 30
    (HERE / 'ASSET_INVENTORY.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    io = {k: int(v) for k, v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    cost = dict(status='PASS_SMALL_METADATA_CHECKS_ONLY', elapsed_seconds=time.monotonic()-START,
                explicit_unique_metadata_source_bytes=_read_bytes, explicit_read_cap_bytes=READ_CAP,
                process_rchar_bytes=io['rchar'], physical_read_bytes=io['read_bytes'],
                peak_rss_bytes=int(status['VmHWM'].split()[0])*1024,
                raw_getrusage_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                rss_source='proc/self/status VmHWM for this executable; raw getrusage includes pre-exec high water',
                cpu_affinity=sorted(os.sched_getaffinity(0)), tensor_reads=0, dispatch=False,
                source_files_read=len(_cache), torch_imported=False, storage_ledger_written=False)
    assert cost['process_rchar_bytes'] < READ_CAP and cost['peak_rss_bytes'] < 1 << 30
    (HERE / 'CAPTURE_COST.json').write_text(json.dumps(cost, indent=2, sort_keys=True)+'\n')
    print(json.dumps(cost, sort_keys=True))


if __name__ == '__main__':
    main()
