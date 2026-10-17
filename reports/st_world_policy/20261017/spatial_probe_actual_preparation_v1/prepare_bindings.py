#!/usr/bin/env python3
"""Build SP0 controls from existing metadata only; never plan, load or dispatch.

The export identity and old image-request hashes supply EXPECTED identities.
The original cache_plan will still verify native shards/index/images in its one
separately admitted real invocation. This script does not approve those reads.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

BASE = Path('/mnt/project/st-world-policy-20261016')
TAG = 'step_005000_1792217229597178910'
TRAIN_SOURCE = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
REG = BASE / 'formal/spatial/registrations/SP0_fulltrain_seed42_official_5k_v1/REGISTRATION.json'
REG_SHA = '85624cb410dab8a531edc3a793ddc052ed1f905c488174289638f55bcf489738'
EXPORT = BASE / 'formal/evaluation/jobs/SP0_step005000_dev1696_seed42_v1/export/IDENTITY.json'
EXPORT_SHA = '7c7dd3c443303e8912dc20c637ee3c764143791e29303fac7e176e1b42b03539'
RECEIPT_SHA = '1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe'
PROTOCOL_SHA = '7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e'
REQUEST_SHA = {'dev': 'e175f0bca0953a49e4c4013af78fa3682af2733cbec4b7df8a8d3446b14847fd',
               'train': '07ed957ef5dd6aa55b3fd35cad0477cb3ffac9490d579e96fd7978be2660e5a0'}
JOB = BASE / 'audits/spatial_probe_actual_SP0_5000_v1'
ROOTS = {s: BASE / f'caches/learned_w_probe_SP0_5000_v1/{s}' for s in ('dev', 'train')}
DIRECTORY_LIMIT = 5 << 20
FINAL_RESERVE = 256 << 10
COUNTER_RESERVE = 128 << 10


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return {**value, 'identity': digest(value)}


def check_seal(value, key='identity'):
    if value[key] != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError('Semantic control identity changed')
    return value


def read(path, expected=None, limit=1 << 20):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('Absolute nonsymlink control required')
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('Control exceeds bounded metadata read')
        raw = f.read(limit + 1); after = os.fstat(f.fileno())
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) > limit or version(before) != version(after) or version(after) != version(path.stat()):
        raise ValueError('Control changed while reading')
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('Control hash changed: ' + str(path))
    return json.loads(raw), {'path': str(path), 'sha256': sha}


@contextmanager
def control_lock(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'CONTROL_WRITE.lock'
    if directory.is_symlink() or path.is_symlink():
        raise ValueError('Symlink control namespace forbidden')
    with path.open('a+b') as lock:
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded control write lock wait exceeded')
                time.sleep(.01)
        yield


def observed_bytes(directory):
    files = list(Path(directory).iterdir())
    if len(files) > 32 or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Only bounded flat controls in each stage directory')
    return sum(p.stat().st_size for p in files)


def write_control(path, value, *, replace=False):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    limit = (128 << 10) if path.name == 'RESULT.json' else (64 << 10) if path.name in ('COUNTERS.json', 'CLI_RESULT.json') else (1 << 20)
    if len(raw) > limit:
        raise ValueError('Serialized control exceeds byte limit')
    reserve = 0 if path.name == 'RESULT.json' else FINAL_RESERVE
    if path.name not in ('RESULT.json', 'COUNTERS.json'):
        reserve += COUNTER_RESERVE
    with control_lock(path.parent):
        if path.is_symlink():
            raise ValueError('Symlink control output forbidden')
        if path.exists() and not replace:
            if path.stat().st_size > limit or path.read_bytes() != raw:
                raise ValueError('Immutable output conflict: ' + str(path))
        else:
            # Account actual serialized bytes and conservative staging/link names
            # under the same lock as logs/counters. Keep the final failure receipt.
            if observed_bytes(path.parent) + 2 * len(raw) + reserve > DIRECTORY_LIMIT:
                raise ValueError('Control directory including staging exceeds byte budget')
            tmp = path.with_name('.' + path.name + '.tmp')
            with tmp.open('xb') as f:
                f.write(raw); f.flush()
                if not replace:
                    os.fsync(f.fileno())
            if replace:
                os.replace(tmp, path)
            else:
                os.link(tmp, path); tmp.unlink()  # publish without replacing an old receipt
                fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_once(path, value):
    return write_control(path, value)


def append_log(path, stream, block):
    with control_lock(Path(path).parent):
        available = max(0, DIRECTORY_LIMIT - observed_bytes(Path(path).parent) - FINAL_RESERVE - COUNTER_RESERVE)
        count = min(len(block), available)
        stream.write(block[:count]); stream.flush()
        return count


def build(runtime, source, output):
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != source:
        raise ValueError('Extraction runtime HEAD differs')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=runtime).strip():
        raise ValueError('Extraction runtime must be clean')
    protocol_path = runtime / 'reports/st_world_policy/20261017/spatial_probe/PROTOCOL.json'
    protocol, protocol_pin = read(protocol_path, PROTOCOL_SHA); check_seal(protocol)
    reg, reg_pin = read(REG, REG_SHA)
    cfg, _ = read(**{'path': reg['artifacts']['config']['path'], 'expected': reg['artifacts']['config']['sha256']})
    # A bounded existing export CONTROL, not weights or the 19 MiB train identity.
    exported, export_pin = read(EXPORT, EXPORT_SHA, 2 << 20); check_seal(exported)
    cp = check_seal(exported['checkpoint'], 'sha256')
    if (cp['tag'] != TAG or cp['completed'] != 5000 or cp['scope'] != 'FORMAL'
            or cp['training_source_sha'] != TRAIN_SOURCE or cp['weights_usable_for_formal_init'] is not True
            or cp['run'] != str(Path(cfg['runtime']['output_root']) / reg['run_id'])
            or cfg['space_time']['cell'] != 'S-P0'):
        raise ValueError('SP0 exact endpoint/training lineage differs')
    receipt, receipt_pin = read(Path(cp['run']) / 'checkpoints' / TAG / 'COMPLETE.json', RECEIPT_SHA)
    check_seal(receipt, 'receipt_identity')
    if (receipt['receipt_identity'] != cp['receipt_identity'] or receipt['state']['identity'] != cp['run_identity']
            or receipt['state']['completed'] != 5000 or cp['native_receipt_sha256'] != RECEIPT_SHA
            or digest({k:v['sha256'] for k,v in receipt['files'].items()}) != cp['native_checkpoint_sha256']):
        raise ValueError('Native small receipt differs')
    # Reproduce the protected AST representation already used by checkpoints.py;
    # source text only. Actual cache_plan still calls the original full checker.
    for name, expected in cp['deployment_source_contract'].items():
        tree = ast.parse((runtime / name).read_text())
        if name.endswith('st_world_model.py'):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpaceTimeStudent')
            cls.body = [n for n in cls.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name != 'forward']
        if hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() != expected:
            raise ValueError('Protected deployment AST differs: ' + name)
    env, env_pin = read(REG.parent / 'ENVIRONMENT.json')
    expected_env = {'base': env, 'scikit_learn': '1.2.2'}
    refs, cache_refs, evidence = {}, {}, [reg_pin, export_pin, receipt_pin, env_pin, protocol_pin]
    for split, count in (('dev', 16), ('train', 64)):
        manifest, manifest_pin = read(protocol['populations'][split]['path'], protocol['populations'][split]['sha256'])
        rows = manifest['scenes']
        if manifest['split'] != split or len(rows) != count or len({r['token'] for r in rows}) != count:
            raise ValueError('Fixed population changed')
        root = Path('/mnt/project/ddp-foresight-artifacts/20260928') / f'student_{split}_v1'
        student, student_pin = read(root / 'identity.json'); check_seal(student)
        requests, requests_pin = read(BASE / f'audits/history_data/smoke_{split}{count}_v1/TEACHER_REQUESTS_SP0_CURRENT.json', REQUEST_SHA[split])
        check_seal(requests)
        if (requests['split'] != split or requests['times_s'] != [0.0] or requests['view_names'] != ['CAM_F0', 'CAM_L0', 'CAM_R0']
                or requests['source_student_identity'] != student['identity']
                or requests['source_index_sha256'] != manifest['source_index_hash']
                or manifest['source_index_hash'] != student['index_sha256']):
            raise ValueError('Current-only request/source identity differs')
        by_token = {r['token']: r for r in requests['scenes']}
        if len(by_token) != count or set(by_token) != {r['token'] for r in rows}:
            raise ValueError('Request population differs')
        images = {}
        for row in rows:
            request = by_token[row['token']]
            record, _ = read(root / 'current' / (row['token'] + '.json'), row['current_record_sha256'])
            if (request['log'] != row['log'] or record['log'] != row['log'] or record['token'] != row['token']
                    or record['identity'] != student['identity'] or len(request['frames']) != 1
                    or request['frames'][0]['requested_s'] != 0):
                raise ValueError('Token/log/current view request differs')
            views = request['frames'][0]['views']
            if record['image_paths'] != [v['path'] for v in views] or len(views) != 3:
                raise ValueError('Current camera order differs')
            images[row['token']] = [[v['path'], v['sha256']] for v in views]
        observed = seal(dict(schema='st_observation_only_v1', current_source=student['identity'],
                            manifest_sha256=manifest_pin['sha256'], split=split, image_pins=images,
                            history=None, targets_read=False, teacher_cache_read=False))
        binding = seal(dict(schema='st_spatial_w_binding_v1', source_sha=source, environment=expected_env,
                            device='cuda:0', training_registration=reg_pin, checkpoint_tag=TAG,
                            checkpoint_identity=cp['sha256'], arm='S-P0', hidden_dim=2048, memory_tokens=144,
                            max_seconds=900, observations=dict(current_root=str(root), manifest=manifest_pin,
                                                               identity=observed['identity'], split=split)))
        signature = seal(dict(schema='st_frozen_spatial_w_v1', binding=binding, checkpoint=cp,
                              observations=observed, rows=rows, precision='FP32_native_master_TF32_off',
                              targets_read=False, memory_mask='all learned W slots valid; no label-derived mask'))
        refs[split] = {'binding': write_once(output / f'{split}_BINDING.json', binding),
                       'expected_signature': write_once(output / f'{split}_EXPECTED_SIGNATURE.json', signature)}
        cache_refs[split] = {'root': str(ROOTS[split]), 'identity': signature['identity']}
        evidence.extend([manifest_pin, student_pin, requests_pin])
    fit = seal(dict(schema='st_spatial_probe_fit_v1', protocol=protocol_pin, caches=cache_refs, arm='S-P0',
                    source_sha=source, environment=expected_env, device='cpu', max_seconds=1200,
                    max_output_bytes=64 << 20))
    fit_pin = write_once(output / 'FIT_REGISTRATION.json', fit)
    result = dict(schema='st_SP0_probe_preparation_v1', status='PREPARED_EXPECTATIONS_NOT_EXECUTED',
                  runtime={'path': str(runtime), 'source_sha': source}, training_source_sha=TRAIN_SOURCE,
                  checkpoint_tag=TAG, checkpoint_identity=cp['sha256'], native_bytes=sum(v['bytes'] for v in receipt['files'].values()),
                  controls=refs, fit_registration=fit_pin, caches=cache_refs, fit_output=str(JOB / 'fit'),
                  job_root=str(JOB), evidence=evidence, dispatch=False, payload_read=False, live_policy_written=False,
                  pending=['independent resource/helper approval', 'fresh lease/GPU/CPU admission',
                           'typed-W storage migration and run-identity-size fix', 'one actual cache_plan per split',
                           'current checkpoint commitments covering outer extraction window'])
    return write_once(output / 'PREPARATION.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', required=True); p.add_argument('--source-sha', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); print(json.dumps(build(a.runtime, a.source_sha, a.output)))


if __name__ == '__main__':
    main()
