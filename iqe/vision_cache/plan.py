"""Bound an access list using the real Base state and original sampler."""
from pathlib import Path
import os
import shutil
import subprocess
import time
import torch
from ..contracts import require
from ..io import read_json, atomic_json, file_hash, digest
from ..query_base import bind_source
from ..training.execution import EpochPlan
from .inputs import current_identity, current_numeric_fields


def mount_info(path):
    path = Path(path).resolve()
    data = subprocess.run(['findmnt', '-T', str(path), '-J', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'],
                          capture_output=True, text=True, check=True)
    import json
    mount = json.loads(data.stdout)['filesystems'][0]
    usage = shutil.disk_usage(path)
    return {'path': str(path), 'mount': mount, 'total_bytes': usage.total, 'free_bytes': usage.free,
            'used_bytes': usage.used, 'observed_unix': time.time(),
            'quota': 'NOT_VERIFIED' if shutil.which('lfs') is None else 'requires lfs quota administrator context'}


def capacity_check(root, required_bytes, *, reserve_bytes, temporary_bytes=0):
    info = mount_info(root)
    require(info['mount']['fstype'] in {'lustre', 'nfs', 'nfs4', 'ceph'}, 'production cache must use verified shared mount')
    info.update(required_bytes=required_bytes, reserve_bytes=reserve_bytes, temporary_bytes=temporary_bytes)
    info['sufficient'] = info['free_bytes'] >= required_bytes + reserve_bytes + temporary_bytes
    return info


def make_plan(config, checkpoint, output, *, scope, updates=200, max_samples=64, records_per_shard=24):
    require(scope in {'pilot', 'representative', 'full'}, 'explicit cache scope required')
    require(1 <= records_per_shard <= 32, 'bounded <=2GiB shard plan required')
    contract = read_json(config['s0']['contract']); bind_source(contract['source_root'], contract['source_commit'])
    receipt = read_json(str(checkpoint)+'.COMPLETE.json')
    require(file_hash(checkpoint) == receipt['checksum'], 'parent checkpoint checksum mismatch')
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    require(state['optimizer_step'] == receipt['optimizer_step'] and state['dependencies'] == receipt['dependencies'],
            'parent state/receipt mismatch')
    require(state['framework_contract_hash'] == contract['framework_contract_hash'], 'foreign Base')
    require(state['dependencies']['original_targets'] == config['data'], 'source data identity changed')
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    fit = state['fit_scene_ids']
    cfg = state['dependencies']['config']
    require(len(fit) == len(set(fit)) and cfg['global_batch_size'] == 32, 'Base fit/batch mismatch')
    sequence = []
    if scope == 'full': selected = list(fit)
    else:
        budget = updates if scope == 'representative' else (max_samples+31)//32
        require(1 <= budget <= 200 and (scope != 'pilot' or 1 <= max_samples <= 64), 'bounded qualification list')
        plan = EpochPlan(len(fit), 32, state['seed'], 0, 1, epoch_batches)
        sequence = [[fit[i] for i in b.entries] for b in plan.iter_from(state['epoch'], state['offset'], budget)]
        selected = list(dict.fromkeys(x for batch in sequence for x in batch))
        if scope == 'pilot': selected = selected[:max_samples]
    root = Path(config['data']['train_root']); index = read_json(root/'index.json')
    by_id = {r['token']: r for r in index}; source_identity = read_json(root/'identity.json')
    rows = []
    for token in selected:
        require(token in by_id, 'fit/source token mismatch')
        path = root/'current'/f'{token}.json'; record = read_json(path)
        require(record['identity'] == source_identity['identity'], 'current record source mismatch')
        _, lang = current_numeric_fields(record)
        rows.append({'scene_id': token, 'role': 'incremental_fit', 'current_ref': str(path),
                     'current_sha256': file_hash(path), 'observation_hash': current_identity(record),
                     'lang_hash': digest(lang), 'image_paths': record['image_paths']})
    from .identity import checkpoint_visual_hash
    value = {'schema': 'iqe_fp32_cache_plan_v1', 'scope': scope, 'parent_checkpoint': str(Path(checkpoint).resolve()),
             'parent_sha256': receipt['checksum'], 'parent_step': state['optimizer_step'],
             'visual_state_sha256': checkpoint_visual_hash(state), 'fit_sha256': digest(fit), 'fit_count': len(fit),
             'source_identity': source_identity, 'source_index_sha256': file_hash(root/'index.json'),
             'framework_contract_hash': contract['framework_contract_hash'], 'records_per_shard': records_per_shard,
             'sample_sequence': sequence, 'rows': rows, 'requested_scenes': len(rows),
             'data': config['data'], 'seed': state['seed'], 'epoch': state['epoch'], 'offset': state['offset'],
             'global_batch': cfg['global_batch_size'], 'microbatch': cfg['micro_batch_size'],
             'max_optimizer_steps': cfg['max_optimizer_steps']}
    value['checkpoint_retention_budget'] = {
        'parent_bytes':Path(checkpoint).stat().st_size, 'every_steps':config['execution']['checkpoint_every_steps'],
        'remaining_count':(cfg['max_optimizer_steps']-state['optimizer_step']+config['execution']['checkpoint_every_steps']-1)//config['execution']['checkpoint_every_steps'],
        'no_deletion_authorized':True}
    value['plan_hash'] = digest(value)
    atomic_json(output, value, immutable=True)
    return {'status': 'PLANNED', 'plan': str(output), 'plan_hash': value['plan_hash'],
            'scenes': len(rows), 'scope': scope, 'parent_step': state['optimizer_step']}


def load_plan(path):
    plan = read_json(path)
    require(plan['schema'] == 'iqe_fp32_cache_plan_v1' and digest({k:v for k,v in plan.items() if k!='plan_hash'}) == plan['plan_hash'],
            'cache plan content hash mismatch')
    require(len(plan['rows']) == plan['requested_scenes'] and
            len({r['scene_id'] for r in plan['rows']}) == len(plan['rows']), 'invalid scene access list')
    require(all(r['role'] == 'incremental_fit' for r in plan['rows']), 'dev/test not allowed in training cache')
    return plan
