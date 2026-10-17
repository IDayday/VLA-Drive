"""Typed <=1 GiB learned-W admission in the original campaign ledger.

This is control validation, not extraction or native checkpoint qualification.
Only bounded pinned JSON is read here. spatial_probe.cache_plan/load_world_checkpoint
remain responsible for actual native shards and observation images before use.
No torch import, second ledger, automatic migration or tensor cache construction.
"""
from pathlib import Path
import math
import re

from tools.st_world_policy import cache_storage_budget as b

SIGNATURE_SCHEMA = 'st_frozen_spatial_w_v1'
PACKAGE_SCHEMA = 'st_frozen_spatial_w_storage_package_v1'
PACKAGE_MAX = b.GIB
MAX_CLAIMS = 8
ARMS = ('S-P0', 'S-P1', 'S-P2', 'S-P3')
AMOUNTS = ('output_bytes', 'staging_bytes', 'metadata_bytes')


def _exact(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        b._fail(name + ' missing/unknown fields')


def _sealed(value, key='identity'):
    b._hash(value.get(key), 'frozen W ' + key)
    if value[key] != b._digest({k:v for k,v in value.items() if k != key}):
        b._fail('frozen W semantic identity differs')
    return value


def _pin(ref):
    from tools.st_world_policy.metadata_storage import pinned_json
    _exact(ref, ('path', 'sha256'), 'frozen W control pin')
    return pinned_json(ref['path'], ref['sha256'])[0]


def _source(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        b._fail('frozen W source must be an exact git commit')


def _token(value):
    if not isinstance(value, str) or not value or any(x in value for x in ('/', '\\', '..')):
        b._fail('unsafe frozen W token/tag/run id')
    return value


def _requirements(scenes, tokens, hidden):
    # Exact current spatial_probe.cache_requirements formula, verified against
    # that public function in tests. Keep this admission module torch-free.
    payload = tokens * hidden * 4 + tokens
    return dict(output_bytes=scenes * (payload + 8192), staging_bytes=payload + (4 << 20),
                metadata_bytes=(scenes + 10) * 65536)


def tensor_schema(binding):
    return {'W': {'dtype': 'float32', 'shape': [binding['memory_tokens'], binding['hidden_dim']]},
            'memory_valid': {'dtype': 'bool', 'shape': [binding['memory_tokens']], 'all_valid': True}}


def validate_signature(signature):
    """Check the actual cache_plan signature and pinned small provenance controls.

    This intentionally never globs/hashes model/W tensors, full observation
    indexes or images. The existing extractor performs those expensive checks.
    """
    _exact(signature, ('schema', 'identity', 'binding', 'checkpoint', 'observations', 'rows',
                      'precision', 'targets_read', 'memory_mask'), 'frozen W signature')
    _sealed(signature)
    if (signature['schema'] != SIGNATURE_SCHEMA or signature['precision'] != 'FP32_native_master_TF32_off'
            or signature['targets_read'] is not False
            or signature['memory_mask'] != 'all learned W slots valid; no label-derived mask'):
        b._fail('frozen W precision/target/mask contract differs')
    binding = _sealed(signature['binding'])
    _exact(binding, ('schema', 'identity', 'source_sha', 'environment', 'device', 'training_registration',
                    'checkpoint_tag', 'checkpoint_identity', 'arm', 'observations', 'hidden_dim',
                    'memory_tokens', 'max_seconds'), 'frozen W binding')
    arm = binding['arm']
    if binding['schema'] != 'st_spatial_w_binding_v1' or arm not in ARMS:
        b._fail('unknown frozen W arm')
    _source(binding['source_sha'])
    environment = binding['environment']
    if (not isinstance(environment, dict) or set(environment) != {'base', 'scikit_learn'}
            or not isinstance(environment['base'], dict)
            or set(environment['base']) != {'schema', 'python', 'packages', 'cuda_runtime', 'cudnn'}
            or environment['base']['schema'] != 'st_training_environment_v1'
            or not isinstance(environment['base']['packages'], dict) or not environment['base']['packages']):
        b._fail('frozen W runtime environment must be explicit')
    if not isinstance(binding['device'], str) or not re.fullmatch(r'cpu|cuda:[0-9]+', binding['device']):
        b._fail('invalid frozen W device')
    tokens, hidden = binding['memory_tokens'], binding['hidden_dim']
    if (type(tokens) is not int or tokens != (288 if arm == 'S-P3' else 144)
            or type(hidden) is not int or hidden < 1
            or type(binding['max_seconds']) not in (int, float)
            or not math.isfinite(binding['max_seconds']) or not 0 < binding['max_seconds'] <= 7200):
        b._fail('invalid frozen W shape/time')
    obs = binding['observations']
    _exact(obs, ('current_root', 'manifest', 'identity', 'split'), 'frozen W observation binding')
    if obs['split'] not in ('train', 'dev'):
        b._fail('frozen W permits only train64/dev16')
    count = 64 if obs['split'] == 'train' else 16
    manifest = _pin(obs['manifest'])
    rows = signature['rows']
    if (manifest.get('schema') != 'st_fixed_scene_list_v1' or manifest.get('split') != obs['split']
            or not isinstance(rows, list) or len(rows) != count or manifest.get('scenes') != rows
            or len({_token(r['token']) for r in rows}) != count):
        b._fail('frozen W fixed observation population differs')
    for row in rows:
        if not isinstance(row.get('log'), str) or not row['log']:
            b._fail('frozen W log identity missing')
        b._hash(row.get('current_record_sha256'), 'current record')
    observed = _sealed(signature['observations'])
    _exact(observed, ('schema', 'identity', 'current_source', 'manifest_sha256', 'split', 'image_pins',
                     'history', 'targets_read', 'teacher_cache_read'), 'frozen W observation identity')
    if (observed['schema'] != 'st_observation_only_v1' or observed['identity'] != obs['identity']
            or observed['manifest_sha256'] != obs['manifest']['sha256'] or observed['split'] != obs['split']
            or observed['history'] is not None or observed['targets_read'] is not False
            or observed['teacher_cache_read'] is not False
            or set(observed['image_pins']) != {r['token'] for r in rows}):
        b._fail('frozen W observation/source/mask identity differs')
    # Small immutable source identity only: no full index or image scan.
    from tools.st_world_policy.metadata_storage import pinned_json, _json, CONTROL_LIMIT, _version
    source_path = b._path(str(b._path(obs['current_root']) / 'identity.json'))
    # No raw SHA in the original observation interface: verify one bounded
    # snapshot by its original semantic hash, without inventing an external pin.
    import os
    import stat
    with source_path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(CONTROL_LIMIT + 1)
        after = os.fstat(stream.fileno())
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or len(raw) > CONTROL_LIMIT
            or _version(before) != _version(after) or _version(after) != _version(source_path.stat())):
        b._fail('frozen W current source control changed/oversized')
    source = _sealed(_json(raw))
    if (source.get('schema') != 'ddpolicy_current_cameras_v1' or source.get('split') != obs['split']
            or source['identity'] != observed['current_source']
            or source.get('index_sha256') != manifest.get('source_index_hash')):
        b._fail('frozen W current source/index identity differs')
    b._hash(source['index_sha256'], 'current source index')
    for pins in observed['image_pins'].values():
        if not isinstance(pins, list) or len(pins) != 3:
            b._fail('frozen W requires original three-view image pins')
        for pin in pins:
            if not isinstance(pin, list) or len(pin) != 2:
                b._fail('frozen W malformed image pin')
            b._path(pin[0]); b._hash(pin[1], 'image pin')
    checkpoint = _sealed(signature['checkpoint'], 'sha256')
    _exact(checkpoint, ('schema', 'run', 'tag', 'run_identity', 'completed', 'receipt_identity',
                       'base_binding', 'native_checkpoint_sha256', 'native_receipt_sha256',
                       'training_source_sha', 'scope', 'weights_usable_for_formal_init',
                       'inference_contract', 'deployment_source_contract', 'sha256'), 'frozen W native checkpoint')
    if (checkpoint['schema'] != 'st_world_deployment_checkpoint_v1' or checkpoint['completed'] != 5000
            or type(checkpoint['completed']) is not int or checkpoint['scope'] != 'FORMAL'
            or checkpoint['weights_usable_for_formal_init'] is not True
            or checkpoint['sha256'] != binding['checkpoint_identity']
            or checkpoint['tag'] != _token(binding['checkpoint_tag'])):
        b._fail('frozen W requires the exact native FORMAL5000 endpoint')
    _source(checkpoint['training_source_sha'])
    for key in ('run_identity', 'receipt_identity', 'base_binding', 'native_checkpoint_sha256', 'native_receipt_sha256'):
        b._hash(checkpoint[key], key)
    if not checkpoint['deployment_source_contract']:
        b._fail('missing frozen W inference source contract')
    for sha in checkpoint['deployment_source_contract'].values():
        b._hash(sha, 'inference source contract')
    reg = _pin(binding['training_registration'])
    cfg = _pin(reg['artifacts']['config'])
    base = _pin(reg['artifacts']['base_binding'])
    _sealed(base, 'binding_sha256')
    if (base.get('schema') != 'st_world_bound_s0_v1'
            or cfg.get('schema') != 'st_training_v1' or cfg['execution']['scope'] != 'FORMAL'
            or cfg['space_time']['cell'] != arm or cfg['data'].get('history_root')
            or base['binding_sha256'] != checkpoint['base_binding']
            or base['parent_identity']['config']['framework']['qwenvl']['vl_hidden_dim'] != hidden
            or base['inference_contract'] != checkpoint['inference_contract']
            or str(b._path(cfg['runtime']['output_root']) / _token(reg['run_id'])) != checkpoint['run']):
        b._fail('frozen W checkpoint/config/base/shape binding differs')
    # Native COMPLETE is a small pinned receipt. Payload verification remains
    # cache_plan's existing responsibility, not a per-record storage operation.
    receipt_path = b._path(checkpoint['run']) / 'checkpoints' / checkpoint['tag'] / 'COMPLETE.json'
    receipt = pinned_json(receipt_path, checkpoint['native_receipt_sha256'])[0]
    _sealed(receipt, 'receipt_identity')
    if (receipt.get('schema') != 'st_native_checkpoint_v1'
            or receipt['receipt_identity'] != checkpoint['receipt_identity']
            or receipt['state']['identity'] != checkpoint['run_identity']
            or receipt['state']['tag'] != checkpoint['tag'] or receipt['state']['completed'] != 5000
            or receipt['world_size'] != cfg['execution']['world_size']
            or b._digest({k:v['sha256'] for k,v in receipt['files'].items()}) != checkpoint['native_checkpoint_sha256']):
        b._fail('frozen W native receipt binding differs')
    world = b._integer(receipt['world_size'], 'native world size', positive=True)
    if world > 64:
        b._fail('frozen W native rank metadata exceeds bounded admission')
    expected = {'mp_rank_00_model_states.pt'} | {
        f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(world)} | {
        f'rng_rank{rank}.pt' for rank in range(world)}
    if (set(receipt['files']) != expected or receipt['state'].get('termination_reason') not in
            ('RUNNING_CHECKPOINT', 'OPERATOR_PAUSED', 'BUDGET_PAUSED', 'MAX_STEPS')):
        b._fail('frozen W native rank inventory/termination differs')
    for name, value in receipt['files'].items():
        _token(name); b._hash(value['sha256'], 'native shard'); b._integer(value['bytes'], 'native shard bytes', positive=True)
    # Full run identity is represented in the native receipt. Its config/source
    # is additionally verified from the small content-addressed run identity.
    identity_path = b._path(str(b._path(checkpoint['run']) / 'identity.json'))
    with identity_path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(CONTROL_LIMIT + 1); after = os.fstat(stream.fileno())
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or len(raw) > CONTROL_LIMIT
            or _version(before) != _version(after) or _version(after) != _version(identity_path.stat())):
        b._fail('frozen W run identity changed/oversized')
    run = _sealed(_json(raw))
    if (run.get('schema') != 'st_world_training_v1' or run['identity'] != checkpoint['run_identity']
            or run['config'] != cfg or run['source_sha'] != checkpoint['training_source_sha']
            or run['registration_sha256'] != binding['training_registration']['sha256']
            or run['base_binding'] != checkpoint['base_binding']
            or run['weights_usable_for_formal_init'] is not True):
        b._fail('frozen W training lineage/source identity differs')
    return binding, _requirements(count, tokens, hidden)


def claim_from_signature(*, root, claim_id, signature_path, signature_sha256):
    """Adapt the unchanged cache_plan signature to the existing reserve API.

    Caller supplies a separately persisted/pinned original cache_plan signature;
    no extraction or output mutation occurs here. Never creates a policy/ledger.
    """
    signature = _pin({'path': str(signature_path), 'sha256': signature_sha256})
    binding, required = validate_signature(signature)
    return dict(claim_id=claim_id, root=str(b._path(root)), cache_identity=signature['identity'],
                identity_kind=SIGNATURE_SCHEMA, signature_path=str(b._path(signature_path)),
                signature_sha256=signature_sha256, tensor_schema=tensor_schema(binding), **required)


def validate_package(core, declarations):
    package = core['frozen_w_package']
    _exact(package, ('schema', 'maximum_bytes', 'baseline_policy_sha256', 'baseline_ledger_sha256',
                     'baseline_claims', 'baseline_legacy_reservations', 'metadata_package_sha256',
                     'members'), 'frozen W package')
    if package['schema'] != PACKAGE_SCHEMA or 'metadata_package' not in core:
        b._fail('frozen W package requires the original metadata package')
    maximum = b._integer(package['maximum_bytes'], 'frozen W package maximum', positive=True)
    if maximum > PACKAGE_MAX:
        b._fail('frozen W package exceeds one GiB')
    for key in ('baseline_policy_sha256', 'baseline_ledger_sha256', 'metadata_package_sha256'):
        b._hash(package[key], key)
    if b._digest(core['metadata_package']) != package['metadata_package_sha256']:
        b._fail('frozen W changed original metadata package')
    baseline = package['baseline_claims']
    if not isinstance(baseline, dict) or any(declarations.get(root) != row for root,row in baseline.items()):
        b._fail('frozen W package changed a baseline claim')
    if any(row.get('identity_kind') == SIGNATURE_SCHEMA for row in baseline.values()):
        b._fail('frozen W tensors cannot be hidden in baseline claims')
    legacy = {r['claim_id']:r for r in core['legacy_reservations']}
    if any(legacy.get(r['claim_id']) != r for r in package['baseline_legacy_reservations']):
        b._fail('frozen W package changed baseline reservations')
    rows = [row for row in declarations.values() if row.get('identity_kind') == SIGNATURE_SCHEMA]
    if package['members'] != {row['root']:row for row in rows}:
        b._fail('frozen W package exact member declarations differ')
    if not 1 <= len(rows) <= MAX_CLAIMS:
        b._fail('frozen W package requires one to eight exact arm/split claims')
    total = sum(b.StorageBudget._total(row) for row in rows)
    if total > maximum:
        b._fail('aggregate frozen W package exceeds its single allowance')
    endpoints, populations, keys = {}, {}, set()
    shared_contract = None
    for row in rows:
        _exact(row, ('claim_id', 'root', 'cache_identity', 'identity_kind', 'signature_path',
                     'signature_sha256', 'tensor_schema', *AMOUNTS), 'frozen W claim')
        signature = _pin({'path':row['signature_path'], 'sha256':row['signature_sha256']})
        binding, required = validate_signature(signature)
        if (signature['identity'] != row['cache_identity'] or row['tensor_schema'] != tensor_schema(binding)
                or any(row[k] != required[k] for k in AMOUNTS)):
            b._fail('frozen W claim/signature/shape/storage amounts differ')
        key = (binding['arm'], binding['observations']['split'])
        if key in keys:
            b._fail('duplicate frozen W arm/split')
        keys.add(key)
        role = binding['observations']['split']
        population = (binding['observations']['identity'], binding['observations']['manifest'])
        if role in populations and populations[role] != population:
            b._fail('frozen W arms must use the same fixed role population')
        populations[role] = population
        common = (signature['checkpoint']['base_binding'], binding['source_sha'], binding['environment'])
        if shared_contract is not None and shared_contract != common:
            b._fail('frozen W arms must share original Base and extraction runtime')
        shared_contract = common
        endpoint = (binding['checkpoint_identity'], binding['training_registration'], binding['source_sha'],
                    binding['environment'], binding['hidden_dim'], binding['memory_tokens'])
        if binding['arm'] in endpoints and endpoints[binding['arm']] != endpoint:
            b._fail('frozen W train/dev must bind the same arm checkpoint/runtime')
        endpoints[binding['arm']] = endpoint
    return total


def validate_migration(old, new, *, old_policy_sha256, old_ledger_sha256):
    old_package, package = old.core.get('frozen_w_package'), new.core.get('frozen_w_package')
    if old_package is not None:
        if (package is None or {k:v for k,v in package.items() if k != 'members'} !=
                {k:v for k,v in old_package.items() if k != 'members'}
                or any(package['members'].get(root) != row for root,row in old_package['members'].items())):
            b._fail('migration changed original frozen W package baseline')
    elif package is not None:
        if ('metadata_package' not in old.core or package['baseline_claims'] != old.declarations
                or package['baseline_legacy_reservations'] != old.core['legacy_reservations']
                or package['baseline_policy_sha256'] != old_policy_sha256
                or package['baseline_ledger_sha256'] != old_ledger_sha256):
            b._fail('frozen W migration baseline not exact')
