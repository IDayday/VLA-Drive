"""Explicit RAM-only teacher dependency for the existing ST trainer.

Preparing metadata never loads weights. Runtime creation is synchronous on the
owning rank after its backend is configured; the existing collective data error
guard remains responsible for all-rank failure. No cache completion is forged.
"""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash, require_hash
from starVLA.dataloader.st_world_targets import TeacherCacheReader, atomic_json
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider

MiB = 1 << 20
LIMIT_FIELDS = {'max_cached_bytes', 'max_scene_bytes', 'max_batch_images', 'max_manifest_bytes'}
INDEXED_SCHEMA = 'st_indexed_teacher_provider_dependency_v1'
PROFILE_SCHEMA = 'st_indexed_provider_profile_admission_v1'
FULL_PROFILE_SCHEMA = 'st_full_index_provider_profile_admission_v1'
CONSUMPTION_SCHEMA = 'st_full_index_profile_consumption_v1'
INDEX_FIELDS = {'complete', 'request_identity', 'index_identity', 'max_metadata_cache_bytes'}
FIXED_CASES = ('fixed8_all_original_order', 'current_request1', 'current_request4',
    'current_reversed8', 'current_tail8_plus_duplicate', 'provider_cold_partial_lru_cold_resume')


def read_small(path):
    path = Path(path)
    if path.stat().st_size > 32 * MiB:
        raise ValueError('Initial on-demand wiring refuses metadata above 32 MiB')
    return json.loads(path.read_text())


def pin(path):
    path = Path(path).resolve()
    if path.stat().st_size > 32 * MiB:
        raise ValueError('Oversized provider metadata')
    return {'path': str(path), 'sha256': file_sha256(path)}


def pinned(record):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError('Provider dependency needs an exact path/checksum pin')
    require_hash(record['sha256'], 'dependency checksum')
    if not Path(record['path']).is_absolute():
        raise ValueError('Provider dependency must be absolute')
    result = read_small(record['path'])
    if file_sha256(record['path']) != record['sha256']:
        raise ValueError('Provider pinned dependency changed')
    return result


def validate_option(option, *, loader_workers, teacher_root, pca_path, scope, config=None):
    if not isinstance(option, dict) or set(option) != {'descriptor', 'pca_channel_scope'}:
        raise ValueError('Teacher provider option has missing/unknown fields')
    for item in [option['descriptor'], *([option['pca_channel_scope']] if pca_path else [])]:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Provider option requires explicit pinned metadata')
        require_hash(item['sha256'], 'provider metadata checksum')
        if not isinstance(item['path'], str) or not Path(item['path']).is_absolute():
            raise ValueError('Provider metadata path must be absolute')
    if not pca_path and option['pca_channel_scope'] is not None:
        raise ValueError('PCA scope without PCA basis')
    if teacher_root is not None or loader_workers != 0:
        raise ValueError('On-demand provider requires teacher_root=null and synchronous loader_workers=0')
    if scope == 'PROFILE' and config is not None:
        try: d = pinned(option['descriptor'])
        except OSError as e: raise ValueError('SMOKE-only without an existing explicit PROFILE descriptor') from e
        if 'profile_admission' not in d:
            raise ValueError('SMOKE-only descriptor cannot be promoted by a PROFILE flag')
        validate_profile_recipe(d['profile_admission'], config)
    elif scope != 'SMOKE':
        raise ValueError('Initial provider wiring is SMOKE-only without explicit bounded PROFILE admission')
    elif Path(option['descriptor']['path']).is_file() and 'profile_admission' in pinned(option['descriptor']):
        raise ValueError('PROFILE descriptor cannot be relabeled as SMOKE')


def validate_training_manifest(option, manifest_pin):
    d=pinned(option['descriptor'])
    if 'profile_admission' in d and d['profile_admission']['data_manifest']!=manifest_pin:
        raise ValueError('Registered PROFILE dataset path/SHA differs from the admitted fixed population')


def load_descriptor(record):
    return validate_descriptor(pinned(record))


def index_binding(root, *, max_metadata_cache_bytes=16 * MiB):
    """Pin only the bounded index receipt; original JSON hashes remain dependencies."""
    complete = pin(Path(root) / 'COMPLETE.json')
    receipt = pinned(complete)
    return {'complete': complete, 'request_identity': receipt['request_identity'],
            'index_identity': receipt['identity'], 'max_metadata_cache_bytes': max_metadata_cache_bytes}


def open_metadata_index(binding, *, existing=None):
    from starVLA.dataloader.st_world_metadata_index import MetadataIndex
    if not isinstance(binding, dict) or set(binding) != INDEX_FIELDS:
        raise ValueError('Indexed provider needs an exact explicit metadata-index binding')
    for name in ('request_identity', 'index_identity'):
        require_hash(binding[name], name)
    complete = binding['complete']
    receipt = pinned(complete)
    if Path(complete['path']).name != 'COMPLETE.json':
        raise ValueError('Metadata index binding must name its original COMPLETE receipt')
    if existing is not None:
        if (not isinstance(existing, MetadataIndex) or existing.root != Path(complete['path']).parent.resolve() or
                existing.receipt != receipt or existing.receipt['identity'] != binding['index_identity'] or
                existing.receipt['request_identity'] != binding['request_identity'] or
                existing.max_metadata_cache_bytes != binding['max_metadata_cache_bytes']):
            raise ValueError('Existing metadata reader differs from the exact pinned index')
        existing.check()
        return existing
    index = MetadataIndex(Path(complete['path']).parent,
        expected_request_identity=binding['request_identity'], expected_index_identity=binding['index_identity'],
        max_metadata_cache_bytes=binding['max_metadata_cache_bytes'])
    try:
        if index.receipt != receipt or file_sha256(complete['path']) != complete['sha256']:
            raise ValueError('Metadata index COMPLETE changed across its pinned verification')
        return index
    except BaseException:
        index.close()
        raise


def close_provider_metadata(provider):
    """Close only optional SQLite previews; never alter teacher/tensor/RNG state."""
    indices = {id(i): i for i in [getattr(provider, 'metadata_index', None),
        *(getattr(s, 'metadata_index', None) for s in provider.sources)] if i is not None}
    for index in indices.values():
        index.close()


def validate_bound_indexed_descriptor(record, provider):
    """Revalidate an already owned provider without duplicate SQLite connections."""
    d = pinned(record)
    if d.get('schema') != INDEXED_SCHEMA:
        raise ValueError('Indexed PCA scope requires an indexed provider descriptor')
    return _validate_indexed_descriptor(d, existing=provider)[0]


def validate_fixed_qualification(plan, qualification, execution):
    """Accept only the exercised fixed execution, never infer legacy parity."""
    from tools.st_world_policy import fixed_batch_teacher, ondemand_teacher, qualify_fixed_batch_teacher
    fixed_batch_teacher.require_matching_qualification(qualification, identity_hash(execution))
    if (plan.get('schema') != 'st_fixed_teacher_qualification_plan_v1' or
            execution.get('schema') != 'st_fixed8_dino_execution_v1' or
            plan.get('batch_policy') != fixed_batch_teacher.FIXED_BATCH_POLICY or
            execution.get('batch_policy') != fixed_batch_teacher.FIXED_BATCH_POLICY or
            plan.get('cases') != list(FIXED_CASES) or set(qualification.get('cases', {})) != set(FIXED_CASES) or
            qualification.get('backend') != plan.get('backend') or
            qualification.get('execution_identity') != identity_hash(execution) or
            execution.get('executor_source_sha256') != file_sha256(fixed_batch_teacher.__file__) or
            execution.get('legacy_executor_source_sha256') != file_sha256(ondemand_teacher.__file__) or
            plan.get('runner_source_sha256') != file_sha256(qualify_fixed_batch_teacher.__file__)):
        raise ValueError('Fixed qualification cases/execution/policy/source do not match the active implementation')
    backend = plan['backend']
    for name in ('tf32', 'autocast', 'deterministic_algorithms', 'deterministic_warn_only',
                 'cublas_workspace_config', 'cudnn_benchmark', 'cudnn_deterministic',
                 'flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled'):
        if name not in backend or execution.get(name) != backend[name]:
            raise ValueError('Fixed qualification backend differs from execution: ' + name)
    if (backend['deterministic_algorithms'] is not True or backend['deterministic_warn_only'] is not False or
            backend['cublas_workspace_config'] != ':4096:8' or backend['tf32'] is not False or
            backend['autocast'] is not False):
        raise ValueError('Fixed qualification backend is not the registered deterministic FP32 policy')
    # Bind the actual archived implementation/config files, without deserializing
    # the full training population or opening any model/target tensor.
    for entry in plan['inputs'] + plan['source_files']:
        path = Path(entry['path'])
        if not path.is_absolute() or path.stat().st_size > 32*MiB or file_sha256(path) != entry['sha256']:
            raise ValueError('Fixed qualification archived source/dependency changed')
    counts = {'fixed8_all_original_order': plan['unique_images'],
        'current_request1': plan['current_unique_images'], 'current_request4': plan['current_unique_images'],
        'current_reversed8': plan['current_unique_images'],
        'current_tail8_plus_duplicate': plan['current_unique_images'] + 1,
        'provider_cold_partial_lru_cold_resume': plan['scenes'] * 3 * 2}
    for name, count in counts.items():
        case = qualification['cases'][name]
        if (type(count) is not int or count <= 0 or case.get('all_masks_equal') is not True or
                len(case.get('records', [])) != count):
            raise ValueError('Fixed qualification case is incomplete: ' + name)
        if name not in ('fixed8_all_original_order', 'provider_cold_partial_lru_cold_resume'):
            if (case.get('fixed8_reference', {}).get('bitwise_equal') is not True or
                    any(r.get('fixed8_reference', {}).get('bitwise_equal') is not True or
                        r['fixed8_reference'].get('max_abs') != 0 for r in case['records'])):
                raise ValueError('Fixed qualification regrouping is not bitwise repeatable')
    partial = qualification['cases']['provider_cold_partial_lru_cold_resume']
    if partial.get('partial_lru_exercised') is not True or partial.get('sampler_equal') is not True:
        raise ValueError('Fixed qualification did not validate partial LRU and sampling continuity')
    if any(r.get('cold_reference', {}).get('bitwise_equal') is not True or
           r['cold_reference'].get('max_abs') != 0 for r in partial['records']):
        raise ValueError('Fixed qualification partial-LRU/cold-resume targets changed')
    fixed = qualification.get('fixed_policy_summary', {})
    legacy = qualification.get('legacy_reference_summary', {})
    if (fixed.get('bitwise_equal') is not True or fixed.get('max_abs') != 0 or
            fixed.get('comparisons', 0) <= 0 or fixed.get('elements', 0) <= 0):
        raise ValueError('Fixed qualification did not compare all finite target tensors exactly')
    legacy_ok = qualification.get('legacy_reference_compatibility')
    if (type(legacy_ok) is not bool or legacy.get('bitwise_equal') is not legacy_ok or
            legacy.get('comparisons', 0) <= 0 or legacy.get('elements', 0) <= 0 or
            (legacy_ok and legacy.get('max_abs') != 0)):
        raise ValueError('Legacy compatibility must retain its independent actual comparison result')
    expected_status = ('FIXED_POLICY_AND_LEGACY_BITWISE_QUALIFIED' if legacy_ok else
                       'FIXED_POLICY_REPEATABLE_LEGACY_REVIEW_REQUIRED')
    if qualification.get('status') != expected_status:
        raise ValueError('Fixed/legacy qualification status is inconsistent')
    stats = qualification.get('compute_stats', {})
    calls = qualification.get('batch_calls', 0)
    if (type(calls) is not int or calls <= 0 or qualification.get('rng_forward_checks') != calls or
            stats.get('calls') != calls or stats.get('compute_image_slots') != calls * 8 or
            qualification.get('image_forwards') != calls * 8 or
            stats.get('requested_images', 0) <= 0 or stats.get('padding_image_slots', -1) < 0 or
            stats['requested_images'] + stats['padding_image_slots'] != calls * 8 or
            qualification.get('frozen_initial_versions_hash') != qualification.get('frozen_final_versions_hash') or
            not qualification.get('frozen_initial_versions_hash')):
        raise ValueError('Fixed qualification RNG/frozen/actual padded compute accounting is incomplete')


def _validate_limits(d):
    if set(d['limits']) != LIMIT_FIELDS:
        raise ValueError('Provider limits must all be explicit')
    if (any(type(v) is not int or v <= 0 for v in d['limits'].values()) or
            d['limits']['max_cached_bytes'] > 2 << 30 or
            d['limits']['max_scene_bytes'] > 128 * MiB or
            d['limits']['max_manifest_bytes'] > 32 * MiB or d['limits']['max_batch_images'] > 8):
        raise ValueError('Initial synchronous provider exceeds declared RAM/batch bounds')


def profile_recipe_identity(cfg):
    value=deepcopy(cfg)
    value['data'].pop('teacher_provider',None)
    value['data']['teacher_root']=None
    value['runtime']['loader_workers']=0
    value['runtime']['output_root']='<isolated-provider-output>'
    return identity_hash(value)


def is_full_profile(admission):
    return isinstance(admission, dict) and admission.get('schema') == FULL_PROFILE_SCHEMA


def full_profile_consumption(cfg):
    """Finite PROFILE ceilings; diagnostic and intended-topology paths stay distinct.

    The new timing case is exactly the registered original 8-rank topology and
    two 20+100 windows. It is never a FORMAL admission or a continuation of the
    completed global2 diagnostic. All native sampler and training math are reused.
    """
    ex = cfg['execution']; rt = cfg['runtime']
    diagnostic = (ex['world_size'] in (1, 2) and ex['micro_batch'] == 1
                  and ex['global_batch'] == 2 and rt['profile_mode'] == 'none')
    timing = (ex.get('scope') == 'PROFILE' and ex['world_size'] == 8 and ex['micro_batch'] == 4 and ex['global_batch'] == 32
              and ex['stop_after'] == 240 and ex['sample_limit'] == 101592
              and rt['profile_mode'] == 'timing' and ex.get('max_updates') == 20000
              and cfg.get('optimizer', {}).get('warmup') == 1000)
    if (not (diagnostic or timing)
            or any(type(ex[k]) is not int for k in ('world_size', 'micro_batch', 'global_batch'))
            or type(ex['sample_limit']) is not int or ex['sample_limit'] <= 256
            or type(ex['stop_after']) is not int or not 1 <= ex['stop_after'] <= 256
            or 'continuation' in cfg):
        raise ValueError('Full-index PROFILE requires global2 diagnostic or fixed world8/global32 timing240')
    from tools.foresight.student_state import validate_rank_batches
    world, batch, micro = ex['world_size'], ex['global_batch'], ex['micro_batch']
    validate_rank_batches(ex['sample_limit'], batch, world, micro)
    n = ex['sample_limit']; batches = (n + batch - 1) // batch
    epochs, remainder = divmod(ex['stop_after'], batches)
    tail_exposures = min(remainder * batch, n)
    exposures = epochs * n + tail_exposures
    per_rank = [epochs * len(range(rank, n, world))
                + len(range(rank, tail_exposures, world)) for rank in range(world)]
    return {'schema': CONSUMPTION_SCHEMA, 'initial_completed_updates': 0, 'initial_epoch': 0,
            'initial_offset': 0, 'cumulative_optimizer_updates': ex['stop_after'],
            'cumulative_scene_exposures': exposures,
            'world_size': world, 'micro_batch': micro, 'global_batch': batch,
            'gradient_accumulation_steps': batch // (world * micro),
            'per_rank_scene_exposures': per_rank,
            'resume_budget': ('timing_requires_fresh_complete_windows_no_resume' if timing
                              else 'same_run_cumulative_not_per_invocation'),
            'formal_global32_throughput_qualified': False}


def _validate_full_population(d, request, source):
    """Stream original selected rows against both immutable indices in order.

    Only one selected/current/future row is resident. Do not parse requests.json
    or selected scenes with json.load/list; MetadataIndex owns their raw hashes.
    The unchanged student dataset still performs its own source/record checks.
    """
    from starVLA.dataloader.st_world_metadata_index import _atomic_rows, version, checked_sha, canonical
    a = d['profile_admission']; population = a['population']
    if (not isinstance(population, dict) or set(population) !=
            {'current_registration', 'current_complete', 'future_complete'}):
        raise ValueError('Full PROFILE needs original current and completed full-future population receipts')
    reg = pinned(population['current_registration']); current = pinned(population['current_complete'])
    future = pinned(population['future_complete']); binding = reg.get('source', {})
    n = request.receipt['scene_count']; header = request.receipt['manifest_header']
    future_name = 'COMPLETE.json'
    future_schema = 'st_future_metadata_complete_v1'
    if future.get('schema') == 'st_future_metadata_finalization_complete_v1':
        from tools.st_world_policy.finalize_future_metadata import validate_completion, COMPLETE_NAME, COMPLETE_SCHEMA
        validate_completion(future, population['future_complete']['path'],
            request_sha256=d['requests']['sha256'], request_identity=request.receipt['request_identity'],
            manifest_header=header, scene_count=n)
        future_name, future_schema = COMPLETE_NAME, COMPLETE_SCHEMA
    selected = a['data_manifest']; path = Path(selected['path'])
    current_root = Path(population['current_registration']['path']).parent
    if (Path(population['current_complete']['path']) != current_root / 'COMPLETE.json'
            or path != current_root / 'SELECTED_SCENES.json'
            or Path(population['future_complete']['path']) != Path(d['requests']['path']).parent / future_name):
        raise ValueError('Full PROFILE receipt paths do not bind the original preparations')
    if (reg.get('schema') != 'st_current_population_registration_v1'
            or reg.get('identity') != identity_hash({k:v for k,v in reg.items() if k != 'identity'})
            or reg.get('scope') != 'all_original_current_observations' or reg.get('max_scenes') != n
            or reg.get('target_times_s') != [0.] or binding.get('scenes') != n
            or current.get('status') != 'COMPLETE' or current.get('scope') != 'current_only_metadata'
            or current.get('registration_identity') != reg['identity'] or current.get('scenes') != n
            or current.get('logs') != binding.get('logs') or current.get('selected_sha256') != selected['sha256']):
        raise ValueError('Full PROFILE original current-population receipt is inconsistent')
    original_selection = {'path': selected['path'], 'sha256': selected['sha256'],
        'current_registration_identity': reg['identity'],
        'current_registration_sha256': population['current_registration']['sha256'],
        'current_complete_sha256': population['current_complete']['sha256']}
    if (future.get('schema') != future_schema or future.get('status') != 'COMPLETE'
            or future.get('scope') != 'full_population' or future.get('scenes') != n
            or future.get('logs') != binding.get('logs') or future.get('target_times_s') != [0., .5, 1., 2., 4.]
            or future.get('request_identity') != request.receipt['request_identity']
            or future.get('files', {}).get('TEACHER_REQUESTS.json') != d['requests']['sha256']
            or future.get('original_selection') != original_selection
            or future.get('registration_identity') != header.get('registration_identity')
            or future.get('tensor_disk_bytes') != 0 or future.get('training_admission') is not False
            or header.get('source') != 'original_raw_log_future_frames_v1'
            or request.receipt['dependencies'].get('source_logs_ledger') != {
                'path': header.get('source_logs_ledger', {}).get('path'),
                'sha256': future.get('files', {}).get('RAW_LOG_PROVENANCE.json')}):
        raise ValueError('Full PROFILE requires matching actual full-future COMPLETE, not a PLAN/subset')
    require_hash(future.get('plan_identity'), 'full future plan identity')
    for field, target in [('source_identity','source_student_identity'),
                          ('source_index_identity','source_index_sha256'), ('partition_sha256','partition_sha256')]:
        if binding.get(field) != header.get(target):
            raise ValueError('Full PROFILE source/partition identity differs: ' + field)
    if (binding.get('split') != 'train' or binding.get('student_root') != pinned(a['recipe'])['data']['root']
            or source.receipt['scene_count'] != n or source.manifest['times_s'] != [0.]
            or a['max_scenes'] != n):
        raise ValueError('Full PROFILE must retain the entire original train population and current index')
    if (not isinstance(selected, dict) or set(selected) != {'path','sha256'} or not path.is_absolute()
            or path.stat().st_size > 32 * MiB):
        raise ValueError('Full PROFILE selected metadata exceeds the unchanged 32 MiB control bound')
    require_hash(selected['sha256'], 'selected manifest checksum')
    digest, stamp = checked_sha(path)
    if digest != selected['sha256']:
        raise ValueError('Full PROFILE selected manifest checksum changed')
    future_rows, current_rows = iter(request.manifest['scenes']), iter(source.manifest['scenes'])
    token_hash = hashlib.sha256(b'['); selection_hash = hashlib.sha256(b'['); logs = set()
    def consume(key, pos, row, encoded):
        if pos >= n:
            raise ValueError('Full PROFILE selection exceeds registered population')
        fr, cr = next(future_rows, None), next(current_rows, None)
        if (fr is None or cr is None or any((row.get('token'), row.get('log')) != (r['token'], r['log']) for r in (fr, cr))
                or fr['timestamp_us'] != cr['timestamp_us'] or fr['frames'][0] != cr['frames'][0]):
            raise ValueError('Full PROFILE selected/request/current order, observation or time differs')
        for field in ('current_record_sha256', 'ego_record_sha256'):
            require_hash(row.get(field), field)
        token_hash.update(((',' if pos else '') + canonical(row['token'])).encode())
        selection_hash.update(((',' if pos else '') + encoded).encode()); logs.add(row['log'])
        if len(logs) > 20000: raise ValueError('Full PROFILE log metadata exceeds existing ledger bound')
    selected_header, counts, _ = _atomic_rows(path, {'scenes'}, consume)
    token_hash.update(b']'); selection_hash.update(b']')
    if (version(path) != stamp or checked_sha(path)[0] != selected['sha256']
            or counts.get('scenes') != n or next(future_rows, None) is not None or next(current_rows, None) is not None
            or selected_header.get('schema') != 'st_fixed_scene_list_v1' or selected_header.get('split') != 'train'
            or selected_header.get('source_index_hash') != header['source_index_sha256']
            or selected_header.get('source_student_identity') != header['source_student_identity']
            or selected_header.get('ego_identity_sha256') != binding.get('ego_identity_sha256')
            or selected_header.get('registration_identity') != reg['identity']
            or selected_header.get('selection_rule') != 'entire_original_index_in_original_order'
            or selection_hash.hexdigest() != header.get('selection_identity')
            or selection_hash.hexdigest() != source.manifest.get('selection_identity')
            or token_hash.hexdigest() != binding.get('ordered_scene_ids_hash') or len(logs) != binding.get('logs')):
        raise ValueError('Full PROFILE complete ordered population/selected source hashes differ')
    request.check(); source.check()


def validate_profile_recipe(admission, cfg):
    fields={'schema','scope','parent_descriptor','source_migration','data_manifest','recipe',
            'reference_recipe','max_scenes','max_updates','new_population_tensor_qualified','formal_approved'}
    full = is_full_profile(admission)
    if full: fields |= {'population', 'consumption'}
    if (not isinstance(admission,dict) or set(admission)!=fields
            or admission['schema'] not in (PROFILE_SCHEMA, FULL_PROFILE_SCHEMA)
            or admission['scope']!='PROFILE' or type(admission['max_scenes']) is not int
            or (admission['max_scenes'] <= 256 if full else admission['max_scenes'] != 256)
            or admission['max_updates']!=256
            or admission['new_population_tensor_qualified'] is not False or admission['formal_approved'] is not False):
        raise ValueError('Invalid bounded PROFILE admission; no formal promotion is supported')
    reference=pinned(admission['recipe'])
    original=pinned(admission['reference_recipe'])
    if (original['execution']['scope']!='SMOKE' or original['execution']['stop_after']>4
            or original['data'].get('teacher_provider',{}).get('descriptor')!=admission['parent_descriptor']):
        raise ValueError('PROFILE reference must be the original exact32 provider SMOKE recipe')
    def mathematical_recipe(c):
        c=deepcopy(c)
        for key in ('scope','sample_limit','stop_after','world_size','micro_batch','global_batch'): c['execution'].pop(key,None)
        for key in ('output_root','max_seconds','save_every','profile_mode','loader_workers'): c['runtime'].pop(key,None)
        c['data'].pop('teacher_provider',None); c['data']['teacher_root']=None
        if 'official_execution' in c['runtime']:
            for key in ('declared_recipe_overrides','adaptation_reason'): c['runtime']['official_execution'].pop(key,None)
        return c
    if mathematical_recipe(reference)!=mathematical_recipe(original):
        raise ValueError('PROFILE changed the original selected model/optimizer/PCA/precision recipe')
    ex=cfg['execution']; rt=cfg['runtime']
    if (ex['scope']!='PROFILE' or not 1<=ex['sample_limit']<=admission['max_scenes'] or not 1<=ex['stop_after']<=256
            or ex['family']!='temporal' or ex['max_updates']!=20000 or cfg['optimizer']['warmup']!=1000
            or rt['deterministic'] is not True or rt['gradient_checkpointing'] is not False
            or rt.get('official_execution',{}).get('preset')!='drivedreamer_f35fe783'
            or cfg['data']['history_root'] is not None
            or profile_recipe_identity(cfg)!=profile_recipe_identity(reference)):
        raise ValueError('PROFILE recipe/population/budget differs from its explicit original-policy binding')
    if full and (ex['sample_limit'] != admission['max_scenes']
            or identity_hash(admission['consumption']) != identity_hash(full_profile_consumption(cfg))):
        raise ValueError('Full PROFILE consumption must preregister exact cumulative updates and exposures')
    return reference


def _qualified_profile_execution(d, manifest, scene_count):
    from tools.st_world_policy.qualify_indexed_provider import validate_receipt
    a=d['profile_admission']; recipe=validate_profile_recipe(a,pinned(a['recipe']))
    parent=pinned(a['parent_descriptor']); migration=pinned(a['source_migration'])
    if (parent.get('schema')!=INDEXED_SCHEMA or 'profile_admission' in parent
            or parent.get('identity')!=identity_hash({k:v for k,v in parent.items() if k!='identity'})
            or migration['parent_descriptor']!=a['parent_descriptor']):
        raise ValueError('PROFILE must derive from one original exact32 indexed descriptor')
    plan=pinned(parent['qualification_plan']); result=pinned(parent['qualification'])
    validate_receipt(plan,result,parent,control_migration=migration)
    if (d['qualification']!=parent['qualification'] or d['qualification_plan']!=parent['qualification_plan']
            or d['sources']!=parent['sources'] or d['execution']!=parent['execution']
            or d['teacher_config']!=parent['teacher_config'] or d['limits']!=parent['limits']
            or d['provider']['contract']!=parent['provider']['contract']):
        raise ValueError('PROFILE changed qualified source/numerical/teacher/LRU contract')
    original=pinned(parent['request_index']['complete'])['manifest_header']
    for key in ('split','source_student_identity','source_index_sha256','partition_sha256','times_s','view_names'):
        if manifest.get(key)!=original.get(key):
            raise ValueError('PROFILE population changed legal source/partition/time/view lineage: '+key)
    if is_full_profile(a):
        if scene_count != a['max_scenes'] or recipe['execution']['sample_limit'] != scene_count:
            raise ValueError('Full PROFILE actual index population differs from registered count')
        return pinned(d['teacher_config']),TeacherFeatureContract.from_dict(d['provider']['contract']),plan
    selected=pinned(a['data_manifest'])
    rows=selected.get('scenes',[])
    if (manifest['split']!='train' or manifest['times_s']!=[0.,.5,1.,2.,4.]
            or not 1<=scene_count<=256 or len(rows)!=scene_count or recipe['execution']['sample_limit']!=scene_count
            or selected.get('schema')!='st_fixed_scene_list_v1' or selected.get('split')!='train'
            or selected.get('source_index_hash')!=manifest['source_index_sha256']):
        raise ValueError('PROFILE fixed dataset/teacher population does not match')
    return pinned(d['teacher_config']),TeacherFeatureContract.from_dict(d['provider']['contract']),plan


def _qualified_execution(d, manifest, scene_count):
    """Old exact admission is unchanged; PROFILE has its own source/population binding."""
    if "profile_admission" in d:
        if d.get("schema") != INDEXED_SCHEMA:
            raise ValueError("PROFILE requires the indexed provider")
        return _qualified_profile_execution(d,manifest,scene_count)
    if manifest['split'] != 'train' or not 1 <= scene_count <= 64:
        raise ValueError('Initial provider wiring is bounded to 64 train scenes')
    cfg, qualification = pinned(d['teacher_config']), pinned(d['qualification'])
    plan = pinned(d['qualification_plan'])
    fixed = d['execution'].get('schema') == 'st_fixed8_dino_execution_v1'
    if plan.get('schema') == 'st_indexed_composition_plan_v1':
        # The prior body qualification is engineering evidence only. A new
        # population/source needs its own actual mixed-layout result.
        from tools.st_world_policy.qualify_indexed_provider import validate_receipt
        if (d.get('schema') != INDEXED_SCHEMA or not fixed or scene_count > 32 or
                manifest['times_s'] != [0., .5, 1., 2., 4.] or
                manifest['identity'] != plan['training_admission']['exact_request_identity']):
            raise ValueError('Indexed composition is exact-population <=32-scene SMOKE only')
        validate_receipt(plan, qualification, d)
        contract = TeacherFeatureContract.from_dict(d['provider']['contract'])
        if (cfg.get('command') != 'build' or cfg.get('variant') not in ('SP2', 'SP3') or
                cfg.get('storage_dtype') != 'float32' or plan['contract'] != contract.to_dict()):
            raise ValueError('Indexed composition teacher config/contract changed')
        return cfg, contract, plan
    if fixed and manifest['times_s'] != [0., .5, 1., 2., 4.]:
        raise ValueError('Fixed qualification requires the complete physical future table')
    plan_schema = 'st_fixed_teacher_qualification_plan_v1' if fixed else 'st_ondemand_qualification_plan_v1'
    if (plan.get('schema') != plan_schema or
            plan.get('identity') != identity_hash({k: v for k, v in plan.items() if k != 'identity'}) or
            qualification.get('plan_identity') != plan['identity'] or
            Path(plan['requests_path']).resolve() != Path(d['requests']['path']).resolve() or
            Path(plan['teacher_config']).resolve() != Path(d['teacher_config']['path']).resolve() or
            not all(any(p['path'] == item['path'] and p['sha256'] == item['sha256'] for p in plan['inputs'])
                    for item in (d['requests'], d['teacher_config']))):
        raise ValueError('Provider request/teacher population is outside completed qualification plan')
    if (qualification.get('errors') != [] or qualification.get('tensor_disk_bytes') != 0 or
            qualification.get('execution') != d['execution'] or
            qualification.get('construction_rng_unchanged') is not True):
        raise ValueError('Provider requires completed real teacher qualification evidence, including numerical failures')
    if fixed:
        validate_fixed_qualification(plan, qualification, d['execution'])
    elif (qualification.get('schema') != 'st_ondemand_teacher_qualification_v1' or
            qualification.get('status') not in ('BITWISE_TARGET_QUALIFIED', 'NUMERICAL_REVIEW_REQUIRED')):
        raise ValueError('Provider requires completed real teacher qualification evidence, including numerical failures')
    if cfg.get('command') != 'build' or cfg.get('variant') not in ('SP2', 'SP3') or cfg.get('storage_dtype') != 'float32':
        raise ValueError('Only the actual native FP32 teacher build config is supported initially')
    contract = TeacherFeatureContract.from_dict(d['provider']['contract'])
    if plan['contract'] != contract.to_dict():
        raise ValueError('Provider contract differs from real qualification plan')
    record_bytes = contract.output_grid_hw[0] * contract.output_grid_hw[1] * (contract.channels * 4 + 1)
    if fixed:
        budget = plan['budget']
        if (d['limits']['max_cached_bytes'] > budget['resident_provider_bytes'] or
                d['limits']['max_scene_bytes'] > budget['max_scene_bytes'] or
                d['limits']['max_batch_images'] != 8):
            raise ValueError('Fixed provider exceeds the actually qualified RAM/scene/batch budget')
    elif d['limits']['max_cached_bytes'] >= record_bytes:
        raise ValueError('Initial trainer requires RAM reuse disabled: partial-LRU batch-layout resume is not qualified')
    execution = d['execution']
    if (execution.get('schema') != ('st_fixed8_dino_execution_v1' if fixed else 'st_ondemand_dino_execution_v1') or
            execution.get('contract_identity') != contract.identity or
            execution.get('max_batch_images') != d['limits']['max_batch_images'] or
            execution.get('teacher', {}).get('compute_dtype') != 'float32' or
            identity_hash(execution['teacher']) != contract.teacher_identity or
            identity_hash(execution['transform']) != contract.transform_identity):
        raise ValueError('Provider execution is not the exact raw FP32 teacher contract')
    return cfg, contract, plan


def validate_descriptor(d):
    """Validate parsed metadata before immutable publication or runtime use."""
    if d.get('schema') == INDEXED_SCHEMA:
        return _validate_indexed_descriptor(d)
    fields = {'schema', 'requests', 'sources', 'teacher_config', 'qualification', 'qualification_plan', 'execution',
              'limits', 'provider', 'identity'}
    if (set(d) != fields or d['schema'] != 'st_teacher_provider_dependency_v1' or
            d['identity'] != identity_hash({k: v for k, v in d.items() if k != 'identity'})):
        raise ValueError('Provider descriptor identity/schema changed')
    _validate_limits(d)
    manifest = pinned(d['requests'])
    cfg, contract, plan = _qualified_execution(d, manifest, len(manifest['scenes']))
    execution = d['execution']
    sources = []
    if not isinstance(d['sources'], list) or len(d['sources']) != 1:
        raise ValueError('Initial trainer requires exactly the qualified COMPLETE current source')
    for source in d['sources']:
        metadata = pinned(source)
        if metadata['identity'] != plan['current_identity']:
            raise ValueError('Provider source differs from qualified current cache; miss layouts would change')
        root = Path(source['path']).parent
        read_small(root / 'requests.json')
        sources.append(TeacherCacheReader(root, expected_identity=metadata['identity'], expected_contract=contract))
    placeholder = SimpleNamespace(contract_identity=contract.identity, execution_identity=identity_hash(execution))
    provider = BoundedTeacherProvider(manifest, contract, sources, executor=placeholder,
        execution_identity=placeholder.execution_identity, **d['limits'])
    if provider.identity != d['provider']:
        raise ValueError('Provider implementation/population/runtime identity changed')
    return d, cfg, provider


def _validate_indexed_descriptor(d, *, existing=None):
    from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader, IndexedBoundedTeacherProvider
    fields = {'schema', 'requests', 'request_index', 'sources', 'teacher_config', 'qualification',
              'qualification_plan', 'execution', 'limits', 'provider', 'identity'}
    if 'profile_admission' in d: fields.add('profile_admission')
    if (set(d) != fields or d['identity'] != identity_hash({k:v for k,v in d.items() if k != 'identity'})):
        raise ValueError('Indexed provider descriptor identity/schema changed')
    _validate_limits(d)
    if not isinstance(d['sources'], list) or len(d['sources']) != 1:
        raise ValueError('Initial indexed trainer requires exactly the qualified COMPLETE current source')
    if existing is not None and (not isinstance(existing, IndexedBoundedTeacherProvider) or len(existing.sources) != 1):
        raise ValueError('Indexed descriptor requires its actual owned provider')
    indices = []
    try:
        index = open_metadata_index(d['request_index'], existing=existing.metadata_index if existing else None)
        if existing is None: indices.append(index)
        if 'profile_admission' in d and not is_full_profile(d['profile_admission']):
            if not 1<=index.receipt['scene_count']<=256:
                raise ValueError('PROFILE request population exceeds 256 before row materialization')
            rows=pinned(d['profile_admission']['data_manifest'])['scenes']
            if len(rows)>256 or [(r['token'],r['log']) for r in rows] != [(r['token'],r['log']) for r in index.manifest['scenes']]:
                raise ValueError('PROFILE dataset order/scene IDs differ from request index')
            for r in rows:
                require_hash(r.get('current_record_sha256'),'PROFILE current record')
                require_hash(r.get('ego_record_sha256'),'PROFILE ego record')
        if index.receipt['cache_identity'] is not None or index.receipt['dependencies']['requests'] != d['requests']:
            raise ValueError('Indexed request source/raw SHA differs from the explicit request pin')
        # A header and count suffice: the index has streamed and validated every
        # original scene. Never materialize its lazy population to rehash JSON.
        cfg, contract, plan = _qualified_execution(d, index.receipt['manifest_header'], index.receipt['scene_count'])
        if d['execution'].get('schema') != 'st_fixed8_dino_execution_v1':
            raise ValueError('Indexed trainer requires the qualified fixed8 execution policy')
        sources = []
        for source in d['sources']:
            if not isinstance(source, dict) or set(source) != {'index', 'cache_identity'}:
                raise ValueError('Indexed source must bind its index and original COMPLETE cache identity')
            require_hash(source['cache_identity'], 'source cache identity')
            source_index = open_metadata_index(source['index'],
                existing=existing.sources[0].metadata_index if existing else None)
            if existing is None: indices.append(source_index)
            if source['cache_identity'] != plan['current_identity']:
                raise ValueError('Provider source differs from qualified current cache; miss layouts would change')
            sources.append(IndexedTeacherCacheReader(source_index, expected_identity=source['cache_identity'],
                expected_contract=contract))
        if is_full_profile(d.get('profile_admission')):
            _validate_full_population(d, index, source_index)
        placeholder = SimpleNamespace(contract_identity=contract.identity, execution_identity=identity_hash(d['execution']))
        provider = IndexedBoundedTeacherProvider(index, contract, sources, executor=placeholder,
            execution_identity=placeholder.execution_identity, **d['limits'])
        if provider.identity != d['provider']:
            raise ValueError('Indexed provider implementation/population/runtime identity changed')
        if existing is not None and existing.identity != provider.identity:
            raise ValueError('Owned indexed provider identity differs from its descriptor')
        return d, cfg, existing if existing is not None else provider
    except BaseException:
        for index in indices:
            index.close()
        raise


def create_provider(record, *, device, executor_factory=None):
    d, cfg, provider = load_descriptor(record)
    if executor_factory is None:
        if d['execution']['schema'] == 'st_fixed8_dino_execution_v1':
            from tools.st_world_policy.fixed_batch_teacher import FixedBatchDINOExecutor
            executor_factory = FixedBatchDINOExecutor
        else:
            from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
            executor_factory = FrozenDINOExecutor
    try:
        executor = executor_factory(cfg['teacher_root'], contract=provider.contract, variant=cfg['variant'],
            device=device, max_batch_images=d['limits']['max_batch_images'],
            expected_execution_identity=provider.execution_identity)
        if (executor.execution != d['execution'] or executor.contract_identity != provider.contract.identity or
                executor.execution_identity != provider.execution_identity):
            raise ValueError('Constructed teacher differs from explicit provider dependency')
    except BaseException:
        close_provider_metadata(provider)
        raise
    provider.executor = executor
    return provider


def runtime_metrics(provider, rank):
    """Invocation counters are observability only, never a consumed cursor."""
    return {'provider_identity': provider.identity['identity'], 'rank': rank,
            'counter_scope': 'this_process_invocation_not_checkpoint_progress',
            **provider.memory_report(), **({'fixed_compute': deepcopy(provider.executor.compute_stats)}
                if hasattr(provider.executor, 'compute_stats') else {})}


def prepare(requests, sources, teacher_config, qualification, output, *, limits):
    """Version metadata only. A numerical review is recorded, never promoted."""
    report = read_small(qualification)
    execution = report.get('execution', {})
    readers = []
    for root in sources:
        root = Path(root)
        metadata = read_small(root / 'identity.json'); read_small(root / 'requests.json')
        readers.append(TeacherCacheReader(root, expected_identity=metadata['identity']))
    if not readers:
        raise ValueError('Initial trainer wiring requires an audited COMPLETE current source')
    contract = readers[0].contract
    placeholder = SimpleNamespace(contract_identity=contract.identity, execution_identity=identity_hash(execution))
    provider = BoundedTeacherProvider(read_small(requests), contract, readers, executor=placeholder,
        execution_identity=placeholder.execution_identity, **limits)
    d = {'schema': 'st_teacher_provider_dependency_v1', 'requests': pin(requests),
         'sources': [pin(r.root / 'identity.json') for r in readers],
         'teacher_config': pin(teacher_config), 'qualification': pin(qualification),
         'qualification_plan': pin(Path(qualification).parent / 'RESOLVED_PLAN.json'),
         'execution': execution, 'limits': limits, 'provider': provider.identity}
    d['identity'] = identity_hash(d)
    # Validate BEFORE publication: a failed request cannot poison an immutable
    # namespace with an unusable descriptor. No teacher tensor/basis is produced.
    validate_descriptor(d)
    atomic_json(output, d, immutable=True)
    return d


def prepare_indexed(request_index, source_indices, teacher_config, qualification, output, *, limits,
                    max_metadata_cache_bytes=16 * MiB, profile_admission=None):
    """Select existing immutable indices only; never build metadata or tensors."""
    from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader, IndexedBoundedTeacherProvider
    if len(source_indices) != 1:
        raise ValueError('Initial indexed trainer requires exactly the qualified COMPLETE current source')
    bindings = [index_binding(root, max_metadata_cache_bytes=max_metadata_cache_bytes)
                for root in [request_index, *source_indices]]
    indices = []
    try:
        for binding in bindings:
            indices.append(open_metadata_index(binding))
        request, source = indices
        contract = TeacherFeatureContract.from_dict(request.receipt['contract'])
        reader = IndexedTeacherCacheReader(source, expected_identity=source.receipt['cache_identity'],
                                          expected_contract=contract)
        execution = read_small(qualification).get('execution', {})
        placeholder = SimpleNamespace(contract_identity=contract.identity, execution_identity=identity_hash(execution))
        provider = IndexedBoundedTeacherProvider(request, contract, [reader], executor=placeholder,
            execution_identity=placeholder.execution_identity, **limits)
        d = {'schema': INDEXED_SCHEMA, 'requests': request.receipt['dependencies']['requests'],
             'request_index': bindings[0], 'sources': [{'index': bindings[1],
                 'cache_identity': source.receipt['cache_identity']}],
             'teacher_config': pin(teacher_config), 'qualification': pin(qualification),
             'qualification_plan': pin(Path(qualification).parent / 'RESOLVED_PLAN.json'),
             'execution': execution, 'limits': limits, 'provider': provider.identity}
        if profile_admission is not None: d['profile_admission']=profile_admission
        d['identity'] = identity_hash(d)
    finally:
        for index in indices:
            index.close()
    _, _, preview = validate_descriptor(d)
    close_provider_metadata(preview)
    atomic_json(output, d, immutable=True)
    return d


def configure_smoke(config_path, descriptor_path, fit_cache_root, output, *, fit_cache_index=None):
    """Retain the selected cell/basis/optimizer; write a NEW diagnostic config."""
    from tools.st_world_policy.training_config import validate_config
    from tools.st_world_policy.provider_pca import register_provider_scope
    cfg = deepcopy(read_small(config_path))
    st = validate_config(cfg)
    is_profile=cfg['execution']['scope']=='PROFILE'
    if not is_profile and (cfg['execution']['scope'] != 'SMOKE' or cfg['execution']['stop_after'] > 4):
        raise ValueError('Supply an already selected SMOKE config of at most four updates; no cell is chosen here')
    if fit_cache_index is not None and not cfg['data']['pca_path']:
        raise ValueError('PCA fit metadata index supplied without an actual frozen PCA basis')
    d, _, provider = load_descriptor(pin(descriptor_path))
    try:
        if is_profile:
            if 'profile_admission' not in d: raise ValueError('SMOKE-only descriptor cannot configure PROFILE')
            validate_profile_recipe(d['profile_admission'],cfg)
        elif 'profile_admission' in d:
            raise ValueError('PROFILE descriptor cannot be relabeled as SMOKE')
        if cfg['execution']['sample_limit'] != len(provider.manifest['scenes']):
            raise ValueError('Selected training config population differs from provider manifest')
        if cfg['runtime']['deterministic'] != d['execution']['deterministic_algorithms']:
            raise ValueError('Teacher qualified execution and student deterministic policy differ')
        output = Path(output).resolve()
        if output == Path(config_path).parent.resolve():
            raise ValueError('Provider diagnostics require a new config/output namespace')
        scope = None
        if cfg['data']['pca_path']:
            if fit_cache_root is None:
                raise ValueError('Explicit original PCA fit cache required; no refit is allowed')
            scope_path = output / 'PROVIDER_CHANNEL_SCOPE.json'
            from starVLA.dataloader.st_world_dataset import projected_teacher_contract
            receipt = read_small(Path(cfg['data']['pca_path']) / 'PCA.json')
            effective = projected_teacher_contract(provider.contract, pca_identity=receipt['identity'],
                                                   dimension=st.teacher.channels)
            if effective != st.teacher:
                raise ValueError('Selected target contract does not use this provider and the same frozen PCA basis')
            close_provider_metadata(provider)  # Only immutable identity fields are used below.
            register_provider_scope(cfg['data']['pca_path'], fit_cache_root, pin(descriptor_path), scope_path,
                                    **({'fit_cache_index': fit_cache_index} if fit_cache_index is not None else {}))
            scope = pin(scope_path)
        elif provider.contract != st.teacher:
            raise ValueError('Selected raw target contract differs from provider')
        cfg['data'].update(teacher_root=None, teacher_provider={
            'descriptor': pin(descriptor_path), 'pca_channel_scope': scope})
        cfg['runtime'].update(loader_workers=0, output_root=str(output / 'runs'))
        validate_config(cfg)
        atomic_json(output / 'training_config.json', cfg, immutable=True)
        lineage = {
            'source_config': pin(config_path), 'new_config': pin(output / 'training_config.json'),
            'provider_descriptor': pin(descriptor_path), 'teacher_manifest_artifact': pin(descriptor_path),
            'numerical_status': pinned(d['qualification'])['status'],
            'scope':cfg['execution']['scope'], 'new_population_tensor_qualified':not is_profile,
            'pca_refitted': False, 'formal_approved': False, 'training_started': False,
            'loader_workers': 0, 'ram_reuse': ('qualified_fixed8_bounded_LRU' if
                d['execution']['schema'] == 'st_fixed8_dino_execution_v1' and
                d['limits']['max_cached_bytes'] >= provider.record_bytes else
                'disabled_pending_partial_eviction_qualification')}
        if d['execution']['schema'] == 'st_fixed8_dino_execution_v1':
            qualification = pinned(d['qualification'])
            lineage['fixed_policy_repeatability'] = qualification['fixed_policy_repeatability']
            lineage['legacy_reference_compatibility'] = qualification['legacy_reference_compatibility']
            lineage['fixed_execution_identity'] = qualification['execution_identity']
            lineage['resident_limit_bytes'] = provider.max_cached_bytes
        atomic_json(output / 'PROVIDER_CONFIG_LINEAGE.json', lineage, immutable=True)
        return str(output / 'training_config.json')
    finally:
        close_provider_metadata(provider)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    requests = p.add_mutually_exclusive_group(required=True)
    requests.add_argument('--requests', help='Original eager request JSON (legacy wiring)')
    requests.add_argument('--request-index', help='Existing verified request metadata-index root; no index build')
    p.add_argument('--source-root', action='append')
    p.add_argument('--source-index', action='append', help='Existing index of the exact qualified COMPLETE current cache')
    p.add_argument('--max-metadata-cache-mib', type=int, default=16,
                   help='Per SQLite index allowance; combined provider allowance <=256 MiB')
    p.add_argument('--teacher-config', required=True)
    p.add_argument('--qualification-result', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-cached-bytes', type=int, default=1,
        help='Legacy must stay below one record; fixed8 may use only its truly qualified LRU budget')
    p.add_argument('--max-scene-mib', type=int, default=64)
    p.add_argument('--max-batch-images', type=int, default=8)
    p.add_argument('--training-config', help='Selected <=4-update SMOKE or explicit bounded PROFILE config; preserves cell and basis')
    p.add_argument('--pca-fit-cache-root', help='Exact existing PCA fitted cache; metadata-only provenance')
    p.add_argument('--pca-fit-cache-index', help='Required existing fit-cache index for the indexed PCA scope')
    p.add_argument('--configured-output', help='New isolated namespace for selectable training_config.json')
    p.add_argument('--profile-parent-descriptor', help='Explicit existing exact32 composition descriptor')
    p.add_argument('--profile-source-migration', help='Pinned reviewed source-control migration JSON')
    p.add_argument('--profile-data-manifest', help='Exact pinned selected dataset; <=256 unless explicit full-population admission')
    p.add_argument('--profile-reference-config', help='Actual original exact32 provider SMOKE config; model/recipe stays fixed')
    p.add_argument('--profile-full-population', action='store_true',
                   help='Explicit whole original train index with <=256 cumulative updates; not FORMAL')
    p.add_argument('--profile-current-registration', help='Original full current population REGISTRATION.json')
    p.add_argument('--profile-current-complete', help='Original full current population COMPLETE.json')
    p.add_argument('--profile-future-complete', help='Actual full_population future preparation COMPLETE.json')
    p.add_argument('--profile-consumption-budget', help='Predeclared cumulative diagnostic or fixed world8 timing240 consumption JSON')
    args = p.parse_args()
    if bool(args.training_config) != bool(args.configured_output):
        p.error('--training-config and --configured-output must be supplied together')
    if args.request_index:
        if args.source_root or not args.source_index:
            p.error('--request-index requires --source-index and rejects --source-root')
        if args.pca_fit_cache_index and not args.training_config:
            p.error('--pca-fit-cache-index requires --training-config')
    elif not args.source_root or args.source_index or args.pca_fit_cache_index or args.max_metadata_cache_mib != 16:
        p.error('Legacy --requests requires --source-root and rejects indexed-only options')
    profile=None
    flags=[args.profile_parent_descriptor,args.profile_source_migration,args.profile_data_manifest,args.profile_reference_config]
    full_flags = [args.profile_current_registration, args.profile_current_complete,
                  args.profile_future_complete, args.profile_consumption_budget]
    if args.profile_full_population != all(bool(x) for x in full_flags) or (any(full_flags) and not args.profile_full_population):
        p.error('Full-index PROFILE requires the explicit flag and all four population/consumption bindings')
    if args.profile_full_population and not all(flags):
        p.error('Full-index PROFILE also requires all four original PROFILE bindings')
    if any(flags):
        if not all(flags) or not args.request_index or not args.training_config:
            p.error('PROFILE requires all four profile bindings, request index and explicit training config')
        profile={'schema':PROFILE_SCHEMA,'scope':'PROFILE','parent_descriptor':pin(flags[0]),
            'source_migration':pin(flags[1]),'data_manifest':pin(flags[2]),'recipe':pin(args.training_config),
            'reference_recipe':pin(flags[3]),
            'max_scenes':256,'max_updates':256,'new_population_tensor_qualified':False,'formal_approved':False}
        if args.profile_full_population:
            profile.update(schema=FULL_PROFILE_SCHEMA,
                max_scenes=read_small(args.training_config)['execution']['sample_limit'],
                population=dict(zip(('current_registration','current_complete','future_complete'),
                                    (pin(x) for x in full_flags[:3]))),
                consumption=read_small(args.profile_consumption_budget))
        validate_profile_recipe(profile,read_small(args.training_config))
    limits = {'max_cached_bytes': args.max_cached_bytes, 'max_scene_bytes': args.max_scene_mib * MiB,
              'max_batch_images': args.max_batch_images, 'max_manifest_bytes': 32 * MiB}
    result = (prepare_indexed(args.request_index, args.source_index, args.teacher_config, args.qualification_result,
              args.output, limits=limits, max_metadata_cache_bytes=args.max_metadata_cache_mib * MiB, profile_admission=profile)
              if args.request_index else
              prepare(args.requests, args.source_root, args.teacher_config, args.qualification_result,
                      args.output, limits=limits))
    print(json.dumps({'descriptor': pin(args.output), 'provider_identity': result['provider']['identity'],
                      'tensor_disk_bytes': 0, 'formal_approved': False}))
    if args.training_config:
        print(json.dumps({'training_config': configure_smoke(args.training_config, args.output,
              args.pca_fit_cache_root, args.configured_output, fit_cache_index=args.pca_fit_cache_index), 'started': False}))


if __name__ == '__main__':
    main()
