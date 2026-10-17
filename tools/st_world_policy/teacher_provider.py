"""Explicit RAM-only teacher dependency for the existing ST trainer.

Preparing metadata never loads weights. Runtime creation is synchronous on the
owning rank after its backend is configured; the existing collective data error
guard remains responsible for all-rank failure. No cache completion is forged.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash, require_hash
from starVLA.dataloader.st_world_targets import TeacherCacheReader, atomic_json
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider

MiB = 1 << 20
LIMIT_FIELDS = {'max_cached_bytes', 'max_scene_bytes', 'max_batch_images', 'max_manifest_bytes'}
INDEXED_SCHEMA = 'st_indexed_teacher_provider_dependency_v1'
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


def validate_option(option, *, loader_workers, teacher_root, pca_path, scope):
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
    if scope != 'SMOKE':
        raise ValueError('Initial provider wiring is SMOKE-only pending real trainer numerical/resume qualification')


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


def _qualified_execution(d, manifest, scene_count):
    """Both metadata representations retain the exact existing finite admission."""
    if manifest['split'] != 'train' or not 1 <= scene_count <= 64:
        raise ValueError('Initial provider wiring is bounded to 64 train scenes')
    cfg, qualification = pinned(d['teacher_config']), pinned(d['qualification'])
    plan = pinned(d['qualification_plan'])
    fixed = d['execution'].get('schema') == 'st_fixed8_dino_execution_v1'
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
                    max_metadata_cache_bytes=16 * MiB):
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
    if cfg['execution']['scope'] != 'SMOKE' or cfg['execution']['stop_after'] > 4:
        raise ValueError('Supply an already selected SMOKE config of at most four updates; no cell is chosen here')
    if fit_cache_index is not None and not cfg['data']['pca_path']:
        raise ValueError('PCA fit metadata index supplied without an actual frozen PCA basis')
    d, _, provider = load_descriptor(pin(descriptor_path))
    try:
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
    p.add_argument('--training-config', help='Existing selected <=4 update SMOKE config; preserves cell and basis')
    p.add_argument('--pca-fit-cache-root', help='Exact existing PCA fitted cache; metadata-only provenance')
    p.add_argument('--pca-fit-cache-index', help='Required existing fit-cache index for the indexed PCA scope')
    p.add_argument('--configured-output', help='New isolated namespace for selectable training_config.json')
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
    limits = {'max_cached_bytes': args.max_cached_bytes, 'max_scene_bytes': args.max_scene_mib * MiB,
              'max_batch_images': args.max_batch_images, 'max_manifest_bytes': 32 * MiB}
    result = (prepare_indexed(args.request_index, args.source_index, args.teacher_config, args.qualification_result,
              args.output, limits=limits, max_metadata_cache_bytes=args.max_metadata_cache_mib * MiB)
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
