"""Explicit application scope for a new provider and an unchanged PCA basis."""
from pathlib import Path

from starVLA.dataloader.st_world_schema import identity_hash, TeacherFeatureContract
from starVLA.dataloader.st_world_targets import atomic_json, TeacherCacheReader
from tools.st_world_policy.teacher_provider import (
    load_descriptor, pinned, pin, read_small, INDEXED_SCHEMA,
    index_binding, open_metadata_index, close_provider_metadata, validate_bound_indexed_descriptor)
from tools.st_world_policy.teacher_pca import TeacherPCA, teacher_channel_family


INDEXED_SCOPE = 'st_pca_indexed_provider_channel_scope_v1'


def _check_indexed_preview_budget(fit_binding, *, descriptor=None, provider=None):
    """Admit the combined connection budget before opening any new index."""
    fit_bytes = fit_binding.get('max_metadata_cache_bytes')
    if type(fit_bytes) is not int or not (16 << 20) <= fit_bytes <= (256 << 20):
        raise ValueError('Invalid fitted-index metadata budget')
    if provider is not None:
        provider_bytes = provider.memory_report()['metadata_cache_budget_bytes']
    else:
        if (descriptor.get('schema') != INDEXED_SCHEMA or
                not isinstance(descriptor.get('sources'), list) or len(descriptor['sources']) != 1):
            raise ValueError('Indexed PCA requires the explicit indexed provider descriptor')
        bindings = [descriptor['request_index'], descriptor['sources'][0]['index']]
        sizes = [binding.get('max_metadata_cache_bytes') for binding in bindings]
        if any(type(n) is not int or not (16 << 20) <= n <= (256 << 20) for n in sizes):
            raise ValueError('Invalid provider-index metadata budget')
        provider_bytes = sum(sizes)
    if fit_bytes + provider_bytes > 256 << 20:
        raise ValueError('Combined provider/PCA preview metadata budget exceeds 256 MiB')


def register_provider_scope(pca_root, fit_cache_root, descriptor, output, *, fit_cache_index=None):
    if pinned(descriptor).get('schema') == INDEXED_SCHEMA:
        if fit_cache_index is None:
            raise ValueError('Indexed provider PCA requires the exact original fitted cache metadata index')
        return _register_indexed_scope(pca_root, fit_cache_root, descriptor, output, fit_cache_index)
    if fit_cache_index is not None:
        raise ValueError('Indexed PCA scope cannot be attached to a legacy provider descriptor')
    pca_root, fit_cache_root = Path(pca_root), Path(fit_cache_root)
    receipt = read_small(pca_root / 'PCA.json')
    fit_metadata = read_small(fit_cache_root / 'identity.json')
    read_small(fit_cache_root / 'requests.json')
    fit = TeacherCacheReader(fit_cache_root, expected_identity=receipt['fit_cache_identity'])
    TeacherPCA.load(pca_root, dimension=receipt['dimensions'][0], expected_identity=receipt['identity'],
                    expected_teacher_contract=receipt['teacher_contract_identity'])
    d, _, provider = load_descriptor(descriptor)
    family = teacher_channel_family(fit_metadata)
    if teacher_channel_family({'contract': provider.contract.to_dict(), 'execution': d['execution']}) != family:
        raise ValueError('Provider raw channel family differs from the frozen fitted basis')
    scope = {'schema': 'st_pca_provider_channel_scope_v1', 'pca_identity': receipt['identity'],
        'pca_receipt': pin(pca_root / 'PCA.json'), 'pca_tensor_sha256': receipt['tensor_sha256'],
        'fit_cache_identity': fit.identity['identity'], 'fit_contract_identity': fit.contract.identity,
        'fit_cache_metadata': pin(fit_cache_root / 'identity.json'),
        'fit_cache_complete': pin(fit_cache_root / 'COMPLETE.json'), 'channel_family': family,
        'provider_descriptor': descriptor, 'provider_identity': provider.identity['identity'],
        'provider_contract_identity': provider.contract.identity, 'statistics_refitted': False}
    scope['identity'] = identity_hash(scope)
    atomic_json(output, scope, immutable=True)
    return scope


def validate_provider_scope(record, receipt, expected_contract, provider):
    scope = pinned(record)
    if scope.get('schema') == INDEXED_SCOPE:
        return _validate_indexed_scope(scope, receipt, expected_contract, provider)
    if scope.get('schema') != 'st_pca_provider_channel_scope_v1':
        raise ValueError('Unknown PCA provider scope schema')
    if (provider.identity.get('schema') != 'st_bounded_teacher_provider_v1' or
            pinned(scope['provider_descriptor']).get('schema') != 'st_teacher_provider_dependency_v1'):
        raise ValueError('Legacy PCA scope cannot bind an indexed provider; explicit indexed scope required')
    if (scope.get('schema') != 'st_pca_provider_channel_scope_v1' or
            scope.get('identity') != identity_hash({k: v for k, v in scope.items() if k != 'identity'}) or
            scope['statistics_refitted'] is not False or scope['pca_identity'] != receipt['identity'] or
            pinned(scope['pca_receipt']) != receipt or scope['pca_tensor_sha256'] != receipt['tensor_sha256'] or
            scope['fit_cache_identity'] != receipt['fit_cache_identity'] or
            scope['fit_contract_identity'] != receipt['teacher_contract_identity'] or
            scope['provider_identity'] != provider.identity['identity'] or
            scope['provider_contract_identity'] != expected_contract):
        raise ValueError('PCA provider scope/basis/target identity mismatch')
    fit = pinned(scope['fit_cache_metadata']); done = pinned(scope['fit_cache_complete'])
    d, _, preview = load_descriptor(scope['provider_descriptor'])
    if (fit['identity'] != scope['fit_cache_identity'] or
            fit['identity'] != identity_hash({k: v for k, v in fit.items() if k != 'identity'}) or
            TeacherFeatureContract.from_dict(fit['contract']).identity != scope['fit_contract_identity'] or
            done['identity'] != fit['identity'] or done.get('status') != 'COMPLETE' or
            done['identity_sha256'] != scope['fit_cache_metadata']['sha256'] or
            preview.identity != provider.identity or
            teacher_channel_family(fit) != scope['channel_family'] or
            teacher_channel_family({'contract': provider.contract.to_dict(), 'execution': d['execution']}) != scope['channel_family']):
        raise ValueError('PCA provider channel provenance changed')
    return scope['identity']


def _register_indexed_scope(pca_root, fit_cache_root, descriptor, output, fit_cache_index):
    pca_root = Path(pca_root)
    receipt = read_small(pca_root / 'PCA.json')
    binding = index_binding(fit_cache_index)
    _check_indexed_preview_budget(binding, descriptor=pinned(descriptor))
    index = open_metadata_index(binding)
    provider = None
    try:
        if (index.receipt['cache_identity'] != receipt['fit_cache_identity'] or
                Path(index.receipt['cache_root']).resolve() != Path(fit_cache_root).resolve() or
                TeacherFeatureContract.from_dict(index.receipt['contract']).identity != receipt['teacher_contract_identity']):
            raise ValueError('Indexed PCA source is not the original fitted COMPLETE cache/contract/root')
        TeacherPCA.load(pca_root, dimension=receipt['dimensions'][0], expected_identity=receipt['identity'],
                        expected_teacher_contract=receipt['teacher_contract_identity'])
        d, _, provider = load_descriptor(descriptor)
        _check_indexed_preview_budget(binding, provider=provider)
        family = teacher_channel_family(index.receipt['cache_header'])
        if teacher_channel_family({'contract': provider.contract.to_dict(), 'execution': d['execution']}) != family:
            raise ValueError('Provider raw channel family differs from the frozen fitted basis')
        scope = {'schema': INDEXED_SCOPE, 'pca_identity': receipt['identity'],
            'pca_receipt': pin(pca_root / 'PCA.json'), 'pca_tensor_sha256': receipt['tensor_sha256'],
            'fit_cache_identity': receipt['fit_cache_identity'], 'fit_contract_identity': receipt['teacher_contract_identity'],
            'fit_cache_index': binding, 'fit_cache_root': str(Path(fit_cache_root).resolve()),
            'fit_cache_metadata': index.receipt['dependencies']['identity'],
            'fit_cache_complete': index.receipt['dependencies']['complete'], 'channel_family': family,
            'provider_descriptor': descriptor, 'provider_identity': provider.identity['identity'],
            'provider_contract_identity': provider.contract.identity, 'statistics_refitted': False}
        scope['identity'] = identity_hash(scope)
        _validate_indexed_scope(scope, receipt, provider.contract.identity, provider, fit_index=index)
        atomic_json(output, scope, immutable=True)
        return scope
    finally:
        index.close()
        if provider is not None:
            close_provider_metadata(provider)


def _validate_indexed_scope(scope, receipt, expected_contract, provider, *, fit_index=None):
    fields = {'schema', 'pca_identity', 'pca_receipt', 'pca_tensor_sha256', 'fit_cache_identity',
        'fit_contract_identity', 'fit_cache_index', 'fit_cache_root', 'fit_cache_metadata', 'fit_cache_complete',
        'channel_family', 'provider_descriptor', 'provider_identity', 'provider_contract_identity',
        'statistics_refitted', 'identity'}
    if (set(scope) != fields or scope['identity'] != identity_hash({k:v for k,v in scope.items() if k != 'identity'}) or
            scope['statistics_refitted'] is not False or scope['pca_identity'] != receipt['identity'] or
            pinned(scope['pca_receipt']) != receipt or scope['pca_tensor_sha256'] != receipt['tensor_sha256'] or
            scope['fit_cache_identity'] != receipt['fit_cache_identity'] or
            scope['fit_contract_identity'] != receipt['teacher_contract_identity'] or
            scope['provider_identity'] != provider.identity['identity'] or
            scope['provider_contract_identity'] != expected_contract or
            provider.identity.get('schema') != 'st_indexed_bounded_teacher_provider_v1'):
        raise ValueError('Indexed PCA provider scope/basis/target identity mismatch')
    # Exact raw-file pins are verified by MetadataIndex. The compact header has
    # no records and is NOT assigned/rehashed as a new original cache identity.
    _check_indexed_preview_budget(scope['fit_cache_index'], provider=provider)
    index = open_metadata_index(scope['fit_cache_index'], existing=fit_index)
    try:
        d = validate_bound_indexed_descriptor(scope['provider_descriptor'], provider)
        fit = index.receipt
        if (d['schema'] != INDEXED_SCHEMA or
                fit['cache_identity'] != scope['fit_cache_identity'] or fit['cache_header'] is None or
                Path(fit['cache_root']).resolve() != Path(scope['fit_cache_root']).resolve() or
                TeacherFeatureContract.from_dict(fit['contract']).identity != scope['fit_contract_identity'] or
                fit['dependencies'].get('identity') != scope['fit_cache_metadata'] or
                fit['dependencies'].get('complete') != scope['fit_cache_complete'] or
                teacher_channel_family(fit['cache_header']) != scope['channel_family'] or
                teacher_channel_family({'contract': provider.contract.to_dict(), 'execution': d['execution']}) != scope['channel_family']):
            raise ValueError('Indexed PCA fitted source/provider channel provenance changed')
        return scope['identity']
    finally:
        if fit_index is None:
            index.close()
