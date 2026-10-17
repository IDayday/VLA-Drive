"""Explicit application scope for a new provider and an unchanged PCA basis."""
from pathlib import Path

from starVLA.dataloader.st_world_schema import identity_hash, TeacherFeatureContract
from starVLA.dataloader.st_world_targets import atomic_json, TeacherCacheReader
from tools.st_world_policy.teacher_provider import load_descriptor, pinned, pin, read_small
from tools.st_world_policy.teacher_pca import TeacherPCA, teacher_channel_family


def register_provider_scope(pca_root, fit_cache_root, descriptor, output):
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
