"""CPU fixtures test wiring, not real-teacher numerical/training equivalence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from starVLA.dataloader.st_world_dataset import STWorldTrainingDataset
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheWriter, TeacherCacheReader
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider
from tools.st_world_policy.teacher_provider import (
    pin, prepare, load_descriptor, create_provider, configure_smoke, validate_option, runtime_metrics)
from tools.st_world_policy.provider_pca import register_provider_scope
from tools.st_world_policy.teacher_pca import TeacherPCA
from tools.st_world_policy.training_config import validate_config
from test_dataset import setup_data
from test_pca import scoped_fixture
from test_training_config import config
from test_ondemand_teacher import fixture as tensor_fixture, Executor


def write(path, value):
    path.write_text(json.dumps(value)); return path


def provider_artifacts(tmp_path):
    readers, pca, receipt = scoped_fixture(tmp_path)
    fit = readers[0]
    manifest = deepcopy(fit.manifest)
    manifest.update(source_student_identity='d'*64, source_index_sha256='e'*64, partition_sha256='f'*64)
    manifest['identity'] = identity_hash({k: v for k, v in manifest.items() if k != 'identity'})
    current = TeacherCacheWriter(tmp_path / 'current', fit.contract, manifest, execution=fit.identity['execution'])
    current.commit_shard(0, torch.zeros(1, 18, 24, 1024), torch.ones(1, 18, 24, dtype=torch.bool)); current.finalize()
    requests = write(tmp_path / 'requests.json', manifest)
    cfg = write(tmp_path / 'teacher_config.json', {'command': 'build', 'variant': 'SP2',
        'storage_dtype': 'float32', 'teacher_root': str(tmp_path / 'unused-weights')})
    execution = {'schema': 'st_ondemand_dino_execution_v1', **fit.identity['execution'],
        'contract_identity': fit.contract.identity, 'max_batch_images': 8, 'deterministic_algorithms': False}
    plan = {'schema': 'st_ondemand_qualification_plan_v1', 'requests_path': str(requests),
        'teacher_config': str(cfg), 'inputs': [pin(requests), pin(cfg)],
        'contract': fit.contract.to_dict(), 'current_identity': current.identity['identity']}
    plan['identity'] = identity_hash(plan)
    write(tmp_path / 'RESOLVED_PLAN.json', plan)
    report = write(tmp_path / 'RESULT.json', {'schema': 'st_ondemand_teacher_qualification_v1',
        'plan_identity': plan['identity'], 'status': 'NUMERICAL_REVIEW_REQUIRED', 'errors': [],
        'tensor_disk_bytes': 0, 'execution': execution, 'construction_rng_unchanged': True})
    descriptor = tmp_path / 'PROVIDER.json'
    d = prepare(requests, [current.root], cfg, report, descriptor,
        limits={'max_cached_bytes': 1, 'max_scene_bytes': 64 << 20,
                'max_batch_images': 8, 'max_manifest_bytes': 32 << 20})
    return SimpleNamespace(descriptor=descriptor, d=d, pca=pca, receipt=receipt,
        fit=fit, requests=requests, report=report, current=current)


def test_descriptor_prepare_has_no_teacher_and_pins_actual_population(tmp_path):
    a = provider_artifacts(tmp_path)
    d, _, p = load_descriptor(pin(a.descriptor))
    assert d == a.d and p.identity['identity'] == d['provider']['identity']
    assert p.max_cached_bytes == 1 and not hasattr(p.executor, 'model')
    assert not (tmp_path / 'unused-weights').exists()
    changed = json.loads(a.requests.read_text()); changed['partition_sha256'] = '1'*64
    write(a.requests, changed)
    with pytest.raises(ValueError, match='pinned dependency changed'):
        load_descriptor(pin(a.descriptor))


def test_runtime_factory_is_explicit_strict_and_no_default_teacher_in_tests(tmp_path):
    a = provider_artifacts(tmp_path); seen = []
    def factory(root, **kwargs):
        seen.append((root, kwargs))
        return SimpleNamespace(execution=a.d['execution'], execution_identity=identity_hash(a.d['execution']),
            contract_identity=a.fit.contract.identity)
    p = create_provider(pin(a.descriptor), device='cpu', executor_factory=factory)
    assert len(seen) == 1 and seen[0][1]['expected_execution_identity'] == p.execution_identity
    def wrong(*args, **kwargs):
        return SimpleNamespace(execution={}, execution_identity='a'*64, contract_identity=p.contract.identity)
    with pytest.raises(ValueError, match='Constructed teacher differs'):
        create_provider(pin(a.descriptor), device='cpu', executor_factory=wrong)


@pytest.mark.parametrize('field,value', [('max_cached_bytes', 2 << 20), ('max_batch_images', 9)])
def test_partial_lru_and_unqualified_compute_batch_refused(tmp_path, field, value):
    a = provider_artifacts(tmp_path); d = deepcopy(a.d); d['limits'][field] = value
    d['identity'] = identity_hash({k:v for k,v in d.items() if k != 'identity'})
    path = write(tmp_path / 'changed.json', d)
    with pytest.raises(ValueError, match='partial-LRU|bounds'):
        load_descriptor(pin(path))


@pytest.mark.parametrize('status', ['STARTING', 'PARTIAL_BUDGET', 'FAILED'])
def test_incomplete_qualification_not_consumed(tmp_path, status):
    a = provider_artifacts(tmp_path); report = json.loads(a.report.read_text()); report['status'] = status
    write(a.report, report); d = deepcopy(a.d); d['qualification'] = pin(a.report)
    d['identity'] = identity_hash({k:v for k,v in d.items() if k != 'identity'})
    path = write(tmp_path / 'changed.json', d)
    with pytest.raises(ValueError, match='completed real teacher qualification'):
        load_descriptor(pin(path))
    unpublished = tmp_path / 'must_not_be_published.json'
    with pytest.raises(ValueError, match='completed real teacher qualification'):
        prepare(a.requests, [a.current.root], a.d['teacher_config']['path'], a.report, unpublished,
                limits=a.d['limits'])
    assert not unpublished.exists()


def test_provider_scope_reuses_exact_basis_without_fake_cache_identity(tmp_path):
    a = provider_artifacts(tmp_path); _, _, p = load_descriptor(pin(a.descriptor))
    before = {n: (a.pca / n).read_bytes() for n in ('PCA.json', 'COMPLETE.json', 'statistics.safetensors')}
    scope_path = tmp_path / 'PROVIDER_SCOPE.json'
    scope = register_provider_scope(a.pca, a.fit.root, pin(a.descriptor), scope_path)
    kwargs = dict(dimension=2, expected_identity=a.receipt['identity'], expected_teacher_contract=p.contract.identity)
    result = TeacherPCA.load(a.pca, **kwargs, provider_scope=pin(scope_path), target_provider=p)
    assert result.channel_scope_identity == scope['identity']
    assert scope['statistics_refitted'] is False
    assert all((a.pca / n).read_bytes() == v for n,v in before.items())
    exact = TeacherPCA.load(a.pca, **kwargs)
    z = torch.arange(1024, dtype=torch.float32).reshape(1,1024)
    assert torch.equal(result.transform(z), exact.transform(z))
    with pytest.raises(ValueError, match='separate from cache identity'):
        TeacherPCA.load(a.pca, **kwargs, provider_scope=pin(scope_path), target_provider=p,
                        target_cache_identity=a.fit.identity['identity'])
    p.identity = dict(p.identity, identity='1'*64)
    with pytest.raises(ValueError, match='identity mismatch'):
        TeacherPCA.load(a.pca, **kwargs, provider_scope=pin(scope_path), target_provider=p)


def test_dataset_injection_matches_old_masks_and_preserves_old_identity(tmp_path):
    args = setup_data(tmp_path)
    old = STWorldTrainingDataset(**args, need_future=True)
    before = deepcopy(old.identity)
    reader = TeacherCacheReader(args['teacher_root'], expected_identity=args['expected_teacher_identity'])
    p = BoundedTeacherProvider(reader.manifest, reader.contract, [reader], max_cached_bytes=1,
        max_scene_bytes=10000, max_batch_images=8, max_manifest_bytes=100000)
    new_args = dict(args, teacher_root=None, expected_teacher_identity=p.identity['identity'], teacher_provider=p)
    new = STWorldTrainingDataset(**new_args, need_future=True)
    assert old.identity == before and STWorldTrainingDataset(**args, need_future=True).identity == before
    assert new.identity['schema'] == 'st_training_dataset_provider_v1'
    assert 'teacher_cache_identity' not in new.identity
    assert new.identity['teacher_provider_identity'] == p.identity['identity']
    a, b = old[0], new[0]
    assert all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    saved = p.state_dict(); p.load_state_dict(saved)
    assert all(torch.equal(b[1][k], new[0][1][k]) for k in b[1])
    assert not b[1]['future_valid'].any()  # True missing physical frames stay missing.
    with pytest.raises(ValueError, match='type/identity'):
        STWorldTrainingDataset(**dict(new_args, expected_teacher_identity=args['expected_teacher_identity']))


def test_config_is_explicit_smoke_synchronous_and_immutable_default(tmp_path):
    value = config(); before = deepcopy(value); validate_config(value); assert value == before
    value['execution'].update(scope='SMOKE', stop_after=2)
    value['runtime'].update(loader_workers=0, profile_mode='none')
    option = {'descriptor': {'path':'/explicit/PROVIDER.json', 'sha256':'a'*64},
              'pca_channel_scope': {'path':'/explicit/SCOPE.json', 'sha256':'b'*64}}
    value['data'].update(teacher_root=None, teacher_provider=option)
    validate_config(value)
    for workers, scope, root in [(1,'SMOKE',None), (0,'FORMAL',None), (0,'SMOKE','/old-cache')]:
        with pytest.raises(ValueError):
            validate_option(option,loader_workers=workers,teacher_root=root,pca_path='/basis',scope=scope)
    wrong = deepcopy(value); wrong['data']['teacher_provider']['fallback_online'] = True
    with pytest.raises(ValueError, match='missing/unknown'):
        validate_config(wrong)


def test_no_ram_mode_keeps_batch_layout_and_rng_at_cold_restore(tmp_path):
    contract, manifest = tensor_fixture(tmp_path)
    class BatchSensitiveExecutor(Executor):
        def compute_batch(self, records, **kwargs):
            batch = super().compute_batch(records, **kwargs)
            # Simulates a real floating-point kernel depending on batch size;
            # this is NOT a numerical tolerance or real teacher comparison.
            batch.features.add_(float(len(records)))
            return batch
    executor = BatchSensitiveExecutor(contract)
    p = BoundedTeacherProvider(manifest,contract,executor=executor,
        execution_identity=executor.execution_identity,max_cached_bytes=1,
        max_scene_bytes=10000,max_batch_images=2,max_manifest_bytes=100000)
    rng = torch.get_rng_state().clone()
    first = p.read_scene('one'); checkpoint = p.state_dict()
    repeated = p.read_scene('one'); p.load_state_dict(checkpoint); resumed = p.read_scene('one')
    assert torch.equal(rng,torch.get_rng_state())
    for name in ('current_dense_features','future_dense_features','current_valid','future_valid'):
        assert torch.equal(first[name],repeated[name]) and torch.equal(first[name],resumed[name])
    assert [len(x) for x in executor.calls] == [2,2,1]*3
    assert p.memory_report()['resident_bytes'] == 0
    metrics = runtime_metrics(p, rank=0)
    assert metrics['batches'] == 9 and metrics['computed_images'] == 15
    assert metrics['immutable_hits'] == 0 and metrics['tensor_disk_bytes'] == 0
    assert metrics['counter_scope'] == 'this_process_invocation_not_checkpoint_progress'
    assert p.state_dict() == checkpoint  # Counters are never training progress.


def test_selectable_config_keeps_cell_basis_recipe_and_rejects_backend_change(tmp_path):
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    a = provider_artifacts(tmp_path)
    value = config(); value['execution'].update(scope='SMOKE',stop_after=2,sample_limit=1)
    value['runtime'].update(deterministic=False,profile_mode='none')
    value['data']['pca_path'] = str(a.pca)
    value['space_time']['teacher'] = projected_teacher_contract(a.fit.contract,
        pca_identity=a.receipt['identity'],dimension=2).to_dict()
    source = write(tmp_path / 'input_config.json', value)
    output = tmp_path / 'diagnostic'
    result = configure_smoke(source, a.descriptor, a.fit.root, output)
    resolved = json.loads(Path(result).read_text())
    assert resolved['space_time'] == value['space_time'] and resolved['optimizer'] == value['optimizer']
    assert resolved['data']['pca_path'] == value['data']['pca_path']
    assert resolved['runtime']['loader_workers'] == 0
    assert json.loads(source.read_text()) == value
    assert resolved['data']['teacher_root'] is None
    assert json.loads((output / 'PROVIDER_CONFIG_LINEAGE.json').read_text())['formal_approved'] is False
    value['runtime']['deterministic'] = True; write(source, value)
    with pytest.raises(ValueError, match='deterministic policy differ'):
        configure_smoke(source, a.descriptor, a.fit.root, tmp_path / 'rejected')
