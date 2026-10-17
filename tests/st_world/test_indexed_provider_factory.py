"""Synthetic finite fixtures only: no real teacher qualification or training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest
import torch

from starVLA.dataloader.st_world_targets import atomic_json, TeacherCacheReader, TeacherCacheWriter
from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_metadata_index import build_index, Rows
from starVLA.dataloader.st_world_ondemand import TeacherBatch
from starVLA.dataloader.st_world_indexed_provider import IndexedBoundedTeacherProvider
from starVLA.dataloader.st_world_schema import identity_hash
from tools.st_world_policy import teacher_provider as tp
from tools.st_world_policy.provider_pca import register_provider_scope, INDEXED_SCOPE
from tools.st_world_policy.teacher_pca import TeacherPCA
from test_fixed_provider_training import fixed_artifacts, prepare_fixed, sign
from test_provider_training import write
from test_training_config import config


def artifacts(root):
    a = fixed_artifacts(root, legacy_ok=True)
    # The older PCA toy fixture lacks the source hashes required by indexing.
    # Make a NEW fully sourced synthetic fit namespace with the same zero toy
    # features/basis; never relax the production index provenance requirements.
    fit_manifest = deepcopy(a.current.manifest)
    fit_manifest['scenes'][0].update(token='fit-only', log='fit-log')
    sign(fit_manifest)
    fit = TeacherCacheWriter(root / 'sourced_fit', a.fit.contract, fit_manifest, execution=a.fit.identity['execution'])
    fit.commit_shard(0, torch.zeros(1,18,24,1024), torch.ones(1,18,24,dtype=torch.bool)); fit.finalize()
    a.fit = fit
    pca = root / 'sourced_pca'; pca.mkdir()
    shutil.copyfile(a.pca / 'statistics.safetensors', pca / 'statistics.safetensors')
    a.receipt = sign(dict(a.receipt, fit_cache_identity=fit.identity['identity']))
    write(pca / 'PCA.json', a.receipt)
    write(pca / 'COMPLETE.json', {'identity':a.receipt['identity'], 'pca_json_sha256':file_sha256(pca / 'PCA.json')})
    a.pca = pca
    # The production index accepts the original atomic_json format. Repin this
    # tiny fixture BEFORE its synthetic qualification; never migrate real pins.
    m = json.loads(a.requests.read_text()); atomic_json(a.requests, m)
    plan = json.loads((root / 'RESOLVED_PLAN.json').read_text())
    plan['inputs'][0] = tp.pin(a.requests); write(root / 'RESOLVED_PLAN.json', sign(plan))
    q = json.loads(a.report.read_text()); q['plan_identity'] = plan['identity']; write(a.report, sign(q))
    prepare_fixed(a)
    a.request_index = root / 'request_index'
    build_index(a.request_index, requests=a.requests, expected_request_identity=m['identity'],
                contract=a.fit.contract, max_scenes=64)
    for name, reader in [('source_index', a.current), ('fit_index', a.fit)]:
        target = root / name
        build_index(target, requests=reader.root / 'requests.json', cache_root=reader.root,
            expected_request_identity=reader.manifest['identity'], expected_cache_identity=reader.identity['identity'],
            contract=reader.contract, max_scenes=64)
        setattr(a, name, target)
    a.indexed_path = root / 'INDEXED_PROVIDER.json'
    return a


def prepare(a, **kwargs):
    return tp.prepare_indexed(a.request_index, [a.source_index], a.d['teacher_config']['path'],
                             a.report, a.indexed_path, limits=a.fixed_limits, **kwargs)


def forbid_eager(monkeypatch, a):
    original = Path.read_text
    forbidden = {a.requests, *(r.root / n for r in (a.current, a.fit) for n in ('identity.json', 'requests.json'))}
    def read_text(path, *args, **kwargs):
        assert path not in forbidden, 'Population-sized JSON was parsed eagerly'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    def fail(*args, **kwargs):
        raise AssertionError('Eager reader or population iteration is forbidden')
    monkeypatch.setattr(TeacherCacheReader, '__init__', fail)
    monkeypatch.setattr(Rows, '__iter__', fail)


def fake_factory(execution):
    class Executor:
        def __init__(self, root, **kw):
            self.execution = execution
            self.execution_identity = identity_hash(execution)
            self.contract_identity = kw['contract'].identity
            self.calls = []
        def compute_batch(self, records, *, contract, execution_identity):
            self.calls.append(tuple(r['key'] for r in records))
            z = torch.stack([torch.full((*contract.output_grid_hw, contract.channels),
                int(r['key'][:4], 16) / 65536, dtype=torch.float32) for r in records])
            return TeacherBatch(self.calls[-1], self.contract_identity, execution_identity,
                                z, torch.ones(z.shape[:-1], dtype=torch.bool))
    return Executor


def test_indexed_factory_no_eager_parse_and_preserves_keys_math_masks_rng_resume(tmp_path, monkeypatch):
    a = artifacts(tmp_path)
    old = tp.create_provider(tp.pin(a.fixed_path), device='cpu', executor_factory=fake_factory(json.loads(a.report.read_text())['execution']))
    forbid_eager(monkeypatch, a)
    d = prepare(a)
    assert prepare(a) == d  # Same metadata identity is immutable/idempotent.
    new = tp.create_provider(tp.pin(a.indexed_path), device='cpu', executor_factory=fake_factory(d['execution']))
    try:
        assert isinstance(new, IndexedBoundedTeacherProvider)
        assert new.identity != old.identity and new.identity['training_admission'] is False
        assert new.identity['request_manifest_identity'] == old.identity['request_manifest_identity']
        token = a.current.manifest['scenes'][0]['token']
        rng = torch.get_rng_state().clone()
        first, second = old.read_scene(token), new.read_scene(token)
        assert old.executor.calls == new.executor.calls
        names = ('current_dense_features','future_dense_features','current_valid','future_valid')
        assert all(torch.equal(first[k], second[k]) for k in names)
        saved = new.state_dict(); new.read_scene(token); new.load_state_dict(saved)
        assert new.memory_report()['resident_bytes'] == 0
        assert all(torch.equal(second[k], new.read_scene(token)[k]) for k in names)
        assert torch.equal(rng, torch.get_rng_state())
        assert new.memory_report()['metadata_cache_budget_bytes'] == 32 << 20
        with pytest.raises(ValueError, match='identity changed'): new.load_state_dict(old.state_dict())
    finally:
        tp.close_provider_metadata(new)


def test_indexed_pca_basis_unchanged_scope_distinct_and_no_eager_fit_records(tmp_path, monkeypatch):
    a = artifacts(tmp_path); prepare(a)
    old_scope = tmp_path / 'OLD_SCOPE.json'
    register_provider_scope(a.pca, a.fit.root, tp.pin(a.fixed_path), old_scope)
    before = {p.name: p.read_bytes() for p in a.pca.iterdir() if p.is_file()}
    exact = TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                          expected_teacher_contract=a.fit.contract.identity)
    forbid_eager(monkeypatch, a)
    scope_path = tmp_path / 'INDEXED_SCOPE.json'
    scope = register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), scope_path,
                                    fit_cache_index=a.fit_index)
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    try:
        assert scope['schema'] == INDEXED_SCOPE and scope['statistics_refitted'] is False
        assert scope['fit_cache_identity'] == a.receipt['fit_cache_identity']
        assert scope['fit_cache_metadata']['path'] == str(a.fit.root / 'identity.json')
        loaded = TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
            expected_teacher_contract=provider.contract.identity, provider_scope=tp.pin(scope_path), target_provider=provider)
        z = torch.arange(3 * 1024, dtype=torch.float32).reshape(3, 1024)
        assert torch.equal(loaded.transform(z), exact.transform(z))
        assert loaded.channel_scope_identity != json.loads(old_scope.read_text())['identity']
        assert before == {p.name: p.read_bytes() for p in a.pca.iterdir() if p.is_file()}
        with pytest.raises(ValueError):
            TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                expected_teacher_contract=provider.contract.identity, provider_scope=tp.pin(old_scope), target_provider=provider)
    finally:
        tp.close_provider_metadata(provider)


@pytest.mark.parametrize('mutation', ['request_sha','request_index_sha','source_identity','source_index_sha',
    'database','original_source','scope_flag','unknown_field','budget','population_identity','wrong_source_index'])
def test_wrong_index_source_and_hash_rejected(tmp_path, mutation):
    a = artifacts(tmp_path); d = prepare(a)
    if mutation == 'request_sha': d['requests']['sha256'] = '1' * 64
    elif mutation == 'request_index_sha': d['request_index']['complete']['sha256'] = '2' * 64
    elif mutation == 'source_identity': d['sources'][0]['cache_identity'] = '3' * 64
    elif mutation == 'source_index_sha': d['sources'][0]['index']['complete']['sha256'] = '4' * 64
    elif mutation == 'database':
        with (a.request_index / 'metadata.sqlite').open('ab') as f: f.write(b'changed')
    elif mutation == 'original_source':
        with (a.current.root / 'requests.json').open('a') as f: f.write(' ')
    elif mutation == 'scope_flag': d['formal_approved'] = True
    elif mutation == 'unknown_field': d['sources'][0]['trust_source'] = True
    elif mutation == 'budget': d['request_index']['max_metadata_cache_bytes'] = 257 << 20
    elif mutation == 'population_identity': d['request_index']['request_identity'] = '5' * 64
    elif mutation == 'wrong_source_index': d['sources'][0]['index'] = tp.index_binding(a.fit_index)
    sign(d); bad = write(tmp_path / 'BAD.json', d)
    with pytest.raises((ValueError, KeyError)): tp.load_descriptor(tp.pin(bad))


@pytest.mark.parametrize('mutation', ['fit_root','fit_index','fit_raw_sha','basis','provider','refit','field'])
def test_indexed_pca_does_not_launder_source_basis_or_provider(tmp_path, mutation):
    a = artifacts(tmp_path); prepare(a)
    scope_path = tmp_path / 'SCOPE.json'
    scope = register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), scope_path,
                                    fit_cache_index=a.fit_index)
    if mutation == 'fit_root': scope['fit_cache_root'] = str(a.current.root)
    elif mutation == 'fit_index': scope['fit_cache_index'] = tp.index_binding(a.source_index)
    elif mutation == 'fit_raw_sha': scope['fit_cache_metadata']['sha256'] = '3' * 64
    elif mutation == 'basis': scope['pca_tensor_sha256'] = '4' * 64
    elif mutation == 'provider': scope['provider_descriptor'] = tp.pin(a.fixed_path)
    elif mutation == 'refit': scope['statistics_refitted'] = True
    elif mutation == 'field': scope['target_cache_identity'] = a.current.identity['identity']
    write(scope_path, sign(scope))
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    try:
        with pytest.raises(ValueError):
            TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                expected_teacher_contract=provider.contract.identity, provider_scope=tp.pin(scope_path), target_provider=provider)
    finally:
        tp.close_provider_metadata(provider)


def test_indexed_smoke_configuration_is_selectable_and_retains_recipe_and_limits(tmp_path):
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    a = artifacts(tmp_path); prepare(a)
    cfg = config(); cfg['execution'].update(scope='SMOKE', stop_after=2, sample_limit=1)
    cfg['runtime'].update(deterministic=True, profile_mode='none')
    cfg['data']['pca_path'] = str(a.pca)
    cfg['space_time']['teacher'] = projected_teacher_contract(a.fit.contract,
        pca_identity=a.receipt['identity'], dimension=2).to_dict()
    path = write(tmp_path / 'INPUT.json', cfg)
    output = tmp_path / 'selected'
    result = tp.configure_smoke(path, a.indexed_path, a.fit.root, output, fit_cache_index=a.fit_index)
    actual = json.loads(Path(result).read_text())
    assert actual['optimizer'] == cfg['optimizer'] and actual['execution'] == cfg['execution']
    assert actual['runtime']['loader_workers'] == 0
    assert actual['data']['teacher_provider']['descriptor'] == tp.pin(a.indexed_path)
    assert json.loads((output / 'PROVIDER_CHANNEL_SCOPE.json').read_text())['schema'] == INDEXED_SCOPE
    with pytest.raises(ValueError, match='original fitted cache metadata index'):
        register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), tmp_path / 'missing.json')
    cfg['execution']['stop_after'] = 5; write(path, cfg)
    with pytest.raises(ValueError, match='at most four updates'):
        tp.configure_smoke(path, a.indexed_path, a.fit.root, tmp_path / 'rejected', fit_cache_index=a.fit_index)


@pytest.mark.parametrize('count', [2, 65])
def test_indexed_population_cannot_borrow_different_finite_qualification(tmp_path, count):
    a = artifacts(tmp_path)
    m = json.loads(a.requests.read_text()); base = m['scenes'][0]
    m['scenes'] = [dict(deepcopy(base), token=f'scene-{i}') for i in range(count)]
    changed = tmp_path / 'changed_requests.json'; atomic_json(changed, sign(m))
    a.request_index = tmp_path / 'changed_index'
    build_index(a.request_index, requests=changed, expected_request_identity=m['identity'],
                contract=a.fit.contract, max_scenes=65)
    with pytest.raises(ValueError, match='outside completed qualification plan|bounded to 64'):
        prepare(a)
    assert not a.indexed_path.exists()


@pytest.mark.parametrize('field', ['completed_all_cases','fixed_policy_repeatability','construction_rng_unchanged'])
def test_indexed_descriptor_does_not_bypass_completed_numerical_qualification(tmp_path, field):
    a = artifacts(tmp_path)
    report = json.loads(a.report.read_text()); report[field] = False; write(a.report, sign(report))
    with pytest.raises(ValueError): prepare(a)
    assert not a.indexed_path.exists()


def test_existing_provider_pca_revalidation_rejects_changed_metadata_version(tmp_path):
    a = artifacts(tmp_path); prepare(a)
    path = tmp_path / 'scope.json'
    register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), path, fit_cache_index=a.fit_index)
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    try:
        with (a.current.root / 'requests.json').open('a') as f: f.write(' ')
        with pytest.raises(ValueError, match='version changed'):
            TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                expected_teacher_contract=provider.contract.identity, provider_scope=tp.pin(path), target_provider=provider)
    finally:
        tp.close_provider_metadata(provider)


def test_actual_indexed_cli_metadata_only_and_strict_arguments(tmp_path):
    a = artifacts(tmp_path)
    command = [sys.executable, '-m', 'tools.st_world_policy.teacher_provider', '--request-index', str(a.request_index),
        '--source-index', str(a.source_index), '--teacher-config', a.d['teacher_config']['path'],
        '--qualification-result', str(a.report), '--output', str(a.indexed_path),
        '--max-cached-bytes', str(a.fixed_limits['max_cached_bytes'])]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result['tensor_disk_bytes'] == 0 and result['formal_approved'] is False
    assert not (tmp_path / 'unused-weights').exists()
    failed = subprocess.run(command + ['--source-root', str(a.current.root)], capture_output=True, text=True, timeout=40)
    assert failed.returncode != 0 and 'rejects --source-root' in failed.stderr
    failed = subprocess.run(command + ['--pca-fit-cache-index', str(a.fit_index)],
                            capture_output=True, text=True, timeout=40)
    assert failed.returncode != 0 and 'requires --training-config' in failed.stderr
