"""Independent admission tests: generated tiny CPU metadata only."""
import sqlite3

import pytest

from test_indexed_provider_factory import artifacts, prepare
from test_fixed_provider_training import sign
from test_provider_training import write
from starVLA.dataloader.st_world_metadata_index import MetadataIndex
from tools.st_world_policy import teacher_provider as tp
from tools.st_world_policy import provider_pca as pp
from tools.st_world_policy.teacher_pca import TeacherPCA


def closed(index):
    try:
        index.db.execute('SELECT 1')
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.mark.parametrize('fit_mib,allowed', [(240, False), (224, True)])
def test_runtime_scope_checks_budget_before_open_and_keeps_owned_readers(tmp_path, monkeypatch, fit_mib, allowed):
    a = artifacts(tmp_path); prepare(a)
    scope_path = tmp_path/'scope.json'
    scope = pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), scope_path,
                                      fit_cache_index=a.fit_index)
    # Provider owns 16 + 16 MiB. 224 + 32 is exactly permitted; 240 + 32 is not.
    scope['fit_cache_index']['max_metadata_cache_bytes'] = fit_mib << 20
    write(scope_path, sign(scope))
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    owned = [provider.metadata_index, provider.sources[0].metadata_index]
    opened = []; original = MetadataIndex.__init__
    def init(index, *args, **kwargs):
        original(index, *args, **kwargs); opened.append(index)
    monkeypatch.setattr(MetadataIndex, '__init__', init)
    def load():
        return TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
            expected_teacher_contract=provider.contract.identity,
            provider_scope=tp.pin(scope_path), target_provider=provider)
    try:
        if allowed:
            assert load().channel_scope_identity == scope['identity']
            assert len(opened) == 1 and closed(opened[0])
        else:
            with pytest.raises(ValueError, match='256 MiB'):
                load()
            assert opened == [], 'Runtime budget rejection came after opening the fitted index'
        assert all(not closed(index) for index in owned)
    finally:
        tp.close_provider_metadata(provider)
        for index in opened: index.close()


@pytest.mark.parametrize('indexed_target', [False, True])
def test_legacy_indexed_mix_rejected_before_eager_fit_or_preview(tmp_path, monkeypatch, indexed_target):
    a = artifacts(tmp_path); prepare(a)
    scope_path = tmp_path/'legacy.json'
    scope = pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.fixed_path), scope_path)
    target_path = a.indexed_path if indexed_target else a.fixed_path
    _, _, provider = tp.load_descriptor(tp.pin(target_path))
    scope['provider_descriptor'] = tp.pin(a.indexed_path)
    scope['provider_identity'] = provider.identity['identity']
    scope['provider_contract_identity'] = provider.contract.identity
    write(scope_path, sign(scope))
    original = pp.pinned
    def no_eager_fit(record):
        assert record not in (scope['fit_cache_metadata'], scope['fit_cache_complete']), 'Eager fit metadata read before schema rejection'
        return original(record)
    def no_preview(*args, **kwargs):
        raise AssertionError('Extra provider preview before schema rejection')
    monkeypatch.setattr(pp, 'pinned', no_eager_fit)
    monkeypatch.setattr(pp, 'load_descriptor', no_preview)
    try:
        with pytest.raises(ValueError):
            TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                expected_teacher_contract=provider.contract.identity,
                provider_scope=tp.pin(scope_path), target_provider=provider)
    finally:
        tp.close_provider_metadata(provider)


def test_registration_exact_combined_budget_is_admitted_and_closed(tmp_path, monkeypatch):
    a = artifacts(tmp_path); prepare(a, max_metadata_cache_bytes=120 << 20)
    # 120 MiB request + 120 MiB source + 16 MiB fitted index = 256 MiB.
    opened = []; original = MetadataIndex.__init__
    def init(index, *args, **kwargs):
        original(index, *args, **kwargs); opened.append(index)
    monkeypatch.setattr(MetadataIndex, '__init__', init)
    path = tmp_path/'scope.json'
    scope = pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), path,
                                      fit_cache_index=a.fit_index)
    assert scope['schema'] == pp.INDEXED_SCOPE and path.is_file()
    assert len(opened) == 3 and all(closed(index) for index in opened)
