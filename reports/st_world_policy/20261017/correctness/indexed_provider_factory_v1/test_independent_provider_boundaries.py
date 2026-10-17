"""Independent tiny CPU fixtures; no actual assets, teacher or training."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from test_indexed_provider_factory import artifacts, prepare
from test_fixed_provider_training import sign
from test_provider_training import write
from starVLA.dataloader.st_world_metadata_index import MetadataIndex
from tools.st_world_policy import teacher_provider as tp
from tools.st_world_policy import provider_pca as pp
from tools.st_world_policy.teacher_pca import TeacherPCA


def is_closed(index):
    try:
        index.db.execute('SELECT 1')
    except sqlite3.ProgrammingError:
        return True
    return False


def track_indices(monkeypatch):
    opened = []; original = MetadataIndex.__init__
    def init(index, *args, **kwargs):
        original(index, *args, **kwargs); opened.append(index)
    monkeypatch.setattr(MetadataIndex, '__init__', init)
    return opened


def test_legacy_scope_cannot_be_rebound_to_indexed_provider(tmp_path, monkeypatch):
    a = artifacts(tmp_path); prepare(a)
    path = tmp_path/'legacy_scope.json'
    scope = pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.fixed_path), path)
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    scope['provider_descriptor'] = tp.pin(a.indexed_path)
    scope['provider_identity'] = provider.identity['identity']
    scope['provider_contract_identity'] = provider.contract.identity
    write(path, sign(scope))
    previews = track_indices(monkeypatch)
    try:
        # A valid indexed target must require the new bounded/indexed scope.
        # Re-signing an old scope may not select eager validation and extra readers.
        with pytest.raises(ValueError):
            TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
                expected_teacher_contract=provider.contract.identity,
                provider_scope=tp.pin(path), target_provider=provider)
    finally:
        tp.close_provider_metadata(provider)
        for index in previews: index.close()


def test_pca_preview_never_opens_over_combined_metadata_allowance(tmp_path, monkeypatch):
    a = artifacts(tmp_path); prepare(a, max_metadata_cache_bytes=128 << 20)
    opened = []; peak = [0]; original = MetadataIndex.__init__
    def init(index, *args, **kwargs):
        original(index, *args, **kwargs); opened.append(index)
        live = sum(i.max_metadata_cache_bytes for i in opened if not is_closed(i))
        peak[0] = max(peak[0], live)
    monkeypatch.setattr(MetadataIndex, '__init__', init)
    with pytest.raises(ValueError, match='256 MiB'):
        pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), tmp_path/'rejected.json',
                                   fit_cache_index=a.fit_index)
    assert not (tmp_path/'rejected.json').exists()
    assert all(is_closed(i) for i in opened)
    assert peak[0] <= 256 << 20, f'Concurrent declared reader allowance {peak[0]}'


def test_indexed_pca_owns_only_fit_preview_and_preserves_live_provider(tmp_path, monkeypatch):
    a = artifacts(tmp_path); prepare(a)
    scope_path = tmp_path/'scope.json'
    opened = track_indices(monkeypatch)
    pp.register_provider_scope(a.pca, a.fit.root, tp.pin(a.indexed_path), scope_path,
                               fit_cache_index=a.fit_index)
    assert len(opened) == 3 and all(is_closed(i) for i in opened)
    _, _, provider = tp.load_descriptor(tp.pin(a.indexed_path))
    before = len(opened)
    owned = [provider.metadata_index, provider.sources[0].metadata_index]
    try:
        TeacherPCA.load(a.pca, dimension=2, expected_identity=a.receipt['identity'],
            expected_teacher_contract=provider.contract.identity,
            provider_scope=tp.pin(scope_path), target_provider=provider)
        assert len(opened)-before == 1  # Fit only; no duplicate request/source.
        assert is_closed(opened[-1]) and all(not is_closed(i) for i in owned)
    finally:
        tp.close_provider_metadata(provider)


@pytest.mark.parametrize('failure', ['executor_exception', 'executor_identity'])
def test_executor_construction_failure_closes_all_index_previews(tmp_path, monkeypatch, failure):
    a = artifacts(tmp_path); prepare(a)
    opened = track_indices(monkeypatch)
    def executor(*args, **kwargs):
        if failure == 'executor_exception': raise RuntimeError('synthetic construction failure')
        from types import SimpleNamespace
        return SimpleNamespace(execution={}, contract_identity='bad', execution_identity='bad')
    with pytest.raises((RuntimeError, ValueError)):
        tp.create_provider(tp.pin(a.indexed_path), device='cpu', executor_factory=executor)
    assert len(opened) == 2 and all(is_closed(i) for i in opened)


@pytest.mark.parametrize('mutation', ['rng', 'partial_lru', 'fixed8_slots', 'legacy_claim'])
def test_indexed_path_retains_original_fixed_execution_gates(tmp_path, mutation):
    a = artifacts(tmp_path)
    q = json.loads(a.report.read_text())
    if mutation == 'rng': q['rng_forward_checks'] -= 1
    elif mutation == 'partial_lru': q['cases'][tp.FIXED_CASES[-1]]['partial_lru_exercised'] = False
    elif mutation == 'fixed8_slots': q['compute_stats']['compute_image_slots'] -= 1
    else: q['legacy_reference_compatibility'] = False
    write(a.report, sign(q))
    with pytest.raises(ValueError): prepare(a)
    assert not a.indexed_path.exists()


def test_exact_request_bytes_cannot_be_replaced_by_same_semantic_json(tmp_path):
    a = artifacts(tmp_path); d = prepare(a)
    original = json.loads(a.requests.read_text())
    a.requests.write_text(json.dumps(original, separators=(',', ':')))
    with pytest.raises(ValueError): tp.validate_descriptor(deepcopy(d))
