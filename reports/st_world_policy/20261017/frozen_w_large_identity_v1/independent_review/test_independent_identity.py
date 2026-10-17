"""Independent CPU-only controls. No real run/ledger/model/image is opened."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'tests' / 'st_world'))
from test_spatial_w_large_identity import document, write
from tools.st_world_policy import cache_storage_budget as b, metadata_storage as m, spatial_w_storage as w


@pytest.fixture(autouse=True)
def clean_projection_lru():
    w._RUN_IDENTITIES.clear()
    yield
    w._RUN_IDENTITIES.clear()


def test_large_exception_does_not_raise_generic_control_limit(tmp_path):
    path = tmp_path / 'identity.json'
    write(path, document(population='a' * (16 << 20)))
    raw = path.read_bytes()
    assert 16 << 20 < len(raw) < 32 << 20
    assert m.CONTROL_LIMIT == 16 << 20
    with pytest.raises(b.StorageBudgetError, match='size/version'):
        m.pinned_json(path, hashlib.sha256(raw).hexdigest())
    result = w._run_identity_projection(path)
    assert result['bytes'] == len(raw)
    assert result['raw_sha256'] == hashlib.sha256(raw).hexdigest()


def test_exact_32mib_valid_json_and_next_byte_rejected(tmp_path):
    path = tmp_path / 'identity.json'
    base = document(population='')
    overhead = len((json.dumps(base, sort_keys=True, indent=2) + '\n').encode())
    value = document(population='a' * ((32 << 20) - overhead))
    write(path, value)
    assert path.stat().st_size == w.RUN_IDENTITY_LIMIT == 32 << 20
    result = w._run_identity_projection(path)
    assert result['identity'] == value['identity']
    assert len(json.dumps(dict(result))) < 2048
    w._RUN_IDENTITIES.clear()
    with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(b.StorageBudgetError, match='oversized'):
        w._run_identity_projection(path)


def test_parallel_lookup_opens_exactly_once_and_keeps_no_population(tmp_path, monkeypatch):
    path = tmp_path / 'identity.json'
    write(path, document(population=[{'index': i} for i in range(100)]))
    original, opened = os.open, []
    def record(value, *args, **kwargs):
        opened.append(str(value))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(w.os, 'open', record)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: w._run_identity_projection(path), range(32)))
    assert opened == [str(path)]
    assert all(value is results[0] for value in results)
    assert len(w._RUN_IDENTITIES) == 1
    assert not any(isinstance(value, (dict, list)) for value in results[0].values())


def test_semantic_population_corruption_cannot_hide_behind_projection_fields(tmp_path):
    path = tmp_path / 'identity.json'
    value = document(population=['original'])
    value['population'] = ['changed without resealing']
    write(path, value)
    with pytest.raises(b.StorageBudgetError, match='semantic identity'):
        w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


@pytest.mark.parametrize('raw', [b'{"schema":"one","schema":"two"}', b'{"number":NaN}'])
def test_json_duplicate_and_nonfinite_rejected_not_cached(tmp_path, raw):
    path = tmp_path / 'identity.json'; path.write_bytes(raw)
    with pytest.raises(b.StorageBudgetError): w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


def test_leaf_symlink_swap_at_open_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'identity.json'; write(path, document())
    replacement = tmp_path / 'replacement.json'; write(replacement, document())
    original = os.open
    def race(value, flags, *args, **kwargs):
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
        path.unlink(); path.symlink_to(replacement)
        return original(value, flags, *args, **kwargs)
    monkeypatch.setattr(w.os, 'open', race)
    with pytest.raises(OSError): w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


def test_parent_symlink_after_opened_snapshot_is_rejected(tmp_path, monkeypatch):
    directory = tmp_path / 'run'; directory.mkdir()
    path = directory / 'identity.json'; write(path, document())
    original = m._json
    def parse(raw):
        result = original(raw)
        directory.rename(tmp_path / 'relocated')
        directory.symlink_to(tmp_path / 'relocated', target_is_directory=True)
        return result
    monkeypatch.setattr(m, '_json', parse)
    with pytest.raises(b.StorageBudgetError, match='symlink'):
        w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


def test_hot_version_change_with_identical_valid_semantics_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'identity.json'; write(path, document())
    w._run_identity_projection(path)
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    monkeypatch.setattr(w.os, 'open', lambda *a, **kw: pytest.fail('hot changed identity reread'))
    with pytest.raises(b.StorageBudgetError, match='changed after validation'):
        w._run_identity_projection(path)


def test_evicted_changed_identity_is_resealed_and_rechecked(tmp_path):
    paths = []
    for i in range(9):
        path = tmp_path / f'identity_{i}.json'; write(path, document(population=[i]))
        w._run_identity_projection(path); paths.append(path)
    assert len(w._RUN_IDENTITIES) == 8 and str(paths[0]) not in w._RUN_IDENTITIES
    value = document(population=['evicted replacement']); value['identity'] = '0' * 64
    write(paths[0], value)
    with pytest.raises(b.StorageBudgetError, match='semantic identity'):
        w._run_identity_projection(paths[0])
    assert len(w._RUN_IDENTITIES) == 8 and str(paths[0]) not in w._RUN_IDENTITIES
