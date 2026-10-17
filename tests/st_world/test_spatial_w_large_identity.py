"""Bounded synthetic controls only; no native weights/images/real ledger."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import spatial_w_storage as w


@pytest.fixture(autouse=True)
def isolated_cache():
    w._RUN_IDENTITIES.clear()
    yield
    w._RUN_IDENTITIES.clear()


def document(**changes):
    value = dict(schema='st_world_training_v1', config={'recipe': 'original'},
                 source_sha='1'*40, registration_sha256='2'*64,
                 base_binding='3'*64, weights_usable_for_formal_init=True)
    value.update(changes)
    value['identity'] = b._digest(value)
    return value


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return value


def test_large_identity_full_hash_once_small_projection_and_hot_stat(tmp_path, monkeypatch):
    path = tmp_path/'identity.json'
    value = write(path, document(population='x' * (m.CONTROL_LIMIT + 1)))
    assert m.CONTROL_LIMIT == 16 << 20  # Other controls are deliberately unchanged.
    assert m.CONTROL_LIMIT < path.stat().st_size < w.RUN_IDENTITY_LIMIT
    original = os.open
    opens = []
    def counted(p, *args, **kwargs):
        opens.append(str(p))
        return original(p, *args, **kwargs)
    monkeypatch.setattr(w.os, 'open', counted)
    projected = w._run_identity_projection(path)
    assert projected['identity'] == value['identity']
    assert projected['raw_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert projected['config_sha256'] == b._digest(value['config'])
    assert len(json.dumps(dict(projected))) < 2048
    assert 'population' not in projected and 'config' not in projected
    for _ in range(10):
        assert w._run_identity_projection(path) is projected
    assert opens == [str(path)]
    with pytest.raises(TypeError):
        projected['identity'] = 'a'*64


@pytest.mark.parametrize('field,change', [
    ('schema', 'unrelated_v1'), ('config', []), ('source_sha', 'missing'),
    ('registration_sha256', 'missing'), ('base_binding', 'missing'),
    ('weights_usable_for_formal_init', False), ('identity', 'a'*64),
])
def test_semantic_and_lineage_invalid_not_cached(tmp_path, field, change):
    path = tmp_path/'identity.json'
    value = document(**{field: change})
    if field == 'identity':
        value[field] = change
    write(path, value)
    with pytest.raises(b.StorageBudgetError):
        w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


@pytest.mark.parametrize('mode', ['same_bytes_replacement', 'inplace_edit', 'hardlink', 'symlink'])
def test_hot_file_change_refused_without_reread(tmp_path, monkeypatch, mode):
    path = tmp_path/'identity.json'
    write(path, document())
    w._run_identity_projection(path)
    if mode == 'same_bytes_replacement':
        other = tmp_path/'replacement.json'; other.write_bytes(path.read_bytes()); other.replace(path)
    elif mode == 'inplace_edit':
        with path.open('a') as f: f.write(' ')
    elif mode == 'hardlink':
        os.link(path, tmp_path/'alias.json')
    else:
        other = tmp_path/'other.json'; path.rename(other); path.symlink_to(other)
    monkeypatch.setattr(w.os, 'open', lambda *a, **k: pytest.fail('changed hot file was reread'))
    with pytest.raises(b.StorageBudgetError):
        w._run_identity_projection(path)


@pytest.mark.parametrize('mode', ['oversize', 'empty', 'fifo', 'hardlink', 'symlink'])
def test_invalid_file_rejected_before_open(tmp_path, monkeypatch, mode):
    path = tmp_path/'identity.json'
    if mode == 'fifo':
        os.mkfifo(path)
    elif mode == 'symlink':
        other = tmp_path/'other.json'; write(other, document()); path.symlink_to(other)
    else:
        write(path, document())
        if mode in ('oversize', 'empty'):
            with path.open('wb') as f: f.truncate(w.RUN_IDENTITY_LIMIT+1 if mode=='oversize' else 0)
        else: os.link(path, tmp_path/'alias.json')
    monkeypatch.setattr(w.os, 'open', lambda *a, **k: pytest.fail('invalid file was opened'))
    with pytest.raises(b.StorageBudgetError):
        w._run_identity_projection(path)


@pytest.mark.parametrize('mode', ['grow', 'replace', 'parse_replace'])
def test_read_and_parse_races_rejected(tmp_path, monkeypatch, mode):
    path = tmp_path/'identity.json'; write(path, document())
    def change():
        if mode == 'grow':
            with path.open('ab') as f: f.write(b' ')
        else:
            other = tmp_path/'replacement.json'; write(other, document(source_sha='9'*40)); other.replace(path)
    if mode == 'parse_replace':
        original = m._json
        def parse(raw):
            result = original(raw); change(); return result
        monkeypatch.setattr(m, '_json', parse)
    else:
        original = os.fdopen
        class RacingReader:
            def __init__(self, file): self.file=file
            def __enter__(self): return self
            def __exit__(self,*a): self.file.close()
            def fileno(self): return self.file.fileno()
            def read(self, amount):
                assert amount == w.RUN_IDENTITY_LIMIT + 1
                result = self.file.read(amount); change(); return result
        monkeypatch.setattr(w.os, 'fdopen', lambda *a, **k: RacingReader(original(*a, **k)))
    with pytest.raises(b.StorageBudgetError, match='changed'):
        w._run_identity_projection(path)
    assert not w._RUN_IDENTITIES


def test_lru_max_eight_and_evicted_identity_fully_revalidated(tmp_path):
    paths=[]
    for i in range(10):
        p=tmp_path/f'run{i}.json'; write(p, document(index=i)); w._run_identity_projection(p); paths.append(p)
        assert len(w._RUN_IDENTITIES) <= 8
    assert list(w._RUN_IDENTITIES) == [str(p) for p in paths[2:]]
    w._run_identity_projection(paths[2])
    assert next(reversed(w._RUN_IDENTITIES)) == str(paths[2])
    write(paths[0], document(index='replacement'))
    assert w._run_identity_projection(paths[0])['identity'] == document(index='replacement')['identity']
    assert len(w._RUN_IDENTITIES) == 8


def test_existing_signature_checks_small_projection_binding(tmp_path, monkeypatch):
    from test_spatial_w_storage import signature_fixture
    s = signature_fixture(tmp_path)
    good = dict(w._run_identity_projection(s.run/'identity.json'))
    assert w.validate_signature(s.signature)[0] == s.binding
    for field in ('identity','config_sha256','source_sha','registration_sha256','base_binding'):
        wrong = {**good, field: 'f'*(40 if field=='source_sha' else 64)}
        monkeypatch.setattr(w, '_run_identity_projection', lambda path, x=wrong: x)
        with pytest.raises(b.StorageBudgetError, match='lineage/source identity'):
            w.validate_signature(s.signature)
