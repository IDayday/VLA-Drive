from types import SimpleNamespace

import pytest

from iqe.io import atomic_json, file_hash
from iqe.specialist_study.diagnostics import scorer_fit


def setup(monkeypatch, tmp_path, status='COMPLETE'):
    from iqe.specialist_study import resource_guard, bank, scorer_training
    config=SimpleNamespace(root=tmp_path)
    folder=tmp_path/'scorers/S00';folder.mkdir(parents=True)
    (folder/'best.pt').write_bytes(b'identity-only fixture, never loaded')
    (tmp_path/'BASE_BINDING.json').write_text('{}')
    atomic_json(folder/'result.json',{'status':status,'optimizer_steps':6000,
        'best_checkpoint_hash':file_hash(folder/'best.pt')})
    monkeypatch.setattr(resource_guard,'check',lambda *a:None)
    monkeypatch.setattr(bank,'load_bank',lambda *a:({'hash':'current-bank'},{}))
    def must_not_load(*a,**k):raise AssertionError('diagnostic reuse must not load a model')
    monkeypatch.setattr(scorer_training,'selected_scorer',must_not_load)
    bindings={'cell':'S00','Base_binding_hash':file_hash(tmp_path/'BASE_BINDING.json'),
        'bank_hash':'current-bank','checkpoint_hash':file_hash(folder/'best.pt')}
    return config,folder,bindings


def test_incomplete_cell_is_not_a_fit_endpoint(monkeypatch,tmp_path):
    config,_,_=setup(monkeypatch,tmp_path,'STOPPED_AT_BOUNDARY')
    with pytest.raises(ValueError,match='completed fixed-budget'):
        scorer_fit(config,'S00','cpu')


def test_completed_fit_reuses_only_same_bound_evidence(monkeypatch,tmp_path):
    config,folder,bindings=setup(monkeypatch,tmp_path)
    report={'bindings':bindings,'optimizer_updates':0}
    atomic_json(folder/'FULL_FIT_DIAGNOSTIC.json',report)
    assert scorer_fit(config,'S00','cpu')==report


def test_changed_bank_refuses_stale_fit_evidence(monkeypatch,tmp_path):
    config,folder,bindings=setup(monkeypatch,tmp_path)
    atomic_json(folder/'FULL_FIT_DIAGNOSTIC.json',{'bindings':{**bindings,'bank_hash':'old-bank'}})
    with pytest.raises(ValueError,match='dependencies changed'):
        scorer_fit(config,'S00','cpu')
