"""Execution-only optimization tests; no synthetic bank is scientific evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from tools.action_effect import build_effect_pairs as builder, verified_data as validation
from tools.action_effect.common import artifact, digest, read_json, write_json
from tests.action_effect.test_verified_data_independent_review import fixture_bank


def test_build_and_seal_both_datasets_without_second_native_population_pass(tmp_path,monkeypatch):
    pair,_,calls,_=fixture_bank(tmp_path,monkeypatch)
    monkeypatch.setattr(builder,'read_population',lambda *_args,**_kwargs:pytest.fail('Second full native pass'))
    result=validation.build_and_validate_datasets(tmp_path)
    assert calls==['full-validation'] and result['reused'] is False
    assert set(result['validation'])=={'pair','repair'}
    for kind in ('pair','repair'):
        path=tmp_path/(kind.upper()+'_DATASET.json')
        reused=validation.validate_compact_dataset(path,kind=kind)
        assert reused['reused'] is True
        receipt=read_json(reused['receipt']['path'])
        assert receipt['dependency_identity']['dataset']==artifact(path)
    assert len(calls)==1


def test_failed_strict_build_does_not_seal_success(tmp_path,monkeypatch):
    fixture_bank(tmp_path,monkeypatch)
    def fail(_,**kwargs):raise ValueError('native body invalid')
    monkeypatch.setattr(builder,'build',fail)
    with pytest.raises(ValueError,match='native body invalid'):
        validation.build_and_validate_datasets(tmp_path)
    assert not list((tmp_path/'validation_receipts').glob('*/*.json'))


def test_pair_and_repair_consumers_share_one_strict_build(tmp_path,monkeypatch):
    pair,_,calls,_=fixture_bank(tmp_path,monkeypatch)
    original=builder.build
    entered=threading.Event()
    def slow(root,**kwargs):
        entered.set();time.sleep(.05);return original(root,**kwargs)
    monkeypatch.setattr(builder,'build',slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(validation.validate_compact_dataset,pair,kind='pair')
        assert entered.wait(2)
        second=pool.submit(validation.validate_compact_dataset,tmp_path/'REPAIR_DATASET.json',kind='repair')
        results=[first.result(timeout=5),second.result(timeout=5)]
    assert len(calls)==1
    assert [r['reused'] for r in results]==[False,True]


def test_source_changed_during_build_never_seals(tmp_path,monkeypatch):
    _,_,_,source=fixture_bank(tmp_path,monkeypatch)
    original=builder.build
    def change(root,**kwargs):
        result=original(root,**kwargs);source['fixture_source.py']='raced';return result
    monkeypatch.setattr(builder,'build',change)
    with pytest.raises(ValueError,match='source changed'):
        validation.build_and_validate_datasets(tmp_path)
    assert not list((tmp_path/'validation_receipts').glob('*/*.json'))


def test_metadata_changed_after_strict_build_never_seals(tmp_path,monkeypatch):
    fixture_bank(tmp_path,monkeypatch)
    original=builder.build
    def corrupt(root,**kwargs):
        result=original(root,**kwargs)
        p=Path(root)/'SCENE.json';p.write_bytes(p.read_bytes()+b' ')
        return result
    monkeypatch.setattr(builder,'build',corrupt)
    with pytest.raises(ValueError,match='changed since native semantic validation'):
        validation.build_and_validate_datasets(tmp_path)
    assert not list((tmp_path/'validation_receipts').glob('*/*.json'))


def test_coherently_rewritten_parents_after_native_validation_never_seal(tmp_path,monkeypatch):
    """Even recomputing every parent pin cannot relabel an old semantic pass."""
    fixture_bank(tmp_path,monkeypatch)
    original=builder.build
    def resave(path,value):
        if 'identity' in value:
            value.pop('identity');value['identity']=digest(value)
        write_json(path,value,immutable=False)
        return artifact(path)
    def rewrite(root,**kwargs):
        result=original(root,**kwargs)
        root=Path(root)
        native=read_json(root/'attempt/NATIVE.json');native['changed_after_validation']=True
        nativepin=resave(root/'attempt/NATIVE.json',native)
        complete=read_json(root/'attempt/COMPLETE.json');complete['canonical']=nativepin
        completepin=resave(root/'attempt/COMPLETE.json',complete)
        scene=read_json(root/'SCENE.json');scene['candidates'][0]['canonical_receipt']=completepin
        scenepin=resave(root/'SCENE.json',scene)
        sr=read_json(root/'SCENE_COMPLETE.json');sr['scene_record']=scenepin
        srpin=resave(root/'SCENE_COMPLETE.json',sr)
        pop=read_json(root/'COMPLETE.json');pop['scenes']=[srpin]
        poppin=resave(root/'COMPLETE.json',pop)
        for kind in ('pair','repair'):
            path=root/(kind.upper()+'_DATASET.json');m=read_json(path);m['population_completion']=poppin
            result[kind+'_dataset']=resave(path,m)
        return result
    monkeypatch.setattr(builder,'build',rewrite)
    with pytest.raises(ValueError,match='changed since native semantic validation'):
        validation.build_and_validate_datasets(tmp_path)
    assert not list((tmp_path/'validation_receipts').glob('*/*.json'))


def test_missing_registered_population_refused_before_compact_write(tmp_path):
    reg={'split':'train','max_samples':1,'population':[]};reg['identity']=digest(reg)
    write_json(tmp_path/'REGISTRATION.json',reg)
    write_json(tmp_path/'COMPLETE.json',{'registration_identity':reg['identity'],'scenes':[{}]})
    with pytest.raises(ValueError,match='Partial/foreign'):
        validation.build_and_validate_datasets(tmp_path)
    assert not (tmp_path/'PAIR_DATASET.json').exists()
    assert not list((tmp_path/'validation_receipts').glob('*/*.json'))
