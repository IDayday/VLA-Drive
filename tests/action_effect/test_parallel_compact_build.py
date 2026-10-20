"""CPU spawn workers preserve strict validation, ordering and compact bytes."""
from pathlib import Path

import pytest

from tools.action_effect import build_effect_pairs as builder, verified_data as validation
from tools.action_effect.common import artifact
from tests.action_effect.test_g0_pipeline import (
    factory, test_scene_transaction_compact_datasets_and_reuse_no_new_calls as _fixture_transaction,
)


def test_spawn_worker_bytes_equal_and_broken_native_cannot_seal(factory,tmp_path,monkeypatch):
    original=builder.build;compared=[]
    def compare(root,**kwargs):
        result=original(root,**kwargs)
        if not compared:
            before={k:artifact(Path(root)/(k.upper()+'_DATASET.json')) for k in ('pair','repair')}
            other=original(root,workers=2)
            assert other==result
            after={k:artifact(Path(root)/(k.upper()+'_DATASET.json')) for k in ('pair','repair')}
            assert before==after
            compared.append(True)
        return result
    monkeypatch.setattr(builder,'build',compare)
    # Existing real-wrapper synthetic transaction fixture finishes by corrupting
    # a native body; the workers must reject it too, without any success seal.
    _fixture_transaction(factory,tmp_path,monkeypatch)
    assert compared==[True]
    root=tmp_path/'population_000001'
    with pytest.raises(ValueError,match='stale dependency'):
        validation.build_and_validate_datasets(root,workers=2)
    assert not list((root/'validation_receipts').glob('*/*.json'))


@pytest.mark.parametrize('workers',[0,5,-1,True,1.5])
def test_out_of_budget_workers_rejected_without_io(tmp_path,workers):
    with pytest.raises(ValueError,match='workers'):
        builder.build(tmp_path,workers=workers)
    with pytest.raises(ValueError,match='workers'):
        validation.build_and_validate_datasets(tmp_path,workers=workers)
    assert not list(tmp_path.iterdir())


def test_worker_version_changed_before_parent_merge_rejected(tmp_path):
    from tools.action_effect.common import write_json
    p=tmp_path/'native.json';write_json(p,{'immutable':'v1'})
    pin=artifact(p);old=validation.stamp(p)
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='after worker'):
        validation._capture_dependency(pin,{},verified_stat=old)
