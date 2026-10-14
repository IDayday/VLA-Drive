from types import SimpleNamespace
import pytest
import torch
from iqe.io import atomic_json
from iqe.specialist_study import context
from iqe.specialist_study.context_reader import VerifiedSpatialReader


def test_checksum_verified_once_per_version_and_payload_lru_is_bounded(tmp_path, monkeypatch):
    config=SimpleNamespace(root=tmp_path)
    atomic_json(tmp_path/'BASE_BINDING.json',{'Base':'immutable'})
    monkeypatch.setattr(context,'spatial_key',lambda c,s:s.observation_hash)
    scenes=[SimpleNamespace(scene_id=str(i),observation_hash='obs'+str(i)) for i in range(3)]
    writer=context.SpatialCache(config)
    for i,scene in enumerate(scenes):writer.write(scene,{'visual':torch.full((3,2,4),float(i))})
    reader=VerifiedSpatialReader(config,capacity=1,verification_capacity=4)
    for scene in scenes+scenes:
        value=reader.read(scene)
        assert torch.equal(value['visual'],writer.read(scene)['visual'])
        assert len(reader.loaded)<=1
    assert reader.statistics['checksum_verifications']==3
    assert reader.statistics['version_reuses']>=3
    assert len(reader.verified)==3
    path=reader.path(scenes[-1]);path.write_bytes(b'corrupt')
    with pytest.raises(Exception,match='changed during training'):reader.read(scenes[-1])


def test_wrong_observation_and_missing_record_refused(tmp_path,monkeypatch):
    config=SimpleNamespace(root=tmp_path);atomic_json(tmp_path/'BASE_BINDING.json',{})
    monkeypatch.setattr(context,'spatial_key',lambda c,s:s.observation_hash)
    scene=SimpleNamespace(scene_id='a',observation_hash='right')
    context.SpatialCache(config).write(scene,{'visual':torch.ones(3,2,4)})
    reader=VerifiedSpatialReader(config)
    with pytest.raises(Exception,match='mismatch/corruption'):
        reader.read(SimpleNamespace(scene_id='a',observation_hash='wrong'))
    with pytest.raises(Exception,match='missing spatial'):
        reader.read(SimpleNamespace(scene_id='missing',observation_hash='right'))
