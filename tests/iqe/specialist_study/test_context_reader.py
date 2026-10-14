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


@pytest.mark.parametrize('lengths', [(2, 2, 2, 2), (2, 3, 1, 2)])
def test_mmap_has_identical_all_tensor_bytes_padding_order_and_rng(tmp_path, monkeypatch, lengths):
    config=SimpleNamespace(root=tmp_path);atomic_json(tmp_path/'BASE_BINDING.json',{})
    monkeypatch.setattr(context,'spatial_key',lambda c,s:s.observation_hash)
    scenes=[SimpleNamespace(scene_id=str(i),observation_hash='obs'+str(i)) for i in range(len(lengths))]
    writer=context.SpatialCache(config)
    for i,(scene,n) in enumerate(zip(scenes,lengths)):
        writer.write(scene,{'visual':torch.arange(3*n*4,dtype=torch.float32).view(3,n,4)+i,
                     'visual_valid':torch.ones(3,n,dtype=torch.bool),'visual_xy':torch.rand(3,n,2),
                     'extra':torch.arange(7)})
    reference=VerifiedSpatialReader(config,capacity=1)
    monkeypatch.setenv('IQE_SPATIAL_READER_MODE','mmap_v1')
    monkeypatch.setenv('IQE_SPATIAL_MMAP_RECORDS','3')
    fast=VerifiedSpatialReader(config,capacity=1,verification_capacity=4)
    rng=torch.get_rng_state().clone()
    for order in (scenes, list(reversed(scenes)), [scenes[0],scenes[0]], scenes[-1:]):
        a,b=reference.batch(order,'cpu'),fast.batch(order,'cpu')
        assert all(torch.equal(a[key],b[key]) and a[key].dtype==b[key].dtype for key in a)
        for scene in order:
            assert all(torch.equal(reference.read(scene)[key],fast.read(scene)[key]) for key in reference.read(scene))
        assert len(fast.loaded)<=3
    assert torch.equal(rng,torch.get_rng_state())
    path=fast.path(scenes[-1]);path.write_bytes(b'corrupt')
    with pytest.raises(Exception,match='changed during training'):fast.read(scenes[-1])


def test_mmap_limits_and_unknown_execution_rejected(tmp_path, monkeypatch):
    config=SimpleNamespace(root=tmp_path);atomic_json(tmp_path/'BASE_BINDING.json',{})
    monkeypatch.setenv('IQE_SPATIAL_READER_MODE','fp16')
    with pytest.raises(Exception,match='unsupported spatial'):VerifiedSpatialReader(config)
    monkeypatch.setenv('IQE_SPATIAL_READER_MODE','mmap_v1')
    monkeypatch.setenv('IQE_SPATIAL_MMAP_RECORDS','65537')
    with pytest.raises(Exception,match='bounded spatial'):VerifiedSpatialReader(config)
