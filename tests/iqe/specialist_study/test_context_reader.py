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


def ram_fixture(tmp_path, monkeypatch):
    import socket
    from iqe.io import read_json, file_hash, digest
    from iqe.specialist_study import ram_stage, binding
    from iqe.specialist_study.context_reader import version
    config=SimpleNamespace(root=tmp_path,identity='frozen_config')
    atomic_json(tmp_path/'BASE_BINDING.json',{'checkpoint_hash':'base'})
    monkeypatch.setattr(context,'spatial_key',lambda c,s:s.observation_hash)
    scenes=[SimpleNamespace(scene_id=str(i),observation_hash='obs'+str(i)) for i in range(3)]
    writer=context.SpatialCache(config)
    for i,scene in enumerate(scenes):
        writer.write(scene,{'visual':torch.arange(3*(i+1)*4,dtype=torch.float32).view(3,i+1,4),
                     'visual_valid':torch.ones(3,i+1,dtype=torch.bool),'visual_xy':torch.rand(3,i+1,2)})
    root=tmp_path/'ram';root.mkdir()
    monkeypatch.setattr(ram_stage,'stage_root',lambda c:root)
    monkeypatch.setattr(binding,'scenes_by_role',lambda c:{'scorer_fit':scenes[:2]})
    records={}
    for scene in scenes[:2]:
        source=writer.path(scene);target=root/source.name;receipt=read_json(str(source)+'.COMPLETE.json')
        ram_stage.copy_verified(source,target,receipt['checksum'])
        atomic_json(str(target)+'.COMPLETE.json',receipt)
        records[scene.scene_id]={'checksum':receipt['checksum'],'source_version':list(version(source)),
              'source_receipt_version':list(version(str(source)+'.COMPLETE.json'))}
    manifest={'schema':'readonly_spatial_tmpfs_v1','status':'COMPLETE','host':socket.gethostname(),'root':str(root),
        'config_identity':config.identity,'Base_binding_hash':file_hash(tmp_path/'BASE_BINDING.json'),
        'source_role':'scorer_fit','source_scene_hash':digest([s.scene_id for s in scenes[:2]]),
        'requested':2,'records':records}
    atomic_json(root/'MANIFEST.json',manifest)
    monkeypatch.setenv('IQE_SPATIAL_READER_MODE','ram_mmap_v1')
    monkeypatch.setenv('IQE_SPATIAL_MMAP_RECORDS','3')
    return config,scenes,root,manifest


def test_ram_stage_preserves_all_bytes_order_padding_rng_and_canonical_validation(tmp_path,monkeypatch):
    config,scenes,root,_=ram_fixture(tmp_path,monkeypatch)
    reference=context.SpatialCache(config)
    reader=VerifiedSpatialReader(config)
    rng=torch.get_rng_state().clone()
    for order in (scenes,list(reversed(scenes)),[scenes[1],scenes[1]],scenes[-1:]):
        expected,actual=reference.batch(order,'cpu'),reader.batch(order,'cpu')
        assert all(torch.equal(expected[k],actual[k]) and expected[k].dtype==actual[k].dtype for k in expected)
    assert reader.path(scenes[0]).parent==root
    assert reader.path(scenes[2])==reference.path(scenes[2])
    assert torch.equal(rng,torch.get_rng_state())
    reference.path(scenes[0]).write_bytes(b'changed canonical source')
    with pytest.raises(Exception,match='source changed'):reader.read(scenes[0])


@pytest.mark.parametrize('change', ['wrong_base','missing_record','wrong_source_checksum','manifest_mutation'])
def test_ram_stage_negative_identity_and_integrity(tmp_path,monkeypatch,change):
    config,scenes,root,manifest=ram_fixture(tmp_path,monkeypatch)
    if change=='wrong_base':
        manifest['Base_binding_hash']='different';atomic_json(root/'MANIFEST.json',manifest)
        with pytest.raises(Exception,match='manifest identity'):VerifiedSpatialReader(config)
        return
    reader=VerifiedSpatialReader(config)
    if change=='missing_record':
        reader.path(scenes[0]).unlink()
        with pytest.raises(Exception,match='missing staged'):reader.read(scenes[0])
    elif change=='wrong_source_checksum':
        from iqe.io import read_json
        p=str(reader.path(scenes[0]))+'.COMPLETE.json';receipt=read_json(p);receipt['checksum']='wrong';atomic_json(p,receipt)
        with pytest.raises(Exception,match='verified canonical source'):reader.read(scenes[0])
    else:
        manifest['host']='wrong';atomic_json(root/'MANIFEST.json',manifest)
        with pytest.raises(Exception,match='manifest changed'):reader.read(scenes[0])
