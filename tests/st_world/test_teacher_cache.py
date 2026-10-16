import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from starVLA.dataloader.st_world_schema import TeacherFeatureContract,identity_hash
from starVLA.dataloader.st_world_history import image_reference
from starVLA.dataloader import st_world_targets as target_io
from starVLA.dataloader.st_world_targets import TeacherCacheWriter,TeacherCacheReader,LegacyC1Reader


def fixture(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(32,18),'blue').save(image)
    ref=image_reference(image)
    c=TeacherFeatureContract(image_hw=(32,32),patch_hw=(16,16),native_grid_hw=(2,2),output_grid_hw=(2,2),
        channels=8,storage_dtype='float32',compute_dtype='float32',view_names=['CAM_F0','CAM_L0','CAM_R0'],
        teacher_identity='a'*64,transform_identity='b'*64,pooling={'kind':'none','kernel_hw':[1,1],'stride_hw':[1,1]})
    m={'schema':'st_teacher_requests_v1','split':'train','times_s':[0.,.5],
       'view_names':list(c.view_names),'timestamp_tolerance_s':.05,'scenes':[
           {'token':'a','log':'l','split':'train','timestamp_us':0,'frames':[
               {'requested_s':0.,'timestamp_us':0,'views':[ref]*3},
               {'requested_s':.5,'timestamp_us':None,'views':[None]*3}]}]}
    m['identity']=identity_hash(m)
    return c,m


def test_atomic_roundtrip_dedup_lossless_and_population(tmp_path):
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m,images_per_shard=2)
    assert len(w.records)==1
    assert w.finalize()['status']=='PARTIAL'
    z=torch.arange(32,dtype=torch.float32).reshape(1,2,2,8);v=torch.ones(1,2,2,dtype=torch.bool)
    with w.shard_lock(0):w.commit_shard(0,z,v)
    with w.shard_lock(0):w.commit_shard(0,z,v)
    done=w.finalize(); assert done['status']=='COMPLETE'
    r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'],expected_contract=c)
    item=r.read_scene('a',expected_split='train')
    assert torch.equal(item['current_dense_features'][0],z[0])
    assert not item['future_valid'].any() and not item['future_dense_features'].any()
    with pytest.raises(ValueError):r.read_scene('a',expected_split='dev')
    with pytest.raises(ValueError):TeacherCacheReader(w.root,expected_identity='f'*64)
    with pytest.raises(ValueError):
        with w.shard_lock(0):w.commit_shard(0,z+1,v)


def test_receipt_crash_window_resume_keeps_orphan(tmp_path,monkeypatch):
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m)
    z=torch.randn(1,2,2,8);v=torch.ones(1,2,2,dtype=torch.bool)
    old=target_io.atomic_json
    def crash(path,*args,**kw):
        if Path(path).name=='shard_000000.json':raise RuntimeError('injected before receipt')
        return old(path,*args,**kw)
    monkeypatch.setattr(target_io,'atomic_json',crash)
    with pytest.raises(RuntimeError):
        with w.shard_lock(0):w.commit_shard(0,z,v)
    assert w.finalize()['status']=='PARTIAL'
    monkeypatch.setattr(target_io,'atomic_json',old)
    resumed=TeacherCacheWriter(w.root,c,m)
    with resumed.shard_lock(0):resumed.commit_shard(0,z,v)
    assert len(list(w.root.glob('*.orphan_*.safetensors')))==1
    assert resumed.finalize()['status']=='COMPLETE'
    r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'])
    assert torch.equal(r.read_image(w.records[0]['key'])[0],z[0])


def test_corruption_and_changed_manifest_refuse(tmp_path):
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m)
    with w.shard_lock(0):w.commit_shard(0,torch.zeros(1,2,2,8),torch.ones(1,2,2,dtype=torch.bool))
    w.finalize();r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'])
    r.read_image(w.records[0]['key'])
    path=w.paths(0)[0]
    with path.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError):r.read_image(w.records[0]['key'])
    changed=copy.deepcopy(m);changed['scenes'][0]['log']='other';changed['identity']=identity_hash({k:v for k,v in changed.items() if k!='identity'})
    with pytest.raises(ValueError):TeacherCacheWriter(w.root,c,changed)


def test_bad_masks_nonfinite_and_fake_native_refused(tmp_path):
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m)
    with pytest.raises(ValueError):w.commit_shard(0,torch.full((1,2,2,8),float('nan')),torch.ones(1,2,2,dtype=torch.bool))
    with pytest.raises(ValueError):w.commit_shard(0,torch.zeros(1,2,2,8),torch.ones(1,2,2))
    fake=tmp_path/'old';fake.mkdir();(fake/'identity.json').write_text(json.dumps({'identity':'x'}))
    with pytest.raises(ValueError):LegacyC1Reader(fake)
