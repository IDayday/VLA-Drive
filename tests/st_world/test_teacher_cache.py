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


def test_verify_rejects_other_valid_scene_population(tmp_path, monkeypatch):
    from tools.st_world_policy import cache_teacher
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m)
    w.commit_shard(0,torch.zeros(1,2,2,8),torch.ones(1,2,2,dtype=torch.bool));w.finalize()
    other=copy.deepcopy(m);other['scenes'][0]['token']='different'
    other['identity']=identity_hash({k:v for k,v in other.items() if k!='identity'})
    requests=tmp_path/'other.json';requests.write_text(json.dumps(other))
    monkeypatch.setattr(cache_teacher,'make_contract',lambda *a,**kw:(c,{},{}))
    args=cache_teacher.parser().parse_args(['verify','--requests',str(requests),
        '--teacher-root',str(tmp_path),'--output',str(w.root),'--variant','SP2',
        '--storage-dtype','float32','--mode','smoke','--max-scenes','1'])
    with pytest.raises(ValueError,match='Requested manifest'):
        cache_teacher.run(args)
    assert not (w.root/'VERIFICATION.json').exists()
    requests.write_text(json.dumps(m))
    result=cache_teacher.run(args)
    assert result['exact_requested_population'] and result['images_verified']==1


def test_current_only_never_loads_future_and_lru_does_not_rehash(tmp_path,monkeypatch):
    c,m=fixture(tmp_path)
    p=tmp_path/'future.png';Image.new('RGB',(32,18),'red').save(p)
    m['scenes'][0]['frames'][1]={'requested_s':.5,'timestamp_us':500000,'views':[image_reference(p)]*3}
    m['identity']=identity_hash({k:v for k,v in m.items() if k!='identity'})
    w=TeacherCacheWriter(tmp_path/'cache',c,m,images_per_shard=1)
    for i in range(2):w.commit_shard(i,torch.ones(1,2,2,8)*i,torch.ones(1,2,2,dtype=torch.bool))
    w.finalize();r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'],max_verified_shards=1)
    hashes=[];original=target_io.file_sha256
    def track(path):
        if str(path).endswith('.safetensors'):hashes.append(str(path))
        return original(path)
    monkeypatch.setattr(target_io,'file_sha256',track)
    current=r.read_scene('a',need_future=False)
    assert current['future_dense_features'].shape==(0,3,2,2,8)
    assert len(hashes)==1
    r.read_scene('a',need_future=True)
    r.read_scene('a',need_future=False)
    assert len(hashes)==2


def test_sp1_derivation_reuses_native_fp32_without_encoder(tmp_path,monkeypatch):
    from tools.st_world_policy import cache_teacher
    from starVLA.dataloader.st_world_schema import identity_hash
    c,m=fixture(tmp_path)
    teacher={'registered':'teacher'};transform={'registered':'transform'}
    value=c.to_dict();value.update(image_hw=[288,384],native_grid_hw=[18,24],output_grid_hw=[18,24],
        teacher_identity=identity_hash(teacher),transform_identity=identity_hash(transform))
    native=TeacherFeatureContract.from_dict(value)
    w=TeacherCacheWriter(tmp_path/'native',native,m,execution={'teacher':teacher,'transform':transform})
    z=torch.arange(18*24*8,dtype=torch.float32).reshape(1,18,24,8)
    w.commit_shard(0,z,torch.ones(1,18,24,dtype=torch.bool));w.finalize()
    p=native.to_dict();p.update(output_grid_hw=[6,8],pooling={'kind':'avg','kernel_hw':[3,3],'stride_hw':[3,3]})
    monkeypatch.setattr(cache_teacher,'make_contract',lambda *a,**kw:(TeacherFeatureContract.from_dict(p),teacher,transform))
    requests=tmp_path/'requests.json';requests.write_text(json.dumps(m))
    args=cache_teacher.parser().parse_args(['derive-pooled','--requests',str(requests),'--teacher-root',str(tmp_path),
        '--output',str(tmp_path/'pooled'),'--variant','SP1','--storage-dtype','float32','--max-scenes','1',
        '--source-cache',str(w.root),'--source-cache-identity',w.identity['identity'],'--min-free-gib','0'])
    result=cache_teacher.run(args)
    assert result['status']=='COMPLETE' and result['teacher_forward_calls']==0
    r=TeacherCacheReader(args.output,expected_identity=result['identity'])
    expected=torch.nn.functional.avg_pool2d(z.permute(0,3,1,2),3,3).permute(0,2,3,1)
    assert torch.equal(r.read_scene('a')['current_dense_features'][0],expected[0])
    assert r.contract.source_cache_identity==w.identity['identity'] and r.contract.identity!=native.identity
    assert cache_teacher.run(args)['identity']==result['identity']
    args.command='verify';assert cache_teacher.run(args)['exact_requested_population']


def test_reader_concurrent_first_hash_is_single_and_spawn_pickle_safe(tmp_path,monkeypatch):
    import pickle
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier,Lock
    c,m=fixture(tmp_path);w=TeacherCacheWriter(tmp_path/'cache',c,m)
    z=torch.randn(1,2,2,8);v=torch.ones(1,2,2,dtype=torch.bool)
    w.commit_shard(0,z,v);w.finalize();r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'],max_verified_shards=1)
    calls=[];guard=Lock();old=target_io.file_sha256
    def count(path):
        if str(path).endswith('.safetensors'):
            with guard:calls.append(str(path))
            time.sleep(.025)
        return old(path)
    monkeypatch.setattr(target_io,'file_sha256',count)
    barrier=Barrier(8)
    def read(_):
        barrier.wait();return r.read_image(w.records[0]['key'])[0]
    with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(read,range(8)))
    assert len(calls)==1 and all(torch.equal(x,z[0]) for x in values)
    child=pickle.loads(pickle.dumps(r))
    assert torch.equal(child.read_image(w.records[0]['key'])[0],z[0])
    assert len(calls)==1  # unchanged verified shard stays verified after spawn.


def test_reader_lru_lookup_eviction_is_atomic_across_threads(tmp_path):
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event,current_thread
    c,m=fixture(tmp_path);p=tmp_path/'second.png';Image.new('RGB',(32,18),'red').save(p)
    m['scenes'][0]['frames'][1]={'requested_s':.5,'timestamp_us':500000,'views':[image_reference(p)]*3}
    m['identity']=identity_hash({k:v for k,v in m.items() if k!='identity'})
    w=TeacherCacheWriter(tmp_path/'cache',c,m,images_per_shard=1)
    for i in range(2):w.commit_shard(i,torch.full((1,2,2,8),float(i)),torch.ones(1,2,2,dtype=torch.bool))
    w.finalize();r=TeacherCacheReader(w.root,expected_identity=w.identity['identity'],max_verified_shards=1)
    entered=Event();release=Event()
    class PausedLookup(OrderedDict):
        def __getitem__(self,key):
            value=super().__getitem__(key)
            if key==0 and current_thread().name.startswith('reader_a') and not entered.is_set():
                entered.set();assert release.wait(3)
            return value
    r._verify(0);r.verified=PausedLookup(r.verified)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='reader_a') as a, ThreadPoolExecutor(max_workers=1) as b:
        first=a.submit(r._verify,0);assert entered.wait(3)
        second=b.submit(r._verify,1)
        # With the original unguarded check/read/move_to_end, second evicts0
        # here and first raises KeyError. The lock holds that eviction back.
        assert not second.done();release.set()
        first.result(timeout=3);second.result(timeout=3)
    assert list(r.verified)==[1] and set(r.verified_versions)=={0,1}
