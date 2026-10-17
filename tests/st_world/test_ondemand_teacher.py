"""Behavior tests; small fixture tensors are NOT real-teacher equivalence evidence."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from starVLA.dataloader.st_world_history import image_reference
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheWriter, TeacherCacheReader, payload_key
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider, TeacherBatch
from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future


def fixture(tmp_path, *, missing_source=False):
    contract = TeacherFeatureContract(image_hw=(32,32), patch_hw=(16,16), native_grid_hw=(2,2),
        output_grid_hw=(2,2), channels=2, storage_dtype='float32', compute_dtype='float32',
        view_names=('CAM_F0','CAM_L0','CAM_R0'), teacher_identity='a'*64, transform_identity='b'*64,
        pooling={'kind':'none','kernel_hw':[1,1],'stride_hw':[1,1]})
    refs=[]
    for i in range(5):
        p=tmp_path/f'{i}.png'; Image.new('RGB',(32,18),(i,0,0)).save(p); refs.append(image_reference(p))
    times=[0.,.5,1.,2.,4.]
    frames=[{'requested_s':t,'timestamp_us':int(t*1e6), 'views':[ref]*3} for t,ref in zip(times,refs)]
    if missing_source:frames[2]={'requested_s':1.,'timestamp_us':None,'views':[None]*3}
    manifest={'schema':'st_teacher_requests_v1','split':'train','times_s':times,
        'view_names':list(contract.view_names),'timestamp_tolerance_s':.05,
        'source_student_identity':'c'*64,'source_index_sha256':'d'*64,'partition_sha256':'e'*64,
        'scenes':[{'token':'one','log':'log-a','split':'train','timestamp_us':0,'frames':frames}]}
    manifest['identity']=identity_hash(manifest)
    return contract,manifest


def sign(m):
    m['identity']=identity_hash({k:v for k,v in m.items() if k!='identity'});return m


def z_for_key(key):
    return torch.arange(8,dtype=torch.float32).reshape(2,2,2)+int(key[:5],16)/10000


class Executor:
    """Deterministic fake arithmetic, deliberately consumes global RNG to test isolation."""
    def __init__(self,c):self.contract_identity=c.identity;self.execution_identity='f'*64;self.calls=[]
    def compute_batch(self,records,*,contract,execution_identity):
        self.calls.append(tuple(r['key'] for r in records))
        random.random();np.random.rand();torch.rand(3)
        z=torch.stack([z_for_key(r['key']) for r in records])
        return TeacherBatch(tuple(r['key'] for r in records),contract.identity,execution_identity,z,torch.ones(z.shape[:-1],dtype=torch.bool))


def provider(c,m,sources=(),executor=True,**kw):
    executor=Executor(c) if executor is True else executor
    options=dict(max_cached_bytes=10000,max_scene_bytes=10000,max_batch_images=2,max_manifest_bytes=100000)
    options.update(kw)
    return BoundedTeacherProvider(m,c,sources,executor=executor,
        execution_identity=executor.execution_identity if executor else None,**options)


def complete_cache(tmp_path,c,m,name='cache'):
    w=TeacherCacheWriter(tmp_path/name,c,m,images_per_shard=2)
    for shard in range(w.shard_count):
        z=torch.stack([z_for_key(r['key']) for r in w.rows(shard)])
        with w.shard_lock(shard):w.commit_shard(shard,z,torch.ones(z.shape[:-1],dtype=torch.bool))
    w.finalize()
    return TeacherCacheReader(w.root,expected_identity=w.identity['identity'],expected_contract=c)


def test_complete_current_reuse_exact_image_sha_and_all_future_parity(tmp_path):
    c,m=fixture(tmp_path,missing_source=True)
    current=deepcopy(m);current['times_s']=[0.];current['scenes'][0]['frames']=current['scenes'][0]['frames'][:1];sign(current)
    cached=complete_cache(tmp_path,c,current)
    p=provider(c,m,[cached]);out=p.read_scene('one',expected_split='train')
    expected=complete_cache(tmp_path,c,m,'full').read_scene('one',expected_split='train')
    for key in ('current_dense_features','current_valid','future_dense_features','future_valid'):
        assert torch.equal(out[key],expected[key])
    assert p.stats['immutable_hits']==1 and p.stats['computed_images']==3
    assert p.stats['tensor_disk_bytes']==0
    # Physical missing time alone has all-false mask; genuine low-valued targets remain valid.
    assert not out['future_valid'][1].any() and out['future_valid'][0].all()
    before=len(p.executor.calls);out['current_dense_features'].zero_()
    assert torch.equal(p.read_scene('one')['current_dense_features'],expected['current_dense_features'])
    assert len(p.executor.calls)==before


def test_content_reuse_across_scene_and_time_keeps_role_population(tmp_path):
    c,m=fixture(tmp_path)
    current=deepcopy(m);current['times_s']=[0.];current['scenes'][0]['token']='other'
    current['scenes'][0]['frames']=[deepcopy(m['scenes'][0]['frames'][2])]
    current['scenes'][0]['frames'][0]['requested_s']=0.
    current['scenes'][0]['timestamp_us']=1000000;sign(current)
    source=complete_cache(tmp_path,c,current)
    p=provider(c,m,[source]);p.read_scene('one')
    assert p.stats['immutable_hits']==1 and p.stats['computed_images']==4
    wrong=deepcopy(m);wrong['partition_sha256']='1'*64;sign(wrong)
    with pytest.raises(ValueError,match='lineage'):provider(c,wrong,[source])
    wrong=deepcopy(m);wrong['split']='dev';wrong['scenes'][0]['split']='dev';sign(wrong)
    with pytest.raises(ValueError,match='lineage'):provider(c,wrong,[source])
    changed=c.to_dict();changed['teacher_identity']='2'*64
    with pytest.raises(ValueError,match='same teacher'):provider(TeacherFeatureContract.from_dict(changed),m,[source])


def test_current_only_no_future_compute_and_no_provider_disk_writes(tmp_path):
    c,m=fixture(tmp_path);p=provider(c,m)
    before=set(tmp_path.rglob('*'));out=p.read_scene('one',need_future=False)
    assert p.stats['computed_images']==1 and out['future_dense_features'].shape==(0,3,2,2,2)
    assert set(tmp_path.rglob('*'))==before
    assert p.memory_report()['tensor_disk_bytes']==0


def test_miss_is_error_not_false_mask_retry_same_request(tmp_path):
    c,m=fixture(tmp_path);p=provider(c,m,executor=None)
    with pytest.raises(FileNotFoundError,match='no admitted executor'):p.read_scene('one')
    class FailsOnce(Executor):
        def compute_batch(self,*a,**kw):
            if not self.calls:self.calls.append(('failed',));raise TimeoutError('temporary producer failure')
            return super().compute_batch(*a,**kw)
    p=provider(c,m,executor=FailsOnce(c))
    with pytest.raises(TimeoutError):p.read_scene('one')
    out=p.read_scene('one')
    assert out['future_valid'].all() and p.stats['failures']==1


@pytest.mark.parametrize('bad', ['nonfinite','wrong_key','wrong_contract','wrong_execution','wrong_dtype','wrong_mask','false_mask','gradient'])
def test_executor_malformed_results_fail_before_valid_label_return(tmp_path,bad):
    c,m=fixture(tmp_path)
    class Broken(Executor):
        def compute_batch(self,*a,**kw):
            r=super().compute_batch(*a,**kw)
            keys,contract,execution,z,mask=r.keys,r.contract_identity,r.execution_identity,r.features,r.valid
            if bad=='nonfinite':z[0,0,0,0]=float('nan')
            if bad=='wrong_key':keys=tuple(reversed(keys))
            if bad=='wrong_contract':contract='0'*64
            if bad=='wrong_execution':execution='0'*64
            if bad=='wrong_dtype':z=z.half()
            if bad=='wrong_mask':mask=mask.float()
            if bad=='false_mask':mask[0,0,0]=False
            if bad=='gradient':z.requires_grad_(True)
            return TeacherBatch(keys,contract,execution,z,mask)
    p=provider(c,m,executor=Broken(c))
    with pytest.raises(ValueError):p.read_scene('one')
    assert p.stats['computed_images']==0


def test_immutable_corruption_never_falls_back_to_compute(tmp_path):
    c,m=fixture(tmp_path);source=complete_cache(tmp_path,c,m);p=provider(c,m,[source])
    path=source.root/'shard_000000.safetensors'
    with path.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError,match='Changed'):p.read_scene('one')
    assert not p.executor.calls


def test_lru_bound_and_cold_resume_do_not_change_targets_or_request_rng(tmp_path):
    c,m=fixture(tmp_path);p=provider(c,m,max_cached_bytes=72) # two records exactly
    random.seed(4);np.random.seed(5);torch.manual_seed(6)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    patch=torch.Generator().manual_seed(7);horizon=torch.Generator().manual_seed(8)
    streams=(patch.get_state(),horizon.get_state())
    first=p.read_scene('one')
    req=sample_requests(1,3,(2,2),2,future=True,patch_generator=patch,horizon_generator=horizon)
    checkpoint=p.state_dict();p.load_state_dict(checkpoint)
    assert p.memory_report()['resident_bytes']==0
    second=p.read_scene('one')
    patch.set_state(streams[0]);horizon.set_state(streams[1])
    req2=sample_requests(1,3,(2,2),2,future=True,patch_generator=patch,horizon_generator=horizon)
    assert torch.equal(req.patch_indices,req2.patch_indices) and torch.equal(req.future_pair,req2.future_pair)
    for key in ('current_dense_features','future_dense_features','future_valid'):
        assert torch.equal(first[key],second[key])
    assert p.memory_report()['peak_resident_bytes']<=72 and p.stats['evictions']>0
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    f1,v1=gather_future(first['future_dense_features'][None],first['future_valid'][None],req)
    f2,v2=gather_future(second['future_dense_features'][None],second['future_valid'][None],req2)
    assert torch.equal(f1,f2) and torch.equal(v1,v2)
    with pytest.raises(ValueError,match='identity'):p.load_state_dict({**checkpoint,'identity':'0'*64})


def test_inference_tensor_result_becomes_normal_and_can_supervise_backward(tmp_path):
    c,m=fixture(tmp_path)
    class Inference(Executor):
        def compute_batch(self,*a,**kw):
            with torch.inference_mode():return super().compute_batch(*a,**kw)
    p=provider(c,m,executor=Inference(c));z=p.read_scene('one')['current_dense_features']
    assert not torch.is_inference(z) and not z.requires_grad
    predicted=torch.nn.Parameter(torch.zeros_like(z));(predicted-z).square().mean().backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum()>0


def test_budget_role_and_worker_rejections(tmp_path):
    c,m=fixture(tmp_path)
    with pytest.raises(ValueError,match='metadata'):provider(c,m,max_manifest_bytes=1)
    with pytest.raises(ValueError,match='64'):provider(c,m,max_batch_images=65)
    with pytest.raises(ValueError,match='output tensor'):provider(c,m,max_scene_bytes=1).read_scene('one')
    p=provider(c,m)
    with pytest.raises(ValueError,match='role'):p.read_scene('one',expected_split='dev')
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError,match='owning main thread'):pool.submit(p.read_scene,'one').result()
    assert not p.executor.calls


def _failure_rank(rank,rendezvous,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=20))
    from tools.st_world_policy.train_world import collective_call
    try:
        folder=Path(root)/str(rank);folder.mkdir();c,m=fixture(folder)
        p=provider(c,m,executor=None if rank==1 else True)
        try:
            collective_call(lambda:p.read_scene('one'),'on-demand teacher labels')
            result={'failed':False}
        except RuntimeError as e:result={'failed':True,'error':str(e)}
        (folder/'result.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


def test_single_rank_miss_fails_all_ranks_before_backward(tmp_path):
    mp.spawn(_failure_rank,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
    a,b=[json.loads((tmp_path/str(i)/'result.json').read_text()) for i in range(2)]
    assert a==b and a['failed'] and 'no admitted executor' in a['error']


def test_real_executor_boundary_rejects_worker_and_legacy_without_teacher_load(tmp_path):
    from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
    c,m=fixture(tmp_path)
    with pytest.raises(ValueError,match='legacy'):
        FrozenDINOExecutor(tmp_path,contract=c,variant='SP0',device='cpu',max_batch_images=1)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError,match='owning main'):
            pool.submit(FrozenDINOExecutor,tmp_path,contract=c,variant='SP2',device='cuda',max_batch_images=1).result()


def test_dino_executor_uses_existing_preprocess_freeze_rng_and_batch_limits(tmp_path,monkeypatch):
    from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
    from tools.st_world_policy import cache_teacher
    from starVLA.model.modules.foresight import dinov3_target_encoder as dino
    c,m=fixture(tmp_path)
    monkeypatch.setattr(cache_teacher,'make_contract',lambda *a,**k:(c,{'fixture':True},{'fixture':True}))
    class TinyFrozen(torch.nn.Module):
        def __init__(self,*a,**kw):
            super().__init__();self.p=torch.nn.Parameter(torch.rand(1),requires_grad=False)
            random.random();np.random.rand()
            self.register_buffer('b',torch.zeros(1));self.patch_size=(16,16);self.feature_dim=2;self.prefix=5
            self.eval()
        def forward(self,pixels):
            assert pixels.dtype==torch.float32 and pixels.shape[1:]==(3,32,32)
            torch.rand(1);random.random();np.random.rand()
            return torch.ones(len(pixels),2,2,2)*self.p
    monkeypatch.setattr(dino,'DINOv3TargetEncoder',TinyFrozen)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    before=torch.get_rng_state().clone();py_before=random.getstate();np_before=np.random.get_state()
    e=FrozenDINOExecutor(tmp_path,contract=c,variant='SP2',device='cpu',max_batch_images=2)
    assert torch.equal(before,torch.get_rng_state())
    assert random.getstate()==py_before and np.array_equal(np.random.get_state()[1],np_before[1])
    ref=m['scenes'][0]['frames'][0]['views'][0];row={'key':payload_key(ref,c),'image':ref}
    r=e.compute_batch((row,),contract=c,execution_identity=e.execution_identity)
    assert torch.equal(before,torch.get_rng_state()) and r.features.dtype==torch.float32
    assert r.valid.all() and not r.features.requires_grad
    with pytest.raises(ValueError,match='keys'):
        e.compute_batch((row,row),contract=c,execution_identity=e.execution_identity)
    with pytest.raises(ValueError,match='budget'):
        e.compute_batch((row,)*3,contract=c,execution_identity=e.execution_identity)
    with torch.no_grad():e.model.p.add_(1)
    with pytest.raises(ValueError,match='mutated'):
        e.compute_batch((row,),contract=c,execution_identity=e.execution_identity)
