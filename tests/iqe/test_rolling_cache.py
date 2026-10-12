"""Bounded off-host storage behavior; CPU fixtures are not real training proof."""
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import json
import torch
import pytest
from iqe.io import ContractError, atomic_json, read_json, digest, file_hash
from iqe.rolling_cache.plan import seal, checked, window_plan, window_sequence
from iqe.rolling_cache.placement import producer_placement
from iqe.rolling_cache.pool import Pool, lease
from iqe.rolling_cache.reader import RollingReader, validate_rolling_certificate
from iqe.vision_cache.storage import ShardWriter, commit_cache, encode_record
from iqe.performance.vision_cache import FrozenVisionPayload
from iqe.training.execution import EpochPlan


def batches(size, batch, seed, epoch):
    ids = torch.randperm(size, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    return [ids[i:i+batch] for i in range(0,size,batch)]


def payload():
    return FrozenVisionPayload(torch.arange(10),torch.arange(10).expand(3,-1).clone(),
        {'history':torch.tensor([7]),'foresight':torch.tensor([8]),'action':torch.tensor([9])},
        torch.arange(24,dtype=torch.float32).reshape(3,8),
        tuple(torch.arange(24,dtype=torch.float32).reshape(3,8)+i+.5 for i in range(3)),torch.tensor([[1,2,2]]*3))


def setup(tmp_path, size=10):
    identity={'test_only':True,'precision':'float32'}
    rows=[{'scene_id':str(i),'observation_hash':f'obs{i}','current_sha256':f'meta{i}','role':'incremental_fit'} for i in range(size)]
    full={'rows':rows,'fit_count':size,'global_batch':4}
    cert=tmp_path/'qualification.json';atomic_json(cert,{'visual_identity_hash':digest(identity)})
    session=seal({'parent_step':10,'epoch':2,'offset':1,'window_updates':2,'max_optimizer_steps':24,
        'seed':42,'lookahead':2,'retained_previous_windows':1,'world_size':2,'reuse_sources':[],
        'qualification':str(cert),'capacity_bytes':1024**2})
    pool=Pool(tmp_path/'pool',session,full,batches)
    request=pool.start_attempt(10,attempt='first')
    return pool,request,identity


def publish(pool, number, identity, external=False):
    plan=pool.plan(number)
    path=(pool.root/'permanent'/str(number)) if external else pool.folder(number)/'data'/'one'
    access={r['scene_id']:{**r,'key':r['scene_id']} for r in plan['rows']}
    writer=ShardWriter(path,'shard',digest(identity))
    for sid in access:writer.add(sid,payload())
    writer.commit();commit_cache(path,identity,access,['shard'],plan_hash=plan['content_hash'])
    pool.publish(plan,{sid:str(path) for sid in access},None if external else path)
    return path


def test_windows_follow_original_order_tail_epoch_rng(tmp_path):
    pool,request,identity=setup(tmp_path)
    before=torch.get_rng_state().clone()
    original=EpochPlan(10,4,42,0,1,batches)
    expected=[list(a.entries) for a in original.iter_from(2,1,14)]
    got=[[int(s) for s in b['scene_ids']] for n in range(7) for b in window_sequence(pool.session,pool.full,n,batches)]
    assert got==expected and any(len(b)==2 for b in got)
    assert torch.equal(before,torch.get_rng_state())


def test_window_beyond_original_schedule_rejected(tmp_path):
    pool,_,_=setup(tmp_path)
    with pytest.raises(ContractError):pool.plan(7)


@pytest.mark.parametrize('host,ok',[('training-vla-zt2-worker-0',False),('training-vlawm-zt-worker-0',False),
    ('training-vlawm-zt3-worker-0',True),('training-vlawm-zt4-worker-0',True),('unknown',False)])
def test_remote_host_enforcement(tmp_path,host,ok):
    p=tmp_path/'policy.json'
    atomic_json(p,{'task_authorizations':{'iqe_v1':{'allowed_hosts':['training-vla-zt2','training-vlawm-zt3','training-vlawm-zt4'],
        'workload_placement':{'base_training_host':'training-vla-zt2','forbid_cache_compute_on_base_host':True,
            'cache_producer_hosts':['training-vlawm-zt3','training-vlawm-zt4']}}},
        'global_dispatch_prohibitions':{'training-vlawm-zt':{}}})
    if ok:assert producer_placement(p,host=host)['visual_compute_on_base'] is False
    else:
        with pytest.raises(ContractError):producer_placement(p,host=host)


def test_slowest_rank_and_resume_ignore_prefetch_progress(tmp_path):
    pool,request,_=setup(tmp_path)
    pool.progress(request,0,16)
    assert pool.desired()[0]==[0,1]
    pool.progress(request,1,14)
    assert pool.desired()[0]==[2,3]
    new=pool.start_attempt(10,attempt='resume-from-durable-10')
    assert pool.desired()[0]==[0,1]
    with pytest.raises(ContractError):pool.progress(request,0,18)
    pool.progress(new,0,10)


def test_ready_requires_complete_and_correct_observations(tmp_path):
    pool,_,identity=setup(tmp_path);plan=pool.plan(0)
    with pytest.raises(ContractError):pool.publish(plan,{})
    root=publish(pool,0,identity)
    assert pool.ready(0)['status']=='WINDOW_COMPLETE'
    atomic_json(root/'INDEX.json',{'modified':True})
    with pytest.raises(ContractError):pool.publish(plan,{r['scene_id']:str(root) for r in plan['rows']},root)


def test_reader_bitwise_no_image_decode_shuffled_parallel_reads(tmp_path):
    pool,request,identity=setup(tmp_path)
    publish(pool,0,identity);publish(pool,1,identity)
    reader=RollingReader(pool,request,rank=0,epoch=2,offset=1,step=10,identity=identity,wait_seconds=.01)
    ids=[r['scene_id'] for r in pool.plan(0)['rows']]
    with patch('PIL.Image.open',side_effect=AssertionError('no decode')), patch('torch.load',side_effect=AssertionError('no pickle')):
        with ThreadPoolExecutor(4) as executor:
            results=list(executor.map(reader.read,list(reversed(ids))+ids[:2]))
    assert all(encode_record(v)==encode_record(payload()) for v in results)
    assert reader.stats['reads']==len(ids)+2
    reader.close()


def test_missing_window_waits_then_fails_without_fallback(tmp_path):
    pool,request,identity=setup(tmp_path)
    reader=RollingReader(pool,request,rank=0,epoch=2,offset=1,step=10,identity=identity,wait_seconds=.01)
    with pytest.raises(TimeoutError,match='no online/zero fallback'):reader.read('0')
    assert reader.stats['visual_computations']==0


def test_original_rank_sequence_including_tail_padding(tmp_path):
    pool,request,identity=setup(tmp_path)
    publish(pool,0,identity);publish(pool,1,identity)
    for rank in range(2):
        reader=RollingReader(pool,request,rank=rank,epoch=2,offset=1,step=10,identity=identity)
        for _ in range(3):
            address=reader.expected.at(reader.epoch,reader.offset)
            ids=[str(i) for i in address.mine]
            for sid in ids:reader.read(sid)
            reader.prepared(ids)
        with pytest.raises(ContractError):reader.prepared(['wrong'])
        reader.close()


def test_evict_only_owned_files_and_protect_inflight(tmp_path):
    pool,request,identity=setup(tmp_path);root=publish(pool,0,identity);permanent=publish(pool,1,identity,external=True)
    original=file_hash(permanent/'shard.bin')
    with pytest.raises(ContractError):pool.evict(0)
    for rank in range(2):pool.progress(request,rank,16)
    with lease(pool.folder(0)/'READ.lock'):
        with pytest.raises(BlockingIOError):pool.evict(0)
        assert (root/'shard.bin').exists()
    assert pool.evict(0)['status']=='EVICTED'
    assert not (root/'shard.bin').exists() and (root/'INDEX.json').exists()
    pool.evict(1)
    assert file_hash(permanent/'shard.bin')==original
    assert pool.ready(0) is None


def test_eviction_fault_after_readiness_removed_is_recoverable(tmp_path):
    pool,request,identity=setup(tmp_path);root=publish(pool,0,identity)
    for rank in range(2):pool.progress(request,rank,16)
    original=Path.unlink
    def fail(path,*args,**kwargs):
        if path.suffix=='.bin':raise OSError('injected unlink interruption')
        return original(path,*args,**kwargs)
    with patch.object(Path,'unlink',fail):
        with pytest.raises(OSError):pool.evict(0)
    assert pool.ready(0) is None and (pool.folder(0)/'EVICTING.json').exists()
    assert pool.evict(0)['status']=='EVICTED' and not (root/'shard.bin').exists()


def test_resume_rebuild_keeps_original_payloads_and_next_ids(tmp_path):
    pool,request,identity=setup(tmp_path);root=publish(pool,0,identity)
    ids=[r['scene_id'] for r in pool.plan(0)['rows']]
    before=RollingReader(pool,request,rank=0,epoch=2,offset=1,step=10,identity=identity)
    hashes=[encode_record(before.read(s)) for s in ids];before.close()
    for rank in range(2):pool.progress(request,rank,16)
    pool.evict(0);new=pool.start_attempt(10,attempt='resume')
    # New immutable generation; old receipts remain as eviction evidence.
    plan=pool.plan(0);new_root=pool.folder(0)/'data'/'two'
    access={r['scene_id']:{**r,'key':r['scene_id']} for r in plan['rows']}
    w=ShardWriter(new_root,'shard',digest(identity))
    for sid in access:w.add(sid,payload())
    w.commit();commit_cache(new_root,identity,access,['shard'],plan_hash=plan['content_hash'])
    pool.publish(plan,{s:str(new_root) for s in ids},new_root)
    after=RollingReader(pool,new,rank=0,epoch=2,offset=1,step=10,identity=identity)
    assert hashes==[encode_record(after.read(s)) for s in ids]
    assert after.expected.at(2,1).mine==EpochPlan(10,4,42,0,2,batches).at(2,1).mine
    after.close()


def test_old_full_cache_certificate_cannot_authorize_rolling_training(tmp_path):
    p=tmp_path/'q.json';atomic_json(p,seal({'schema':'iqe_fp32_cache_qualification_v1','status':'PASS'}))
    with pytest.raises(ContractError):validate_rolling_certificate(p,{}, {'content_hash':'session'})


def _rank_failure(rank, rendezvous, root, session, full, request, identity, bad, good):
    import torch.distributed as dist
    from datetime import timedelta
    from iqe.training.execution import collective_check
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=20))
    pool=Pool(root,session,full,batches)
    reader=RollingReader(pool,request,rank=rank,epoch=2,offset=1,step=10,identity=identity,wait_seconds=.02)
    error=None
    try:reader.read(bad if rank==0 else good)
    except Exception as exc:error=repr(exc)
    try:
        collective_check(error)
        result='incorrect_success'
    except ContractError:result='GROUP_FAILED_WITHOUT_FALLBACK'
    atomic_json(Path(root)/f'rank_failure_{rank}.json',{'status':result})
    reader.close();pool.close();dist.destroy_process_group()


def test_one_rank_cache_miss_exits_entire_gloo_group(tmp_path):
    import torch.multiprocessing as mp
    pool,request,identity=setup(tmp_path);publish(pool,0,identity)
    ready=set(pool.ready(0)['refs']);bad=next(str(i) for i in range(10) if str(i) not in ready)
    mp.spawn(_rank_failure,args=(str(tmp_path/'rendezvous'),str(pool.root),pool.session,pool.full,request,identity,bad,next(iter(ready))),nprocs=2,join=True)
    assert all(read_json(pool.root/f'rank_failure_{rank}.json')['status']=='GROUP_FAILED_WITHOUT_FALLBACK' for rank in range(2))


def test_interrupted_eviction_then_checkpoint_rewind(tmp_path):
    pool,request,identity=setup(tmp_path);root=publish(pool,0,identity)
    for rank in range(2):pool.progress(request,rank,16)
    original=Path.unlink
    def fail(path,*args,**kwargs):
        if path.name=='READY.json':raise OSError('after intent before readiness unlink')
        return original(path,*args,**kwargs)
    with patch.object(Path,'unlink',fail):
        with pytest.raises(OSError):pool.evict(0)
    assert pool.ready(0) is None
    pool.start_attempt(10,attempt='rewind')
    assert pool.evict(0)['status']=='EVICTED'
    assert not (root/'shard.bin').exists()
    assert not (pool.folder(0)/'EVICTING.json').exists()


def test_stale_partial_reclaimed_with_interrupted_intent(tmp_path):
    pool,request,_=setup(tmp_path)
    folder=pool.folder(5);root=folder/'data'/'orphan';root.mkdir(parents=True)
    atomic_json(folder/'BUILDING.json',seal({'root':str(root),'plan_hash':'old'}))
    (root/'.test.partial').write_bytes(b'payload')
    original=Path.unlink
    def fail(path,*args,**kwargs):
        if path.suffix=='.partial':raise OSError('interrupted partial cleanup')
        return original(path,*args,**kwargs)
    with patch.object(Path,'unlink',fail):
        with pytest.raises(OSError):pool.discard_stale_partial(5)
    for rank in range(2):pool.progress(request,rank,20)
    assert pool.discard_stale_partial(5)['status']=='PARTIAL_EVICTED'
    assert not (root/'.test.partial').exists() and list(folder.glob('ABANDONED_*'))


def test_rolling_read_leaves_no_deleted_open_descriptor(tmp_path):
    pool,request,identity=setup(tmp_path);root=publish(pool,0,identity)
    reader=RollingReader(pool,request,rank=0,epoch=2,offset=1,step=10,identity=identity)
    reader.read(pool.plan(0)['rows'][0]['scene_id'])
    assert not reader.readers[str(root)].handles
    reader.close()


def test_streaming_window_commit_reuses_receipts_not_full_shard_reads(tmp_path):
    from iqe.rolling_cache.producer import commit_window_cache
    from iqe.vision_cache.storage import CacheReader
    identity={'test':'fp32'};root=tmp_path/'stream';w=ShardWriter(root,'shard',digest(identity))
    w.add('key',payload());w.commit()
    access={'s':{'key':'key'}}
    original=file_hash
    def forbid_bin(path):
        assert Path(path).suffix!='.bin'
        return original(path)
    with patch('iqe.rolling_cache.producer.file_hash',forbid_bin):
        first=commit_window_cache(root,identity,access,['shard'],'plan')
        assert first==commit_window_cache(root,identity,access,['shard'],'plan')
    r=CacheReader(root);assert encode_record(r.read('s'))==encode_record(payload());r.close()
    with (root/'shard.bin').open('r+b') as f:f.write(b'X')
    r=CacheReader(root)
    with pytest.raises(ContractError,match='corrupt'):r.read('s')
    r.close()


def test_streaming_window_partial_commit_requires_full_hash_recovery(tmp_path):
    from iqe.rolling_cache.producer import committed_shard
    root=tmp_path/'recovery';identity={'test':'fp32'};w=ShardWriter(root,'shard',digest(identity))
    w.add('key',payload());w.commit();(root/'shard.bin.COMPLETE.json').unlink()
    with (root/'shard.bin').open('r+b') as f:f.write(b'X')
    with pytest.raises(ContractError,match='checksum'):committed_shard(root,'shard',digest(identity))
    assert not (root/'shard.bin.COMPLETE.json').exists()


def test_obsolete_producer_request_replans_without_error(tmp_path):
    from iqe.rolling_cache.pool import build_window
    pool,request,_=setup(tmp_path)
    for rank in range(2):pool.progress(request,rank,16)
    with patch('iqe.rolling_cache.placement.producer_placement'),patch('iqe.vision_cache.cli.hold_check'):
        result=build_window(pool,0,model_loader=lambda _:pytest.fail('must not load'),
            rate_mib_s=1,reserve_gib=1,max_gib=1,hold_file='unused',worker=15,workers=16)
    assert result['status']=='SKIPPED_REPLAN'
