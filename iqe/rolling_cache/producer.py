"""Window-sized frozen production without repeated full-shard readback.

The qualified codec computes record and full-shard SHA256 while writing, fsyncs
data, and atomically commits its intent and receipt. Window publication validates
those receipts, lengths and byte ranges. Every consumer still verifies every
record's bytes. Interrupted pre-receipt commits use full-hash recovery. This
removes redundant multi-GB readback by every producer, not integrity checks.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import time
import torch
from ..contracts import require
from ..io import atomic_json, digest, file_hash, lock, read_json
from ..training.execution import OrderedPrefetch
from ..vision_cache.builder import FrozenFront, build_access
from ..vision_cache.identity import visual_identity, tensor_state_hash
from ..vision_cache.plan import capacity_check, load_plan
from ..vision_cache.storage import SCHEMA, ShardWriter, shard_lease, recover_shard


def committed_shard(root, sid, identity_hash):
    path = Path(root)/(sid+'.bin'); receipt = Path(str(path)+'.COMPLETE.json')
    if not receipt.exists():
        return recover_shard(root, sid, identity_hash) if path.exists() else None
    value = read_json(receipt)
    require(value == read_json(str(path)+'.INTENT.json') and value['schema'] == SCHEMA and
            value['identity_hash'] == identity_hash and path.is_file() and not path.is_symlink() and
            path.stat().st_size == value['bytes'], 'committed shard identity/size mismatch')
    cursor = 0
    for row in sorted(value['records'].values(), key=lambda x:x['offset']):
        require(row['shard'] == path.name and row['offset'] == cursor and row['length'] > 0 and
                len(row['sha256']) == 64, 'invalid committed record range')
        cursor += row['length']
    require(cursor == value['bytes'] and len(value['sha256']) == 64, 'incomplete committed shard')
    return value


def commit_window_cache(root, identity, access, ids, plan_hash):
    root = Path(root)
    with lock(root/'GLOBAL_COMMIT.lock'):
        if (root/'COMPLETE.json').exists():
            complete=read_json(root/'COMPLETE.json')
            require(complete['plan_hash'] == plan_hash and complete['identity_hash'] == digest(identity) and
                    complete['index_sha256'] == file_hash(root/'INDEX.json'), 'foreign existing window cache')
            return complete
        records, shards = {}, {}
        for sid in ids:
            value=committed_shard(root,sid,digest(identity))
            require(value is not None, 'window cannot commit partial shards')
            require(not (records.keys() & value['records'].keys()), 'duplicate payload key')
            records.update(value['records'])
            shards[sid+'.bin']={k:value[k] for k in ('sha256','bytes')}
        require(set(records) == {v['key'] for v in access.values()}, 'incomplete window coverage')
        index={'schema':SCHEMA,'identity':identity,'plan_hash':plan_hash,'access':access,
               'records':records,'shards':shards}
        atomic_json(root/'INDEX.json',index,immutable=True)
        receipt={'status':'COMPLETE','schema':SCHEMA,'index_sha256':file_hash(root/'INDEX.json'),
            'identity_hash':digest(identity),'plan_hash':plan_hash,'scenes':len(access),
            'unique_payloads':len(records),'bytes':sum(v['bytes'] for v in shards.values())}
        atomic_json(root/'COMPLETE.json',receipt,immutable=True)
        atomic_json(root/'INTEGRITY_PROTOCOL.json',{
            'writer':'qualified ShardWriter: streaming full SHA256 + record SHA256 + fsync + atomic receipt',
            'publication':'commit/intent agreement, complete byte coverage, identity, file size, exact key set',
            'reader':'qualified CacheReader: immutable file stamps and SHA256 of every accessed record',
            'interrupted_commit':'full-shard SHA256 recovery before promoting intent',
            'repeated_full_readback_per_window':False},immutable=True)
        return receipt


def build_window_shards(model, contract, plan_path, root, *, worker, workers, max_gib,
                        reserve_gib, rate_mib_s, deadline=None, hold_file):
    from .placement import producer_placement
    from ..vision_cache.cli import hold_check
    producer_placement(); hold_check(hold_file)
    plan=load_plan(plan_path);root=Path(root)
    began=time.monotonic();timings={}
    def timed(name, fn, *args):
        at=time.monotonic();result=fn(*args)
        timings[name]=timings.get(name,0.)+time.monotonic()-at
        return result
    identity=timed('frozen_identity',visual_identity,model,contract)
    require(identity['visual_state_sha256']==plan['visual_state_sha256'],'frozen visual changed')
    atomic_json(root/'IDENTITY.json',identity,immutable=True)
    with lock(root/'ACCESS_STAGING_BUILD.lock'):
        access=timed('access_hashing_or_wait',build_access,plan,identity,root)
    unique={}
    for row in access.values():unique.setdefault(row['key'],row)
    items=list(unique.items());size=plan['records_per_shard']
    groups=[items[i:i+size] for i in range(0,len(items),size)]
    ids=[f'shard_{i:06d}_{digest([k for k,_ in group])[:16]}' for i,group in enumerate(groups)]
    preferred=list(range(worker,len(groups),workers))
    order=preferred+[i for i in range(len(groups)) if i not in preferred]
    written=completed=0;stop=[False]
    previous=signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    front=FrozenFront(model)
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    from starVLA.dataloader.foresight_dataset import current_observation
    observe=getattr(current_observation,'_online_original',current_observation)
    def read_observation(item):
        _,row=item
        require(file_hash(row['current_ref'])==row['current_sha256'],'source metadata changed')
        for p,expected in zip(row['image_paths'],row['image_sha256_ordered']):
            require(file_hash(p)==expected,'source image changed')
        return observe(read_json(row['current_ref']))
    try:
        with ThreadPoolExecutor(max_workers=4) as io:
            for index in order:
                if stop[0] or (deadline is not None and time.monotonic()>=deadline):break
                producer_placement();hold_check(hold_file)
                sid=ids[index]
                if (root/(sid+'.bin.COMPLETE.json')).exists():continue
                try:
                    with shard_lease(root,sid):
                        if committed_shard(root,sid,digest(identity)) is not None:continue
                        # Reserve before writing; no speculative overcommit or
                        # deletion of permanent data/checkpoints.
                        require(written+1024**3 <= max_gib*1024**3,'producer write budget exhausted')
                        require(capacity_check(root,0,reserve_bytes=int(reserve_gib*1024**3),
                            temporary_bytes=1024**3)['sufficient'],'shared reserve unavailable')
                        chunks=[groups[index][i:i+4] for i in range(0,len(groups[index]),4)]
                        # At most sixteen producers, each with <=1GiB temporary
                        # shard: fits the session's original 16GiB write reserve.
                        writer=ShardWriter(root,sid,digest(identity),max_bytes=1024**3)
                        try:
                            with OrderedPrefetch(chunks,lambda chunk:list(io.map(read_observation,chunk)),depth=1) as ahead:
                                for _ in chunks:
                                    chunk,observations=timed('exposed_image_load_wait',ahead.pop)
                                    payloads=timed('frozen_visual_processor_and_D2H',front,observations)
                                    for (key,_),payload in zip(chunk,payloads):timed('codec_and_write',writer.add,key,payload)
                            receipt=timed('shard_fsync_commit',writer.commit)
                        except BaseException:writer.abort();raise
                        written+=receipt['bytes'];completed+=1
                except BlockingIOError:continue
                elapsed=time.monotonic()-began
                atomic_json(root/f'FAST_PROGRESS_{worker}.json',{'worker':worker,'completed_shards':completed,
                    'bytes_written':written,'seconds':elapsed,'timings_inclusive_not_additive':timings,
                    'visual_calls':front.calls,'updated_unix':time.time()})
                # Per-worker ceiling includes compute: no speed claim from an
                # unbounded write burst. Overlap is explicit, sample order fixed.
                while not stop[0] and written/(rate_mib_s*1024**2) > time.monotonic()-began:
                    if deadline is not None and time.monotonic()>=deadline:break
                    time.sleep(.05)
        require(torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in
            zip(cuda_rng,torch.cuda.get_rng_state_all())),'frozen producer consumed RNG')
        require(tensor_state_hash(front.visual.state_dict())==identity['visual_state_sha256'],'frozen buffer/parameter drift')
    finally:front.close();signal.signal(signal.SIGTERM,previous)
    if all((root/(sid+'.bin.COMPLETE.json')).exists() for sid in ids):
        result = timed('window_manifest_commit',commit_window_cache,root,identity,access,ids,plan['plan_hash'])
    else: result = {'status':'OPERATOR_PAUSED' if stop[0] else 'PARTIAL','bytes_written':written}
    atomic_json(root/f'FAST_RESULT_{worker}.json',{'status':result['status'],'worker':worker,
        'completed_shards':completed,'bytes_written':written,'seconds':time.monotonic()-began,
        'timings_inclusive_not_additive':timings,'visual_calls':front.calls,'updated_unix':time.time()})
    return {**result, 'stop_requested':stop[0]}
