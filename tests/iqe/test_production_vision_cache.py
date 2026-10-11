"""Behavior/fault coverage; these CPU fixtures are not real-image integration."""
from pathlib import Path
from unittest.mock import patch
import json
import os
import struct
import torch
import pytest
from iqe.io import ContractError, read_json, atomic_json, digest, file_hash
from iqe.performance.vision_cache import FrozenVisionPayload, combine_payloads
from iqe.vision_cache.storage import (encode_record, decode_record, ShardWriter, CacheReader,
    commit_cache, recover_shard, shard_lease)
from iqe.vision_cache.builder import split_payloads
from iqe.vision_cache.identity import payload_key, tensor_state_hash
from iqe.vision_cache.runtime import ProductionOptions, ProductionRuntime
from iqe.vision_cache.cli import hold_check, parser


def payload(n=10):
    return FrozenVisionPayload(torch.arange(n),torch.arange(n).expand(3,-1).clone(),
        {'history':torch.tensor([n-3]),'foresight':torch.tensor([n-2]),'action':torch.tensor([n-1])},
        torch.arange(24,dtype=torch.float32).reshape(3,8),
        tuple(torch.arange(24,dtype=torch.float32).reshape(3,8)+i+.5 for i in range(3)),
        torch.tensor([[1,2,2]]*3))


def assert_payload_equal(a,b):
    assert torch.equal(a.input_ids,b.input_ids)
    assert torch.equal(a.position_ids,b.position_ids)
    assert torch.equal(a.image_grid_thw,b.image_grid_thw)
    assert torch.equal(a.image_embeddings,b.image_embeddings)
    for x,y in zip(a.deepstack,b.deepstack):assert torch.equal(x,y)
    assert a.slots.keys()==b.slots.keys()
    for n in a.slots:assert torch.equal(a.slots[n],b.slots[n])


def build(root, count=5):
    identity={'frozen':'test_tensor_only','precision':'float32'}; access={}; ids=[]
    for i in range(count):
        sid=f'shard_{i:06d}'; w=ShardWriter(root,sid,digest(identity),max_bytes=100000)
        w.add(str(i),payload(10+i));w.commit();ids.append(sid)
        access[f'scene{i}']={'key':str(i),'role':'incremental_fit','observation_hash':str(i)}
    commit_cache(root,identity,access,ids,plan_hash='fixture')
    return identity


def test_record_bitwise_roundtrip_all_deepstacks():
    p=payload(); q=decode_record(encode_record(p));assert_payload_equal(p,q)
    assert not any(t.requires_grad or torch.is_inference(t) for t in [q.image_embeddings,*q.deepstack])


@pytest.mark.parametrize('bad',['bf16','nan','inference','missing_deepstack'])
def test_record_rejects_foreign_numerics(bad):
    p=payload()
    if bad=='bf16':p.image_embeddings=p.image_embeddings.bfloat16()
    elif bad=='nan':p.deepstack[1][0,0]=float('nan')
    elif bad=='inference':
        with torch.inference_mode():p.image_embeddings=p.image_embeddings.clone()
    else:p.deepstack=p.deepstack[:2]
    with pytest.raises(ContractError):encode_record(p)


def test_sharded_reader_order_duplicate_tail_and_bounded_fds(tmp_path):
    identity=build(tmp_path);r=CacheReader(tmp_path,identity=identity,max_open_shards=2)
    with patch('PIL.Image.open',side_effect=AssertionError('image opened')), patch('torch.load',side_effect=AssertionError('whole shard deserialized')):
        for i in [4,0,2,2,1,4,0,3]:assert_payload_equal(r.read(f'scene{i}'),payload(10+i))
    assert r.stats['max_open_handles']==2 and len(r.handles)==2
    assert r.stats['reads']==8
    r.close();assert not r.handles


def test_reader_consumes_only_selected_record_bytes(tmp_path):
    ident={'fixture':True};w=ShardWriter(tmp_path,'shard',digest(ident))
    w.add('a',payload());w.add('b',payload(11));rec=w.commit()
    commit_cache(tmp_path,ident,{'x':{'key':'a'},'y':{'key':'b'}},['shard'],plan_hash='x')
    r=CacheReader(tmp_path);r.read('y')
    assert r.stats['bytes']==rec['records']['b']['length']<rec['bytes']


@pytest.mark.parametrize('which',['index','shard','missing','identity','role'])
def test_cache_failures_never_zero_or_fallback(tmp_path,which):
    identity=build(tmp_path)
    if which=='index':
        x=read_json(tmp_path/'INDEX.json');x['access']['injected']={'key':'0'};atomic_json(tmp_path/'INDEX.json',x)
        with pytest.raises(ContractError):CacheReader(tmp_path)
    elif which=='identity':
        with pytest.raises(ContractError):CacheReader(tmp_path,identity={'different':True})
    else:
        r=CacheReader(tmp_path,allowed_scene_ids=['scene0'])
        if which=='role':
            with pytest.raises(ContractError):r.read('scene1')
        elif which=='missing':
            (tmp_path/'shard_000000.bin').unlink()
            with pytest.raises(FileNotFoundError):r.read('scene0')
        else:
            p=tmp_path/'shard_000000.bin'
            with p.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
            with pytest.raises(ContractError):r.read('scene0')


def test_partial_never_complete_and_intent_window_recovers(tmp_path):
    identity={'fixture':True};w=ShardWriter(tmp_path,'one',digest(identity));w.add('k',payload())
    with patch('iqe.vision_cache.storage.atomic_json',side_effect=lambda p,v,**kw:
               (_ for _ in ()).throw(RuntimeError('after rename')) if str(p).endswith('COMPLETE.json') else atomic_json(p,v,**kw)):
        with pytest.raises(RuntimeError):w.commit()
    assert (tmp_path/'one.bin').exists() and not (tmp_path/'one.bin.COMPLETE.json').exists()
    with pytest.raises(FileNotFoundError):CacheReader(tmp_path)
    restored=recover_shard(tmp_path,'one',digest(identity));assert restored['records']['k']
    commit_cache(tmp_path,identity,{'s':{'key':'k'}},['one'],plan_hash='p')
    assert_payload_equal(CacheReader(tmp_path).read('s'),payload())


def test_incomplete_coverage_and_conflicting_lease_rejected(tmp_path):
    identity={'fixture':True};w=ShardWriter(tmp_path,'one',digest(identity));w.add('k',payload());w.commit()
    with pytest.raises(ContractError):commit_cache(tmp_path,identity,{'s':{'key':'other'}},['one'],plan_hash='p')
    with shard_lease(tmp_path,'one'):
        with pytest.raises(BlockingIOError):
            with shard_lease(tmp_path,'one'):raise AssertionError('lease not exclusive')


def test_commit_idempotent_and_rank_independent(tmp_path):
    identity=build(tmp_path,2);old=file_hash(tmp_path/'COMPLETE.json')
    index=read_json(tmp_path/'INDEX.json')
    commit_cache(tmp_path,identity,index['access'],['shard_000000','shard_000001'],plan_hash='fixture')
    assert file_hash(tmp_path/'COMPLETE.json')==old
    assert 'rank' not in json.dumps(index)


@pytest.mark.parametrize('lengths',[[10],[10,14,10,12],[10,14]])
def test_left_padding_full_slots_grid_and_regroup(lengths):
    values=[payload(n) for n in lengths]
    ids,mask,pos,slots,images,deep=combine_payloads(values,99,'cpu',values[0].slots)
    for i,n in enumerate(lengths):
        offset=max(lengths)-n
        assert mask[i].sum()==n and (ids[i,:offset]==99).all()
        assert torch.equal(pos[:,i,offset:],values[i].position_ids)
        for key in slots:assert torch.equal(slots[key][i],values[i].slots[key]+offset)
    assert len(deep)==3 and len(images)==3*len(lengths)


def test_key_ignores_trainable_state_but_changes_image_prompt_precision():
    hashes=['1'*64,'2'*64,'3'*64];identity={'visual':'frozen','precision':'float32'}
    a=payload_key(hashes,'prompt',identity)
    assert a==payload_key(hashes,'prompt',dict(identity))
    assert a!=payload_key(hashes[::-1],'prompt',identity)
    assert a!=payload_key(hashes,'changed',identity)
    assert a!=payload_key(hashes,'prompt',{**identity,'precision':'bfloat16'})
    assert tensor_state_hash({'x':torch.ones(2)})!=tensor_state_hash({'x':torch.zeros(2)})


def test_operator_hold_and_conditional_release(tmp_path):
    p=tmp_path/'hold.json';atomic_json(p,{'hold':True,'mode':'PAUSED'})
    with pytest.raises(ContractError):hold_check(p)
    atomic_json(p,{'hold':True,'mode':'CONDITIONAL_UNTIL_CAPACITY_SUFFICIENT'})
    assert hold_check(p)==file_hash(p)


def test_strict_production_options_reject_bf16_and_from_scratch(tmp_path):
    import yaml
    p=tmp_path/'options.yaml'
    for raw in [{'visual_numeric_contract':'bf16','cache_root':'x'}, {'auto_from_scratch':True,'cache_root':'x'},
                {'execution_weights':'bf16_master','cache_root':'x'}, {'cache_root':None}]:
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises((ContractError,TypeError)):ProductionOptions.load(p)


def test_cli_real_help_subcommands(capsys):
    for command in ['cache-plan','cache-build','cache-verify','cache-benchmark','train-base-cached','resume-base-cached','checkpoint-roundtrip']:
        with pytest.raises(SystemExit) as e:parser().parse_args([command,'--help'])
        assert e.value.code==0


def test_read_detects_post_read_file_mutation(tmp_path):
    build(tmp_path,1);r=CacheReader(tmp_path);r.read('scene0')
    p=tmp_path/'shard_000000.bin'
    with p.open('r+b') as f:f.seek(-2,2);f.write(b'\x12')
    with pytest.raises(ContractError,match='immutable'):r.read('scene0')


def test_access_resume_preserves_planned_order_after_canonical_json(tmp_path):
    from iqe.vision_cache.builder import build_access
    identity={'fixture':True}
    rows=[{'scene_id':key} for key in ['z','a','m']]
    access={key:{'key':key} for key in ['z','a','m']}
    atomic_json(tmp_path/'ACCESS_STAGING.json',{'plan_hash':'p','identity_hash':digest(identity),
        'access':access,'content_hash':digest(access)})
    assert list(read_json(tmp_path/'ACCESS_STAGING.json')['access']) == ['a','m','z']
    assert list(build_access({'plan_hash':'p','rows':rows},identity,tmp_path)) == ['z','a','m']


def test_tail_write_budget_scales_with_actual_record_count():
    from iqe.vision_cache.builder import projected_shard_bytes
    full=projected_shard_bytes(0,0,24,24)
    assert projected_shard_bytes(0,0,16,24) == full*16/24


def test_lossless_shard_reuse_is_hardlinked_idempotent_and_content_bound(tmp_path):
    from iqe.vision_cache.reuse import inspect_reuse,link_reuse
    source=tmp_path/'source';target=tmp_path/'target';target.mkdir()
    identity=build(source,3)
    access=read_json(source/'INDEX.json')['access']
    plan=[dict(scene_id=k,**v) for k,v in access.items()]
    sources=inspect_reuse([source],identity,plan)
    keys,shards,size=link_reuse(target,sources,identity,access)
    assert keys=={'0','1','2'} and size==read_json(source/'COMPLETE.json')['bytes']
    assert link_reuse(target,sources,identity,access)==(keys,shards,size)
    for old,sid in zip(sorted(source.glob('*.bin')),shards):
        assert os.path.samefile(old,target/(sid+'.bin'))
    commit_cache(target,identity,access,shards,plan_hash='full_fixture')
    assert_payload_equal(CacheReader(target).read('scene2'),payload(12))
    with pytest.raises(ContractError,match='role'):
        inspect_reuse([source],identity,[{**r,'role':'stage_val'} for r in plan])
    with pytest.raises(ContractError,match='overlapping'):
        link_reuse(tmp_path,sources+sources,identity,access)


def test_diagnostic_binding_never_promotes_performance_checkpoint():
    from iqe.vision_cache.binding import diagnostic_binding
    state={'mode':'performance_v2','kind':'iqe_query_base','training_visual_source':'cached_fp32',
        'inference_visual_source':'online_fp32','visual_numeric_contract':'legacy_fp32',
        'performance_identity':{'production_cache_runtime':{'fixture':True}},
        'base_provenance':{'fixture':True},'parent_checkpoint':'parent',
        'learned_driving_weights_loaded':False}
    config={'execution':{'scientific_expansion':False}}
    value=diagnostic_binding(state,'smoke',config)
    assert value['diagnostic_only'] and not value['formal_base_qualified'] and not value['permits_deployment']
    for mode,cfg,st in [('full',config,state),('smoke',{'execution':{'scientific_expansion':True}},state),
                        ('smoke',config,{**state,'learned_driving_weights_loaded':None})]:
        with pytest.raises(ContractError):diagnostic_binding(st,mode,cfg)


def _one_rank_cache_failure(rank, path, rendezvous, result):
    from datetime import timedelta
    import torch.distributed as dist
    from iqe.training.execution import collective_check
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=20))
    error=None
    try:CacheReader(path).read('scene_missing' if rank else 'scene0')
    except Exception as ex:error=repr(ex)
    try:
        collective_check(error)
        raise AssertionError('one-rank missing cache was silently ignored')
    except ContractError as ex:
        assert 'scene not authorized' in str(ex)
        atomic_json(Path(result)/f'{rank}.json',{'global_failure':True})
    finally:dist.destroy_process_group()


def test_one_rank_cache_failure_exits_all_ranks_without_deadlock(tmp_path):
    import torch.multiprocessing as mp
    build(tmp_path/'cache',1)
    mp.spawn(_one_rank_cache_failure,args=(str(tmp_path/'cache'),str(tmp_path/'rendezvous'),str(tmp_path/'result')),
             nprocs=2,join=True)
    assert all(read_json(tmp_path/'result'/f'{i}.json')['global_failure'] for i in range(2))
