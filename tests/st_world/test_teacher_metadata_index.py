"""Synthetic metadata/tensor behavior only; no real-teacher numerical claim."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest
import torch

from starVLA.dataloader.st_world_metadata_index import build_index, MetadataIndex, _atomic_rows, canonical, pack, unpack
from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader, IndexedBoundedTeacherProvider
from starVLA.dataloader.st_world_targets import atomic_json, TeacherCacheReader
from starVLA.dataloader.st_world_schema import identity_hash
from tests.st_world.test_ondemand_teacher import fixture, complete_cache, Executor, sign, provider


def request_index(tmp_path, c, m, name='request_index', **kw):
    p = tmp_path / (name+'.json')
    atomic_json(p, m)
    root = tmp_path/name
    build_index(root, requests=p, expected_request_identity=m['identity'], contract=c, max_scenes=8, **kw)
    return MetadataIndex(root, expected_request_identity=m['identity'], expected_index_identity=json.loads((root/'COMPLETE.json').read_text())['identity'])


def cache_index(tmp_path, cache, name='source_index'):
    root = tmp_path/name
    build_index(root, requests=cache.root/'requests.json', cache_root=cache.root,
        expected_request_identity=cache.manifest['identity'], expected_cache_identity=cache.identity['identity'],
        contract=cache.contract, max_scenes=8)
    return MetadataIndex(root, expected_request_identity=cache.manifest['identity'], expected_index_identity=json.loads((root/'COMPLETE.json').read_text())['identity'])


def test_atomic_json_stream_hash_matches_original(tmp_path):
    c,m = fixture(tmp_path)
    m['unicode'] = '测试😀'; sign(m)
    p = tmp_path/'request.json'; atomic_json(p,m)
    got=[]
    header,counts,digest = _atomic_rows(p, {'scenes'}, lambda name,pos,row,encoded: got.append(row))
    assert dict(header,scenes=got) == m and digest == m['identity']
    assert counts == {'scenes':1}


@pytest.mark.parametrize('mutation', ['compact','truncated','hash','duplicate','row_budget'])
def test_bad_json_never_publishes(tmp_path, mutation):
    c,m = fixture(tmp_path)
    p=tmp_path/'requests.json'; out=tmp_path/'index'
    if mutation=='row_budget':
        m['scenes'][0]['unexpected_padding']='x'*70000;sign(m)
    atomic_json(p,m)
    text=p.read_text()
    if mutation=='compact':text=json.dumps(m)
    if mutation=='truncated':text=text[:-10]
    if mutation=='hash':text=text.replace('log-a','log-b')
    if mutation=='duplicate':text=text.replace('  "split": "train",','  "split": "train",\n  "split": "train",')
    p.write_text(text)
    with pytest.raises((ValueError,KeyError,json.JSONDecodeError)):
        build_index(out,requests=p,expected_request_identity=m['identity'],contract=c,max_scenes=8)
    assert not (out/'COMPLETE.json').exists()


def test_rows_records_shards_match_original_reader(tmp_path):
    c,m=fixture(tmp_path,missing_source=True)
    other=deepcopy(m['scenes'][0]);other['token']='two';other['log']='log-b'
    m['scenes'].append(other);sign(m)
    old=complete_cache(tmp_path,c,m)
    idx=cache_index(tmp_path,old)
    new=IndexedTeacherCacheReader(idx,expected_identity=old.identity['identity'],expected_contract=c)
    assert list(new.manifest['scenes']) == old.manifest['scenes']
    assert list(new.identity['records']) == old.identity['records']
    assert dict(new.locations) == old.locations and dict(new.receipt_hashes) == old.receipt_hashes
    assert dict(new.scenes) == old.scenes
    for token in ('two','one','two'):
        a,b=old.read_scene(token),new.read_scene(token)
        for key in ('current_dense_features','current_valid','future_dense_features','future_valid'):
            assert torch.equal(a[key],b[key])
    new.verify_population(m)
    changed=deepcopy(m);changed['scenes'].reverse();sign(changed)
    with pytest.raises(ValueError,match='population'):new.verify_population(changed)


def test_index_build_opens_no_tensor_or_images(tmp_path,monkeypatch):
    c,m=fixture(tmp_path);old=complete_cache(tmp_path,c,m)
    actual=Path.open
    def guarded(self,*args,**kwargs):
        assert self.suffix not in ('.png','.safetensors','.jpg')
        return actual(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    idx=cache_index(tmp_path,old)
    assert idx.receipt['image_files_opened']==0 and idx.receipt['tensor_files_opened']==0


@pytest.mark.parametrize('field',['source_student_identity','source_index_sha256','partition_sha256'])
def test_pure_compute_requires_all_source_hashes(tmp_path,field):
    c,m=fixture(tmp_path);del m[field];sign(m)
    with pytest.raises(ValueError,match=field):request_index(tmp_path,c,m)


def test_indexed_provider_preserves_target_masks_rng_and_cold_resume(tmp_path):
    c,m=fixture(tmp_path,missing_source=True)
    current=deepcopy(m);current['times_s']=[0.];current['scenes'][0]['frames']=current['scenes'][0]['frames'][:1];sign(current)
    cache=complete_cache(tmp_path,c,current)
    src_idx=cache_index(tmp_path,cache);src=IndexedTeacherCacheReader(src_idx,expected_identity=cache.identity['identity'])
    idx=request_index(tmp_path,c,m)
    exe=Executor(c)
    new=IndexedBoundedTeacherProvider(idx,c,[src],executor=exe,execution_identity=exe.execution_identity,
        max_cached_bytes=36,max_scene_bytes=10000,max_batch_images=2)
    old=provider(c,m,[cache],max_cached_bytes=36)
    rng=torch.get_rng_state().clone()
    a,b=new.read_scene('one'),old.read_scene('one')
    assert torch.equal(rng,torch.get_rng_state())
    for key in ('current_dense_features','current_valid','future_dense_features','future_valid'):
        assert torch.equal(a[key],b[key])
    assert not a['future_valid'][1].any()
    assert new.stats['immutable_hits']==1 and new.stats['computed_images']==3
    state=new.state_dict();new.read_scene('one');new.load_state_dict(state)
    assert new.memory_report()['resident_bytes']==0
    assert new.memory_report()['metadata_connections']==2
    assert new.memory_report()['metadata_cache_budget_bytes']==32*(1<<20)
    assert torch.equal(new.read_scene('one')['future_dense_features'],a['future_dense_features'])
    assert old.identity['identity']!=new.identity['identity']
    with pytest.raises(ValueError,match='identity changed'):new.load_state_dict(old.state_dict())
    new.verify_population(idx)


def test_source_role_and_contract_cannot_be_hidden_by_index(tmp_path):
    c,m=fixture(tmp_path);old=complete_cache(tmp_path,c,m)
    s=IndexedTeacherCacheReader(cache_index(tmp_path,old),expected_identity=old.identity['identity'])
    wrong=deepcopy(m);wrong['partition_sha256']='1'*64;sign(wrong)
    index=request_index(tmp_path,c,wrong)
    with pytest.raises(ValueError,match='lineage'):
        IndexedBoundedTeacherProvider(index,c,[s],max_cached_bytes=1,max_scene_bytes=10000,max_batch_images=2)


@pytest.mark.parametrize('which',['requests','database'])
def test_post_open_mutation_is_rejected(tmp_path,which):
    c,m=fixture(tmp_path);idx=request_index(tmp_path,c,m)
    path=tmp_path/'request_index.json' if which=='requests' else idx.root/'metadata.sqlite'
    with path.open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='version changed'):idx.scenes['one']


def test_checksum_and_incomplete_reject(tmp_path):
    c,m=fixture(tmp_path);idx=request_index(tmp_path,c,m);root=idx.root;idx.close()
    with (root/'metadata.sqlite').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='checksum'):MetadataIndex(root,expected_request_identity=m['identity'],expected_index_identity=json.loads((root/'COMPLETE.json').read_text())['identity'])
    empty=tmp_path/'empty';empty.mkdir()
    with pytest.raises(FileNotFoundError):MetadataIndex(empty,expected_request_identity=m['identity'],expected_index_identity='1'*64)


def test_original_tensor_corruption_still_rejected_without_recompute(tmp_path):
    c,m=fixture(tmp_path);cache=complete_cache(tmp_path,c,m)
    idx=cache_index(tmp_path,cache);reader=IndexedTeacherCacheReader(idx,expected_identity=cache.identity['identity'])
    index=request_index(tmp_path,c,m);exe=Executor(c)
    p=IndexedBoundedTeacherProvider(index,c,[reader],executor=exe,execution_identity=exe.execution_identity,
        max_cached_bytes=1,max_scene_bytes=10000,max_batch_images=2)
    with (cache.root/'shard_000000.safetensors').open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError,match='Changed teacher shard'):p.read_scene('one')
    assert not exe.calls


def test_budget_population_no_subset_and_idempotence(tmp_path):
    c,m=fixture(tmp_path);idx=request_index(tmp_path,c,m);d=idx.receipt
    again=build_index(idx.root,requests=tmp_path/'request_index.json',expected_request_identity=m['identity'],
        contract=c,max_scenes=1)
    assert again==d
    m['scenes'].append(dict(m['scenes'][0],token='two'));sign(m)
    requests=tmp_path/'two.json';atomic_json(requests,m)
    with pytest.raises(ValueError,match='max_scenes'):
        build_index(tmp_path/'limited',requests=requests,expected_request_identity=m['identity'],contract=c,max_scenes=1)
    assert not (tmp_path/'limited/COMPLETE.json').exists()
    with pytest.raises(ValueError,match='max_metadata_cache_bytes'):
        MetadataIndex(idx.root,expected_request_identity=d['request_identity'],expected_index_identity=d['identity'],max_metadata_cache_bytes=257*(1<<20))


def test_partial_publication_keeps_evidence_and_rebuilds(tmp_path,monkeypatch):
    from starVLA.dataloader import st_world_metadata_index as module
    c,m=fixture(tmp_path);p=tmp_path/'requests.json';atomic_json(p,m);root=tmp_path/'index'
    real=module.atomic_json
    def fail(*args,**kwargs):raise RuntimeError('receipt window')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(RuntimeError,match='receipt window'):
        build_index(root,requests=p,expected_request_identity=m['identity'],contract=c,max_scenes=1)
    assert (root/'metadata.sqlite').exists() and not (root/'COMPLETE.json').exists()
    monkeypatch.setattr(module,'atomic_json',real)
    build_index(root,requests=p,expected_request_identity=m['identity'],contract=c,max_scenes=1)
    assert len(list(root.glob('metadata.orphan.*.sqlite')))==1
    with MetadataIndex(root,expected_request_identity=m['identity'],expected_index_identity=json.loads((root/'COMPLETE.json').read_text())['identity']) as idx:assert idx.scenes['one']==m['scenes'][0]


def test_cache_coverage_reordering_rejected_even_with_forged_internal_hashes(tmp_path):
    from starVLA.dataloader.st_world_history import file_sha256
    c,m=fixture(tmp_path);cache=complete_cache(tmp_path,c,m)
    d=deepcopy(cache.identity);d['records'].reverse();sign(d)
    atomic_json(cache.root/'identity.json',d)
    done=json.loads((cache.root/'COMPLETE.json').read_text());done.update(identity=d['identity'],identity_sha256=file_sha256(cache.root/'identity.json'))
    atomic_json(cache.root/'COMPLETE.json',done)
    with pytest.raises(ValueError,match='coverage/order'):
        build_index(tmp_path/'index',requests=cache.root/'requests.json',cache_root=cache.root,
            expected_request_identity=m['identity'],expected_cache_identity=d['identity'],contract=c,max_scenes=1)


def test_actual_cli_contract_and_verify_are_bounded(tmp_path):
    from tools.st_world_policy.index_teacher_metadata import parser, run
    from starVLA.dataloader.st_world_history import file_sha256
    c,m=fixture(tmp_path);cache=complete_cache(tmp_path,c,m);out=tmp_path/'cli_index'
    args=parser().parse_args(['build','--cache-root',str(cache.root),
        '--expected-cache-identity',cache.identity['identity'],'--expected-request-identity',m['identity'],
        '--contract-receipt',str(cache.root/'identity.json'),
        '--contract-receipt-sha256',file_sha256(cache.root/'identity.json'),
        '--output',str(out),'--max-scenes','1'])
    result=run(args)
    assert result['scene_count']==1
    verify=parser().parse_args(['verify','--index',str(out),'--expected-request-identity',m['identity'],
        '--expected-index-identity',result['identity']])
    assert run(verify)['status']=='METADATA_VERIFIED'
    args.contract_receipt_sha256='1'*64
    with pytest.raises(ValueError,match='checksum'):run(args)


def test_absent_valid_label_remains_error_not_false_mask(tmp_path):
    c,m=fixture(tmp_path,missing_source=True);idx=request_index(tmp_path,c,m)
    p=IndexedBoundedTeacherProvider(idx,c,max_cached_bytes=1,max_scene_bytes=10000,max_batch_images=8)
    with pytest.raises(FileNotFoundError,match='no cached label'):p.read_scene('one')
    assert p.stats['failures']==1


def test_metadata_caps_sum_per_rank_and_old_unbounded_source_rejected(tmp_path):
    c,m=fixture(tmp_path);cache=complete_cache(tmp_path,c,m);source_index=cache_index(tmp_path,cache)
    idx=request_index(tmp_path,c,m)
    with pytest.raises(ValueError,match='indexed COMPLETE'):
        IndexedBoundedTeacherProvider(idx,c,[cache],max_cached_bytes=1,max_scene_bytes=10000,max_batch_images=8)
    high=MetadataIndex(source_index.root,expected_request_identity=m['identity'],expected_index_identity=source_index.receipt['identity'],max_metadata_cache_bytes=256*(1<<20))
    reader=IndexedTeacherCacheReader(high,expected_identity=cache.identity['identity'])
    with pytest.raises(ValueError,match='combined metadata'):
        IndexedBoundedTeacherProvider(idx,c,[reader],max_cached_bytes=1,max_scene_bytes=10000,max_batch_images=8)


def test_metadata_compression_roundtrip_and_bomb_boundary():
    text=canonical({'x':'测试😀'*100,'array':[1,2,3]})
    assert unpack(pack(text))==text
    for body in [pack('x'*65537),pack(text)[:-1],pack(text)+b'garbage']:
        with pytest.raises(ValueError,match='decompression'):unpack(body)
