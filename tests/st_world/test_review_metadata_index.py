"""Independent metadata-only boundary tests: no image/tensor files or teacher."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from starVLA.dataloader.st_world_metadata_index import build_index, MetadataIndex, pack
from starVLA.dataloader.st_world_indexed_provider import IndexedBoundedTeacherProvider, IndexedTeacherCacheReader
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import atomic_json, unique_image_records
from starVLA.dataloader.st_world_history import file_sha256


def sign(value):
    value['identity'] = identity_hash({k:v for k,v in value.items() if k != 'identity'})
    return value


@pytest.fixture(autouse=True)
def forbid_payload_open(monkeypatch):
    original = Path.open
    def checked(path, *a, **kw):
        assert path.suffix not in {'.png','.jpg','.jpeg','.pt','.pth','.safetensors'}, path
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', checked)


def fixture():
    contract = TeacherFeatureContract(image_hw=(32,32), patch_hw=(16,16), native_grid_hw=(2,2),
        output_grid_hw=(2,2), channels=2, storage_dtype='float32', compute_dtype='float32',
        view_names=('CAM_F0','CAM_L0','CAM_R0'), teacher_identity='a'*64, transform_identity='b'*64,
        pooling={'kind':'none','kernel_hw':[1,1],'stride_hw':[1,1]})
    times = [0., .5, 1., 2., 4.]
    refs = [dict(path=f'/NONEXISTENT_METADATA_FIXTURE/{i}.png',sha256=f'{i+1:064x}',
                 bytes=8,mtime_ns=1,original_wh=[32,18]) for i in range(5)]
    frames = [dict(requested_s=t,timestamp_us=int(t*1e6),views=[ref]*3) for t,ref in zip(times,refs)]
    manifest = dict(schema='st_teacher_requests_v1',split='train',times_s=times,
        view_names=list(contract.view_names),timestamp_tolerance_s=.05,
        source_student_identity='c'*64,source_index_sha256='d'*64,partition_sha256='e'*64,
        scenes=[dict(token='one',log='log-a',split='train',timestamp_us=0,frames=frames)])
    return contract, sign(manifest)


def request_index(tmp_path, contract, manifest, name='index'):
    source = tmp_path/(name+'.json'); atomic_json(source, manifest)
    receipt = build_index(tmp_path/name,requests=source,expected_request_identity=manifest['identity'],contract=contract,max_scenes=8)
    return MetadataIndex(tmp_path/name,expected_request_identity=manifest['identity'],expected_index_identity=receipt['identity'])


def cache_metadata(tmp_path, contract, manifest):
    root = tmp_path/'fake_cache_metadata';root.mkdir()
    records = unique_image_records(manifest,contract)
    metadata = sign(dict(schema='st_teacher_cache_v1',contract=contract.to_dict(),
        request_manifest_identity=manifest['identity'],records=records,images_per_shard=2,
        shards=(len(records)+1)//2,execution={'metadata_only_fixture':True}))
    atomic_json(root/'requests.json',manifest);atomic_json(root/'identity.json',metadata)
    atomic_json(root/'COMPLETE.json',dict(identity=metadata['identity'],status='COMPLETE',
        identity_sha256=file_sha256(root/'identity.json'),requests_sha256=file_sha256(root/'requests.json'),
        scenes=len(manifest['scenes']),unique_images=len(records),
        shards=[dict(shard=i,receipt_sha256=f'{i+11:064x}') for i in range(metadata['shards'])]))
    receipt = build_index(tmp_path/'source_index',requests=root/'requests.json',cache_root=root,
        expected_request_identity=manifest['identity'],expected_cache_identity=metadata['identity'],
        contract=contract,max_scenes=8)
    index = MetadataIndex(tmp_path/'source_index',expected_request_identity=manifest['identity'],expected_index_identity=receipt['identity'])
    return metadata,index


def test_full_ordinal_duplicate_scene_content_and_tail_mapping(tmp_path):
    c,m=fixture();other=deepcopy(m['scenes'][0]);other['token']='two';m['scenes'].append(other);sign(m)
    metadata,index=cache_metadata(tmp_path,c,m)
    reader=IndexedTeacherCacheReader(index,expected_identity=metadata['identity'],expected_contract=c)
    assert len(reader.locations)==5 and len(reader.scenes)==2
    for i,record in enumerate(metadata['records']):
        assert reader.locations[record['key']]==divmod(i,2)
        assert reader.identity['records'][i]==record
    assert reader.identity['records'][-1:]==metadata['records'][-1:]
    assert reader.manifest['scenes'][::-1]==m['scenes'][::-1]
    assert list(reader.receipt_hashes)==[0,1,2]
    index.close()


@pytest.mark.parametrize('field',['source_student_identity','source_index_sha256','partition_sha256'])
def test_zero_source_requires_lineage_even_with_valid_semantic_checksum(tmp_path,field):
    c,m=fixture();m[field]=None;sign(m)
    with pytest.raises(ValueError,match=field):request_index(tmp_path,c,m)
    assert not (tmp_path/'index/COMPLETE.json').exists()


def test_cold_metadata_identity_restoration_cannot_accept_other_population(tmp_path):
    c,m=fixture();idx=request_index(tmp_path,c,m)
    provider=IndexedBoundedTeacherProvider(idx,c,max_cached_bytes=1,max_scene_bytes=4096,max_batch_images=8)
    state=provider.state_dict();provider._resident['sentinel']=(None,None);provider._bytes=1
    provider.load_state_dict(state);assert not provider._resident and provider._bytes==0
    altered=deepcopy(state);altered['identity']='0'*64
    with pytest.raises(ValueError,match='identity changed'):provider.load_state_dict(altered)
    idx.close()


def test_source_role_rejected_without_touching_data(tmp_path):
    c,m=fixture();metadata,source=cache_metadata(tmp_path,c,m)
    m['split']='dev';m['scenes'][0]['split']='dev';sign(m)
    index=request_index(tmp_path,c,m)
    cache=IndexedTeacherCacheReader(source,expected_identity=metadata['identity'])
    with pytest.raises(ValueError,match='lineage'):
        IndexedBoundedTeacherProvider(index,c,[cache],max_cached_bytes=1,max_scene_bytes=4096,max_batch_images=8)
    index.close();source.close()


def test_row_checksum_rejects_corruption_even_if_db_receipt_is_recomputed(tmp_path):
    c,m=fixture();idx=request_index(tmp_path,c,m);root=idx.root;idx.close()
    with sqlite3.connect(root/'metadata.sqlite') as db:db.execute('UPDATE scenes SET body=? WHERE pos=0',(pack('{"token":"tampered"}'),))
    receipt=json.loads((root/'COMPLETE.json').read_text());receipt['database_sha256']=file_sha256(root/'metadata.sqlite');receipt['database_bytes']=(root/'metadata.sqlite').stat().st_size;sign(receipt)
    atomic_json(root/'COMPLETE.json',receipt)
    with MetadataIndex(root,expected_request_identity=m['identity'],expected_index_identity=receipt['identity']) as opened:
        with pytest.raises(ValueError,match='record corruption'):opened.scenes['one']


@pytest.mark.parametrize('relative',['index.json','index/metadata.sqlite','index/COMPLETE.json'])
def test_access_rejects_changed_original_index_or_receipt(tmp_path,relative):
    c,m=fixture();idx=request_index(tmp_path,c,m)
    with (tmp_path/relative).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='version changed'):idx.scenes['one']
    idx.close()


def test_actual_cli_verify_no_tensor_and_reject_wrong_expected_identity(tmp_path):
    from tools.st_world_policy.index_teacher_metadata import parser,run
    c,m=fixture();idx=request_index(tmp_path,c,m)
    argv=['verify','--index',str(idx.root),'--expected-request-identity',m['identity'],'--expected-index-identity',idx.receipt['identity']]
    result=run(parser().parse_args(argv));assert result['status']=='METADATA_VERIFIED' and result['training_admission'] is False
    argv[-1]='0'*64
    with pytest.raises(ValueError,match='checksum'):run(parser().parse_args(argv))
    idx.close()


def test_no_partial_population_published(tmp_path):
    c,m=fixture();m['scenes'].append(dict(m['scenes'][0],token='two'));sign(m)
    source=tmp_path/'requests.json';atomic_json(source,m)
    with pytest.raises(ValueError,match='max_scenes'):
        build_index(tmp_path/'limited',requests=source,expected_request_identity=m['identity'],contract=c,max_scenes=1)
    assert not (tmp_path/'limited/COMPLETE.json').exists()


def test_registered_budget_and_multi_connection_sum_not_silent(tmp_path):
    c,m=fixture();metadata,source=cache_metadata(tmp_path,c,m);idx=request_index(tmp_path,c,m)
    huge=MetadataIndex(source.root,expected_request_identity=m['identity'],expected_index_identity=source.receipt['identity'],max_metadata_cache_bytes=256*(1<<20))
    reader=IndexedTeacherCacheReader(huge,expected_identity=metadata['identity'])
    with pytest.raises(ValueError,match='combined metadata'):
        IndexedBoundedTeacherProvider(idx,c,[reader],max_cached_bytes=1,max_scene_bytes=4096,max_batch_images=8)
    idx.close();source.close();huge.close()


def test_cli_contract_must_be_the_same_bytes_that_passed_pin(tmp_path,monkeypatch):
    """A concurrent atomic replacement must not change the approved teacher."""
    from tools.st_world_policy import index_teacher_metadata as cli
    c,m=fixture();requests=tmp_path/'requests.json';atomic_json(requests,m)
    contract_receipt=tmp_path/'contract_receipt.json'
    approved={'schema':'st_fixed_teacher_qualification_plan_v1','contract':c.to_dict()}
    atomic_json(contract_receipt,approved);approved_sha=file_sha256(contract_receipt)
    args=cli.parser().parse_args(['build','--requests',str(requests),
        '--expected-request-identity',m['identity'],'--contract-receipt',str(contract_receipt),
        '--contract-receipt-sha256',approved_sha,'--output',str(tmp_path/'index'),
        '--max-scenes','1'])
    original=Path.open
    replaced=False
    class ReplaceAfterRead:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):self.handle.__enter__();return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def __getattr__(self,key):return getattr(self.handle,key)
        def read(self,*args,**kwargs):
            nonlocal replaced
            value=self.handle.read(*args,**kwargs)
            if not replaced:
                replaced=True
                replacement=deepcopy(approved);replacement['contract']['teacher_identity']='9'*64
                atomic_json(contract_receipt,replacement)
            return value
    def checked_open(path,*args,**kwargs):
        handle=original(path,*args,**kwargs)
        mode=args[0] if args else kwargs.get('mode','r')
        return ReplaceAfterRead(handle) if Path(path)==contract_receipt and 'r' in mode else handle
    monkeypatch.setattr(Path,'open',checked_open)
    try:
        cli.run(args)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ('checksum','changed','contract'))
        assert not (tmp_path/'index/COMPLETE.json').exists()
    else:
        complete=json.loads((tmp_path/'index/COMPLETE.json').read_text())
        assert complete['contract']==approved['contract'], 'Published contract differs from the bytes that passed the pin'
    assert replaced
