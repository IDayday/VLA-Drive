"""Only synthetic metadata; no original image/log/checkpoint/cache is read."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.st_world_policy import history_shards as packing
from tools.st_world_policy import prepare_future_metadata as bounded
from tools.st_world_policy import prepare_causal_history as prepare
from starVLA.dataloader import st_world_history as history
from starVLA.dataloader.st_world_schema import identity_hash


def make_budget(root):
    return bounded.PreparationBudget(root,dict(max_seconds=90,
        max_rss_bytes=2<<30,max_metadata_bytes=32<<20,
        max_source_read_bytes=2<<30,read_mib_per_second=100000))


def valid_records():
    rows=[]
    for i in range(256):
        token=f't{i:04d}'; log=f'log{i//128}'
        frame={'token':token,'timestamp':1000000,
            'ego2global_rotation':[1,0,0,0],
            'ego2global_translation':[0,0,0]}
        current={'token':token,'log':log,'timestamp':1000000,
            'image_paths':['synthetic/F','synthetic/L','synthetic/R']}
        # No historical frame/image is supplied. Missing input remains masked.
        rows.append(history.build_history_record([frame],token,log,current,[],
            raw_log_sha256='a'*64))
    return rows


def write_manifest(root,rows,*,packed):
    entries=[]
    if packed:
        writer=packing.HistoryShards(root,make_budget(root),max_shards=4)
    for row in rows:
        relative='history/'+row['token']+'.json';path=root/relative
        if packed:writer.add(path,row);digest=writer.digest(path)
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(packing.record_bytes(row))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({'token':row['token'],'log':row['log'],
            'record':relative,'sha256':digest})
    # A nontrivial selection order must survive a grouped writing order.
    entries=entries[::2]+entries[1::2]
    value={'schema':'st_history_manifest_v1','scenes':entries}
    if packed:value=writer.manifest(value)
    else:value['identity']=identity_hash(value)
    path=root/'HISTORY_MANIFEST.json'
    path.write_text(json.dumps(value,sort_keys=True))
    (root/'COMPLETE.json').write_text(json.dumps({
        'history_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    return value


def test_256_validated_records_match_original_layout_and_exact_order(tmp_path):
    rows=valid_records();old=tmp_path/'files';new=tmp_path/'packed'
    old.mkdir();new.mkdir()
    original=write_manifest(old,rows,packed=False)
    packed=write_manifest(new,rows,packed=True)
    r0=history.HistoryReader(old,original['identity'])
    r1=history.HistoryReader(new,packed['identity'])
    assert original['identity']!=packed['identity']
    assert list(r0.index)==list(r1.index)
    assert len(list(new.glob('history_shards/*.jsonseq')))==2
    for row in rows:
        assert r0.read(row['token'])==r1.read(row['token'])==row
    before={p.name:(p.stat().st_ino,p.read_bytes()) for p in new.glob('history_shards/*.jsonseq')}
    assert write_manifest(new,rows,packed=True)==packed
    assert before=={p.name:(p.stat().st_ino,p.read_bytes()) for p in new.glob('history_shards/*.jsonseq')}


def test_exact_bounded_single_record_read_and_corruption_rejection(tmp_path):
    rows=valid_records();manifest=write_manifest(tmp_path,rows,packed=True)
    reader=history.HistoryReader(tmp_path,manifest['identity'])
    token=rows[3]['token'];entry=reader.index[token]
    path=tmp_path/entry['record'];requests=[];original=Path.open
    class Watching:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        def __getattr__(self,name):return getattr(self.f,name)
        def read(self,n=-1):requests.append(n);return self.f.read(n)
    def opener(self,*a,**kw):
        f=original(self,*a,**kw)
        return Watching(f) if self==path else f
    with patch.object(Path,'open',opener):assert reader.read(token)==rows[3]
    assert requests==[entry['record_length']]
    with original(path,'r+b') as f:
        f.seek(entry['record_offset']);f.write(b'!')
    with pytest.raises(ValueError,match='checksum'):reader.read(token)
    with pytest.raises(ValueError,match='conflict'):write_manifest(tmp_path,rows,packed=True)


@pytest.mark.parametrize('field,value', [('record_offset',-1),
    ('record_length',0),('record_length',(1<<20)+1),
    ('record_offset',True),('record_layout','unknown')])
def test_invalid_record_layout_and_ranges_rejected(tmp_path,field,value):
    rows=valid_records()[:1];manifest=write_manifest(tmp_path,rows,packed=True)
    reader=history.HistoryReader(tmp_path,manifest['identity'])
    reader.index[rows[0]['token']][field]=value
    with pytest.raises(ValueError):reader.read(rows[0]['token'])


def test_reproduce_path_swap_can_read_external_file(tmp_path):
    root=tmp_path/'root';root.mkdir();row=valid_records()[0]
    manifest=write_manifest(root,[row],packed=True)
    reader=history.HistoryReader(root,manifest['identity'])
    path=root/reader.index[row['token']]['record']
    external=tmp_path/'outside.jsonseq';external.write_bytes(path.read_bytes())
    original=Path.open;swapped=[]
    def opener(self,*a,**kw):
        if self==path and not swapped:
            path.unlink();path.symlink_to(external);swapped.append(True)
        return original(self,*a,**kw)
    # Defect characterization: validation accepts external bytes after a swap.
    with patch.object(Path,'open',opener):
        actual=reader.read(row['token'])
    assert swapped and actual==row and path.is_symlink()


def test_reproduce_fifo_reaches_blocking_open_without_type_guard(tmp_path):
    row=valid_records()[0];manifest=write_manifest(tmp_path,[row],packed=True)
    reader=history.HistoryReader(tmp_path,manifest['identity'])
    path=tmp_path/reader.index[row['token']]['record']
    path.unlink();os.mkfifo(path)
    original=Path.open;attempted=[]
    def opener(self,*a,**kw):
        if self==path:
            attempted.append(a)
            raise RuntimeError('fixture intercepts otherwise blocking FIFO open')
        return original(self,*a,**kw)
    # Avoid an actual hanging FIFO: capture the production open before execution.
    with patch.object(Path,'open',opener),pytest.raises(RuntimeError,match='blocking FIFO'):
        reader.read(row['token'])
    assert attempted==[('rb',)]


def test_reproduce_files_command_accepts_packed_only_inventory_registration():
    args=SimpleNamespace(action='build',pin_selection=None,scene_list='/fixture',
        target_times=[0.],mode='registered',max_scenes=64,record_layout='files')
    guard=SimpleNamespace(registration={'record_layout':'packed_log_v1',
        'inventory_strategy':'immutable_history_shards_v1'},limits={
        'max_disk_bytes':32<<20,'max_seconds':900,'max_rss_bytes':4<<30})
    with patch.object(prepare.storage,'_ACTIVE_GUARD',guard), \
         patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''), \
         patch.object(os,'sched_getaffinity',return_value={6,7}):
        # This is a mismatched recipe and should be rejected, but is accepted.
        prepare.validate(args)

