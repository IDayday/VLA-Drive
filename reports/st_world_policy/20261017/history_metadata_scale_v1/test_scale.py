import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.st_world_policy import history_shards as h
from tools.st_world_policy import prepare_future_metadata as bounded
from tools.st_world_policy import metadata_storage as m
from starVLA.dataloader import st_world_history as history
from starVLA.dataloader.st_world_schema import identity_hash


def budget(root):
    return bounded.PreparationBudget(root,dict(max_seconds=60,max_rss_bytes=2<<30,max_metadata_bytes=32<<20,max_source_read_bytes=2<<30,read_mib_per_second=100000))

def population():
    return [dict(token=f't{i:04d}',log='log0',offsets_s=[-1.,-.5],frames=[{'valid':False}],marker=i) for i in range(256)]

def build(root,rows):
    writer=h.HistoryShards(root,budget(root),max_shards=4)
    entries=[]
    for row in rows:
        path=root/'history'/f'{row["token"]}.json';writer.add(path,row)
        entries.append(dict(token=row['token'],log=row['log'],record=str(path.relative_to(root)),sha256=writer.digest(path)))
    manifest=writer.manifest({'schema':'fixture','scenes':entries})
    return writer,manifest


def test_256_records_two_shards_order_bounded_reads_and_resume(tmp_path):
    rows=population();w,manifest=build(tmp_path,rows)
    assert w.shards==2 and len(list(tmp_path.rglob('*.jsonseq')))==2
    assert [r['token'] for r in manifest['scenes']]==[r['token'] for r in rows]
    before={p.name:(p.stat().st_ino,p.read_bytes()) for p in tmp_path.rglob('*.jsonseq')}
    w2,resumed=build(tmp_path,rows)
    assert manifest==resumed
    assert before=={p.name:(p.stat().st_ino,p.read_bytes()) for p in tmp_path.rglob('*.jsonseq')}
    reader=history.HistoryReader.__new__(history.HistoryReader);reader.root=tmp_path;reader.index={r['token']:r for r in manifest['scenes']}
    with patch.object(history,'validate_history_record',side_effect=lambda x:x):
        assert [reader.read(r['token']) for r in rows]==rows
    assert sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())<32<<20


def test_corrupt_truncated_foreign_and_orphan_recovery(tmp_path):
    rows=population();w,manifest=build(tmp_path,rows)
    orphan=tmp_path/'history_shards/orphan.partial';orphan.write_bytes(b'failed preserved')
    before=orphan.read_bytes();w2,again=build(tmp_path,rows);assert again==manifest and orphan.read_bytes()==before
    file=next(tmp_path.rglob('*.jsonseq'));raw=file.read_bytes();file.write_bytes(b'!'+raw[1:])
    with pytest.raises(ValueError,match='conflict'):build(tmp_path,rows)
    reader=history.HistoryReader.__new__(history.HistoryReader);reader.root=tmp_path;reader.index={r['token']:r for r in manifest['scenes']}
    with pytest.raises(ValueError,match='checksum'):reader.read(rows[0]['token'])
    file.write_bytes(raw[:2])
    with pytest.raises(ValueError,match='Truncated'):reader.read(rows[0]['token'])


def test_shard_budget_is_not_unbounded_entry_cap(tmp_path):
    w=h.HistoryShards(tmp_path,budget(tmp_path),max_shards=1)
    for row in population()[:129]:w.add(tmp_path/'history'/f'{row["token"]}.json',row)
    with pytest.raises(ValueError,match='count budget'):w.flush()
    with pytest.raises(ValueError):h.HistoryShards(tmp_path,budget(tmp_path),max_shards=100000)


def test_read_callbacks_do_not_census_per_record_but_publication_always_does(tmp_path):
    reg=dict(writer_kind='history_preparation',record_layout='packed_log_v1',inventory_strategy='immutable_history_shards_v1',limits=dict(max_disk_bytes=32<<20,max_rss_bytes=2<<30,max_seconds=60,max_entries=20000,max_depth=1),file_patterns=['.*'],directory_patterns=['history_shards'])
    fake=SimpleNamespace(_total=lambda _:32<<20,_check_disk=lambda *a,**k:None,core={'free_reserve_bytes':0})
    scans=[]
    def census(*a,**k):scans.append(1);return {'charged_bytes':0,'entries':[]}
    with patch.object(m,'tree_inventory',side_effect=census),patch.object(m,'process_tree_rss',return_value=1),patch.object(m,'_checkpoint_reserve',return_value=0):
        guard=m.MetadataWriteContext(fake,{},tmp_path,{},reg,'unused','unused')
        for _ in range(256):guard.check()
        assert len(scans)==1
        guard.check(before_commit=True);assert len(scans)==2
        guard._inventory_at-=5;guard.check();assert len(scans)==3
        guard.check(before_commit=True);assert len(scans)==4
        reg['writer_kind']='future_preparation'
        with pytest.raises(ValueError):m._resources(reg)
