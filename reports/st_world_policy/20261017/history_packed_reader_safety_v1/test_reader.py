import os
from pathlib import Path
import time
from unittest.mock import patch
import pytest
from starVLA.dataloader import st_world_history as h


def asset(tmp_path):
    root=tmp_path/'root';folder=root/'history_shards';folder.mkdir(parents=True)
    path=folder/'a.jsonseq';path.write_bytes(b'prefix{"a":1}suffix')
    return root,path


def test_actual_regular_range(tmp_path):
    root,path=asset(tmp_path)
    assert h.read_packed_history_range(root,'history_shards/a.jsonseq',6,7)==b'{"a":1}'

@pytest.mark.parametrize('kind',['fifo','directory','hardlink','leaf_symlink','parent_symlink'])
def test_nonregular_and_alias_refuse_without_block(tmp_path,kind):
    root,path=asset(tmp_path);outside=tmp_path/'outside';outside.write_bytes(path.read_bytes())
    path.unlink()
    if kind=='fifo':os.mkfifo(path)
    if kind=='directory':path.mkdir()
    if kind=='hardlink':os.link(outside,path)
    if kind=='leaf_symlink':path.symlink_to(outside)
    if kind=='parent_symlink':
        path.parent.rmdir();path.parent.symlink_to(tmp_path)
    start=time.monotonic()
    with pytest.raises(ValueError):h.read_packed_history_range(root,'history_shards/a.jsonseq',6,7)
    assert time.monotonic()-start<1


def test_leaf_swap_at_open_never_reads_outside(tmp_path):
    root,path=asset(tmp_path);outside=tmp_path/'outside';outside.write_bytes(b'prefix{"a":1}suffix')
    original=os.open;seen=[]
    def raced(name,flags,*args,**kwargs):
        if name=='a.jsonseq':
            path.unlink();path.symlink_to(outside);seen.append(flags)
        return original(name,flags,*args,**kwargs)
    with patch.object(h.os,'open',side_effect=raced),patch.object(h.os,'pread',side_effect=AssertionError('outside read')):
        with pytest.raises(ValueError,match='Unsafe'):h.read_packed_history_range(root,'history_shards/a.jsonseq',6,7)
    assert seen and seen[0]&os.O_NOFOLLOW and seen[0]&os.O_NONBLOCK

@pytest.mark.parametrize('change',['replace','inplace','hardlink','ancestor'])
def test_after_open_mutations_fail_before_return(tmp_path,change):
    root,path=asset(tmp_path);original=os.pread
    def raced(fd,n,offset):
        data=original(fd,n,offset)
        if change=='replace':
            path.unlink();path.write_bytes(b'prefix{"a":1}suffix')
        if change=='inplace':
            before=path.stat();path.write_bytes(b'prefix{"a":2}suffix')
            # This filesystem can coalesce same-tick mtime/ctime changes.
            # Exercise version detection with an explicit observable change.
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000000))
        if change=='hardlink':os.link(path,tmp_path/'alias')
        if change=='ancestor':
            path.parent.rename(root/'moved');(root/'history_shards').symlink_to(root/'moved')
        return data
    with patch.object(h.os,'pread',side_effect=raced),pytest.raises(ValueError):
        h.read_packed_history_range(root,'history_shards/a.jsonseq',6,7)

@pytest.mark.parametrize('relative,offset,length',[('../outside',0,1),('/outside',0,1),('history_shards/a.jsonseq',-1,1),('history_shards/a.jsonseq',0,2<<20),('history_shards/a.jsonseq',999,1)])
def test_lexical_and_range_reject(tmp_path,relative,offset,length):
    root,path=asset(tmp_path)
    with pytest.raises(ValueError):h.read_packed_history_range(root,relative,offset,length)


def test_files_mode_cannot_borrow_packed_census_registration():
    from types import SimpleNamespace
    from tools.st_world_policy import prepare_causal_history as writer
    args=SimpleNamespace(action='build',pin_selection=None,scene_list='/fixed',target_times=[0.],mode='registered',max_scenes=64,record_layout='files')
    guard=SimpleNamespace(registration={},limits={'max_disk_bytes':32<<20,'max_seconds':900,'max_rss_bytes':4<<30})
    with patch.object(writer.storage,'_ACTIVE_GUARD',guard),patch.object(writer.os,'sched_getaffinity',return_value={6,7}),patch.dict(writer.os.environ,CUDA_VISIBLE_DEVICES=''):
        writer.validate(args)  # Legacy files registration remains valid.
        guard.registration={'record_layout':'packed_log_v1','inventory_strategy':'immutable_history_shards_v1'}
        with pytest.raises(ValueError,match='Files command'):writer.validate(args)
        guard.registration={'record_layout':'files','inventory_strategy':'immutable_history_shards_v1'}
        with pytest.raises(ValueError,match='Files command'):writer.validate(args)
