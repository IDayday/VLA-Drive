"""Synthetic bounded resource checks; no live claims, ledger, raw logs or images."""
import hashlib,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.st_world_policy import history_shards as h,prepare_future_metadata as bounded,metadata_storage as m
from starVLA.dataloader import st_world_history as history


def budget(root,**updates):
 limits=dict(max_seconds=60,max_rss_bytes=2<<30,max_metadata_bytes=16<<20,max_source_read_bytes=16<<20,read_mib_per_second=100000)
 limits.update(updates);return bounded.PreparationBudget(root,limits)


def packed(root,rows,**kwargs):
 writer=h.HistoryShards(root,budget(root,**kwargs),max_shards=4)
 entries=[]
 for row in rows:
  path=root/'history'/f'{row["token"]}.json';writer.add(path,row)
  assert writer.size<=h.MAX_SHARD_BYTES and len(writer.pending)<=h.MAX_SHARD_RECORDS
  entries.append({'token':row['token'],'log':row['log'],'record':str(path.relative_to(root))})
 return writer,writer.manifest({'scenes':entries})


def test_byte_bound_and_resume_reads_are_charged(tmp_path):
 rows=[{'token':f't{i}','log':'l','padding':'x'*(900<<10)} for i in range(5)]
 writer,manifest=packed(tmp_path,rows)
 files=list(tmp_path.rglob('*.jsonseq'))
 assert len(files)==2 and all(0<p.stat().st_size<=4<<20 for p in files)
 assert sum(len(h.record_bytes(r)) for r in rows)==sum(p.stat().st_size for p in files)
 before={p.name:(p.stat().st_ino,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
 again,manifest2=packed(tmp_path,rows)
 assert manifest==manifest2
 assert before=={p.name:(p.stat().st_ino,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
 assert again.budget.bytes>=sum(p.stat().st_size for p in files)
 with pytest.raises(bounded.BudgetExceeded,match='SOURCE_READ_BYTES_BUDGET'):
  packed(tmp_path,rows,max_source_read_bytes=2<<20)
 assert before=={p.name:(p.stat().st_ino,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}


def test_retained_partial_is_in_resumed_disk_budget(tmp_path):
 (tmp_path/'interrupted.partial').write_bytes(b'x'*(2<<20))
 with pytest.raises(bounded.BudgetExceeded,match='METADATA_DISK_BUDGET'):
  budget(tmp_path,max_metadata_bytes=1<<20)
 assert (tmp_path/'interrupted.partial').stat().st_size==2<<20


def test_record_limit_fails_without_publication(tmp_path):
 writer=h.HistoryShards(tmp_path,budget(tmp_path),max_shards=1)
 with pytest.raises(ValueError,match='record exceeds'):
  writer.add(tmp_path/'history/big.json',{'token':'big','log':'l','padding':'x'*(1<<20)})
 assert writer.size==0 and writer.shards==0 and not list(tmp_path.iterdir())


def test_single_pread_and_fd_bound_even_on_errors(tmp_path,monkeypatch):
 folder=tmp_path/'history_shards';folder.mkdir()
 path=folder/'a.jsonseq';path.write_bytes(b'p'*128 + b'z'*(4<<20))
 before=len(list(Path('/proc/self/fd').iterdir())); live=set();peak=0; reads=[]
 original_open,original_close,original_pread=os.open,os.close,os.pread
 def opened(*a,**kw):
  nonlocal peak
  fd=original_open(*a,**kw);live.add(fd);peak=max(peak,len(live));return fd
 def closed(fd):
  live.discard(fd);return original_close(fd)
 def pread(fd,n,off):
  reads.append((n,off));return original_pread(fd,n,off)
 with patch.object(history.os,'open',side_effect=opened),patch.object(history.os,'close',side_effect=closed),patch.object(history.os,'pread',side_effect=pread):
  for i in range(100):
   assert history.read_packed_history_range(tmp_path,'history_shards/a.jsonseq',128,4096)==b'z'*4096
  for i in range(10):
   with pytest.raises(ValueError,match='Truncated'):
    history.read_packed_history_range(tmp_path,'history_shards/a.jsonseq',9<<20,4096)
 assert not live and peak<=4
 assert reads==[(4096,128)]*100
 assert len(list(Path('/proc/self/fd').iterdir()))==before


def registration():
 return dict(writer_kind='history_preparation',record_layout='packed_log_v1',inventory_strategy='immutable_history_shards_v1',limits=dict(max_disk_bytes=4<<20,max_rss_bytes=2<<30,max_seconds=60,max_entries=20000,max_depth=1),file_patterns=['.*'],directory_patterns=['history_shards'])


def test_cached_census_still_checks_rss_storage_and_precommit_disk(tmp_path,monkeypatch):
 clock=[10.];charged=[0];counts={'scan':0,'rss':0,'reserve':0,'disk':0}
 def census(*a,**kw):counts['scan']+=1;return {'charged_bytes':charged[0],'entries':[]}
 def rss(*a):counts['rss']+=1;return 1
 def reserve(*a):counts['reserve']+=1;return 0
 def disk(*a,**kw):counts['disk']+=1
 fake=SimpleNamespace(_total=lambda _:4<<20,_check_disk=disk,core={'free_reserve_bytes':0})
 monkeypatch.setattr(m.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(m,'tree_inventory',census)
 monkeypatch.setattr(m,'process_tree_rss',rss);monkeypatch.setattr(m,'_checkpoint_reserve',reserve)
 guard=m.MetadataWriteContext(fake,{},tmp_path,{},registration(),'fixture','fixture')
 guard.check()
 for i in range(1,50):clock[0]=10+i/10;guard.check()
 assert counts=={'scan':1,'rss':50,'reserve':50,'disk':50}
 clock[0]=15.;guard.check();assert counts['scan']==2
 charged[0]=5<<20
 with pytest.raises(ValueError,match='allocation exceeds'):guard.check(before_commit=True)
 assert counts['scan']==3 and guard.failed
 with pytest.raises(ValueError,match='already failed'):guard.check()

@pytest.mark.parametrize('kind,layout,strategy',[('future_preparation','packed_log_v1','immutable_history_shards_v1'),('current_index','packed_log_v1','immutable_history_shards_v1'),('history_preparation','files','immutable_history_shards_v1'),('history_preparation','packed_log_v1','other')])
def test_census_scope_cannot_expand_to_other_writer(kind,layout,strategy):
 reg=registration();reg.update(writer_kind=kind,record_layout=layout,inventory_strategy=strategy)
 with pytest.raises(ValueError):m._resources(reg)


def test_original_global_entry_cap_not_relaxed():
 reg=registration();reg['limits']['max_entries']=20001
 with pytest.raises(ValueError,match='resource limit'):m._resources(reg)
