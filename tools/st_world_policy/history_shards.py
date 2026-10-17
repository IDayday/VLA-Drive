"""Immutable bounded history record-range storage; record semantics remain unchanged."""
import hashlib
import json
from pathlib import Path
from tools.st_world_policy import prepare_future_metadata as bounded

LAYOUT='packed_log_v1'
MAX_RECORD_BYTES=1<<20
MAX_SHARD_BYTES=4<<20
MAX_SHARD_RECORDS=128


def record_bytes(value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>MAX_RECORD_BYTES:raise ValueError('History record exceeds bounded schema storage')
    return raw


class HistoryShards:
    def __init__(self,root,budget,*,max_shards):
        if type(max_shards) is not int or not 1<=max_shards<=4096:raise ValueError('Explicit bounded shard allowance required')
        self.root=Path(root);self.budget=budget;self.max_shards=max_shards
        self.rows={};self.pending=[];self.size=0;self.log=None;self.shards=0

    def add(self,path,value):
        token=value['token'];log=value['log']
        if token in self.rows:raise ValueError('Duplicate history scene')
        if Path(path)!=self.root/'history'/f'{token}.json':raise ValueError('Unexpected original record destination')
        raw=record_bytes(value)
        if self.pending and (log!=self.log or len(self.pending)>=MAX_SHARD_RECORDS or self.size+len(raw)>MAX_SHARD_BYTES):self.flush()
        self.log=log
        self.pending.append((token,raw));self.size+=len(raw)
        self.rows[token]={'sha256':hashlib.sha256(raw).hexdigest()}
        self.budget.check()

    def digest(self,path):
        token=Path(path).stem
        if Path(path).parent==self.root/'history' and token in self.rows:return self.rows[token]['sha256']
        return bounded.budget_sha(path,self.budget)

    def flush(self):
        if not self.pending:return
        if self.shards>=self.max_shards:raise ValueError('History shard count budget exhausted')
        name=f'history_shards/{self.shards:06d}_{hashlib.sha256(self.log.encode()).hexdigest()[:16]}.jsonseq'
        bounded.write_chunks(self.root/name,[r for _,r in self.pending],self.budget)
        offset=0
        for token,raw in self.pending:
            self.rows[token].update(record=name,record_offset=offset,record_length=len(raw),record_layout=LAYOUT)
            offset+=len(raw)
        self.shards+=1;self.pending=[];self.size=0

    def manifest(self,value):
        self.flush()
        result=dict(value,scenes=[dict(row,**self.rows[row['token']]) for row in value['scenes']])
        from starVLA.dataloader.st_world_schema import identity_hash
        result['identity']=identity_hash({k:v for k,v in result.items() if k!='identity'})
        return result
