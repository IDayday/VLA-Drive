"""Lossless physical reuse of already verified shards on the same shared mount."""
from pathlib import Path
import os
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json
from .storage import CacheReader, recover_shard


def inspect_reuse(roots, identity, planned_rows):
    expected={r['scene_id']:r for r in planned_rows}; sources=[]
    for path in roots:
        reader=CacheReader(path,identity=identity)
        require(set(reader.index['access']) <= expected.keys(), 'reuse cache contains foreign role/population')
        for token,row in reader.index['access'].items():
            require(row['observation_hash']==expected[token]['observation_hash'] and row['role']==expected[token]['role'],
                    'reuse current observation/role mismatch')
        sources.append({'root':str(Path(path).resolve()),'index_sha256':reader.complete['index_sha256'],
                        'index':reader.index})
        reader.close()
    return sources


def link_reuse(root,sources,identity,access):
    root=Path(root);keys={v['key'] for v in access.values()}; admitted=set(); shard_ids=[]; reused_bytes=0
    for source in sources:
        grouped={}
        for key,row in source['index']['records'].items():grouped.setdefault(row['shard'],{})[key]=row
        for name,records in grouped.items():
            require(set(records)<=keys,'image bytes/key changed since reusable cache was built')
            require(not set(records)&admitted,'overlapping reused payloads; select a non-overlapping source bank')
            old=Path(source['root'])/name;meta=source['index']['shards'][name]
            require(file_hash(old)==meta['sha256'],'reused shard checksum mismatch')
            sid='reuse_'+meta['sha256'];target=root/(sid+'.bin')
            require(old.stat().st_dev==root.stat().st_dev,'reuse must share filesystem; no implicit large copy')
            new_records={key:{**row,'shard':target.name} for key,row in records.items()}
            receipt={'schema':source['index']['schema'],'identity_hash':digest(identity),'bytes':meta['bytes'],
                     'sha256':meta['sha256'],'records':new_records}
            atomic_json(str(target)+'.INTENT.json',receipt,immutable=True)
            if not target.exists():os.link(old,target)
            require(os.path.samefile(old,target),'reused path must retain exact immutable file identity')
            atomic_json(str(target)+'.COMPLETE.json',receipt,immutable=True)
            admitted.update(records);shard_ids.append(sid);reused_bytes+=meta['bytes']
    atomic_json(root/'REUSE.json',{'sources':[{k:v for k,v in r.items() if k!='index'} for r in sources],
                                 'payloads':len(admitted),'physically_shared_bytes':reused_bytes,
                                 'logical_bytes_not_double_counted':True},immutable=True)
    return admitted,shard_ids,reused_bytes
