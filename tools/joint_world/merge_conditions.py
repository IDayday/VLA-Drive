"""Merge complete frozen current-feature shards without opening any labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','shards','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();tokens=json.loads(Path(a.manifest).read_text());out=Path(a.output)
    if len(tokens)!=len(set(tokens)):raise ValueError('Duplicate requested tokens')
    if out.exists():raise FileExistsError('Use a fresh immutable merge directory')
    manifests=[(path,json.loads((path/'manifest.json').read_text())) for path in sorted(Path(a.shards).glob('conditions_shard*'))]
    if not manifests:raise ValueError('No complete shards')
    reference=manifests[0][1]['identity'];records={}
    for root,m in manifests:
        for key in ['schema_version','code_sha','world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract','frozen_upstream','targets_loaded']:
            if m['identity'][key]!=reference[key]:raise ValueError('Shard identity mismatch: '+key)
        if m['failed']:raise ValueError('Shard failures must be resolved, never discarded')
        for row in m['records']:
            token=row['token'];path=root/(token+'.pt')
            if token in records:raise ValueError('Duplicate shard token')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed condition file')
            records[token]=dict(row,identity_sha256=m['identity_sha256'],source=str(path.resolve()))
    if set(tokens)!=set(records):raise ValueError('Missing/extra requested conditions')
    identity=dict(reference,source_shard_identities=[m['identity_sha256'] for _,m in manifests],
                  manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest())
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True)
    for token in tokens:(out/(token+'.pt')).symlink_to(records[token]['source'])
    (out/'manifest.json').write_text(json.dumps({'identity':identity,'identity_sha256':identity_sha,
        'records':[records[token] for token in tokens],'failed':0},indent=2))
    print(json.dumps({'scenes':len(tokens),'identity_sha256':identity_sha,'labels_opened':False,'output':str(out)}))


if __name__=='__main__':main()
