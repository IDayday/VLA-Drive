"""Merge immutable current-feature shards and independently completed full targets."""
import argparse
import hashlib
import json
from pathlib import Path
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','shards','original-targets','rebuilt-overflow','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();tokens=json.loads(Path(a.manifest).read_text());root=Path(a.output)
    if root.exists():raise FileExistsError('Use a fresh corpus directory')
    manifests=[(path,json.loads((path/'manifest.json').read_text())) for path in sorted(Path(a.shards).glob('conditions_shard*'))]
    if len(manifests)!=3:raise ValueError('Expected all3 complete shards')
    keyset=['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract','frozen_upstream','targets_loaded']
    reference=manifests[0][1]['identity'];records={}
    for path,m in manifests:
        for key in keyset:
            if m['identity'][key]!=reference[key]:raise ValueError('Shard identity mismatch: '+key)
        for r in m['records']:
            if r['token'] in records:raise ValueError('Duplicate shard token')
            records[r['token']]=dict(r,identity_sha256=m['identity_sha256'],source=str((path/(r['token']+'.pt')).resolve()))
    if set(tokens)!=set(records):raise ValueError('Missing/extra corpus scenes')
    original=json.loads((Path(a.original_targets)/'audit.json').read_text());original_rows={r['token']:r for r in original['records']}
    rebuilt=json.loads((Path(a.rebuilt_overflow)/'audit.json').read_text());rebuilt_rows={r['token']:r for r in rebuilt['records']}
    if rebuilt['failures']:raise ValueError('Incomplete overflow target rebuild')
    root.mkdir();cache=root/'conditions';cache.mkdir();targets=root/'world_targets';targets.mkdir()
    for key in ['targets','observations']:(targets/key).mkdir()
    merged=[];target_rows=[]
    for token in tokens:
        record=records[token];src=Path(record['source'])
        if hashlib.sha256(src.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed frozen feature shard')
        os.symlink(src,cache/(token+'.pt'));merged.append(record)
        target_root=Path(a.rebuilt_overflow) if original_rows[token]['overflow'] else Path(a.original_targets)
        audit=rebuilt_rows[token] if original_rows[token]['overflow'] else original_rows[token]
        if audit['overflow']:raise ValueError('Full GT required')
        for key,ext in [('targets','.pt'),('observations','.npz')]:os.symlink((target_root/key/(token+ext)).resolve(),targets/key/(token+ext))
        target_rows.append(audit)
    identity=dict(reference,source_shard_identities=[m['identity_sha256'] for _,m in manifests],manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest())
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    (cache/'manifest.json').write_text(json.dumps({'identity':identity,'identity_sha256':identity_hash,'records':merged,'failed':0},indent=2))
    (targets/'audit.json').write_text(json.dumps({'requested':len(tokens),'completed':len(tokens),'failures':[],'overflow_scenes':0,'records':target_rows,'construction':'immutable symlinks: nonoverflow originalv6 + raw-log rebuilt full overflow'},indent=2))
    print(json.dumps({'scenes':len(tokens),'rebuilt_overflow':len(rebuilt_rows),'cache':str(cache),'targets':str(targets)}))


if __name__=='__main__':main()
