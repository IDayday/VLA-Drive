"""Bounded integrity read of immutable compact features or completed labels.

Warms the host's normal filesystem page cache without copying a dataset,
changing the reader, touching tensors/RNG, or changing mount settings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--features',type=Path);source.add_argument('--labels',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();resources=check_host();start=time.monotonic();cpu=time.process_time()
    if a.output.exists():raise FileExistsError('Preserve previous pre-read receipt')
    root=a.features or a.labels;meta=read(root/'identity.json')
    if not 1<=a.threads<=(8 if a.labels else 2):raise ValueError('Bounded IO required')
    if (root/'COMPLETE.json').exists():
        complete=read(root/'COMPLETE.json')
        if complete['identity']!=meta['identity']:raise ValueError('Foreign completion receipt')
    elif a.labels:
        # Pre-read may overlap the serial global verifier after all producers
        # exit. This tool never writes COMPLETE or authorizes training itself.
        shards=[read(p) for p in sorted(root.glob('COMPLETE_rank*.json'))]
        if not shards or any(s['identity']!=meta['identity'] or s['errors'] or s['scenes']!=s['expected'] for s in shards) or sum(s['scenes'] for s in shards)!=len(meta['index']):
            raise ValueError('All immutable label producer shards must be complete')
    else:raise ValueError('Complete immutable feature source required')
    receipts=sorted((root/'records').glob('*.json'))
    if a.labels and {p.stem for p in receipts}!={r['token'] for r in meta['index']}:
        raise ValueError('Label population differs from its registered identity')
    def check(receipt):
        if time.monotonic()-start>a.max_seconds:return None
        row=read(receipt)
        path=root/('scenes' if a.labels else 'chunks')/(receipt.stem+('.npz' if a.labels else '.safetensors'))
        if row['identity']!=meta['identity'] or digest(path)!=row['sha256']:raise ValueError('Immutable cache bytes changed')
        return path.stat().st_size
    with ThreadPoolExecutor(a.threads) as pool:values=list(pool.map(check,receipts))
    done=sum(x is not None for x in values);size=sum(x or 0 for x in values);elapsed=time.monotonic()-start
    atomic_json(a.output,{'complete':done==len(receipts),'chunks':done,'expected_chunks':len(receipts),
        'bytes_verified':size,'seconds':elapsed,'bytes_per_second':size/elapsed,'resources':resources,
        'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        ('label_identity' if a.labels else 'feature_identity'):meta['identity'],
        'CPU_process_seconds':time.process_time()-cpu,
        'new_persistent_data_bytes':0,'training_code_data_order_RNG_or_mount_changed':False,
        'scope':'Normal OS page-cache priming; a performance benefit must be measured, not assumed'})


if __name__=='__main__':main()
