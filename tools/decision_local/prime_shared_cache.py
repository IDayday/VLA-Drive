"""Bounded sequential integrity read of immutable compact training chunks.

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
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();resources=check_host();start=time.monotonic()
    if a.output.exists():raise FileExistsError('Preserve previous pre-read receipt')
    meta=read(a.features/'identity.json');complete=read(a.features/'COMPLETE.json')
    if complete['identity']!=meta['identity'] or not 1<=a.threads<=2:raise ValueError('Complete immutable source and bounded IO required')
    receipts=sorted((a.features/'records').glob('*.json'))
    def check(receipt):
        if time.monotonic()-start>a.max_seconds:return None
        row=read(receipt);path=a.features/'chunks'/(receipt.stem+'.safetensors')
        if row['identity']!=meta['identity'] or digest(path)!=row['sha256']:raise ValueError('Immutable feature bytes changed')
        return path.stat().st_size
    with ThreadPoolExecutor(a.threads) as pool:values=list(pool.map(check,receipts))
    done=sum(x is not None for x in values);size=sum(x or 0 for x in values);elapsed=time.monotonic()-start
    atomic_json(a.output,{'complete':done==len(receipts),'chunks':done,'expected_chunks':len(receipts),
        'bytes_verified':size,'seconds':elapsed,'bytes_per_second':size/elapsed,'resources':resources,
        'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'feature_identity':meta['identity'],
        'new_persistent_data_bytes':0,'training_code_data_order_RNG_or_mount_changed':False,
        'scope':'Normal OS page-cache priming; a performance benefit must be measured, not assumed'})


if __name__=='__main__':main()
