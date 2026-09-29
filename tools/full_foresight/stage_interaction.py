"""Copy immutable MAE labels onto local storage, with byte integrity checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError('Bounded I/O workers required')
    source,out=Path(a.source),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ident=json.loads((source/'identity.json').read_text());index=json.loads((source/'index.json').read_text())
    done=json.loads((source/'COMPLETE.json').read_text())
    if ident['schema']!='foresight_interaction_latent_v1' or identity_hash(index)!=ident['index_hash'] or done['scenes']!=len(index) or done['failed']:
        raise ValueError('Incomplete source MAE export')
    if shutil.disk_usage(out).free<100*2**30+len(index)*20000:raise RuntimeError('Keep100GiB local reserve')
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise ValueError('Foreign local targets')
    atomic_json(out/'identity.json',ident);atomic_json(out/'index.json',index);(out/'targets').mkdir(exist_ok=True)
    def copy(row):
        src=source/'targets'/(row['token']+'.pt');dest=out/'targets'/src.name;digest=file_sha256(src)
        if not dest.exists():
            temp=dest.with_suffix(f'.{os.getpid()}.tmp');shutil.copy2(src,temp)
            if file_sha256(temp)!=digest:raise ValueError('MAE target copy corruption')
            temp.replace(dest)
        elif file_sha256(dest)!=digest:raise ValueError('Existing local MAE target changed')
        return row['token'],digest
    started=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:hashes=dict(pool.map(copy,index))
    atomic_json(out/'file_hashes.json',hashes)
    atomic_json(out/'local_replica.json',{'host':socket.gethostname(),'identity':ident['identity'],'scenes':len(index),
        'hash_manifest_sha256':file_sha256(out/'file_hashes.json'),'seconds':time.time()-started})
    atomic_json(out/'COMPLETE.json',done)

if __name__=='__main__':main()
