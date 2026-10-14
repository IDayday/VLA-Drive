"""One train-only current-representation mean, never a GT or dev normalizer."""
import argparse
from pathlib import Path
import sys
import time
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,digest,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();meta=read(a.features/'identity.json')
    if meta['split']!='train' or not (a.features/'COMPLETE.json').exists():raise ValueError('Complete train-only feature source required')
    if a.output.exists():raise FileExistsError('Immutable mean')
    sums={};count=0;start=time.monotonic();torch.set_num_threads(2)
    for receipt in sorted((a.features/'records').glob('*.json')):
        r=read(receipt);path=a.features/'chunks'/(receipt.stem+'.safetensors')
        if digest(path)!=r['sha256']:raise ValueError('Changed current representation')
        data=load_file(str(path))
        for key in ('W','H_A'):sums[key]=sums.get(key,0)+data[key].double().sum(0)
        count+=len(r['tokens'])
    if count!=len(meta['index']):raise ValueError('Mean population incomplete')
    payload={k:(v/count).float() for k,v in sums.items()}|{'anchor_identity':meta['anchor_identity'],'split':'train','scenes':count,'features':meta['identity']}
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');torch.save(payload,temp);temp.replace(a.output)
    atomic_json(a.output.with_suffix('.json'),{'scenes':count,'CPU_seconds':time.monotonic()-start,'sha256':digest(a.output),'source_features':meta['identity']})


if __name__=='__main__':main()
