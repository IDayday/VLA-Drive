"""Compare actual ZeRO pre-clip gradients, retaining every failed tensor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.comparison import compare_named
from starVLA.rl.flow_grpo.transactions import atomic_json


def load(root, update):
    path = Path(root)/f'optimizer_gradients/update_{update:06d}/rank_0'
    entries = json.loads((path/'manifest.json').read_text())['parameters']
    if len(entries) != 359 or any(not r['present'] or not r['finite'] or r['dtype'] != 'torch.float32' for r in entries):
        raise ValueError('incomplete actual optimizer gradient evidence')
    def read(r): return r['name'], torch.load(path/r['file'], weights_only=True)
    with ThreadPoolExecutor(max_workers=4) as pool: return dict(pool.map(read, entries))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    summary={'status':'PASS','scope':'real safe_get_full_grad before clip/Adam, both inner epochs','updates':{}}
    for u in (1,2):
        left,right=load(a.left,u),load(a.right,u)
        r=compare_named(left,right);atomic_json(out/f'update{u}.json',r)
        summary['updates'][u]={'status':r['status'],'modules':r['modules']}
        if r['status']!='PASS':summary['status']='FAIL'
        atomic_json(out/'summary.json',summary);print(json.dumps(summary['updates'][u]),flush=True)
        del left,right
    if summary['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
