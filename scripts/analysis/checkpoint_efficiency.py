"""Read actual native gradients/Adam states for a checkpointing-only comparison.

This observer neither edits checkpoints nor relaxes the historical chunk gate.
Different checkpointing profiles intentionally cannot exact-resume each other.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.analysis.frozen_training_summary import initial_bank_equal, summarize
from scripts.flow_grpo.compare_boundaries import flatten
from starVLA.rl.flow_grpo.config import config_hash
from starVLA.rl.flow_grpo.loading import file_sha


def tensor_stats(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'equal': False, 'reason': 'shape/dtype differs'}
    finite = True
    nonidentical = 0
    max_abs = a2 = b2 = d2 = dot = 0.
    for x, y in zip(a.detach().flatten().split(1_000_000), b.detach().flatten().split(1_000_000)):
        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        nonidentical += int((x != y).sum())
        xd, yd = x.double(), y.double()
        delta = xd-yd
        max_abs = max(max_abs, float(delta.abs().max()) if delta.numel() else 0.)
        a2 += float(xd.square().sum()); b2 += float(yd.square().sum())
        d2 += float(delta.square().sum()); dot += float((xd*yd).sum())
    near_zero = a2 <= a.numel()*1e-24
    return {'equal': finite and nonidentical == 0, 'finite': finite, 'numel': a.numel(),
            'dtype': str(a.dtype), 'nonidentical': nonidentical, 'max_abs': max_abs,
            'reference_norm': math.sqrt(a2), 'difference_norm': math.sqrt(d2),
            'relative_l2': None if near_zero else math.sqrt(d2/a2),
            'cosine': dot/math.sqrt(a2*b2) if not near_zero and b2 else None,
            'norm_ratio': None if near_zero else math.sqrt(b2/a2), 'near_zero': near_zero}


def compare_tree(a, b):
    left, right = dict(flatten(a)), dict(flatten(b))
    result = {}
    for key in sorted(left.keys() | right.keys()):
        x, y = left.get(key), right.get(key)
        if key not in left or key not in right:
            result[key] = {'equal': False, 'reason': 'missing key'}
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            # Exact finite fast path avoids double precision scans of large
            # identical Adam partitions. Unequal tensors retain full statistics.
            if x.dtype == y.dtype and torch.equal(x, y) and torch.isfinite(x).all():
                result[key] = {'equal': True, 'finite': True, 'numel': x.numel(),
                               'dtype': str(x.dtype), 'nonidentical': 0, 'max_abs': 0.}
            else:
                result[key] = tensor_stats(x, y)
        else:
            result[key] = {'equal': type(x) is type(y) and x == y}
    return result


def compare(left, right, difference='activation_checkpointing'):
    cfg = [json.loads((p/'rl_config.json').read_text()) for p in (left, right)]
    expected = {'activation_checkpointing': [True, False], 'overlap_reward_reference': [False, True]}
    if difference not in expected or [c['runtime'].get(difference, False) for c in cfg] != expected[difference]:
        raise ValueError('incorrect declared runtime comparison')
    cfg[1]['runtime'][difference] = cfg[0]['runtime'][difference]
    if config_hash(cfg[0]) != config_hash(cfg[1]):
        raise ValueError('configuration differs beyond declared runtime flag and allowed output/budget settings')
    world = len(json.loads((left/'training.jsonl').read_text().splitlines()[0])['ranks'])
    bank = initial_bank_equal(left, right, world)
    bank.pop('advantages_excluded')
    for rank in range(world):
        a,b=[torch.load(p/f'rollout_rank{rank}_v0.pt',map_location='cpu',weights_only=False) for p in (left,right)]
        if any(not torch.equal(x.advantages,y.advantages) for x,y in zip(a,b)):
            raise ValueError('initial advantages differ')
    bank['advantages_equal'] = True
    report = {'status':'PASS', 'scope':difference+' only, two native BF16/ZeRO2 updates, zero tolerance; not chunk acceptance',
              'world_size': world,
              'initial_bank':bank, 'gradients':{}, 'boundaries':{}, 'runs':{}}
    for update in (1,2):
        folders=[p/f'optimizer_gradients/update_{update:06d}/rank_0' for p in (left,right)]
        manifests=[json.loads((p/'manifest.json').read_text()) for p in folders]
        entries=[{r['name']:r for r in m['parameters']} for m in manifests]
        if entries[0].keys()!=entries[1].keys():raise ValueError('gradient coverage changed')
        rows={}
        for name in entries[0]:
            a,b=[torch.load(p/e[name]['file'],map_location='cpu',weights_only=True) for p,e in zip(folders,entries)]
            rows[name]=compare_tree({'gradient':a},{'gradient':b})['/gradient']
        report['gradients'][str(update)]=rows
        boundaries=[p/f'checkpoints/update_{update:06d}' for p in (left,right)]
        if not all((p/'COMPLETE').is_file() for p in boundaries):raise ValueError('incomplete boundary')
        files=[{str(x.relative_to(p)) for x in p.glob('*/*.pt')} for p in boundaries]
        if files[0]!=files[1] or len(files[0])!=world+1:raise ValueError('model/optimizer inventory mismatch')
        states={}
        for name in sorted(files[0]):
            a,b=[torch.load(p/name,map_location='cpu',weights_only=False,mmap=True) for p in boundaries]
            states[name]=compare_tree(a,b)
            del a,b
        report['boundaries'][str(update)]=states
        if not all(v['equal'] for v in rows.values()) or not all(v['equal'] for f in states.values() for v in f.values()):
            report['status']='FAIL'
    for label,path in [('baseline',left),('candidate',right)]:
        run=summarize(path)
        timings=[]
        for row in run['rows']:
            phases=row['phase_wall_seconds_max']
            timings.append({'update':row['update'],'update_seconds':row['max_update_seconds'],
                'observer_nested_seconds':phases.get('optimizer_gradient_observer_nested',0.),
                'phases_max':phases,'peak_gpu_gib':max(row['peak_gpu_gib_by_rank'])})
        report['runs'][label]={'path':str(path),'fixed_behavior_reuse':run['fixed_behavior_reuse'],
            'trainable_numel':run['trainable_numel'],'immutable':run['immutable'],'timings':timings,
            'training_sha256':run['training_sha256']}
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--on',required=True);parser.add_argument('--off',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--difference',choices=['activation_checkpointing','overlap_reward_reference'],default='activation_checkpointing')
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    result=compare(Path(args.on),Path(args.off),args.difference);result['observer_sha256']=file_sha(__file__)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':result['status'],'runs':result['runs']}))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
