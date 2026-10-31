"""Compare completed disposable native execution qualifications, never models.

Checks all ranks, effective updates, samples, losses, native model/Adam shards
and committed RNG/builder state. Source/config identities are allowed to differ
only in the explicitly named execution tuning record, not scientific fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch


def canonical_tree(value):
    if isinstance(value,torch.Tensor):
        raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value,np.ndarray):
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):canonical_tree(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical_tree(x)for x in value]
    if isinstance(value,np.generic):return value.item()
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(type(value))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root);paths=[root/n for n in('baseline','optimized')]
    cps=[Path(json.loads((q/'LATEST.json').read_text())['checkpoint'])for q in paths]
    identities=[json.loads((q/'identity.json').read_text())for q in cps]
    for v in identities:
        v['execution'].pop('qualification_tuning',None)
        v['execution'].pop('execution_tuning',None)
    if identities[0]!=identities[1]:raise ValueError('Non-execution identity changed')
    world=identities[0]['world'];torch.set_num_threads(4);rows=[];timings=[]
    all_metrics=[]
    for path in paths:
        ranks=[]
        for rank in range(world):
            values=[json.loads(l)for l in(path/f'metrics_rank{rank}.jsonl').read_text().splitlines()if l.strip()]
            start=values[0]['updates']
            if [v['updates']for v in values]!=list(range(start,start+len(values))):raise ValueError('Missing/duplicate updates')
            ranks.append(values)
        all_metrics.append(ranks)
        timings.append([max(v[i]['seconds']for v in ranks)for i in range(len(ranks[0]))])
    for rank in range(world):
        a,b=all_metrics[0][rank],all_metrics[1][rank]
        if len(a)!=len(b):raise ValueError('Different effective update budget')
        for x,y in zip(a,b):
            for key in('updates','cursor','phase','sample_ids','post_positions','losses','lr_used','lr'):
                if x[key]!=y[key]:raise ValueError(f'Rank{rank}/update{x["updates"]}/{key} mismatch')
        left=torch.load(cps[0]/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
        right=torch.load(cps[1]/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
        for key in('state','builder','rng'):
            if canonical_tree(left[key])!=canonical_tree(right[key]):raise ValueError(f'Rank{rank} committed {key} changed')
    inventories=[json.loads((q/'files.json').read_text())for q in cps]
    names=[n for n in inventories[0]if n.startswith('native/')and n.endswith('.pt')]
    for name in names:
        identical=inventories[0][name]==inventories[1][name]
        if not identical:
            hashes=[]
            for checkpoint in cps:
                payload=torch.load(checkpoint/name,map_location='cpu',weights_only=False)
                value=canonical_tree(payload)
                hashes.append(hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest())
                del payload,value
            if hashes[0]!=hashes[1]:raise ValueError(f'Native tensor/optimizer payload differs: {name}')
        rows.append(dict(file=name,serialized_bytes_identical=identical,full_native_state_equivalent=True))
    discard=min(2,len(timings[0])-1)
    medians=[statistics.median(v[discard:])for v in timings]
    result=dict(status='PASS',world=world,architecture_id=identities[0]['architecture_id'],source_files=identities[0]['source_files'],logical_global_scene_batch=32,effective_updates=len(timings[0]),
        actual_optimizer_updates_and_native_states_compared=True,sample_cursor_loss_RNG_builder_exact=True,
        native_payloads=rows,raw_max_rank_step_wall_seconds=dict(zip(('baseline','optimized'),timings)),
        discard_first_updates=discard,median_max_rank_wall_seconds=dict(zip(('baseline','optimized'),medians)),
        observed_end_to_end_speedup=medians[0]/medians[1],
        starting_update=all_metrics[0][0][0]['updates']-1,
        timed_updates_after_discard=len(timings[0])-discard,
        caveat=f'Finite {world}-rank qualification on shared GPUs, global32. Starting update is recorded separately; initialization provenance must be read from the qualification invocation. Very few timed updates and changing cohosts do not establish production speedup or long-run ETA.')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(out.read_text())


if __name__=='__main__':main()
