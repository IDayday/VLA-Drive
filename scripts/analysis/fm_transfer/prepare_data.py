"""Seal original GT labels for full TRAIN; reuse corrected observation cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import torch
from starVLA.rl.constrained_flow_noise.loading import ReplayLabels,DEFAULT_CONFIG,DEFAULT_SOURCE
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.simwam_recipe.contracts import POPULATION,RAW_CONDITION,CONDITION_SHA,file_sha
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=16);a=parser.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    pop=json.loads(Path(POPULATION).read_text());tokens=pop['train_tokens']
    cache=json.loads((Path(RAW_CONDITION)/'COMPLETE.json').read_text())
    if cache['count']!=len(tokens) or cache['identity_sha256']!=CONDITION_SHA:
        raise ValueError('Complete deployment-aligned TRAIN conditions required')
    _,sft=resolve_config(DEFAULT_CONFIG,overrides=['checkpoint_contract.source_config='+DEFAULT_SOURCE])
    labels=ReplayLabels(sft)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(labels.get,tokens))
    target=torch.stack(rows)
    if target.shape!=(len(tokens),8,4) or not torch.isfinite(target).all():raise ValueError('Invalid original action labels')
    atomic_torch(out/'labels.pt',target)
    manifest=dict(schema='fm_transfer_original_supervised_TRAIN_v1',tokens=tokens,
        population_identity=pop['identity_sha256'],population_manifest_sha256=file_sha(POPULATION),
        labels_sha256=file_sha(out/'labels.pt'),label_source='unchanged ReplayLabels and original action preprocessing',
        condition_identity=CONDITION_SHA,actor_features='raw observation condition only; labels never actor input')
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'COMPLETE',dict(status='COMPLETE',manifest_sha256=file_sha(out/'manifest.json'),
        scenes=len(tokens),wall_seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE',scenes=len(tokens),wall_seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
