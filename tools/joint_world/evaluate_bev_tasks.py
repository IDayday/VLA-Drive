"""Evaluate trained BEV bridge task heads separately from deployed planning metrics."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch import nn

from starVLA.model.modules.structured_world.contracts import WorldTargets
from tools.joint_world.bev_cache import BEVFeatureStore
from tools.joint_world.planner_runtime import CachedCurrentPlanner, file_sha256
from tools.joint_world.train_corpus import Corpus
from tools.joint_world.train_bev_tasks import evaluate
from tools.structured_world_v1p1.budget import start, record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','cache','targets','data-root','bev-index','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not saved['identity']['bev_enabled']:raise ValueError('Checkpoint has no BEV task heads')
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'checkpoint_sha256':file_sha256(a.checkpoint),'training_step':saved['step'],
              'task_supervision':saved['identity']['bev_task_supervision'],
              'scope':'Oracle current occupied-cell displacement and current occupancy; not actor detection or planning PDMS.'}
    start(a.ledger,a.run_id,0,identity)
    try:
        ds=Corpus(a.cache,a.targets,a.data_root);store=BEVFeatureStore(a.bev_index)
        source=json.loads((Path(saved['identity']['arguments']['cache'])/'manifest.json').read_text())['identity']
        for key in ['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract']:
            if ds.manifest['identity'][key]!=source[key]:raise ValueError('Current feature identity mismatch: '+key)
        if not {r['token'] for r in ds.records}<=set(store.records):raise ValueError('Missing current BEV')
        limit=len(ds) if a.limit is None else a.limit
        if not 1<=limit<=len(ds):raise ValueError('Invalid explicit scene limit')
        identity.update(cache_identity=ds.manifest['identity_sha256'],label_fingerprint=ds.target_fingerprint,
                        bev_index_sha256=store.identity_sha256,scenes=limit)
        model=CachedCurrentPlanner(ds[0]['cache']['context'].shape[-1],saved['identity']['graph_config'],True).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True)
        modules=nn.ModuleDict({'encoder':model.bev_encoder,'fusion':model.bev_fusion,'interaction':model.interaction_head})
        def samples():
            for i in range(limit):
                if not record(a.ledger,a.run_id,0):raise RuntimeError('GPU budget reached')
                original=ds[i];bev=store[original['cache']['token']]
                target=WorldTargets(**{k:v.cuda() if torch.is_tensor(v) else v for k,v in vars(original['target']).items()})
                yield dict(original,target=target,bev={k:v.cuda() if torch.is_tensor(v) else v for k,v in bev.items()})
        evaluate(modules,samples(),out,'metrics')
        identity['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        (out/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
        record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
