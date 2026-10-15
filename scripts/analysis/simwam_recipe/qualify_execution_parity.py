"""Compare execution repairs with an already qualified unchanged normal path."""
import argparse
import json
from pathlib import Path
import torch
from scripts.analysis.simwam_recipe.qualify_recovery import equal
from starVLA.rl.simwam_recipe.lineage import scientific_identity
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    roots=[Path(a.reference),Path(a.candidate)]
    ids=[json.loads((r/'identity.json').read_text()) for r in roots]
    if scientific_identity(ids[0])!=scientific_identity(ids[1]):raise ValueError('Scientific configuration changed')
    ck='checkpoints/update000008_inner0'
    def load(root,rel):return torch.load(root/rel,map_location='cpu',weights_only=False)
    equal(*[load(r,ck+'/optimizer.pt') for r in roots])
    for rank in range(ids[0]['world']):
        own=[load(r,ck+f'/rank{rank}.pt') for r in roots]
        equal(own[0]['rng'],own[1]['rng'],f'rank{rank}.RNG')
        for k,v in own[0]['state'].items():
            if not k.startswith('seconds_') and k!='phase':equal(v,own[1]['state'][k],k)
        for collection in (1,2):
            rr=[load(r,f'behavior/collection{collection:06}/rank{rank}.pt') for r in roots]
            for k in ('chain','timesteps','deltas','stoch_idx','old_logp','advantages','adv_mask','global_std','rewards','old_probability','token'):
                equal(rr[0][k],rr[1][k],f'rank{rank}.{k}')
        metrics=[[json.loads(l) for l in (r/f'metrics_rank{rank}.jsonl').read_text().splitlines()] for r in roots]
        for i in range(8):
            for k in ('loss','gradient_norm','ratio_max_abs_from_one','metrics','lr'):
                equal(metrics[0][i][k],metrics[1][i][k],f'rank{rank}.update{i+1}.{k}')
    result=dict(status='PASS',world=ids[0]['world'],scientific_identity=scientific_identity(ids[0])['sha256'],
        source_identities=[x['sha256'] for x in ids],reference=str(roots[0]),candidate=str(roots[1]),
        normal_chain_probability_advantages_exact=True,optimizer_adapters_scheduler_RNG_exact=True,
        loss_gradient_norm_ratio_LR_exact=True,metadata_and_wall_time_excluded=True,
        checkpoint_seals=[file_sha(r/ck/'files.json') for r in roots])
    atomic_json(Path(a.output),result);print(json.dumps(result))


if __name__=='__main__':main()
