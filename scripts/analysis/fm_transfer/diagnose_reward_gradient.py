"""Source vs canonical reward gradients on the SAME saved real B batches.

Calls the existing full source loss unchanged; PG = total - unmasked anchor.
No optimizer step, sampling, scoring, or counterfactual trained policy claim.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from scripts.analysis.fm_transfer.diagnose_saved_batch import TransferDiagnostic
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.reward_readouts import readouts
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.simwam_recipe.checkpoint import device_tree
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def cosine(a,b):
    denom=a.norm()*b.norm()
    return float(a@b/denom) if denom>0 else None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require_dispatch();configure_numerics();torch.set_num_threads(1);start=time.monotonic()
    if a.output.exists():raise FileExistsError(a.output)
    identity=json.loads((a.run/'identity.json').read_text())
    variant=identity['recipe']['supervised_transfer']['variant']
    if variant!='B_SOURCE_ACTION_DIT':raise ValueError('Bounded diagnostic registered for B only')
    root=a.run.parent.parent; runtime=TransferDiagnostic()
    args=SimpleNamespace(gradient_strata_registration=None,
        hard_manifest=str(root/'hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json'),
        sft_screen=str(root/'screens/B_SOURCE_ACTION_DIT_SFT_update100000'),collection=1250)
    runtime.prepare(args,identity);actor=runtime.make_actor(runtime.cfg)
    parameters=[p for p in actor.parameters() if p.requires_grad];count=sum(p.numel() for p in parameters)
    results=[]
    for update in (5000,20000):
        collection=update//4;args.collection=collection;runtime.prepare(args,identity)
        cp=a.run/f'checkpoints/update{update:06}_inner0'
        seal=json.loads((cp/'COMPLETE').read_text());files=json.loads((cp/'files.json').read_text())
        if (file_sha(cp/'files.json')!=seal['files_sha256'] or file_sha(cp/'optimizer.pt')!=files['optimizer.pt'] or
                seal['updates']!=update or seal['phase'] not in ('READY','COMPLETE')):
            raise ValueError('Incomplete checkpoint')
        saved=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
        if {int(v['step']) for v in saved['optimizer']['state'].values()}!={update}:raise ValueError('Wrong optimizer')
        restore_adapters(actor,saved['adapters'])
        rows=[];paths=[];source=[];canonical=[]
        for rank in range(8):
            path=a.run/'behavior'/f'collection{collection:06}'/f'rank{rank}.pt';paths.append(path)
            row=torch.load(path,map_location='cpu',weights_only=False);runtime.validate_batch(rank,row)
            if row['behavior_version']['run']!=identity['sha256']:raise ValueError('Behavior identity mismatch')
            values=[readouts(r) for r in row['records']]
            source.append(torch.tensor([v['train_reward_source'] for v in values]))
            canonical.append(torch.tensor([v['canonical_PDMS'] for v in values]));rows.append(row)
        s=advantages(torch.stack(source));c=advantages(torch.stack(canonical))
        gradients={k:torch.zeros(count) for k in ('source_total','canonical_total','anchor')}
        for rank,cpu in enumerate(rows):
            torch.testing.assert_close(cpu['advantages'],s['advantages'].reshape(8,8)[rank],rtol=0,atol=0)
            row=device_tree(cpu,actor.device)
            for kind in gradients:
                changed=dict(row)
                if kind=='canonical_total':
                    changed['advantages']=c['advantages'].reshape(8,8)[rank].to(actor.device)
                    changed['adv_mask']=c['adv_mask'].reshape(8,8)[rank].to(actor.device)
                elif kind=='anchor':
                    changed['advantages']=torch.zeros_like(row['advantages'])
                    changed['adv_mask']=torch.zeros_like(row['adv_mask'])
                loss,_=actor(changed)
                grads=torch.autograd.grad(loss/8,parameters)
                gradients[kind]+=torch.cat([g.detach().float().flatten().cpu() for g in grads])
        anchor=gradients['anchor'];source_pg=gradients['source_total']-anchor;canonical_pg=gradients['canonical_total']-anchor
        results.append(dict(update=update,collection=collection,scenes=8,candidates=64,
            checkpoint_files_sha256=seal['files_sha256'],behavior_payload_sha256=[file_sha(p) for p in paths],
            source_vs_canonical_PG_cosine=cosine(source_pg,canonical_pg),
            source_vs_canonical_total_cosine=cosine(gradients['source_total'],gradients['canonical_total']),
            source_PG_norm=float(source_pg.norm()),canonical_PG_norm=float(canonical_pg.norm()),
            beta50_anchor_norm=float(anchor.norm()),source_PG_anchor_cosine=cosine(source_pg,anchor),
            canonical_PG_anchor_cosine=cosine(canonical_pg,anchor),
            source_PG_total_cosine=cosine(source_pg,gradients['source_total']),
            canonical_PG_total_cosine=cosine(canonical_pg,gradients['canonical_total'])))
        for name,param in actor.named_parameters():
            if param.grad is not None:raise ValueError('Autograd left persistent gradients')
            if param.requires_grad and not torch.equal(param.detach().cpu(),saved['adapters'][name]):
                raise ValueError('Read-only diagnostic modified LoRA')
        del saved,gradients
    result=dict(status='COMPLETE_SAME_CHAIN_REWARD_GRADIENT_COUNTERFACTUAL',run_identity=identity['sha256'],
        source_sha256=file_sha(__file__),milestones=results,parameters=count,wall_seconds=time.monotonic()-start,
        new_candidates=0,new_scoring=0,optimizer_updates=0,adapters_unchanged=True,
        limits=['One saved global batch per fixed milestone; not a population gradient estimate.',
                'Checkpoint after inner4 compared against its preceding behavior, so clipping is real.',
                'These are loss gradients, not Adam displacements or a separately trained policy.',
                'Native source loss including unclipped source effective gradient norm is unchanged.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,result)
    print(json.dumps(dict(status=result['status'],milestones=results)))


if __name__=='__main__':main()
