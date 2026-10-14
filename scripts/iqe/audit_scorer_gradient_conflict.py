#!/usr/bin/env python3
"""Read-only pointwise-gradient audit on a fixed natural D_S sample."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from iqe.contracts import require
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.robust_scorer.config import load
from iqe.robust_scorer.runner import create
from iqe.robust_scorer.objectives import terms
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.scorer_training import fetch_context
from iqe.specialist_study.training import runtime
from iqe.specialist_study.resource_guard import check


def comparison(first, second, selected):
    a = torch.cat([first[name].double().flatten() for name in selected])
    b = torch.cat([second[name].double().flatten() for name in selected])
    na, nb = a.norm().item(), b.norm().item()
    return dict(first_norm=na, second_norm=nb,
                cosine=(a.dot(b).item()/(na*nb)) if na*nb > 0 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--scenes', type=int, default=128)
    args = p.parse_args()
    require(1 <= args.scenes <= 128, 'maximum128 natural D_S scenes')
    config = load(args.config)
    check(config, 'benchmark-system')
    device = runtime(args.device)
    source = config.source()
    meta, bank = load_bank(source, 'deploy', 'scorer_fit')
    all_scenes = scenes_by_role(source)['scorer_fit']
    indices = sorted(range(len(all_scenes)), key=lambda i: digest([20261014, all_scenes[i].scene_id]))[:args.scenes]
    scenes = [all_scenes[i] for i in indices]
    folder = config.root/'qualification/pointwise_gradient_conflict'
    registration = dict(role='scorer_fit', scene_ids=[s.scene_id for s in scenes],
                        bank_hash=meta['hash'], selection='seeded hash before reading per-scene outcome',
                        cell='R10', endpoints=['initial', 'stage_val_selected', 'terminal_6000'],
                        optimizer_steps=0, production_RNG_modified=False,
                        pointwise_only=True, ranking_not_in_this_diagnostic=True)
    atomic_json(folder/'REGISTRATION.json', registration, immutable=True)
    if (folder/'REPORT.json').exists():
        old=read_json(folder/'REPORT.json')
        require(old['registration_hash']==file_hash(folder/'REGISTRATION.json'), 'audit inputs changed')
        print(old); return
    cache = ReusedContext(source, capacity=args.scenes)
    rows = []
    for endpoint in registration['endpoints']:
        model, init = create(config, 'R10')
        require(init == read_json(config.root/'scorers/R10/SCORER_INITIALIZATION.json'), 'initializer changed')
        checkpoint_hash = None
        if endpoint == 'stage_val_selected':
            checkpoint=config.root/'scorers/R10/best.pt'
            require(file_hash(checkpoint)==read_json(config.root/'scorers/R10/result.json')['best_checkpoint_hash'], 'best changed')
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            checkpoint_hash=file_hash(checkpoint)
        elif endpoint == 'terminal_6000':
            checkpoint=config.root/'scorers/R10/step_006000.pt'
            receipt=read_json(str(checkpoint)+'.COMPLETE.json')
            require(file_hash(checkpoint)==receipt['checksum'], 'terminal checkpoint changed')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            require(state['optimizer_step']==6000, 'not the terminal checkpoint')
            model.load_state_dict(state['module'], strict=True)
            checkpoint_hash=receipt['checksum']
        model.to(device).train()
        before=module_hash(model)
        shared={n:p for n,p in model.named_parameters() if not n.startswith(('value_head.','component_head.','nonzero_head.'))}
        gradients={task:{n:torch.zeros_like(v,device='cpu') for n,v in shared.items()} for task in ('value','nonzero','components')}
        losses={task:0. for task in gradients}
        for start in range(0,len(scenes),16):
            chunk=scenes[start:start+16];ix=indices[start:start+16]
            context=fetch_context(source,chunk,device,cache)
            batch={key:torch.from_numpy(bank[key][ix].copy()).to(device) for key in
                   ('trajectories','scores','valid','components','component_valid')}
            batch.update(context=context,natural=torch.ones(len(chunk),device=device,dtype=torch.bool),
                         comparison=torch.zeros(len(chunk),device=device,dtype=torch.bool))
            output=model(context,batch['trajectories'])
            objective=terms(output,batch,'zero_hurdle')
            for task in gradients:
                value=objective[task].numerator/len(scenes)
                losses[task]+=float(value.detach())
                grad=torch.autograd.grad(value,tuple(shared.values()),retain_graph=task!='components',allow_unused=False)
                for name,g in zip(shared,grad):
                    require(torch.isfinite(g).all(), 'nonfinite diagnostic gradient')
                    gradients[task][name]+=g.detach().cpu()
        groups={'all_shared':list(shared)}
        for prefix in ('world_projection','action_projection','ego_projection','trajectory_encoder','decoder.layers.0','decoder.layers.1'):
            groups[prefix]=[n for n in shared if n.startswith(prefix+'.')]
        result={name:{'event_vs_value':comparison(gradients['nonzero'],gradients['value'],names),
                      'event_vs_components':comparison(gradients['nonzero'],gradients['components'],names)}
                for name,names in groups.items() if names}
        require(module_hash(model)==before, 'gradient audit changed weights or buffers')
        rows.append(dict(endpoint=endpoint,checkpoint_hash=checkpoint_hash,state_hash=before,
                         pointwise_losses=losses,gradient_comparisons=result,parameters_unchanged=True))
        del model,gradients
    report=dict(registration_hash=file_hash(folder/'REGISTRATION.json'),rows=rows,optimizer_steps=0,
                inference_or_deployment_policy_modified=False,source_hash=file_hash(__file__),
                limitation='Local gradient alignment on a fixed natural D_S subset; not causal proof or heldout success.')
    atomic_json(folder/'REPORT.json',report,immutable=True)
    print(report)


if __name__=='__main__': main()
