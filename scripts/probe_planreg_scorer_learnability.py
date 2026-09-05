#!/usr/bin/env python3
"""Opt-in train-only, frozen-upstream scorer fit diagnostic. Not a formal run."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from omegaconf import OmegaConf
from planreg_audit_runtime import load_formal_training_agent, collate_samples, to_device_non_paths, sha256_file
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.EpisodeDrive.scorer_replay import FrozenScorerCacheContract, exact_cached_scorer_forward, group_identity
from navsim.agents.EpisodeDrive.score_module.scorer import aggregate_drivor_pdm_score
from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import get_sub_score


def main():
    parser = argparse.ArgumentParser()
    for name in ('config','checkpoint','cache-root','metric-cache','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--steps',type=int, default=150)
    parser.add_argument('--scenes',type=int, default=16)
    parser.add_argument('--seed',type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.steps <= 150:
        raise ValueError('Bounded probe permits 1..150 updates, not a new formal schedule')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device=torch.device('cuda:0')
    cfg, agent, provenance = load_formal_training_agent(args.config, args.checkpoint, device=device)
    agent.remove_training_only_world_model()
    agent.eval().requires_grad_(False)
    head=agent.action_head
    OmegaConf.update(head._config, 'return_memory_fields', True, force_add=True)
    scorer_modules={name:getattr(head,name) for name in ('pos_embed','scorer_attention','scorer')}
    upstream={'backbone':agent.backbone, **{name:mod for name,mod in head.named_children() if name not in scorer_modules}}
    contract=FrozenScorerCacheContract(upstream,dict(head.named_parameters(recurse=False)))
    metric=MetricCacheLoader(args.metric_cache)
    scenes,egos,proposals,truth,identities,tokens=[],[],[],[],[],[]
    dependency=[]
    for log in sorted(set(cfg.train_logs)):
        directory=args.cache_root/log
        if not directory.is_dir():
            continue
        for target_path in sorted(directory.glob('*/trajectory_target_planreg_wm_v1.gz')):
            feature_path=target_path.parent/'internvl_feature.gz'
            if not feature_path.is_file():
                continue
            features=load_feature_target_from_pickle(feature_path)
            targets=load_feature_target_from_pickle(target_path)
            token=targets['token']
            if token not in metric.metric_cache_paths:
                continue
            batch,_=collate_samples([(features,targets)])
            with torch.inference_mode():
                pred=agent(to_device_non_paths(batch,device))
            coordinates=pred['proposals'][0].float().cpu().numpy()
            path=Path(metric.metric_cache_paths[token])
            labels=np.asarray(get_sub_score(str(path),coordinates,True)[0],dtype=np.float32)
            if len(tokens)<2:
                alone=np.asarray(get_sub_score(str(path),coordinates[:1],True)[0])
                dependency.append({'token':token,'singleton_vs_full_max_abs_diff':float(abs(alone[0]-labels[0]).max())})
            evaluator_sha=sha256_file(ROOT/'navsim/agents/EpisodeDrive/score_module/compute_navsim_score.py')
            identities.append(group_identity(token,coordinates,evaluator_sha,provenance['checkpoint_sha256'],sha256_file(path)))
            scenes.append(pred['language_feature'].detach().cpu())
            egos.append(pred['ego_feature'].detach().cpu())
            proposals.append(pred['proposals'].detach().cpu())
            truth.append(torch.from_numpy(labels)[None])
            tokens.append(token)
            break
        if len(tokens)>=args.scenes:
            break
    if len(tokens)!=args.scenes:
        raise RuntimeError('Insufficient train-only scenes')
    # Persistent cache is legal only inside this exact frozen-upstream probe.
    arrays={'scene':torch.cat(scenes).clone(), 'ego':torch.cat(egos).clone(),
            'proposals':torch.cat(proposals).clone(), 'labels':torch.cat(truth)}
    torch.save({'tokens':tokens, **arrays},args.output/'frozen_train_bank.pt')
    bank={k:v.to(device).clone() for k,v in arrays.items()}
    for module in scorer_modules.values():
        module.requires_grad_(True)
    params=[p for m in scorer_modules.values() for p in m.parameters()]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    def evaluate(train=False):
        contract.validate()
        for module in scorer_modules.values():
            module.train(train)
        logits=exact_cached_scorer_forward(head,bank['scene'],bank['ego'],bank['proposals'])
        # Exact online evaluator supplies FP64 labels. Retain that boundary:
        # the unchanged upstream binary-label mapping mutates local cast views.
        _components, loss, *_ = agent.loss.score_loss(logits,None,None,None,bank['labels'].double(),None,None,None,None)
        log_score=aggregate_drivor_pdm_score(logits,head._config)
        selected=bank['labels'][torch.arange(len(tokens),device=device),log_score.argmax(-1),-1].mean()
        return loss, selected
    with torch.no_grad():
        initial,selected=evaluate()
    report={**provenance,'run_type':'scorer_learnability_probe','split':'trainval',
            'seed':args.seed,'scene_count':len(tokens),'tokens':tokens,'candidate_groups':identities,
            'frozen_feature_bank_sha256':sha256_file(args.output/'frozen_train_bank.pt'),
            'group_dependency_checks':dependency,'full_group_preserved':True,
            'navtest_labels_read':False,'formal_sampler_changed':False,
            'initial_exact_scorer_loss':float(initial), 'initial_train_selected_pdms':float(selected),
            'lr':1e-4,'additional_optimizer_steps':args.steps,'history':[]}
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss,selected=evaluate(train=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite exact scorer probe loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite scorer probe gradients')
        optimizer.step()
        if step%10==0 or step+1==args.steps:
            report['history'].append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
    with torch.no_grad():
        final,selected=evaluate()
    report.update(final_exact_scorer_loss=float(final),final_train_selected_pdms=float(selected),
                  train_oracle_at_64=float(bank['labels'][...,-1].max(-1).values.mean()),
                  conclusion_scope='Fit on these training scenes only; not generalization or deployment PDMS')
    torch.save({name:module.state_dict() for name,module in scorer_modules.items()},args.output/'probe_scorer.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
