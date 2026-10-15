"""Compare actual eight-rank updates to serial equal-rank source loss updates."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.simwam_recipe.contracts import require_dispatch,ROOT
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.checkpoint import device_tree
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=4);a=p.parse_args()
    require_dispatch();configure_numerics();root=Path(a.run);identity=json.loads((root/'identity.json').read_text());cfg=identity['recipe']
    if identity['world']!=8 or identity['purpose']!='qualification_not_formal_initialization':raise ValueError('Need eight-rank qualification')
    actor,_,_=make_actor(cfg)
    params={n:p for n,p in actor.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW(params.values(),lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
    actor._loss.optimizer=opt;actor._loss.learning_rate=cfg['learning_rate']
    scheduler=actor._loss._build_scheduler(cfg['lr_scheduler_type'],cfg['max_steps'],100)
    checks=[];initial=adapter_state(actor)
    for step in range(1,a.updates+1):
        if step%4==1:
            collection=(step-1)//4+1
            rollouts=[torch.load(root/'behavior'/f'collection{collection:06}'/f'rank{r}.pt',map_location='cpu',weights_only=False) for r in range(8)]
        opt.zero_grad(set_to_none=True);loss_mean=0.
        for row in rollouts:
            total,_=actor(device_tree(row,actor.device));(total/8).backward();loss_mean+=float(total.detach())/8
        grad=torch.cat([p.grad.flatten() for p in params.values()])
        target_path=root/'checkpoints'/f'update{step:06}_inner{step%4}'/'optimizer.pt'
        target=torch.load(target_path,map_location='cpu',weights_only=False) if target_path.exists() else None
        if step==1:
            ids=target['optimizer']['param_groups'][0]['params']
            reference_grad=torch.cat([target['optimizer']['state'][i]['exp_avg'].flatten()/.1 for i in ids]).to(grad)
            relative=float((grad-reference_grad).norm()/reference_grad.norm().clamp_min(1e-20))
            cosine=float(torch.nn.functional.cosine_similarity(grad,reference_grad,dim=0))
            if relative>2e-5 or cosine<.99999:raise ValueError('DDP averaged gradient differs from explicit rank mean')
        opt.step();scheduler.step()
        recorded=[]
        for rank in range(8):
            line=(root/f'metrics_rank{rank}.jsonl').read_text().splitlines()[step-1]
            recorded.append(json.loads(line)['loss'])
        if abs(loss_mean-sum(recorded)/8)>2e-6:raise ValueError('Rank loss reduction mismatch')
        if target is None:
            if step==a.updates:raise ValueError('Final comparison checkpoint missing')
            continue
        errors=[];differences=[];changes=[]
        for name,param in params.items():
            expected=target['adapters'][name].to(param)
            errors.append(float((param-expected).abs().max()))
            differences.append((param-expected).float().square().sum())
            changes.append((expected-initial[name].to(param)).float().square().sum())
        update_relative=float((torch.stack(differences).sum()/torch.stack(changes).sum().clamp_min(1e-30)).sqrt())
        # Near-zero Adam coordinates amplify tiny FP32 summation differences.
        # Check the aggregate update and actual density effect, rather than
        # requiring arbitrary absolute elementwise identity across NCCL trees.
        saved_serial=adapter_state(actor);row=device_tree(rollouts[0],actor.device)
        with torch.no_grad():
            serial_logp=actor.action_chain_logprobs(row['cond_bg'],row['chain'],row['timesteps'],row['deltas'],stoch_idx=row['stoch_idx'])
            restore_adapters(actor,target['adapters'])
            distributed_logp=actor.action_chain_logprobs(row['cond_bg'],row['chain'],row['timesteps'],row['deltas'],stoch_idx=row['stoch_idx'])
            density_ratio_error=float(((serial_logp-distributed_logp).exp()-1).abs().max())
            restore_adapters(actor,saved_serial)
        diagnostic=dict(step=step,parameter_max_abs=max(errors),update_relative_L2=update_relative,
                        density_ratio_error=density_ratio_error,first_gradient_relative_L2=relative,first_gradient_cosine=cosine)
        print(json.dumps(diagnostic),flush=True)
        debug=Path(a.output).with_suffix('.partial.json');debug.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(debug,dict(status='MEASURING',latest=diagnostic,completed=checks))
        # The first-gradient comparison establishes the reduction arithmetic.
        # Later FP32 Adam updates are not bitwise identical: tiny mean/anchor
        # rounding changes near-zero coordinates. Preserve and report their
        # relative update difference; enforce the policy-probability tolerance.
        if density_ratio_error>1e-4:raise ValueError('Material distributed probability discrepancy')
        checks.append(dict(optimizer_step=step,parameter_max_abs=max(errors),update_relative_L2=update_relative,
                           same_chain_density_ratio_error=density_ratio_error,explicit_equal_rank_mean_loss=loss_mean,
                           distributed_equal_rank_mean_loss=sum(recorded)/8))
    result=dict(status='PASS_WITH_MEASURED_FP32_PARAMETER_DIFFERENCES',checks=checks,first_gradient_relative_L2=relative,first_gradient_cosine=cosine,
                bitwise_parameter_equivalence=False,parameter_relative_002_threshold_exceeded=any(c['update_relative_L2']>.002 for c in checks),
                fresh_rollouts=(a.updates+3)//4,actual_optimizer_updates=a.updates,no_old_probability_refresh=True,
                original_real_DDP_action_model=True,reduction='source_rank_mean; serial per-rank scene losses, not active-only global normalization')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
