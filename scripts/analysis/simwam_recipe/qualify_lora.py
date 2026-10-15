"""Actual DDP attention coverage, zero adapter, and merged ODE checks."""
import argparse
import copy
import json
from pathlib import Path
import torch
from starVLA.rl.simwam_recipe.contracts import ROOT,require_dispatch,load_recipe,seed_for
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.simwam_recipe.lora_adapter import merged_state_dict,LoRALinear
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from .export_policy import load_adapters


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();configure_numerics();cfg=load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
    actor,_,_=make_actor(cfg);store=ProjectedConditions(Path(ROOT)/'projected_condition',actor.action_expert)
    tokens=json.loads((Path(ROOT)/'qualification/real_TRAIN_tokens.json').read_text())['tokens'][:2]
    calls={n:0 for n,m in actor.action_expert.named_modules() if isinstance(m,LoRALinear)}
    def hook(name):
        def call(*_):calls[name]+=1
        return call
    hooks=[m.register_forward_hook(hook(n)) for n,m in actor.action_expert.named_modules() if isinstance(m,LoRALinear)]
    checks=[]
    with torch.no_grad():
        for token in tokens:
            c=store.get(token);z=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(seed_for('original_ode_initial',2101,token)))
            with torch.autocast('cuda',dtype=torch.float32):
                before=actor.action_expert._euler_sample(z,c);original=actor.ref_action_expert._euler_sample(z,c)
            if not torch.equal(before,original):raise ValueError('Zero-LoRA changed original native ODE')
            checks.append(dict(zero_LoRA_native_ODE_max_abs=float((before-original).abs().max())))
    if not all(calls.values()) or len(calls)!=96:raise ValueError('Unused or missing attention projection')
    for h in hooks:h.remove()
    identity,complete=load_adapters(a.checkpoint,actor)
    merged=copy.deepcopy(actor.ref_action_expert);merged.load_state_dict(merged_state_dict(actor.action_expert),strict=True)
    changed_native=[]
    with torch.no_grad():
        for token in tokens:
            c=store.get(token);z=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(seed_for('original_ode_initial',2101,token)))
            with torch.autocast('cuda',dtype=torch.float32):
                unmerged=actor.action_expert._euler_sample(z,c);deployed=merged._euler_sample(z,c)
                reference=actor.ref_action_expert._euler_sample(z,c)
            error=float((unmerged-deployed).abs().max())
            torch.testing.assert_close(unmerged,deployed,atol=2e-5,rtol=2e-5)
            changed_native.append(float((deployed-reference).abs().max()))
            checks.append(dict(trained_LoRA_merge_native_ODE_max_abs=error))
            for g in (1,8):
                x=z.float().expand(g,-1,-1).contiguous();t=torch.full((g,),642,device='cuda',dtype=torch.long)
                y=actor.action_expert.predict_velocity(x,t,c.expand(g,-1,-1).contiguous())
                y2=merged.predict_velocity(x,t,c.expand(g,-1,-1).contiguous())
                torch.testing.assert_close(y,y2,atol=2e-5,rtol=2e-5)
                checks.append(dict(candidate_batch=g,merged_velocity_max_abs=float((y-y2).abs().max())))
    result=dict(status='PASS',checks=checks,projection_calls=calls,parameter_contract=actor.closure,
                original_architecture_unchanged=True,source_checkpoint=a.checkpoint,updates=complete['updates'],
                normalized_native_ODE_change_from_SFT=changed_native,not_a_planning_score=True)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status='PASS',used_projections=len(calls),checks=checks)))


if __name__=='__main__':main()
