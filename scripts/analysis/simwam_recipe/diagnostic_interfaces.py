"""Explicit, non-formal native-grid / unclipped-math interfaces.

The formal make_actor/train entry never uses this class. These switches are
named algorithm changes, not silent fixes to either registered source recipe.
The bounded acceptance performs no scoring or optimizer update.
"""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.simwam_recipe.trainer_adapter import RecipeActor
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.contracts import ROOT,load_recipe,require_dispatch
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


class NativeDDPGrid(WanContinuousFlowMatchScheduler):
    def build_inference_schedule(self,num_inference_steps,device,dtype,shift_override=None):
        if num_inference_steps!=10 or shift_override is not None:raise ValueError('Native diagnostic is original K10 only')
        return (torch.tensor([1000*(1-k/10) for k in range(10)],device=device,dtype=dtype),
                torch.full((10,),-.1,device=device,dtype=dtype))


class DiagnosticActor(RecipeActor):
    def __init__(self,head,cfg):
        self.diagnostic_grid='source_shift5';self.diagnostic_probability='source_relative_clamp'
        super().__init__(head,cfg)

    def set_diagnostic(self,*,grid,probability):
        if grid not in ('source_shift5','native_ddp_grid'):raise ValueError(grid)
        if probability not in ('source_relative_clamp','unclipped_math'):raise ValueError(probability)
        self.diagnostic_grid=grid;self.diagnostic_probability=probability
        self.infer_action_scheduler=NativeDDPGrid(1000,shift=1) if grid=='native_ddp_grid' else WanContinuousFlowMatchScheduler(1000,shift=5)
        ts,_=self.infer_action_scheduler.build_inference_schedule(10,self.device,torch.float32)
        self.grpo_sigma_max_guard=float(ts[1])/1000
        self.grpo_randn_clip=float('inf') if probability=='unclipped_math' else 5.

    def action_velocity(self,x,t,cond,use_ref=False):
        if self.diagnostic_grid=='source_shift5':return super().action_velocity(x,t,cond,use_ref=use_ref)
        index=((1000-t.float())/100).round().long()
        if ((1000-index*100)-t).abs().max()>1e-4 or (index<0).any() or (index>9).any():raise ValueError('Native grid timestep mismatch')
        # Match original _euler_sample Python step -> bucket conversion, avoiding
        # extra subtraction/cast rounding from reversing a FP32 sigma value.
        table=torch.tensor([int(k/10*1000) for k in range(10)],device=x.device)
        head=self.ref_action_expert if use_ref else self.action_expert
        self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
        with torch.autocast(self.device.type,enabled=False):return -head.predict_velocity(x.float(),table[index],cond['context'].float())

    def action_chain_logprobs(self,cond,chain,timesteps,deltas,*,stoch_idx=None,return_terms=False,use_ref=False):
        logp,terms=super().action_chain_logprobs(cond,chain,timesteps,deltas,stoch_idx=stoch_idx,return_terms=True,use_ref=use_ref)
        if self.diagnostic_probability=='unclipped_math':
            idx=self._normalize_stoch_idx(stoch_idx,len(timesteps))
            nxt=chain[:,torch.tensor(idx,device=chain.device)+1]
            logp=torch.distributions.Normal(terms['mean'].float(),terms['std'][None,:,None,None]).log_prob(nxt.float()).mean((-1,-2))
        return (logp,terms) if return_terms else logp


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args();require_dispatch();configure_numerics()
    from starVLA.rl.constrained_flow_noise.loading import load_action_closure
    cfg=load_recipe('configs/flow_rl/simwam_source_ddp.yaml');closure,_,_=load_action_closure(dtype=torch.float32)
    actor=DiagnosticActor(closure.action_model,cfg)
    token=json.loads(Path(ROOT+'/qualification/real_TRAIN_tokens.json').read_text())['tokens'][0]
    condition=ProjectedConditions(Path(ROOT)/'projected_condition',actor.action_expert).get(token)
    root=torch.randn((1,8,4),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
    actor.set_diagnostic(grid='native_ddp_grid',probability='source_relative_clamp')
    with torch.no_grad():
        chain,ts,dt,idx=actor.sample_action_chain({'context':condition},8,init_actions=root,deterministic=True)
        native=actor.action_expert._euler_sample(root,condition)
    error=float((chain[:,-1]-native).abs().max())
    if error!=0:raise ValueError('Native diagnostic changed original deterministic ODE')
    actor.set_diagnostic(grid='source_shift5',probability='unclipped_math')
    with torch.no_grad():
        chain,ts,dt,idx=actor.sample_action_chain({'context':condition},8,generator=torch.Generator().manual_seed(42))
    lp,terms=actor.action_chain_logprobs({'context':condition},chain,ts,dt,stoch_idx=idx,return_terms=True)
    expected=torch.distributions.Normal(terms['mean'],terms['std'][None,:,None,None]).log_prob(chain[:,idx+1]).mean((-1,-2))
    if not torch.equal(lp,expected):raise ValueError('Unclipped mathematical density differs')
    rows=[]
    for k,(t,delta) in enumerate(zip(ts,dt)):
        sigma=float(t)/1000;_,std=actor._sde_mean_std(chain[:,k],torch.zeros_like(chain[:,k]),sigma,float(delta))
        rows.append(dict(step=k,sigma=sigma,t_DDP=1-sigma,bucket=int(((1-t/1000)*1000).long()),delta_DDP=-float(delta),
                         injected_transition_std=std if k in idx.tolist() else 0.,stochastic=k in idx.tolist()))
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',purpose='diagnostic interfaces only, not a third formal experiment',
                native_mean_ODE_max_abs=error,unclipped_Normal_exact=True,new_scoring_calls=0,optimizer_updates=0,
                source_shift5_grid=rows,source_relative_clamp_remains_formal=True)
    atomic_json(output,result);print(json.dumps(dict(status='PASS',output=str(output))))


if __name__=='__main__':main()
