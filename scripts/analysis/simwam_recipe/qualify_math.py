"""Compare directly loaded locked source methods with the isolated port.

No Wan weights are loaded. A mock expert exercises the actual source call path;
explicit independent calculations check signs, tails, reduction and gradients.
"""
import argparse
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from starVLA.rl.simwam_recipe.vendor import action_math,trainer_math
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.contracts import load_recipe,file_sha,ROOT
from starVLA.rl.flow_grpo.transactions import atomic_json


def original_class(path,name,namespace,methods):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    cls.bases=[];cls.decorator_list=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert len(cls.body)==len(methods)
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    scope=dict(namespace)
    exec(compile(module,str(path),'exec'),scope)
    return scope[name]


class Expert(nn.Module):
    def __init__(self,dim):
        super().__init__();self.action_dim=dim
        for n in ('q','k','v','o'):setattr(self,n,nn.Linear(dim,dim))
    def forward(self,x,t,c):
        return self.o(torch.tanh(self.q(x)+self.k(c)+self.v(x*.25+t[:,None,None]/1000)))


def actor_class(mixin):
    class Actor(mixin,nn.Module):
        def __init__(self,cfg,dtype=torch.float32,dim=3):
            nn.Module.__init__(self);torch.manual_seed(42)
            self.action_expert=Expert(dim).to(dtype)
            self.device=torch.device('cpu');self.torch_dtype=dtype;self.grpo_action_dtype=dtype
            self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5)
            self.configure_grpo(cfg['grpo']);self.eval()
        def action_velocity(self,x,t,c,use_ref=False):
            return (self.ref_action_expert if use_ref else self.action_expert)(x,t,c['context'])
    return Actor


def run(source,config):
    lock=json.loads(Path('reports/ddp_simwam_recipe_20261015/upstream_lock.json').read_text())
    for rel,sha in lock['simwam']['files'].items():assert file_sha(source/rel)==sha
    for rel,sha in lock['vendor'].items():assert file_sha(rel)==sha
    om=original_class(source/'src/simwam/models/wan22/simwam_grpo.py','ActionGRPOMixin',vars(action_math),lock['extraction']['action_methods'])
    ot=original_class(source/'src/simwam/trainer_grpo.py','SimWAMGRPOTrainer',vars(trainer_math),lock['extraction']['trainer_methods'])
    cfg=load_recipe(config);cfg['grpo']['train']['kl_dim_weights']=[.2,1,1]
    results=[]
    def compare(label,x,y,atol=0,rtol=0):
        torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
        results.append(dict(check=label,max_abs=float((x.float()-y.float()).abs().max())))
    for dtype in (torch.float32,torch.bfloat16):
        a=actor_class(action_math.ActionGRPOMixin)(cfg,dtype);b=actor_class(om)(cfg,dtype)
        cond={'context':torch.randn(16,8,3).to(dtype)}
        ca,ta,da,ia=a.sample_action_chain(cond,8,generator=torch.Generator().manual_seed(123))
        cb,tb,db,ib=b.sample_action_chain(cond,8,generator=torch.Generator().manual_seed(123))
        assert ia.tolist()==[7,8,9] and ca.shape==(16,11,8,3)
        for name,x,y in [('grid',ta,tb),('deltas',da,db),('chain',ca,cb)]:compare(str(dtype)+'_'+name,x,y)
        for k in [0,7,8,9]:
            x=ca[:,k];v=a.action_velocity(x,ta[k].expand(16),cond)
            mu,std=a._sde_mean_std(x,v,float(ta[k])/1000,float(da[k]))
            mo,so=b._sde_mean_std(x,v,float(ta[k])/1000,float(da[k]));compare('mean_source_'+str(k),mu,mo);assert std==so
            sig=a.grpo_sigma_max_guard if float(ta[k])/1000>=1 else float(ta[k])/1000
            g=.1*math.sqrt(sig/(1-sig));delta=float(da[k])
            # Independent FP64 arithmetic, plus DDP time/velocity reversal.
            manual=x.double()*(1+g*g/(2*sig)*delta)+(-v.double())*(1+g*g*(1-sig)/(2*sig))*(-delta)
            compare('mean_fp64_sign_'+str(k),mu.double(),manual,atol=5e-7,rtol=2e-7)
            assert std==max(g*math.sqrt(-delta),.001)
        # Include source probability-clamp tails explicitly.
        ca[:,10,0,0]+=2
        la,terms=a.action_chain_logprobs(cond,ca,ta,da,stoch_idx=ia,return_terms=True)
        lb=b.action_chain_logprobs(cond,ca,ta,da,stoch_idx=ia)
        compare('raw_to_relative_clamp_'+str(dtype),la,lb)
        manual=[];tail=0
        for j,k in enumerate(ia.tolist()):
            mean,std=a._sde_mean_std(ca[:,k],terms['vel'][:,j],float(ta[k])/1000,float(da[k]))
            raw=torch.distributions.Normal(mean.to(dtype).float(),torch.tensor(std)).log_prob(ca[:,k+1].float())
            peak=-math.log(std)-.5*math.log(2*math.pi);tail+=int((raw<peak-7).sum())
            manual.append(raw.clamp(peak-7,peak).mean((1,2)))
        compare('density_reduction_'+str(dtype),la,torch.stack(manual,1));assert tail>0
        rewards=torch.tensor([[1,1,1,1,1,1,1,1],[0,.1,.2,.3,.6,.7,.8,.9]])
        stats=advantages(rewards);loss=SourceLoss(cfg)
        ref=ot();ref.__dict__.update(loss.__dict__);ref.accelerator=SimpleNamespace(gather=lambda t:t)
        expected=ref._compute_advantages(rewards)
        compare('advantages_vs_source',stats['advantages'],expected['advantages'],atol=2e-7)
        compare('tie_mask_vs_source',stats['adv_mask'],expected['adv_mask'])
        rollout=dict(chain=ca,timesteps=ta,deltas=da,stoch_idx=ia,cond_bg=cond,
                     advantages=stats['advantages'],adv_mask=stats['adv_mask'],old_logp=la.detach())
        # Change the current policy; reference and saved old probability stay frozen.
        with torch.no_grad():
            for actor in (a,b):
                for name,p in actor.named_parameters():
                    if name.endswith('lora_B'):p.add_(.004)
        total,metrics=loss._policy_loss(a,rollout);other,ometrics=ref._policy_loss(b,rollout)
        compare('total_source_'+str(dtype),total,other)
        total.backward();other.backward()
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            assert n==m
            if p.requires_grad:compare('gradient_'+str(dtype)+'_'+n,p.grad,q.grad)
            else:assert p.grad is None
        new,terms=a.action_chain_logprobs(cond,ca,ta,da,stoch_idx=ia,return_terms=True)
        _,refterms=a.action_chain_logprobs(cond,ca,ta,da,stoch_idx=ia,return_terms=True,use_ref=True)
        w=torch.tensor([.2,1,1]).view(1,1,1,3)
        anchor=((terms['mean']-refterms['mean'])**2*w).sum((2,3))/(8*2.2)
        ratio=(new-rollout['old_logp']).exp();adv=stats['advantages'][:,None].clamp(-5,5)
        mask=stats['adv_mask'][:,None];denom=mask.sum().clamp(min=1)*3
        pg=(torch.maximum(-ratio*adv,-ratio.clamp(.995,1.005)*adv)*mask).sum()/denom
        compare('independent_loss_'+str(dtype),total,pg+50*anchor.mean(),atol=1e-7,rtol=1e-6)
    # Global std is over raw rewards, not centered residuals; tied decimal groups exactly zero.
    for value in [0.,1.,.1,.833333333]:
        z=advantages(torch.full((2,8),value));assert torch.count_nonzero(z['advantages'])==0
    r=torch.tensor([[1.]*8,[0.]*7+[.1]])
    z=advantages(r);compare('global_raw_std',z['global_std'],r.double().std(unbiased=False))
    assert abs(float(z['global_std'])-float((r-r.mean(1,keepdim=True)).std(unbiased=False)))>.1
    # 4D heading-pair adaptation has the same total logical heading weight,
    # but does not claim equivalence to angular squared error.
    d=torch.tensor([2.,3.,4.,6.]);mapped=(d*d*torch.tensor([.2,1,.5,.5])).sum()/2.2
    logical=(.2*d[0]**2+d[1]**2+.5*(d[2]**2+d[3]**2))/2.2
    compare('heading_pair_average',mapped,logical)
    # Actual source losses/gradients: tied ranks retain their anchor, and their
    # zero PG is not renormalized away by the world reduction.
    for world in (1,2,8):
        for all_tied in (False,True):
            model=actor_class(action_math.ActionGRPOMixin)(cfg)
            cond={'context':torch.randn(8,8,3)}
            chain,ts,dt,idx=model.sample_action_chain(cond,8,generator=torch.Generator().manual_seed(301))
            old=model.action_chain_logprobs(cond,chain,ts,dt,stoch_idx=idx).detach()
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if name.endswith('lora_B'):param.add_(.0001)
            raw=torch.stack([torch.ones(8)*.1 if all_tied or rank%2 else torch.linspace(.05,.95,8) for rank in range(world)])
            local_losses=[];manual_pg=[];anchors=[]
            for rank in range(world):
                local=advantages(raw[rank:rank+1],all_rewards=raw)
                row=dict(chain=chain,timesteps=ts,deltas=dt,stoch_idx=idx,cond_bg=cond,old_logp=old,**local)
                total,_=SourceLoss(cfg)._policy_loss(model,row);local_losses.append(total)
                lp,terms=model.action_chain_logprobs(cond,chain,ts,dt,stoch_idx=idx,return_terms=True)
                with torch.no_grad():_,refterms=model.action_chain_logprobs(cond,chain,ts,dt,stoch_idx=idx,return_terms=True,use_ref=True)
                ratio=(lp-old).exp();ad=local['advantages'].clamp(-5,5)[:,None]
                pg=torch.maximum(-ratio*ad,-ratio.clamp(.995,1.005)*ad).mean()
                if not local['adv_mask'].any():pg=pg*0
                weight=torch.tensor([.2,1,1]).view(1,1,1,3)
                anchor=(((terms['mean']-refterms['mean'])**2*weight).sum((-1,-2))/(8*2.2)).mean()
                manual_pg.append(pg);anchors.append(anchor)
            actual=torch.stack(local_losses).mean();expected=torch.stack(manual_pg).sum()/world+50*torch.stack(anchors).mean()
            compare(f'actual_source_rank_loss_world{world}_alltied{all_tied}',actual,expected,atol=1e-7)
            params=[p for p in model.parameters() if p.requires_grad]
            left=torch.autograd.grad(actual,params,retain_graph=True)
            right=torch.autograd.grad(expected,params)
            compare(f'actual_source_rank_gradient_world{world}_alltied{all_tied}',torch.cat([x.flatten() for x in left]),torch.cat([x.flatten() for x in right]),atol=3e-7,rtol=2e-5)
            if all_tied:
                assert torch.stack(manual_pg).abs().sum()==0
                assert float(torch.stack(anchors).mean())>0
    return dict(status='PASS',scope='locked upstream 3D forward/loss/gradients and explicit 4D mapping; not driving performance',
                checks=results,source_commit=lock['simwam']['commit'],preserved_empirical_clamps=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',default=ROOT+'/upstream/SimWAM')
    p.add_argument('--config',default='configs/flow_rl/simwam_source_ddp.yaml')
    p.add_argument('--output',default=ROOT+'/qualification/math.json');a=p.parse_args()
    result=run(Path(a.source),a.config);Path(a.output).parent.mkdir(parents=True,exist_ok=True);atomic_json(Path(a.output),result)
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),output=a.output)))


if __name__=='__main__':main()
