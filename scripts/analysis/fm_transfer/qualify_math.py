"""Compare source ActionDiT forward and all input/parameter gradients directly."""
import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
import types
import torch
from starVLA.rl.fm_transfer.vendor.action_dit import ActionDiT
from starVLA.rl.fm_transfer.model import TransferPolicy
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def original_action_class(root):
    # Import the real pinned files, mocking ONLY the logging/package bootstrap.
    # No oracle math is copied into this test.
    for name,path in [('simwam',root/'src/simwam'),('simwam.models',root/'src/simwam/models'),
        ('simwam.models.wan22',root/'src/simwam/models/wan22'),('simwam.utils',root/'src/simwam/utils')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    log=types.ModuleType('simwam.utils.logging_config');log.get_logger=logging.getLogger
    sys.modules[log.__name__]=log
    return importlib.import_module('simwam.models.wan22.action_dit').ActionDiT


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    root=Path(a.upstream);Oracle=original_action_class(root);torch.manual_seed(42)
    cases=[]
    for dim in (3,4):
        cfg=dict(hidden_dim=16,action_dim=dim,ffn_dim=32,text_dim=24,freq_dim=8,eps=1e-6,
            num_heads=2,attn_head_dim=8,num_layers=2,use_gradient_checkpointing=False)
        source=Oracle(**cfg).float();adapted=ActionDiT(**cfg).float();adapted.load_state_dict(source.state_dict(),strict=True)
        for batch in (1,2):
            x=torch.randn(batch,8,dim);context=torch.randn(batch,8,24);t=torch.linspace(357.142857,681.81818,batch)
            copies=[]
            for model in (source,adapted):
                model.zero_grad(set_to_none=True);xx=x.clone().requires_grad_();cc=context.clone().requires_grad_()
                y=model(xx,t,cc);y.square().mean().backward()
                copies.append((y.detach(),xx.grad,cc.grad,{n:v.grad for n,v in model.named_parameters()}))
            left,right=copies
            for i in range(3):torch.testing.assert_close(left[i],right[i],rtol=0,atol=0)
            assert set(left[3])==set(right[3])
            for n in left[3]:
                assert left[3][n] is not None,n
                torch.testing.assert_close(left[3][n],right[3][n],rtol=0,atol=0)
            cases.append(dict(dim=dim,batch=batch,forward_exact=True,input_gradients_exact=True,
                parameter_gradients_exact=True,parameter_tensors=len(left[3])))
    sched=WanContinuousFlowMatchScheduler(1000,5.)
    sigma=torch.tensor([.01,.25,.5,.9]);y=torch.randn(4,8,4);eps=torch.randn_like(y)
    torch.testing.assert_close(sched.add_noise(y,eps,sigma*1000),(1-sigma[:,None,None])*y+sigma[:,None,None]*eps)
    torch.testing.assert_close(sched.training_target(y,eps,sigma*1000),eps-y,rtol=0,atol=0)
    weight=sched.training_weight(sigma*1000);assert torch.isfinite(weight).all() and (weight>=0).all()
    # Entire four-dimensional adapter loss, including repeats, scene reduction,
    # velocity sign, source time weights and all shared gradients.
    cfg.update(action_dim=4,text_dim=2048)
    oracle=Oracle(**cfg).float();head=ActionDiT(**cfg).float()
    head.load_state_dict(oracle.state_dict(),strict=True)
    adapter=TransferPolicy(head,'source_action_dit')
    raw=torch.randn(2,8,2048);target=torch.randn(2,8,4);epsilon=torch.randn(4,8,4)
    sigma=torch.tensor([.05,.35,.65,.95]);yy=target.repeat_interleave(2,0)
    xx=sched.add_noise(yy,epsilon,sigma*1000)
    direct=oracle(xx,sigma*1000,raw.repeat_interleave(2,0))
    expected=((direct-(epsilon-yy)).square().mean((1,2))*sched.training_weight(sigma*1000)).mean()
    actual,_=adapter(raw,target,epsilon=epsilon,sigma=sigma,repeat=2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.backward();expected.backward()
    for (n,x),(m,y) in zip(head.named_parameters(),oracle.named_parameters()):
        assert n==m
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    result=dict(status='PASS',scope='small exact source ActionDiT oracle; does not qualify full-model training',cases=cases,
        source_action_sha256=file_sha(root/'src/simwam/models/wan22/action_dit.py'),
        adapter_action_sha256=file_sha(Path(__file__).parents[3]/'starVLA/rl/fm_transfer/vendor/action_dit.py'),
        FM_sign_and_interpolation='PASS',full_adapter_weighted_loss_and_gradients='EXACT',
        no_Wan_video_VAE_or_MoT_instantiated=True)
    atomic_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
