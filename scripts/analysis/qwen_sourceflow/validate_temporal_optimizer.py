"""Two CPU ranks: inherited/sharded Adam, exact RNG resume and publish failures."""
import argparse,copy,json,os
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.qwen_sourceflow.temporal_optimizer import build_optimizer,inherit_parent_adam,validate_local_adam,set_temporal_stage
from starVLA.rl.qwen_sourceflow.il_optimization import parameter_groups
from starVLA.rl.qwen_sourceflow import temporal_checkpoint as cp
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.qwen=nn.ModuleDict(dict(visual=nn.Linear(5,6),language=nn.Linear(6,6)))
        self.provider=nn.Linear(6,7);self.policy=nn.Linear(7,2)
    def forward(self,x,y,d=None,head_only=False):
        h=self.qwen['language'](self.qwen['visual'](x)).tanh();loss=x.new_zeros(()) if head_only else (self.policy(self.provider(h))-y).square().mean()
        if d is not None:loss=loss+(self.current_dino_head(h)-d).square().mean()
        return loss


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);dist.init_process_group('gloo');common.init_control_group();torch.set_num_threads(1);torch.manual_seed(42)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    cfg=dict(learning_rates=dict(qwen_language_merger=1e-3,qwen_vision=1e-4,action_interfaces=1e-2),betas=[.9,.95],eps=1e-8,weight_decay=.01,head_only_updates=2)
    model=Tiny();parent_opt=torch.optim.AdamW(parameter_groups(model,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    for _ in range(2):
        x=torch.randn(2,5);y=torch.randn(2,2);parent_opt.zero_grad();model(x,y).backward();parent_opt.step()
    payload=dict(optimizer=parent_opt.state_dict());steps={n:2 for n,_ in model.named_parameters()}
    model.current_dino_head=nn.Linear(6,3);reference=copy.deepcopy(model)
    optimizer=build_optimizer(model,cfg);inherit_parent_adam(model,optimizer,copy.deepcopy(payload))
    groups=parameter_groups(reference,cfg);heads={id(p) for p in reference.current_dino_head.parameters()}
    for g in groups:g['params']=[p for p in g['params'] if id(p) not in heads]
    groups.append(dict(params=list(reference.current_dino_head.parameters()),lr=.01,name='current_dino'))
    refopt=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    old=copy.deepcopy(payload['optimizer']);offset=sum(len(g['params']) for g in old['param_groups'])
    head={k:v for k,v in refopt.param_groups[-1].items() if k!='params'};head['params']=list(range(offset,offset+2));head['initial_lr']=.01
    old['param_groups'].append(head);refopt.load_state_dict(old)
    sched=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.);ddp=None;stage=None
    identity=dict(schema='native_temporal_DINO_IL_v1',qualification=True,world=2,parent_steps=steps,config=cfg)
    identity['sha256']=fingerprint(identity)
    torch.manual_seed(100+rank);saved=None;errors=[];inputs=[]
    def update(step,compare):
        nonlocal ddp,stage
        desired='head_calibration' if step<=2 else 'joint'
        if stage!=desired:
            if ddp is not None:del ddp
            stage=set_temporal_stage(model,step-1,2);set_temporal_stage(reference,step-1,2)
            ddp=DDP(model,find_unused_parameters=True)
        x=torch.randn(1,5);y=torch.randn(1,2);d=torch.randn(1,3)
        optimizer.zero_grad(set_to_none=True);ddp(x,y,d,head_only=step<=2).backward();optimizer.step();sched.step()
        validate_local_adam(model,optimizer,step,steps,2)
        if compare:
            all_inputs=common.gather((x,y,d));refopt.zero_grad()
            reference(*(torch.cat([z[i] for z in all_inputs]) for i in range(3)),head_only=step<=2).backward();refopt.step()
            errors.append(max(float((p-q).abs().max()) for p,q in zip(model.parameters(),reference.parameters())))
        return x.clone(),y.clone(),d.clone()
    for step in range(1,5):
        values=update(step,True)
        if step==2:saved=cp.save(root,model,optimizer,sched,dict(updates=step,cursor=100+2*step,phase='READY'),identity)
        if step>2:inputs.append(values)
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    state=cp.restore(saved,model,optimizer,sched,identity)
    for step in [3,4]:
        values=update(step,False)
        assert all(torch.equal(x,y) for x,y in zip(values,inputs[step-3])),'RNG/input resume changed'
    resume_error=max(float((p-expected[n]).abs().max()) for n,p in model.named_parameters())
    assert max(errors)<2e-7 and resume_error==0.,(errors,resume_error)
    rejected=False
    try:cp.save(root,model,optimizer,sched,dict(updates=4,cursor=108,phase='COLLECTING' if rank==0 else 'READY'),identity)
    except RuntimeError:rejected=True
    assert rejected and json.loads((root/'latest.json').read_text())['updates']==2
    original=common.atomic_torch
    def fail(path,payload):
        if rank==1 and Path(path).name=='rank1.pt':raise OSError('injected rank1 write failure')
        return original(path,payload)
    common.atomic_torch=fail;rejected=False
    try:cp.save(root,model,optimizer,sched,dict(updates=4,cursor=108,phase='READY'),identity)
    except RuntimeError:rejected=True
    finally:common.atomic_torch=original
    assert rejected and json.loads((root/'latest.json').read_text())['updates']==2
    assert not (root/'checkpoints/update000004/COMPLETE').exists()
    new=cp.save(root,model,optimizer,sched,dict(updates=4,cursor=108,phase='READY'),identity)
    if rank==0:
        atomic_json(root/'QUALIFICATION.json',dict(status='PASS',world=2,device='CPU',same_inherited_Adam_update_max_abs=max(errors),
            exact_resume_parameter_max_abs=resume_error,RNG_and_next_inputs_exact=True,mixed_phase_rejected_globally=True,
            rank_write_failure_preserves_latest=True,new_attempt_after_incomplete_passed=True,checkpoint=new))
        print((root/'QUALIFICATION.json').read_text(),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
