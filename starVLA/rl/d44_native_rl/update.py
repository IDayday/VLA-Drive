"""One native optimizer update, shared by the trainer and cycle benchmarks."""
import contextlib
import time
import torch
from .scheduler import scene_lr
from .stages import stage_recipe


def optimizer_update(ddp,actor,optimizer,scheduler,batch,cfg,scene_epochs,world,trainables,*,bulk_metrics=False):
    begin=time.monotonic();optimizer.zero_grad(set_to_none=True);totals={};valid_kl_sum=0.
    meter=None
    if bulk_metrics:
        from starVLA.rl.d46_fpo_engineering.device_metrics import DeviceMetrics
        meter=DeviceMetrics()
    multiplier=scene_lr(scene_epochs,warmup=cfg['warmup_epochs'],hold=cfg['hold_until_epoch'],end=cfg['cosine_end_epoch'],floor=cfg['min_lr_ratio'])
    for group in optimizer.param_groups:
        stage_multiplier=.5 if cfg['arm']=='D1'and batch['epoch']>=2 and group['name']=='text_lora'else 1.
        group['lr']=group['initial_lr']*multiplier*stage_multiplier
        if cfg['arm']=='D6':
            current_stage=stage_recipe(cfg,batch['epoch'])[1]
            rate=current_stage.text_lr if group['name']=='text_lora'else current_stage.film_lr if group['name']=='film_lora'else 2e-5 if current_stage.post else 0.
            warm=min(1.,max(1e-4,scene_epochs/cfg['warmup_epochs']));group['lr']=rate*warm
    for j,r in enumerate(batch['scenes']):
        sync=ddp.no_sync()if world>1 and j<len(batch['scenes'])-1 else contextlib.nullcontext()
        with sync:
            loss,metrics=ddp(r)
            weighted=loss*(0. if r['padding']else world/batch['valid_scenes']);weighted.backward()
        if meter is not None:meter.add(metrics,valid=not r['padding'])
        else:
            for key,value in metrics.items():totals[key]=totals.get(key,0.)+float(value.detach())/len(batch['scenes'])
            if not r['padding']:valid_kl_sum+=float(metrics['reference_KL_joint'].detach())
    if meter is not None:totals,valid_kl_sum=meter.drain()
    if cfg['arm']=='D6':
        for group in optimizer.param_groups:
            if group['lr']==0:
                for parameter in group['params']:parameter.grad=None
    norm=torch.nn.utils.clip_grad_norm_(trainables,1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite native gradient')
    grad_roles={}
    for group in optimizer.param_groups:
        grad_roles[group['name']]=float(torch.stack([p.grad.float().norm()for p in group['params']if p.grad is not None]).norm())if any(p.grad is not None for p in group['params'])else None
    optimizer.step();scheduler.step()
    if norm.device.type=='cuda':torch.cuda.synchronize()
    return dict(totals=totals,valid_kl_sum=valid_kl_sum,norm=norm,grad_roles=grad_roles,seconds=time.monotonic()-begin)
