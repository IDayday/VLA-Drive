"""Explicitly allocated tiny real-Scope single/torchrun DDP diagnostic, no scorer.

Usage after operator allocation:
 torchrun --nproc_per_node=2 tests/performance_nominal_gpu_probe.py --device cuda --output DIR
 CPU same-hook rehearsal: torchrun --nproc_per_node=2 ... --device cpu --output DIR
Only genuine actor/Memory/autograd plumbing is tested; official physics scoring
and full52.7M throughput/performance are outside this tiny mocked-score probe.
"""
import argparse,copy,json,os,sys
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_world_performance_nominal import actual,primary,Service,Loss
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.rl.qwen_world_performance import nominal_improvement as N,ode_target as O
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);configure_numerics();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
 device=torch.device(f'cuda:{local}'if a.device=='cuda'else'cpu')
 if a.device=='cuda':torch.cuda.set_device(device)
 if world>1:dist.init_process_group('nccl'if a.device=='cuda'else'gloo',timeout=timedelta(seconds=180))
 try:
  actor,conditions,c=actual(device);service=Service(nominal_good=bool(rank%2));leaf=primary(actor,conditions,service)
  nominal=O.probe_nominal(actor,conditions,'t0','TRAIN_log',service,leaf['behavior_version'])
  raw=O.condition_for_batch(actor,leaf['cond_bg'],1)
  with actor._context_call(raw)as condition:
   prepared=actor._prepare(condition,False)
   with torch.autocast(device.type,dtype=torch.bfloat16):deployed=CleanXFlowPolicy(actor.action_expert).sample_ode(prepared,nominal['chain'][:,0].to(device))
  assert torch.equal(deployed.detach().cpu(),nominal['clean'])
  replay=O.predict_clean_x(actor,leaf['cond_bg'],nominal['last_state'].to(device),nominal['last_sigma'].to(device));assert torch.equal(replay.detach().cpu(),nominal['clean'])
  N.attach_material(actor,leaf,conditions,service,exposure_epoch=.5,rank=rank,nominal=nominal);N.validate_material(leaf,rank=rank)
  actor._loss=Loss(c);actor._loss.auxiliary_state=N.new_calibration_state()
  model=DDP(actor,device_ids=[local]if a.device=='cuda'else None,broadcast_buffers=False,find_unused_parameters=True)if world>1 else actor
  params=[p for p in actor.parameters()if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5);rows=[]
  for u in range(2):
   opt.zero_grad(set_to_none=True)
   for micro in range(2):
    with(model.no_sync()if world>1 and micro==0 else nullcontext()):
     before=actor.current_memory_builds;loss,metrics=model(leaf);(loss/2).backward()
    assert actor.current_memory_builds-before==(3 if rank%2==0 else 1)
    if u==0 and micro==0 and rank%2==0:
     norms=actor._loss.auxiliary_state['last_norms'];assert norms['cfm']>0 and norms['ode']>0
    assert actor._probability_audit['current']['mean'].shape[0]==16
    rows.append(dict(update=u,micro=micro,loss=float(loss.detach()),enabled=bool(leaf['nominal_improvement']['enabled']),
     measured=float(metrics.get('nominal_gradient_norm_measured',loss.detach()*0)),memory_builds=actor.current_memory_builds,
     cfm_gradient_norm_last=actor._loss.auxiliary_state['last_norms']['cfm']if actor._loss.auxiliary_state['last_norms']else None,
     ode_gradient_norm_last=actor._loss.auxiliary_state['last_norms']['ode']if actor._loss.auxiliary_state['last_norms']else None))
   assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in params)
   for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
    assert sum(float(p.grad.norm())for n,p in actor.named_parameters()if n.startswith(prefix)and p.requires_grad)>0
   torch.nn.utils.clip_grad_norm_(params,1);opt.step()
  #The component observer never adds optimizersteps or replaces DDP reduction.
  assert all(int(v['step'])==2 for v in opt.state.values())
  out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
  value=dict(status='PASS',rank=rank,world=world,device=str(device),BF16=True,B1_deployment_last_head_exact=True,
   rows=rows,calibration_state=actor._loss.auxiliary_state,optimizer_steps=2,
   limits='actualTinyScope32hidden/2layers/frozenfakeQwen; mockedcanonicalfeedback; notfull52.7Mproductionqualification orPDMS evidence')
  (out/f'rank{rank}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
  if world>1:dist.barrier()
  if rank==0:(out/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',world=world,device=a.device,DDP='2microbackwards/optimizerstep; conditionalrank0auxon/rank1off; autograd.grad insideforward; nohookreentry'),indent=2)+'\n')
 finally:
  if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
