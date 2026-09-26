"""Two actual GPUs: real Qwen training, empty target rank, serialized optimizer/RNG resume."""
import argparse,copy,json,os,random
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from runtime import load_baseline,load_dataset,load_world_batch,seed_all
from budget import reserve,record
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def main():
 p=argparse.ArgumentParser()
 for n in ['checkpoint','vlm','data-root','manifest','target-cache','output','ledger']:p.add_argument('--'+n,required=True)
 a=p.parse_args();dist.init_process_group('nccl');rank=dist.get_rank();torch.cuda.set_device(rank)
 if rank==0:reserve(a.ledger,'real_ddp_resume_v4',3,vars(a))
 dist.barrier();seed_all(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
 cfg={'enabled':True,'agent_tokens':32,'lambda_ego':0.,'lambda_cls':1.,'lambda_box':1.,'lambda_motion':1.}
 policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval();ddp=DistributedDataParallel(policy,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
 ds=load_dataset(agent,a.manifest,a.data_root,2);example=ds[rank];inputs,targets=load_world_batch([example],a.target_cache)
 if rank==0:
  t=targets[0];targets=[replace(t,current_boxes=t.current_boxes[:0],current_classes=t.current_classes[:0],track_ids=(),future_xy_in_ego_t0=t.future_xy_in_ego_t0[:0],future_valid_mask=t.future_valid_mask[:0],current_supervision_mask=t.current_supervision_mask[:0],box_valid_mask=t.box_valid_mask[:0],overflow=0)]
 params=[p for p in ddp.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-4)
 def step(module,opt):
  opt.zero_grad(set_to_none=True);result=module([example],inputs,targets);result['loss'].backward();step.gradient=module.module.reader.queries.grad.detach().clone();opt.step();return float(result['loss'].detach())
 first=step(ddp,optimizer)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items() if not k.startswith('baseline.')}
 payload={'state':state,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
 torch.save(payload,out/f'rank{rank}.pt')
 second=step(ddp,optimizer);expected_gradient=step.gradient.clone()
 reference={k:v.detach().clone() for k,v in policy.state_dict().items() if not k.startswith('baseline.')}
 clone=StructuredWorldPolicy(agent.model,cfg).cuda().eval()
 loaded=torch.load(out/f'rank{rank}.pt',weights_only=False,map_location='cpu')
 for name,module in [('reader',clone.reader),('heads',clone.heads)]:module.load_state_dict({k[len(name)+1:]:v for k,v in loaded['state'].items() if k.startswith(name+'.')},strict=True)
 resumed=DistributedDataParallel(clone,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
 resumed_optimizer=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=1e-4);resumed_optimizer.load_state_dict(loaded['optimizer'])
 torch.set_rng_state(loaded['torch_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng']);random.setstate(loaded['python_rng']);np.random.set_state(loaded['numpy_rng'])
 replay=step(resumed,resumed_optimizer);gradient_error=float((step.gradient-expected_gradient).abs().max());max_error=0.
 for key,value in clone.state_dict().items():
  if key in reference:
   max_error=max(max_error,float((value-reference[key]).abs().max()))
 (out/f'rank{rank}.json').write_text(json.dumps({'rank':rank,'first_loss':first,'second_loss':second,'resumed_loss':replay,'max_parameter_difference':max_error,'query_gradient_difference':gradient_error,'deterministic_math_attention':True,'real_current_token':example['token'],'empty_target_rank':rank==0,'status':'PASS' if max_error==0 else 'FAIL','resume_boundary':'serialized per-rank state/RNG, reconstructed DDP model and optimizer, same 2-GPU topology; no mid-step support'},indent=2))
 print('RESUME_DEBUG',rank,first,second,replay,max_error,gradient_error,flush=True)
 dist.barrier()
 if rank==0:record(a.ledger,'real_ddp_resume_v4',3,'complete')
 dist.destroy_process_group()

if __name__=='__main__':main()
