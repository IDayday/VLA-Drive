"""Saved-batch execution qualification, not fresh training or a new RL method.

Keep each full G8 group intact. A second worker evaluates the third stochastic
step while the first evaluates two. Weighting and DDP averaging recover the
original three-step mean, including tied-group masks and the SFT anchor.
"""
import argparse,json,os,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.qwen_sourceflow.contracts import ROOT,require_dispatch,configure_native_numerics,file_sha
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.conditions import FrozenConditions
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.simwam_recipe.checkpoint import device_tree,atomic_torch
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters,adapter_state
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


class StepShard(nn.Module):
    def __init__(self,actor,positions,factor):super().__init__();self.actor=actor;self.positions=positions;self.factor=factor
    def forward(self,rollout):
        local=dict(rollout,stoch_idx=rollout['stoch_idx'][self.positions],old_logp=rollout['old_logp'][:,self.positions])
        total,metrics=self.actor(local)
        return total*self.factor,metrics


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--native-run',dest='run',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--collection',type=int,default=5000);p.add_argument('--logical-world',type=int,default=8)
    p.add_argument('--shard-steps',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    host=require_dispatch();world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if a.logical_world not in [1,8] or world!=a.logical_world*(2 if a.shard_steps else 1):raise ValueError('Preserve logical groups')
    configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300));started=time.perf_counter();run=Path(a.run);ck=Path(a.checkpoint)
    identity=json.loads((run/'identity.json').read_text());cfg=identity['recipe'];seal=json.loads((ck/'COMPLETE').read_text());files=json.loads((ck/'files.json').read_text())
    if file_sha(ck/'files.json')!=seal['files_sha256'] or file_sha(ck/'optimizer.pt')!=files['optimizer.pt']:raise ValueError('Unsealed native optimizer')
    model,parent=load_il(identity['IL_parent']['path'],qualification=True,device=device,verify=rank==0)
    conditions=FrozenConditions(model,RawSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN'),parent['files']['model.pt'])
    actor=NativeRecipeActor(model.policy.action_expert,cfg,conditions);payload=torch.load(ck/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,payload['adapters']);trainable=[p for p in actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(payload['optimizer']);del payload,model
    logical=rank%a.logical_world;path=run/'behavior'/f'collection{a.collection:06d}'/f'rank{logical}.pt'
    rollout=torch.load(path,map_location='cpu',weights_only=False)
    if rollout['chain_identity']!=rollout_digest(rollout) or rollout['behavior_version']['run']!=identity['sha256']:raise ValueError('Changed saved chain')
    positions=([0,1] if rank<a.logical_world else [2]) if a.shard_steps else [0,1,2]
    factor=world/a.logical_world*len(positions)/3
    rollout=device_tree(rollout,device);actor.materialize(rollout['cond_bg'])
    wrapped=StepShard(actor,positions,factor).eval()
    ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25) if world>1 else wrapped
    ddp.eval();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);target=out/f'rank{rank}.json'
    if target.exists():raise FileExistsError(target)
    rows=[];dist.barrier();setup=time.perf_counter()-started
    try:
        for inner in range(4):
            dist.barrier();torch.cuda.synchronize();begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            total,metrics=ddp(rollout);total.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):raise ValueError('Missing/nonfinite copied gradient')
            if inner==0 and rank==0:first=torch.cat([p.grad.detach().reshape(-1) for p in trainable]).cpu()
            optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            row=dict(inner=inner+1,wall_seconds=elapsed,weighted_local_loss=float(total.detach()),raw_local_metrics={k:float(v) for k,v in metrics.items()},
                positions=positions,stochastic_steps=rollout['stoch_idx'][positions].tolist(),DDP_loss_factor=factor,peak_GPU_bytes=torch.cuda.max_memory_allocated())
            rows.append(row);atomic_json(target,dict(status='RUNNING',host=host,rank=rank,world=world,rows=rows))
            if inner==0 and rank==0:atomic_torch(out/'first_global_gradient.pt',first);del first
        if rank==0:atomic_torch(out/'final_adapters.pt',adapter_state(actor))
        atomic_json(target,dict(status='PASS',qualification_only=True,no_new_rollout_or_score=True,host=host,rank=rank,world=world,logical_world=a.logical_world,
            shard_steps=a.shard_steps,rows=rows,setup_seconds=setup,total_wall_seconds=time.perf_counter()-started,
            checkpoint=str(ck),checkpoint_optimizer_sha256=files['optimizer.pt'],saved_behavior_sha256=file_sha(path),parent_IL_sha256=parent['files']['model.pt'],script_sha256=file_sha(__file__),
            semantics='Fixed saved G8, old probability and global advantages; source loss unchanged; four copied Adam updates; qualification weights never formal initialization'))
        dist.barrier()
        if rank==0:atomic_json(out/'COMPLETE.json',dict(status='PASS',world=world,logical_world=a.logical_world,qualification_only=True))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
