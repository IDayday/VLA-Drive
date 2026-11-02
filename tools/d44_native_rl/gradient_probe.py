"""Component gradients on one saved real TRAIN scene; no new policy candidates."""
import argparse
from pathlib import Path
import torch
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.deployment import inspect
from starVLA.rl.qwen_world_abcd_research.checkpoint import device_tree
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import require_dispatch


def main():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    configure_numerics();torch.set_num_threads(2)
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--behavior',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();identity,_,_=inspect(a.checkpoint);cfg=identity['recipe'];stream=DataStream(cfg,cfg['global_scenes'])
    model,info=load(cfg['parent_checkpoint'],'cuda',readouts=cfg['post_feedback']);actor=D44Actor(model,source_config(cfg),cfg)
    model.decision.activation_checkpointing=False;model.decision.checkpoint_short_suffix=False
    restore_adapters(actor,torch.load(Path(a.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False)['adapters'])
    object.__setattr__(actor,'conditions',DConditions(model,stream.data,info,group=cfg['group']))
    batch=torch.load(a.behavior,map_location='cpu',weights_only=False)
    candidates=[r for r in batch['scenes']if 'teacher'in r or cfg['post_feedback']]
    if not candidates:raise ValueError('No auxiliary-bearing real TRAIN scene in saved batch')
    r=device_tree(candidates[0],torch.device('cuda'));r['expect_fresh']=False
    total,metrics,parts=actor(r,return_components=True);params=[p for p in actor.parameters()if p.requires_grad]
    vectors={}
    for name,loss in parts.items():
        values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(v if v is not None else torch.zeros_like(p)).detach().float().flatten()for p,v in zip(params,values)])
    norms={k:float(v.norm())for k,v in vectors.items()};cos={}
    for name in ('reference','auxiliary'):
        den=norms['policy']*norms[name]
        cos[name+'_versus_policy']=float(vectors[name].dot(vectors['policy'])/den)if den else None
    atomic_json(a.output,dict(status='PASS',schema='D44_saved_TRAIN_component_gradients_v1',checkpoint=a.checkpoint,
        behavior=a.behavior,token=r['token'],gradient_norms=norms,cosines=cos,
        weighted_auxiliary_to_policy=norms['auxiliary']/norms['policy']if norms['policy']else None,
        loss_values={k:float(v.detach())for k,v in parts.items()},scorer_calls=0,new_candidates=0,optimizer_updates=0,
        interpretation='One saved TRAIN batch only; zero-gradient cosine is undefined, no test-driven coefficient selection'))


if __name__=='__main__':main()
