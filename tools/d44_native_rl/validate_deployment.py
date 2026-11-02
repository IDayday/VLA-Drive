"""Real updated policy: full R model, pruned model and compact export agree."""
import argparse
import os
from pathlib import Path
import torch
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.deployment import inspect
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.qwen_world_compare.contracts import require_dispatch
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.d44_native_rl.export import load_export


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--export',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(2);require_dispatch()
    identity,_,_=inspect(a.checkpoint);cfg=identity['recipe'];stream=DataStream(cfg,cfg['global_scenes'])
    model,info=load(cfg['parent_checkpoint'],'cuda',readouts=cfg['post_feedback']);actor=D44Actor(model,source_config(cfg),cfg)
    restore_adapters(actor,torch.load(Path(a.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False)['adapters'])
    token=stream.tokens[0];obs=stream.data.observation(token);inputs=QwenObservationProcessor(str(QWEN_PATH))([obs])
    ego=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn(1,8,3,generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token))).cuda()
    before=model.predict(inputs,ego,noise.clone());pruned=prune_deployment(model,copy_model=False)
    after=pruned.predict(inputs,ego,noise.clone());exported,meta=load_export(a.export,'cuda');result=exported.predict(inputs,ego,noise.clone())
    for field in ('normalized_action','requested_trajectory'):
        if not torch.equal(before[field],after[field])or not torch.equal(after[field],result[field]):raise ValueError('Pruning/export changed '+field)
    atomic_json(a.output,dict(status='PASS',checkpoint=a.checkpoint,export=a.export,token=token,original_R_model_to_pruned_max_abs=0.,
        pruned_to_compact_export_max_abs=0.,model_state_sha256=meta['model_state_sha256'],new_scorer_calls=0,
        actual_B1_inference_calls=3,new_training_samples=0,R16=False,CoT=False,scorer_required_for_deployment=False))


if __name__=='__main__':main()
