"""Native snapshot reconstructs execution precision after real scope restore."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import yaml
from world_postcot.test_scope_rl import FakeQwen
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,SCOPE_SCHEMA
from starVLA.rl.qwen_world_postcot.deployment import load_scope_trainable_state
from starVLA.rl.qwen_world_postcot.rl_actor import install_deployment_lora
from starVLA.rl.qwen_world_postcot.snapshot_navtest import load_native_snapshot
from starVLA.rl.qwen_world_safety_exploration.precision_recipe import build_recipe,export_model_config
from starVLA.rl.qwen_world_safety_exploration.precision_runtime import enable_projection_precision
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state

MODEL_CONFIG=dict(memory=dict(input_dim=64,hidden_dim=32,num_layers=1,num_heads=4,selected_layers=[1,2,3]),
 predictive=dict(input_dim=32,hidden_dim=16,num_layers=1,num_heads=4),
 action=dict(hidden_dim=32,num_layers=2,num_heads=4,ffn_dim=128))

def model(config):
 torch.manual_seed(42)
 return QwenWorldPostCoT(None,config=copy.deepcopy(config),train_heads=False,qwen_encoder=FakeQwen()).eval()

def predict(policy,noise):
 ego=torch.tensor([[1.,0.,0.,0.,1.,0.,0.,0.]])
 return policy.predict(dict(current=True),ego,noise)

def test_real_scope_native_snapshot_matches_deployment_B1_ODE_precision_and_legacy(tmp_path):
 torch.set_num_threads(1)
 base=yaml.safe_load(Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/configs/SAFETY_G8_CONTROL.yaml').read_text())
 noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(97))
 for precision in (True,False):
  cfg=build_recipe(base)if precision else copy.deepcopy(base)
  trained=model(MODEL_CONFIG)
  actor=ScopeRecipeActor(trained.actor,cfg,scene_provider=trained.scene_provider,predictive_provider=trained.predictive_provider)
  if precision:enable_projection_precision(actor)
  with torch.no_grad():
   for name,p in actor.named_parameters():
    if p.requires_grad:p.add_(.013 if name.endswith('lora_B')else .003)
  payload=adapter_state(actor);root=tmp_path/str(precision);root.mkdir()
  identity=dict(schema=SCOPE_SCHEMA,recipe=cfg,parameter_contract=dict(trainable_names=sorted(payload)))
  identity=json.loads(json.dumps(identity));(root/'identity.json').write_text(json.dumps(identity))
  torch.save(dict(adapters=payload),root/'optimizer.pt')
  info=dict(model_sha256='tiny_IL70',identity=dict(model_config=MODEL_CONFIG))
  parent=dict(path='mock_original_complete_IL70')
  export=dict(RL_parent=identity,RL_checkpoint=str(root))
  #Mock only complete-Qwen asset loading/device placement. The snapshot helper
  #constructs the genuine ScopeActor and restore_adapters copies actual tensors.
  with patch('starVLA.rl.qwen_world_postcot.loading.load_deployment_checkpoint',side_effect=lambda *a,**k:(model(MODEL_CONFIG),info)),patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self):
   native,native_info=load_native_snapshot(parent,export)
  assert native.actor.output_projection_fp32 is precision and native_info==info
  for name,value in payload.items():
   target=native.actor.get_parameter(name[len('action_expert.'):])if name.startswith('action_expert.')else native.get_parameter(name)
   assert torch.equal(target,value)
  #Independently follow the real deployment config/adapter mapping, not the
  #snapshot loader, and use the actual complete-model single-candidate predict.
  deployed=model(export_model_config(MODEL_CONFIG,identity))
  install_deployment_lora(deployed.actor,cfg['grpo']);load_scope_trainable_state(deployed,payload,MEMORY)
  deployed.eval().requires_grad_(False)
  a=predict(native,noise);b=predict(deployed,noise)
  assert a['normalized_action'].shape==(1,8,3)
  for key in ('normalized_action','requested_trajectory'):assert torch.equal(a[key],b[key])
  if precision:
   native.actor.output_projection_fp32=False
   wrong=predict(native,noise)
   assert not torch.equal(wrong['normalized_action'],b['normalized_action'])
   #Malformed opt-in is rejected by the actual precision recipe validator.
   bad=copy.deepcopy(identity);bad['recipe']['research']['precision_training']['output_projection_fp32']=False
   (root/'identity.json').write_text(json.dumps(bad))
   with patch('starVLA.rl.qwen_world_postcot.loading.load_deployment_checkpoint',side_effect=lambda *a,**k:(model(MODEL_CONFIG),info)),patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self),pytest.raises(ValueError):
    load_native_snapshot(parent,dict(export,RL_parent=bad))
