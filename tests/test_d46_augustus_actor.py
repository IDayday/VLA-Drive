"""Actual tiny native Qwen behavior, gradients and permanent-reference checks."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.qwen_world_compare.policy_adapter import CONDITION_SCHEMA
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d46_augustus.config import defaults
from starVLA.rl.d46_augustus.actor import AugustusActor
from starVLA.rl.d46_augustus.collector import collect_scene
from starVLA.rl.d46_augustus.runtime import credit,advantages
from tools.d46_world16.qualify_native import SyntheticService


def tiny(experiment):
    path=Path(__file__).parent/'world_compare/test_d_qwen_flow_adapter.py'
    spec=importlib.util.spec_from_file_location('augustus_tiny_native',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=FourFrameworkVLA('D',config={'d':{'qualification':True,'activation_checkpointing':False}},
        train_heads=False,qwen_encoder=SimpleNamespace(model=module.model().qwen_model))
    model=prune_deployment(model,copy_model=False);parent=copy.deepcopy(model)
    cfg=defaults(experiment);actor=AugustusActor(model,source_config(cfg),cfg)
    inputs=dict(input_ids=torch.tensor([[3,5,7,9]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    current=actor.action_expert.encode_current(inputs,torch.zeros(1,8),('TRAIN_CPU_0',))
    condition=dict(schema=CONDITION_SCHEMA,context=torch.zeros(8,1,1),current=current)
    return actor,parent,cfg,inputs,condition


@pytest.mark.parametrize('experiment',['AUG1_FULL','AUG2_LORA'])
def test_real_native_cleanx_JaT_update_preserves_parent_reference(experiment):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(42);actor,parent,cfg,inputs,condition=tiny(experiment)
        noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101))
        with torch.no_grad():
            a=actor.action_expert.predict(inputs,torch.zeros(1,8),noise)
            b=parent.predict(inputs,torch.zeros(1,8),noise)
        assert torch.equal(a['requested_trajectory'],b['requested_trajectory'])
        reference={n:p.detach().clone()for n,p in actor.ref_action_expert.named_parameters()}
        reference_ids={id(p)for p in actor.ref_action_expert.parameters()}
        assert all(id(p)not in reference_ids for p in actor.parameters()if p.requires_grad)
        generator=torch.Generator().manual_seed(420000)
        scene=collect_scene(actor,condition,'TRAIN_CPU_0','TRAIN_cpu',SyntheticService(),generator,cfg,0.,{'update':0})
        raw,_=credit([scene],None,0.);adv,_=advantages(raw,torch.tensor([True]),8,1)
        scene.update(advantages=adv[0],padding=False,beta=.001,expect_fresh=True)
        loss,metrics=actor(scene)
        traced,traced_metrics,parts=actor(scene,return_components=True)
        torch.testing.assert_close(traced,loss,rtol=0,atol=0)
        assert parts['logratio'].shape==(8,3)and parts['logratio'].requires_grad
        assert parts['reference_KL_steps'].shape==(8,3)
        loss.backward()
        assert metrics['fresh_ratio_error']==0 and torch.isfinite(loss)
        gradients=[p.grad for p in actor.parameters()if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all()for g in gradients)
        assert sum(float(g.square().sum())for g in gradients)>0
        opt=torch.optim.AdamW([p for p in actor.parameters()if p.requires_grad],lr=1e-6)
        opt.step()
        assert all(torch.equal(reference[n],p)for n,p in actor.ref_action_expert.named_parameters())
        assert scene['chain'].shape==(8,4,8,3)
        assert torch.equal(scene['chain'][:,0],scene['chain'][:1,0].expand(8,-1,-1))
        assert not any(p.requires_grad for p in actor.ref_action_expert.parameters())
    finally:torch.set_num_threads(previous)
