"""Numerical precision, genuine scope gradients and deployment-path parity."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from world_postcot.test_scope_rl import TinyModel
from test_world_verified_retention import Data, Service
from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY
from starVLA.rl.qwen_world_postcot.deployment import load_scope_trainable_state
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit
from starVLA.rl.qwen_world_safety_exploration import precision_recipe as C
from starVLA.rl.qwen_world_safety_exploration.precision_runtime import enable_projection_precision
from starVLA.rl.qwen_world_safety_exploration.standardized_recipe import StandardSchedulerLossMixin
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss


def config():
    root = Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101')
    return C.build_recipe(yaml.safe_load((root/'configs/SAFETY_G8_CONTROL.yaml').read_text()))


def test_projection_compute_not_posthoc_cast_and_default_unchanged():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    head = CleanXActor(hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64)
    keys = list(head.state_dict())
    hidden = torch.randn(8, 8, 32, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        legacy = head.predict_x0(hidden)
        expected_legacy = head.action_output(hidden).float()
    assert torch.equal(legacy, expected_legacy)
    head.output_projection_fp32 = True
    with torch.autocast('cpu', dtype=torch.bfloat16):
        current = head.predict_x0(hidden)
    expected = torch.nn.functional.linear(hidden, head.action_output.weight, head.action_output.bias)
    assert current.dtype == torch.float32 and torch.equal(current, expected)
    assert not torch.equal(current, legacy) and keys == list(head.state_dict())
    current.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_real_scope_rollout_replay_memory_gradient_and_single_ODE_deployment_parity():
    torch.set_num_threads(1)
    cfg = config()
    torch.manual_seed(42)
    model = TinyModel()
    actor = ScopeRecipeActor(model.actor, cfg, scene_provider=model.scene_provider,
                            predictive_provider=model.predictive_provider)
    enable_projection_precision(actor)
    class Loss(StandardSchedulerLossMixin, ResearchLoss):pass
    actor._loss = Loss(cfg)
    cond = GroupMemoryConditions(model, Data(), 'fixture_IL70', actor, group_size=8,
                                 processor=lambda obs: dict(current=True))
    object.__setattr__(actor, 'conditions', cond)
    actor.record_sampling_terms = True
    rollout = collect_group(actor, cond.get('t0', 'cpu'), 't0', 'log0', Service(),
        torch.Generator().manual_seed(83), group_size=8, variant='R_CANONICAL',
        version=dict(run='FP32_fixture', update=0, adapter_sha256='fixture'))
    assert rollout['qualification_sampling_check']['status'] == 'PASS'
    GroupCredit(8).finalize(rollout)
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = actor(rollout)
        loss.backward()
        for prefix in ['action_expert.', 'scene_provider.', 'predictive_provider.']:
            selected = [p for n,p in actor.named_parameters() if p.requires_grad and n.startswith(prefix)]
            assert selected and all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected)
            assert sum(float(p.grad.abs().sum()) for p in selected) > 0
        optimizer.step()
    # The real deployment loader copies exactly the same nonzero LoRA/memory
    # tensors; the execution flag is reconstructed from explicit export config.
    torch.manual_seed(42)
    deployed = TinyModel()
    wrapper = ScopeRecipeActor(deployed.actor, cfg, scene_provider=deployed.scene_provider,
                               predictive_provider=deployed.predictive_provider)
    deployed.actor.output_projection_fp32 = True
    load_scope_trainable_state(deployed,
        {n:p.detach().clone() for n,p in actor.named_parameters() if p.requires_grad}, MEMORY)
    root = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(97))
    ego = torch.tensor(Data().observation('t0')['proprio'])[None]
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        train_memory = actor.predictive_provider(actor.scene_provider(model.qwen.features, ego))
        deploy_memory = deployed.predictive_provider(deployed.scene_provider(deployed.qwen.features, ego))
        tc = actor.action_expert.prepare_context(train_memory.actor_memory, train_memory.actor_valid, ego)
        dc = deployed.actor.prepare_context(deploy_memory.actor_memory, deploy_memory.actor_valid, ego)
        a = CleanXFlowPolicy(actor.action_expert).sample_ode(tc, root)
        b = CleanXFlowPolicy(deployed.actor).sample_ode(dc, root)
    assert torch.equal(a, b)


def test_explicit_export_config_and_initial_evaluation_gate():
    cfg = config()
    parent = dict(action=dict(hidden_dim=32))
    rl = dict(recipe=cfg)
    result = C.export_model_config(parent, rl)
    assert result['action']['output_projection_fp32'] is True
    assert 'output_projection_fp32' not in parent['action']
    C.validate_export_precision(dict(model_config=result, RL_parent=rl,
                                    IL_parent=dict(identity=dict(model_config=parent))))
    with pytest.raises(ValueError):
        C.validate_export_precision(dict(model_config=parent, RL_parent=rl,
                                        IL_parent=dict(identity=dict(model_config=parent))))
    assert C.export_model_config(parent, dict(recipe={})) == parent
    assert cfg['evaluate_updates'][0] == 0
    from tools.qwen_world_safety_exploration.improvement_campaign import gates, budget
    assert gates('fp32_output', 20000)[0] == 0 and budget('fp32_output', 2000)['candidate_queries'] == 32000
