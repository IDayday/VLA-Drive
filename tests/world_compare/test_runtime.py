"""Shared recipe/runtime and recoverable frozen-input descriptor lifecycle."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_loading_deployment import TinyModel
from starVLA.model.modules.qwen_world_compare.contracts import CurrentContext
from starVLA.rl.qwen_world_compare import runtime
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor, DESCRIPTOR_SCHEMA
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe


class Data:
    tokens = ['s0', 's1']
    sha = 'sealed_TRAIN'
    def observation(self, token):
        return dict(proprio=[0.]*8, images={}, navigation='straight', token=token)


class CurrentModel(TinyModel):
    def __init__(self, architecture):
        super().__init__(architecture, train_heads=False)
        if architecture != 'D':
            self.qwen = torch.nn.Linear(1, 1)
            self.scene_provider = torch.nn.Linear(1, 1)
        self.current_calls = 0

    def encode_current(self, batch, ego, scene_identity=()):
        self.current_calls += 1
        if self.architecture_id == 'D':
            encoded = self.decision.encode_current(batch, ego)
            return CurrentContext(None, None, ego, encoded, tuple(scene_identity), 'initial', 'CPU', 'native')
        # Frozen observation modules are genuinely used in this small oracle.
        value = self.scene_provider(self.qwen(torch.ones(1, 1)))
        scene = value.reshape(1, 1, 1).expand(1, 160, 16)
        return CurrentContext(scene, torch.ones(1, 160, dtype=torch.bool), ego, None,
                              tuple(scene_identity), 'initial', 'CPU', 'C160')


def processor(_):
    return dict(input_ids=torch.tensor([[3, 5, 7, 9]]), attention_mask=torch.ones(1, 4, dtype=torch.long))


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def test_common_recipe_changes_only_explicit_architecture_identity_and_metadata(tmp_path):
    base = expected_cleanx_recipe()
    path = tmp_path/'recipe.yaml';path.write_text(yaml.safe_dump(runtime.expected_recipe()))
    for architecture in ('A', 'B', 'C', 'D'):
        cfg = runtime.load_recipe(path, architecture)
        assert cfg['architecture_id'] == architecture
        for key in ('grpo', 'max_steps', 'learning_rate', 'optimizer_betas', 'optimizer_eps',
                    'gradient_clipping_effective', 'DDP_bucket_cap_mb', 'evaluation_seed'):
            assert cfg[key] == base[key]
        assert cfg['max_steps'] == 50000
        assert cfg['grpo']['train']['num_inner_epochs'] == 4
    cfg = runtime.expected_recipe();cfg['grpo']['sample']['noise_level'] = .2
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='locked common'):
        runtime.load_recipe(path, 'A')


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_token_descriptor_frozen_inputs_once_but_current_prefix_every_call(architecture):
    model = CurrentModel(architecture)
    cfg = runtime.expected_recipe(architecture)
    actor = FourFrameworkRecipeActor(model, cfg)
    provider = runtime.FrozenCurrentConditions(model, Data(),
        dict(model_sha256='IL_parent', identity={'same_IL':architecture}), processor=processor)
    object.__setattr__(actor, 'conditions', provider)
    descriptor = provider.get('s0')
    assert set(descriptor) == {'schema', 'token', 'identity', 'repeats'}
    assert descriptor['schema'] == DESCRIPTOR_SCHEMA and descriptor['repeats'] == 8
    chain, times, deltas, steps = actor.sample_action_chain(descriptor, 8,
        generator=torch.Generator().manual_seed(42))
    old = actor.action_chain_logprobs(descriptor, chain, times, deltas, stoch_idx=steps).detach()
    logp = actor.action_chain_logprobs(descriptor, chain, times, deltas, stoch_idx=steps)
    assert (torch.exp(logp-old)-1).abs().max() < 1e-4
    logp.mean().backward()
    assert model.current_calls == provider.prefill_calls == 1
    assert actor.prepared_context_builds['current'] == 3
    assert actor._prepared_scope is None
    if architecture == 'D':
        assert not provider.cached.encoded.inputs_embeds.requires_grad
        assert any(p.requires_grad for p in model.decision.qwen_model.language_model.parameters())
        assert not any(p.requires_grad for p in model.decision.qwen_model.visual.parameters())
    else:
        assert not provider.cached.scene.requires_grad
        assert not any(p.requires_grad for p in model.qwen.parameters())
        assert not any(p.requires_grad for p in model.scene_provider.parameters())
    # Restore materialization starts from the identical descriptor. It does not
    # restore a serialized stale P/W/text KV graph.
    restored = runtime.FrozenCurrentConditions(model, Data(),
        dict(model_sha256='IL_parent', identity={'same_IL':architecture}), processor=processor)
    assert restored.sha == provider.sha
    torch.testing.assert_close(restored.materialize(descriptor, torch.device('cpu'))['current'].ego,
                               provider.cached.ego, atol=0, rtol=0)
    restored.materialize(restored.get('s1'), torch.device('cpu'))
    assert restored.prefill_calls == 2


def test_descriptor_forbids_outside_population_and_changed_parent_identity():
    model = CurrentModel('B')
    first = runtime.FrozenCurrentConditions(model, Data(),
        dict(model_sha256='parentA', identity={'IL':'A'}), processor=processor)
    other = runtime.FrozenCurrentConditions(model, Data(),
        dict(model_sha256='parentB', identity={'IL':'B'}), processor=processor)
    with pytest.raises(ValueError, match='outside'):
        first.get('NAVTEST_scene')
    with pytest.raises(ValueError, match='identity changed'):
        other.materialize(first.get('s0'), torch.device('cpu'))


def test_runtime_does_not_accept_another_architecture_or_unselected_formal_parent(tmp_path, monkeypatch):
    path = tmp_path/'rl.yaml';path.write_text(yaml.safe_dump(runtime.expected_recipe()))
    identity = dict(architecture_id='B', config={'data_root':'/sealed_data'})
    monkeypatch.setattr(runtime, 'inspect_IL_checkpoint', lambda *a, **k:
                        dict(identity=identity, model_sha256='IL'))
    args = SimpleNamespace(config=str(path), architecture='A', il_checkpoint='/native',
                           data=None, qualification=False, parent_selection=None)
    with pytest.raises(ValueError, match='different architecture'):
        runtime.FourFrameworkRuntime().load_recipe(args)
    args.architecture = 'B'
    with pytest.raises(ValueError, match='full100k'):
        runtime.FourFrameworkRuntime().load_recipe(args)
    assert args.data == str(Path('/sealed_data/TRAIN').resolve())
