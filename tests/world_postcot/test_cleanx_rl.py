import copy
import json

import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.model.modules.qwen_world_postcot.memory import StudentMemory
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.rl_actor import CleanXRecipeActor, install_deployment_lora
from starVLA.rl.qwen_world_postcot.rl_conditions import FrozenWorldConditions
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe, load_cleanx_recipe
from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler
from scripts.analysis.simwam_recipe.train import parameter_hash


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_actor(*, constant=False):
    torch.manual_seed(42)
    head = CleanXActor(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=128)
    with torch.no_grad():
        for block in head.blocks:
            block.modulation[-1].bias[64:96].fill_(0.1)
            block.modulation[-1].bias[160:192].fill_(0.1)
        if constant:
            head.action_output.weight.zero_()
            head.action_output.bias.fill_(0.25)
    return CleanXRecipeActor(head, expected_cleanx_recipe())


def condition(batch=8):
    base = torch.randn(1, 9, 32)
    valid = torch.ones(1, 9, dtype=torch.bool)
    ego = torch.randn(1, 8)
    return dict(context=base.expand(batch, -1, -1), valid=valid.expand(batch, -1),
        ego=ego.expand(batch, -1), base_memory=base, base_valid=valid, base_ego=ego)


def test_actor_closure_and_clean_x_single_velocity_conversion():
    actor = make_actor(constant=True)
    assert len(actor.closure['adapters']) == 16
    assert len(actor.closure['trainable_names']) == 32
    assert all(parameter.dtype == torch.float32 for parameter in actor.parameters())
    assert all(not parameter.requires_grad for parameter in actor.ref_action_expert.parameters())
    state = torch.randn(8, 8, 3)
    timestep = torch.full((8,), 400.0)
    velocity = actor.action_velocity(state, timestep, condition())
    torch.testing.assert_close(velocity, (state - 0.25) / 0.4, rtol=0, atol=0)
    with pytest.raises(ValueError, match='strictly positive'):
        actor.action_velocity(state, timestep * 0, condition())


def test_ode_matches_clean_x_and_never_converts_at_sigma_zero():
    actor = make_actor(constant=True)
    cond = condition()
    noise = torch.randn(8, 8, 3)
    chain, times, _, stochastic = actor.sample_action_chain(cond, 8, deterministic=True, init_actions=noise)
    assert chain.shape == (8, 11, 8, 3) and chain.dtype == torch.float32
    assert stochastic.numel() == 0 and bool((times > 0).all())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        prepared = actor.action_expert.prepare_context(cond['base_memory'], cond['base_valid'], cond['base_ego'])
        prepared = prepared.index_select(torch.zeros(8, dtype=torch.long))
        direct = CleanXFlowPolicy(actor.action_expert).sample_ode(prepared, noise)
    torch.testing.assert_close(chain[:, -1], direct, rtol=0, atol=1e-6)
    assert actor._prepared_scope is None and actor.prepared_context_builds['current'] == 1
    assert actor.prepared_context_source_batches == [1]


def test_stochastic_sampling_recompute_ratio_and_mean_parity():
    actor = make_actor()
    actor.record_sampling_terms = True
    cond = condition()
    with torch.no_grad():
        chain, times, deltas, indices = actor.sample_action_chain(cond, 8, generator=torch.Generator().manual_seed(7))
        old = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=indices)
    recomputed = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=indices)
    assert indices.tolist() == [7, 8, 9] and len(actor.sampling_terms) == 3
    assert torch.equal((recomputed - old).exp(), torch.ones_like(old))
    terms = actor._probability_audit['current']
    for index, (mean, std) in enumerate(actor.sampling_terms):
        assert torch.equal(mean, terms['mean'][:, index])
        assert float(torch.tensor(std)) == float(terms['std'][index])
    assert torch.isfinite(terms['mean']).all() and torch.isfinite(terms['std']).all()
    assert actor.prepared_context_builds['current'] == 3
    assert actor._prepared_scope is None


def test_real_source_policy_loss_updates_lora_not_reference_or_base():
    actor = make_actor()
    cond = condition()
    with torch.no_grad():
        chain, times, deltas, indices = actor.sample_action_chain(cond, 8, generator=torch.Generator().manual_seed(3))
        old = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=indices).detach()
    rollout = dict(cond_bg=cond, chain=chain, timesteps=times, deltas=deltas, stoch_idx=indices,
                   old_logp=old, advantages=torch.linspace(-1, 1, 8), adv_mask=torch.ones(8, dtype=torch.bool))
    before = parameter_hash(actor, trainable=False)
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    initial = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = actor(rollout)
        assert torch.isfinite(loss) and torch.isfinite(metrics['policy_loss'])
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        assert actor._prepared_scope is None
    assert any(not torch.equal(old, parameter) for old, parameter in zip(initial, trainable))
    assert parameter_hash(actor, trainable=False) == before
    assert actor.prepared_context_builds['reference'] == 3
    assert actor.prepared_context_builds['current'] == 5


def test_trainable_context_kv_rebuilt_after_each_update():
    actor = make_actor()
    cond = condition()
    with torch.no_grad():
        chain, times, deltas, indices = actor.sample_action_chain(cond, 8, generator=torch.Generator().manual_seed(4))
    _, first = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=indices, return_terms=True)
    count = actor.prepared_context_builds['current']
    with torch.no_grad():
        actor.action_expert.blocks[0].cross_attn.k.lora_B.add_(0.25)
        actor.action_expert.blocks[0].cross_attn.v.lora_B.add_(0.25)
    _, second = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=indices, return_terms=True)
    assert actor.prepared_context_builds['current'] == count + 1
    assert not torch.equal(first['mean'], second['mean'])
    assert actor._prepared_scope is None


def test_deployment_preserves_lora_execution_and_strict_load():
    actor = make_actor()
    with torch.no_grad():
        for name, parameter in actor.action_expert.named_parameters():
            if name.endswith('lora_B'):
                parameter.normal_(std=0.01)
    deployed = CleanXActor(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=128)
    install_deployment_lora(deployed, expected_cleanx_recipe()['grpo'])
    deployed.load_state_dict(actor.action_expert.state_dict(), strict=True)
    cond = condition(batch=1)
    state, sigma = torch.randn(1, 8, 3), torch.tensor([0.3])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        a = actor.action_expert(state, sigma, actor.action_expert.prepare_context(
            cond['base_memory'], cond['base_valid'], cond['base_ego']))
        b = deployed(state, sigma, deployed.prepare_context(cond['base_memory'], cond['base_valid'], cond['base_ego']))
    assert torch.equal(a, b)
    assert all(not parameter.requires_grad for parameter in deployed.parameters())


def test_context_rejects_future_and_metric_fields():
    actor = make_actor()
    with pytest.raises(ValueError, match='non-deployment'):
        actor.materialize(dict(condition(), GT_future=torch.randn(1, 8, 3)))
    with pytest.raises(ValueError, match='non-deployment'):
        actor.materialize(dict(condition(), PDMS=torch.ones(1)))


@pytest.mark.parametrize('residual_fp32',[False,True])
def test_frozen_current_encoder_cache_has_no_projected_kv_or_labels(residual_fp32):
    class Data:
        tokens, sha = ['a', 'b'], 'data'
        calls = 0
        def observation(self, token):
            self.calls += 1
            return dict(images={}, proprio=[float(token == 'b')] * 8, navigation='go')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen = nn.Linear(8, 32)
            self.scene_provider = nn.Linear(32, 32)
            self.predictive_provider = nn.Linear(32, 32)
        def encode_condition(self, batch, ego):
            base = self.scene_provider(self.qwen(ego))[:, None]
            if residual_fp32:base=base.float()+0.0001
            self.last_base=base.detach()
            future = self.predictive_provider(base)
            valid = torch.ones(1, 192, dtype=torch.bool)
            return StudentMemory(base.expand(-1, 144, -1), base.expand(-1, 16, -1),
                future.expand(-1, 32, -1), valid[:, :144], valid[:, :16], valid[:, :32], ego=ego)

    model, data = Model(), Data()
    conditions = FrozenWorldConditions(model, data, 'parent', processor=lambda observations: {})
    descriptor = conditions.get('a', torch.device('cpu'))
    one = conditions.materialize(descriptor, torch.device('cpu'))
    two = conditions.materialize(descriptor, torch.device('cpu'))
    assert conditions.prefill_calls == data.calls == 1
    assert set(conditions.cached) == {'base_memory', 'base_valid', 'base_ego'}
    assert one['context'].shape == (8, 192, 32)
    assert one['context'].dtype==(torch.float32 if residual_fp32 else torch.bfloat16)
    assert torch.equal(one['base_memory'][:,0],model.last_base[:,0])
    if residual_fp32:assert not torch.equal(one['base_memory'][:,0],model.last_base[:,0].bfloat16().float())
    assert one['context'].data_ptr() == one['base_memory'].data_ptr()
    assert one['context'].stride(0) == 0 and torch.equal(one['context'], two['context'])
    assert not one['base_memory'].requires_grad
    conditions.materialize(conditions.get('b', torch.device('cpu')), torch.device('cpu'))
    assert conditions.prefill_calls == data.calls == 2
    with pytest.raises(ValueError, match='identity changed'):
        conditions.materialize(dict(descriptor, identity='other'), torch.device('cpu'))


def stream_fixture(tmp_path, monkeypatch, *, overlap=False):
    from starVLA.rl.qwen_world_postcot import data
    root = tmp_path/'data'
    (root/'TRAIN').mkdir(parents=True)
    population_path, original_path = tmp_path/'population.json', tmp_path/'source.json'
    tokens = ['t'+str(index) for index in range(7)]
    logs = {token: 'train_0_20' for token in tokens}
    logs.update(d='dev_0_20', n='test_0_20')
    if overlap:
        logs[tokens[-1]] = 'dev_100_120'
    population_path.write_text(json.dumps(dict(token_logs=logs)))
    original_path.write_text(json.dumps(dict(tokens=['d', *reversed(tokens), 'n'])))
    roles = dict(schema='world_postcot_roles_v1', population_sha256=file_sha(population_path),
        train_tokens=tokens, dev_tokens=['d'], navtest_tokens=['n'])
    (root/'DATA_PARTITION.json').write_text(json.dumps(roles))
    train = dict(cohort='TRAIN', tokens=tokens, role_partition_sha256=fingerprint(roles))
    (root/'TRAIN/manifest.json').write_text(json.dumps(train))
    (root/'TRAIN/COMPLETE').write_text(json.dumps(dict(manifest_sha256=file_sha(root/'TRAIN/manifest.json'))))
    manifest = dict(schema='world_postcot_rl_tokens_v1', tokens=list(reversed(tokens)), count=7,
        parent_manifest_sha256=file_sha(original_path), role_partition_sha256=fingerprint(roles),
        train_manifest_sha256=file_sha(root/'TRAIN/manifest.json'))
    path = root/'RL_HARD.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(data, 'POPULATION', population_path)
    monkeypatch.setattr(data, 'SOURCE_HARD_MANIFEST', original_path)
    return path


def test_rl_stream_source_order_padding_and_reconstruction(tmp_path, monkeypatch):
    from starVLA.rl.qwen_world_postcot.data import WorldRLTokenStream
    path = stream_fixture(tmp_path, monkeypatch)
    stream = WorldRLTokenStream(path, 2)
    source = ResumableEpochSampler(stream.tokens, seed=42, batch_size=1, num_processes=2)
    for epoch in range(3):
        source.set_epoch(epoch)
        expected = list(source)
        expected += expected[:1]
        for offset in range(0, 8, 2):
            for rank in range(2):
                token, position = stream.at(epoch * 8 + offset, rank)
                assert token == stream.tokens[expected[offset + rank]]
                assert position == dict(epoch=epoch, index=offset + rank, padding=offset + rank >= 7)
    resumed = WorldRLTokenStream(path, 2)
    assert resumed.at(18, 1) == stream.at(18, 1)
    assert stream.identity['scene_count'] == 7 and stream.identity['physical_log_disjoint']
    with pytest.raises(ValueError, match='cursor'):
        stream.at(3, 0)


def test_rl_stream_rejects_physical_overlap(tmp_path, monkeypatch):
    from starVLA.rl.qwen_world_postcot.data import WorldRLTokenStream
    path = stream_fixture(tmp_path, monkeypatch, overlap=True)
    with pytest.raises(ValueError, match='Physical log leakage'):
        WorldRLTokenStream(path, 2)


def test_rl_stream_rejects_reordered_or_changed_bank(tmp_path, monkeypatch):
    from starVLA.rl.qwen_world_postcot.data import WorldRLTokenStream
    path = stream_fixture(tmp_path, monkeypatch)
    value = json.loads(path.read_text())
    value['tokens'].reverse()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='membership/order'):
        WorldRLTokenStream(path, 2)


def test_explicit_new_recipe_is_loaded_and_velocity_variant_rejected(tmp_path):
    import yaml
    path = tmp_path/'rl.yaml'
    cfg = expected_cleanx_recipe()
    path.write_text(yaml.safe_dump(cfg))
    assert load_cleanx_recipe(path) == cfg
    assert cfg['gradient_clipping_effective'] == 1 and cfg['action_forward_dtype'] == 'bfloat16'
    cfg['prediction_type'] = 'velocity'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='registered complete contract'):
        load_cleanx_recipe(path)
