"""Behavior tests for direct original-DiT FPO, not driving performance claims."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from starVLA.model.modules.flow_posttraining import (
    CFMSamples, DiTFlowPolicy, FeedbackBatch, FeedbackReuseGuard, FlowRandomStreams,
    FPOConfig, FrozenPolicySnapshot, PolicyIdentity, RolloutBatch, cfm_coordinate_mean,
    fpo_ratio, fpo_surrogate, global_action_count, group_centered_advantages,
    module_hash, sample_cfm, sample_rollout_group,
    CoupledFPOBatch, prepare_fpo_batch,
)


def original_tiny_head():
    # Real production class/attention/ActionEncoder/projection, reduced width
    # solely for a bounded CPU behavior test. Not a substitute for real S0 smoke.
    from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
    head = SimpleNamespace(
        hidden_size=16, DiTConfig=dict(input_embedding_dim=16, num_layers=2,
                                      num_attention_heads=2, attention_head_dim=8),
        diffusion_model_cfg=dict(output_dim=16, cross_attention_dim=16, dropout=.2,
                                 final_dropout=True, interleave_self_attention=True),
        action_dim=4, action_horizon=8, num_inference_timesteps=10,
        noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999,
        num_timestep_buckets=1000, repeated_diffusion_steps=8,
        add_pos_embed=True, max_seq_len=8,
    )
    torch.manual_seed(1207)
    return FlowmatchingActionHead(SimpleNamespace(framework=SimpleNamespace(
        action_model=head, qwenvl=SimpleNamespace(vl_hidden_dim=12))))


def context(batch=2):
    return torch.randn((batch, 3, 12), generator=torch.Generator().manual_seed(199))


def rollouts(policy=None, batch=2):
    policy = DiTFlowPolicy(original_tiny_head()) if policy is None else policy
    old = FrozenPolicySnapshot(policy, version=3, optimizer_step=17, base_binding_hash="a"*64)
    streams = FlowRandomStreams(99)
    rollout = sample_rollout_group(old, context(batch), scene_ids=[f"scene_{i}" for i in range(batch)],
                                   observation_hashes=["b"*64]*batch, noise_generator=streams["rollout_noise"])
    return policy, old, streams, rollout


def feedback(rollout, *, rewards=None):
    rewards = torch.tensor([[0., .2, .8, 1.]]).expand(len(rollout.scene_ids), -1).clone() if rewards is None else rewards
    return FeedbackBatch(rollout.rollout_id, "c"*64, tuple(["d"*64]*len(rollout.scene_ids)),
                         "C1-PDMS", rewards, torch.ones_like(rewards, dtype=torch.bool),
                         ("valid_scored",)*rewards.numel())


def test_official_arithmetic_coordinate_mean_and_exp_order():
    torch.manual_seed(51)
    old = torch.rand(3, 50, dtype=torch.float64)
    current = torch.rand(3, 50, dtype=torch.float64, requires_grad=True)
    config = FPOConfig()
    actual = fpo_ratio(old, current, config)
    reference = torch.exp(torch.clamp(torch.clamp(old-current, -3, 3).mean(dim=1), -3, 3))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    ga = torch.autograd.grad(actual.sum(), current, retain_graph=True)[0]
    gb = torch.autograd.grad(reference.sum(), current)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert not torch.equal(actual, torch.clamp(old-current, -3, 3).exp().mean(1))
    velocity = torch.randn(2, 4, 8, 4, dtype=torch.float64, requires_grad=True)
    endpoint, noise = torch.randn_like(velocity), torch.randn_like(velocity)
    actual_cfm = cfm_coordinate_mean(velocity, endpoint, noise)
    torch.testing.assert_close(actual_cfm, ((velocity-(endpoint-noise))**2).reshape(2, 4, 32).mean(-1), rtol=0, atol=0)
    torch.testing.assert_close(actual_cfm*32, ((velocity-(endpoint-noise))**2).sum((-2,-1)))


def test_actual_head_initial_ratio_one_despite_original_dropout_and_modes_restore():
    policy, old, streams, rollout = rollouts()
    policy.train()
    condition = context().repeat_interleave(4, 0).requires_grad_()
    actions = rollout.raw_actions.flatten(0, 1)
    cfg = FPOConfig(mc_samples=3)
    samples = sample_cfm(actions, cfg, noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    states_before = torch.get_rng_state().clone()
    old_cfm = old(condition, actions, samples, mc_chunk=2)
    current = policy(condition, actions, samples, mc_chunk=2)
    assert torch.equal(torch.get_rng_state(), states_before)
    assert torch.equal(fpo_ratio(old_cfm, current, cfg), torch.ones(8))
    assert policy.training and policy.action_head.model.training
    assert all(not m.training for m in old.modules())
    current.sum().backward()
    assert condition.grad is None
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert all(p.grad is None and not p.requires_grad for p in old.parameters())
    old.verify()


@pytest.mark.parametrize("sign", [1., -1.])
def test_real_dit_gradient_direction_and_parameters_change(sign):
    policy, old, streams, rollout = rollouts(batch=1)
    actions = rollout.raw_actions.flatten(0, 1)
    condition = context(1).repeat_interleave(4, 0)
    cfg = FPOConfig(mc_samples=2)
    draws = sample_cfm(actions, cfg, noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    initial = old(condition, actions, draws)
    before = module_hash(policy)
    current = policy(condition, actions, draws)
    loss, _ = fpo_surrogate(initial, current, torch.full((4,), sign), torch.ones(4, dtype=torch.bool), cfg, global_count=4)
    loss.backward()
    assert policy.action_head.qwen_proj.layer1.weight.grad.norm() > 0
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-5)
    optimizer.step()
    change = (policy(condition, actions, draws).detach().mean()-initial.mean()).item()
    assert change * sign < 0
    assert module_hash(policy) != before
    old.verify()


def test_zero_variance_policy_gradient_keeps_original_beta_repeat8_bc():
    policy, old, streams, rollout = rollouts(batch=1)
    cfg = FPOConfig(mc_samples=2)
    actions = rollout.raw_actions.flatten(0, 1)
    condition = context(1)
    draws = sample_cfm(actions, cfg, noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    adv = group_centered_advantages(torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool)).flatten()
    before_rng = deepcopy(streams.state_dict())
    repeated_condition = condition.repeat_interleave(4, 0)
    current = policy(repeated_condition, actions, draws)
    policy_loss, _ = fpo_surrogate(old(repeated_condition, actions, draws), current, adv,
                                 torch.ones(4, dtype=torch.bool), cfg, global_count=4)
    assert policy_loss.item() == 0
    policy_loss.backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in policy.parameters())
    policy.zero_grad()
    policy.train()
    target = torch.randn(1, 8, 4)
    dropout_state = torch.get_rng_state().clone()
    bc = policy.bc_anchor(condition, target, noise_generator=streams["bc_noise"],
                          time_generator=streams["bc_time"], global_scene_count=1)
    # Execute the original head with exact saved independent noise/time and RNG.
    reference_streams = FlowRandomStreams(99)
    reference_streams.load_state_dict(before_rng)
    y = target.repeat(8, 1, 1)
    eps = torch.randn(y.shape, generator=reference_streams["bc_noise"])
    t = (.999-torch.rand(8, generator=reference_streams["bc_time"]).pow(1/1.5))/.999
    torch.set_rng_state(dropout_state)
    reference = policy.action_head(condition.repeat(8,1,1), y, noise=eps, times=t)
    torch.testing.assert_close(bc, reference, rtol=0, atol=0)
    bc.backward()
    assert sum(p.grad.abs().sum() for p in policy.parameters()) > 0


def test_rollout_actual_ten_steps_projection_once_and_no_clipping():
    policy = DiTFlowPolicy(original_tiny_head())
    with torch.no_grad():
        policy.action_head.action_decoder.layer2.bias.fill_(5.)
    counts = dict(project=0, dit=0)
    old = FrozenPolicySnapshot(policy, version=0, optimizer_step=0, base_binding_hash="a"*64)
    hp = old.policy.action_head.qwen_proj.register_forward_hook(lambda *args: counts.__setitem__("project", counts["project"]+1))
    hd = old.policy.action_head.model.register_forward_hook(lambda *args: counts.__setitem__("dit", counts["dit"]+1))
    batch = sample_rollout_group(old, context(), scene_ids=("a", "b"), observation_hashes=("a"*64,)*2,
                                 noise_generator=torch.Generator().manual_seed(77))
    hp.remove(); hd.remove()
    assert counts == dict(project=1, dit=10)
    assert batch.raw_actions.abs().max() > 1.
    with torch.no_grad():
        reference = old.policy.action_head.predict_action(context().repeat_interleave(4,0),
                                                           initial_noise=batch.initial_noise.flatten(0,1))
    torch.testing.assert_close(batch.raw_actions.flatten(0,1), reference, rtol=0, atol=0)
    restored = RolloutBatch.from_state_dict(batch.state_dict())
    assert restored.rollout_id == batch.rollout_id
    with pytest.raises(TypeError):
        sample_rollout_group(old, context(), scene_ids=("a","b"), observation_hashes=("a"*64,)*2,
                             noise_generator=torch.Generator(), gt=torch.zeros(2,8,4))


def test_checkpoint_velocity_recomputation_keeps_policy_eval_mode():
    plain = DiTFlowPolicy(original_tiny_head()).train()
    saved = deepcopy(plain).train()
    actions = torch.randn(2,8,4)
    cfg = FPOConfig(mc_samples=3)
    streams = FlowRandomStreams(88)
    draws = sample_cfm(actions, cfg, noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    state = torch.get_rng_state().clone()
    plain_loss = plain(context(), actions, draws, mc_chunk=2).sum()
    saved_loss = saved(context(), actions, draws, mc_chunk=2, checkpoint_velocity=True).sum()
    assert saved.training
    plain_loss.backward(); saved_loss.backward()
    torch.testing.assert_close(saved_loss, plain_loss, rtol=0, atol=0)
    for a, b in zip(plain.parameters(), saved.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_rng_coupling_and_resume_are_exact_and_streams_independent():
    streams = FlowRandomStreams(33)
    saved = deepcopy(streams.state_dict())
    reference = FlowRandomStreams(33)
    reference.load_state_dict(saved)
    torch.randn(1000, generator=streams["rollout_noise"])
    actions = torch.zeros(2,8,4)
    first = sample_cfm(actions, FPOConfig(mc_samples=3), noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    second = sample_cfm(actions, FPOConfig(mc_samples=3), noise_generator=reference["policy_noise"], time_generator=reference["policy_time"])
    assert torch.equal(first.noise, second.noise) and torch.equal(first.times, second.times)
    stored = CFMSamples.from_state_dict(first.state_dict(), device="cpu")
    assert torch.equal(stored.noise, first.noise)
    with pytest.raises(ValueError):
        FlowRandomStreams(34).load_state_dict(saved)
    with pytest.raises(ValueError):
        sample_cfm(actions, FPOConfig(mc_samples=2), noise_generator=streams["policy_noise"], time_generator=streams["policy_noise"])


def test_feedback_failures_hash_units_and_stale_reuse_are_rejected():
    _, old, _, rollout = rollouts(batch=1)
    truth = feedback(rollout)
    guard = FeedbackReuseGuard(score_protocol_hash="c"*64)
    kwargs = dict(active_old=old.identity, optimizer_step=17, expected_context_hashes=truth.context_hashes)
    guard.check(rollout, truth, **kwargs)
    with pytest.raises(ValueError, match="protocol/context"):
        guard.check(rollout, truth, **{**kwargs,"expected_context_hashes":("e"*64,)})
    for invalid in (replace(truth, rewards=truth.rewards*100), replace(truth, rollout_id="f"*64),
                    replace(truth, statuses=("transient_scoring_failure",)*4),
                    replace(truth, rewards=truth.rewards.requires_grad_())):
        with pytest.raises(ValueError):
            invalid.validate(rollout)
    truth = feedback(rollout)
    partial = replace(truth, valid=torch.tensor([[1,1,0,1]],dtype=torch.bool),
                      statuses=("valid_scored","valid_scored","transient_scoring_failure","valid_scored"))
    partial.validate(rollout, require_complete=False)
    with pytest.raises(ValueError, match="Incomplete"):
        guard.check(rollout, partial, **kwargs)
    guard.commit_consumed(rollout, truth, **kwargs)
    saved = guard.state_dict()
    recovered = FeedbackReuseGuard(score_protocol_hash="c"*64)
    recovered.load_state_dict(saved)
    with pytest.raises(ValueError, match="already consumed"):
        recovered.check(rollout, truth, **kwargs)
    with pytest.raises(ValueError, match="staleness"):
        recovered.check(rollout, truth, **{**kwargs,"optimizer_step":18})
    with pytest.raises(ValueError, match="Stale policy"):
        recovered.check(rollout, truth, **{**kwargs,"active_old":replace(old.identity,version=4)})
    recovered.retire_before(18)
    with pytest.raises(ValueError, match="staleness"):
        recovered.check(rollout, truth, **kwargs)
    recovered_truth = FeedbackBatch.from_state_dict(truth.state_dict(), rollout)
    assert torch.equal(recovered_truth.rewards, truth.rewards)
    altered = RolloutBatch.from_state_dict(rollout.state_dict())
    altered.raw_actions[0,0,0,0] += .1
    with pytest.raises(ValueError, match="content changed"):
        altered.validate()


def test_frozen_snapshot_strict_state_and_storage_are_preserved():
    policy, old, _, _ = rollouts(batch=1)
    reference = FrozenPolicySnapshot(policy, version=0, optimizer_step=0, base_binding_hash="a"*64, role="reference")
    old_hash = old.identity.state_hash
    with torch.no_grad():
        next(policy.parameters()).add_(.5)
    old.train(True); reference.train(True)
    old.verify(); reference.verify()
    resumed = FrozenPolicySnapshot(policy, version=9, optimizer_step=9, base_binding_hash="a"*64)
    resumed.load_state_dict(old.state_dict())
    assert resumed.identity.state_hash == old_hash
    with pytest.raises(ValueError, match="strict"):
        resumed.load_state_dict(old.state_dict(), strict=False)


def test_saved_cfm_identity_cannot_mix_noise_time_actions_context_or_policy():
    policy,old,streams,rollout=rollouts(batch=1)
    comparison=prepare_fpo_batch(old,rollout,context(1),config=FPOConfig(mc_samples=2),streams=streams)
    restored=CoupledFPOBatch.from_state_dict(comparison.state_dict(),rollout)
    current=restored.current_losses(policy,rollout,context(1))
    assert torch.equal(fpo_ratio(restored.old_cfm,current,restored.config),torch.ones(4))
    with pytest.raises(ValueError,match='context differs'):
        restored.current_losses(policy,rollout,context(1)+1)
    restored.samples.times[0,0]+=.01
    with pytest.raises(ValueError,match='coupling content'):
        restored.current_losses(policy,rollout,context(1))


def _complete_small_update(policy,optimizer,streams,sampler,step):
    ids=torch.randperm(9,generator=sampler)[:1]
    condition=context(1)+ids[:,None,None].float()/10
    old=FrozenPolicySnapshot(policy,version=step,optimizer_step=step,base_binding_hash='a'*64)
    rollout=sample_rollout_group(old,condition,scene_ids=[str(x) for x in ids.tolist()],
                                 observation_hashes=['b'*64],noise_generator=streams['rollout_noise'])
    truth=feedback(rollout)
    paired=prepare_fpo_batch(old,rollout,condition,config=FPOConfig(mc_samples=2),streams=streams)
    advantage=group_centered_advantages(truth.rewards,truth.valid).flatten()
    loss,_=fpo_surrogate(paired.old_cfm,paired.current_losses(policy,rollout,condition),advantage,
                         truth.valid.flatten(),paired.config,global_count=4)
    policy.train()
    target=torch.sin(torch.arange(32).reshape(1,8,4).float()/7+ids[:,None,None])
    bc=policy.bc_anchor(condition,target,noise_generator=streams['bc_noise'],time_generator=streams['bc_time'],
                        global_scene_count=1)
    optimizer.zero_grad(); total=bc+.2*loss; total.backward()
    gradients=[p.grad.detach().clone() for p in policy.parameters()]
    optimizer.step()
    return ids,total.detach(),gradients,rollout.rollout_id,paired.comparison_id


def _assert_state_equal(left,right):
    if torch.is_tensor(left):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: _assert_state_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): _assert_state_equal(a,b)
    else: assert left==right


def test_actual_head_optimizer_dropout_rng_and_consumed_sampler_resume(tmp_path):
    policy=DiTFlowPolicy(original_tiny_head()).train()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    streams=FlowRandomStreams(771)
    sampler=torch.Generator().manual_seed(772)
    _complete_small_update(policy,optimizer,streams,sampler,0)
    saved=dict(model=deepcopy(policy.state_dict()),optimizer=deepcopy(optimizer.state_dict()),
               streams=streams.state_dict(),sampler=sampler.get_state(),torch_rng=torch.get_rng_state(),step=1)
    path=tmp_path/'checkpoint.pt'; torch.save(saved,path)
    expected=_complete_small_update(policy,optimizer,streams,sampler,1)
    expected_rng=torch.get_rng_state().clone()
    resumed=DiTFlowPolicy(original_tiny_head()).train()  # construction changes global RNG deliberately
    new_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-4)
    new_streams=FlowRandomStreams(771)
    new_sampler=torch.Generator()
    loaded=torch.load(path,weights_only=True)
    resumed.load_state_dict(loaded['model'],strict=True)
    new_optimizer.load_state_dict(loaded['optimizer'])
    new_streams.load_state_dict(loaded['streams']); new_sampler.set_state(loaded['sampler'])
    torch.set_rng_state(loaded['torch_rng'])  # restored LAST, never reseeded afterwards
    actual=_complete_small_update(resumed,new_optimizer,new_streams,new_sampler,loaded['step'])
    _assert_state_equal(actual,expected)
    _assert_state_equal(resumed.state_dict(),policy.state_dict())
    _assert_state_equal(new_optimizer.state_dict(),optimizer.state_dict())
    _assert_state_equal(new_streams.state_dict(),streams.state_dict())
    assert torch.equal(new_sampler.get_state(),sampler.get_state())
    assert torch.equal(torch.get_rng_state(),expected_rng)


def test_invalid_and_zero_mask_math_and_configuration():
    cfg = FPOConfig(mc_samples=2)
    old = torch.zeros(3,2)
    current = torch.zeros(3,2,requires_grad=True)
    loss, _ = fpo_surrogate(old,current,torch.full((3,),float('nan')),torch.zeros(3,dtype=torch.bool),cfg,global_count=0)
    loss.backward()
    assert loss.item() == 0 and torch.equal(current.grad,torch.zeros_like(current))
    with pytest.raises(ValueError):
        group_centered_advantages(torch.zeros(2,4),torch.zeros(2,4,dtype=torch.bool))
    with pytest.raises(ValueError):
        fpo_ratio(old.requires_grad_(),current,cfg)
    with pytest.raises(ValueError):
        FPOConfig.from_dict({"mc_samples":2})
    with pytest.raises(ValueError):
        FPOConfig(coordinate_reduction="sum")
    with pytest.raises(ValueError):
        fpo_surrogate(old.detach(),current,torch.zeros(3),torch.ones(3,dtype=torch.bool),cfg,global_count=3,world_size=0)
    with pytest.raises(ValueError):
        fpo_surrogate(old.detach(),current,torch.zeros(3),torch.ones(3,dtype=torch.bool),cfg,global_count=3.5)
    assert FPOConfig.from_dict(cfg.to_dict()) == cfg


class ArithmeticPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.05))
    def forward(self,x):
        return (self.weight*x).square()+.1


def arithmetic_data():
    x=torch.arange(12,dtype=torch.float32).reshape(6,2)/12
    old=(torch.tensor(.04)*x).square()+.1
    advantage=torch.tensor([1.,-.3,.6,-1.,0.,.2])
    valid=torch.tensor([1,1,1,0,0,0],dtype=torch.bool)
    return x,old,advantage,valid


def test_microbatch_sum_matches_global_objective_gradient():
    x,old,adv,valid=arithmetic_data()
    full,micro=ArithmeticPolicy(),ArithmeticPolicy()
    count=global_action_count(valid,distributed=False)
    cfg=FPOConfig(mc_samples=2)
    expected,_=fpo_surrogate(old,full(x),adv,valid,cfg,global_count=count)
    expected.backward()
    value=0.
    for sl in (slice(0,1),slice(1,4),slice(4,6)):
        loss,_=fpo_surrogate(old[sl],micro(x[sl]),adv[sl],valid[sl],cfg,global_count=count)
        loss.backward(); value+=loss.item()
    torch.testing.assert_close(torch.tensor(value),expected.detach())
    torch.testing.assert_close(micro.weight.grad,full.weight.grad)


def _ddp_worker(rank,init_file,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        x,old,adv,valid=arithmetic_data()
        sl=slice(rank*3,(rank+1)*3)
        policy=DistributedDataParallel(ArithmeticPolicy())
        count=global_action_count(valid[sl])
        loss,_=fpo_surrogate(old[sl],policy(x[sl]),adv[sl],valid[sl],FPOConfig(mc_samples=2),global_count=count)
        loss.backward()
        loss_log=loss.detach().clone(); dist.all_reduce(loss_log); loss_log/=2
        torch.save(dict(gradient=policy.module.weight.grad,value=loss_log,count=count),output+str(rank)+'.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_with_entirely_invalid_rank(tmp_path):
    init=str(tmp_path/'rendezvous'); output=str(tmp_path/'rank')
    mp.spawn(_ddp_worker,args=(init,output),nprocs=2,join=True)
    x,old,adv,valid=arithmetic_data(); model=ArithmeticPolicy()
    loss,_=fpo_surrogate(old,model(x),adv,valid,FPOConfig(mc_samples=2),global_count=3)
    loss.backward()
    for rank in range(2):
        actual=torch.load(output+str(rank)+'.pt',weights_only=True)
        torch.testing.assert_close(actual['gradient'],model.weight.grad)
        torch.testing.assert_close(actual['value'],loss.detach())
        assert actual['count'].item()==3
