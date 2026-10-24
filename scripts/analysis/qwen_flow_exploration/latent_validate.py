"""Finite CPU arithmetic and native tiny-generator qualification for root PPO."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal, Independent, kl_divergence
from starVLA.rl.qwen_flow_exploration.latent.gaussian import GaussianParameters, joint_log_prob, joint_kl, sample_roots
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy, condition_features
from starVLA.rl.qwen_flow_exploration.latent.frozen import freeze_generator, frozen_digest, decode_roots
from starVLA.rl.qwen_flow_exploration.latent.objective import latent_ppo_loss
from starVLA.rl.qwen_flow_exploration.latent.training import LatentBehaviorBatch, train_collected_batch
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.source_advantages import advantages as source_advantages


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    checks = []
    for b in [1, 2]:
        for g in [1, 16]:
            mean = torch.randn(b, 8, 3, dtype=torch.float64)
            log_std = torch.randn_like(mean)*.2
            params = GaussianParameters(mean, log_std)
            roots = sample_roots(params, g, torch.Generator().manual_seed(31))
            actual = joint_log_prob(params, roots)
            reference = Independent(Normal(mean[:, None], log_std[:, None].exp()), 2).log_prob(roots)
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
            other = GaussianParameters(mean*.3, log_std*.7)
            correct_kl = kl_divergence(Independent(Normal(mean, log_std.exp()), 2),
                                      Independent(Normal(other.mean, other.log_std.exp()), 2))
            torch.testing.assert_close(joint_kl(params, other), correct_kl, rtol=1e-12, atol=1e-12)
    checks.append('B1/2 G1/16 exact24D log density and analytic KL match torch Independent Normal')
    old = GaussianParameters(torch.zeros(1,8,3), torch.zeros(1,8,3))
    probes = sample_roots(old, 16, torch.Generator().manual_seed(93))
    old_lp = joint_log_prob(old, probes)
    assert not torch.equal(joint_log_prob(GaussianParameters(old.mean+.01, old.log_std), probes), old_lp)
    assert not torch.equal(joint_log_prob(GaussianParameters(old.mean, old.log_std+.01), probes), old_lp)
    checks.append('Changing only mean or only standard deviation changes the actual joint ratio')
    condition = dict(context=torch.randn(2, 5, 40), context_mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]], dtype=torch.bool),
        visual_mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]], dtype=torch.bool),
        visual_kv=[dict(k=torch.randn(2, 7, 24), v=torch.randn(2, 7, 24)) for _ in range(3)])
    features = condition_features(condition)
    changed = copy.deepcopy(condition)
    changed['context'][0, 3:] = 1e9
    changed['visual_kv'][-1]['v'][0, 4:] = -1e9
    assert torch.equal(condition_features(changed), features)
    checks.append('Pooled legal observation features ignore padding and retain no generator graph')
    policy = LatentNoisePolicy(features.shape[-1])
    global_policy = LatentNoisePolicy(features.shape[-1], conditional=False)
    for actor in [policy, global_policy]:
        params = actor(features)
        assert not params.mean.any() and not params.log_std.any()
        assert not joint_kl(params, params.standard_prior()).any()
        roots = sample_roots(params, 16, torch.Generator().manual_seed(32))
        assert not roots.requires_grad
        old = joint_log_prob(params.detached(), roots).detach()
        assert torch.equal((joint_log_prob(params, roots)-old).exp(), torch.ones_like(old))
    checks.append('Conditional/global initialization is exact original N0I, joint prior KL0 and old/current ratio1')
    old_params = policy(features).detached()
    old_logp = joint_log_prob(old_params, roots).detach()
    constant = LatentBehaviorBatch.from_collected(policy, features, roots, torch.full((2, 16), .9),
        old_parameters=old_params, old_joint_logp=old_logp, advantage_callback=source_advantages)
    assert not constant.advantages.any()
    zero_loss, metrics = latent_ppo_loss(policy(features), roots, constant.old_joint_logp,
        constant.advantages, old_parameters=constant.old_parameters, clip_range=.2, prior_coefficient=.01)
    assert float(zero_loss) == 0 and metrics['PG'] == 0
    checks.append('Constant groups have exact zero advantage/loss at prior, without fake signal or NaN')
    # Known score-function gradients, without clipping or group standardization.
    mean = torch.full((1, 8, 3), .2, dtype=torch.float64, requires_grad=True)
    log_std = torch.full((1, 8, 3), .3, dtype=torch.float64, requires_grad=True)
    parameters = GaussianParameters(mean, log_std)
    samples = sample_roots(parameters, 131072, torch.Generator().manual_seed(91))
    reward = -(samples-.5).square().mean((-1,-2))
    centered = reward-reward.mean(1, keepdim=True)
    loss = -(centered.detach()*joint_log_prob(parameters, samples)).mean()
    loss.backward()
    expected_mu = 2*(mean.detach()-.5)/24
    expected_std = 2*torch.exp(2*log_std.detach())/24
    assert float((mean.grad-expected_mu).abs().max()) < .006
    assert float((log_std.grad-expected_std).abs().max()) < .008
    assert (mean.grad < 0).all() and (log_std.grad > 0).all()
    checks.append('Independent unclipped score-function test: mean moves toward optimum and excessive variance decreases')
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    rewards = -(roots-.5).square().mean((-1,-2))
    batch = LatentBehaviorBatch.from_collected(policy, features, roots, rewards,
        old_parameters=old_params, old_joint_logp=old_logp, advantage_callback=source_advantages)
    batch_hash = batch.digest()
    baseline_policy = copy.deepcopy(policy)
    before = tensor_digest(policy.state_dict())
    continuous = train_collected_batch(policy, optimizer, batch, clip_range=.2, prior_coefficient=.01)
    assert len(continuous) == 4 and tensor_digest(policy.state_dict()) != before and batch.digest() == batch_hash
    resumed_policy = copy.deepcopy(baseline_policy)
    resumed_optimizer = torch.optim.Adam(resumed_policy.parameters(), lr=1e-4)
    class Interrupted(Exception): pass
    saved = {}
    def interrupt(inner, behavior, metrics):
        if inner == 1:
            saved['policy'] = copy.deepcopy(resumed_policy.state_dict())
            saved['optimizer'] = copy.deepcopy(resumed_optimizer.state_dict())
            raise Interrupted()
    try:
        train_collected_batch(resumed_policy, resumed_optimizer, batch, checkpoint_callback=interrupt)
    except Interrupted:
        pass
    resumed_policy.load_state_dict(saved['policy'])
    resumed_optimizer.load_state_dict(saved['optimizer'])
    train_collected_batch(resumed_policy, resumed_optimizer, batch, start_inner=1)
    assert tensor_digest(resumed_policy.state_dict()) == tensor_digest(policy.state_dict())
    assert tensor_digest(resumed_optimizer.state_dict()) == tensor_digest(optimizer.state_dict())
    checks.append('Four real optimizer steps keep old batch fixed; inner1 interruption resumes bitwise policy/Adam state')
    head = ActionDiT(action_dim=3, hidden_dim=32, ffn_dim=64, text_dim=40, freq_dim=16,
                     num_heads=2, attn_head_dim=12, num_layers=3, eps=1e-6, use_gradient_checkpointing=False)
    model = nn.Module()
    model.policy = SourceFlowPolicy(head)
    model.register_buffer('frozen_qualification_buffer', torch.tensor([3.]))
    model.register_buffer('nonpersistent_rotary_test_buffer', torch.tensor([7.]), persistent=False)
    freeze_generator(model)
    frozen_before = frozen_digest(model)
    model.nonpersistent_rotary_test_buffer.add_(1)
    assert frozen_digest(model) != frozen_before
    model.nonpersistent_rotary_test_buffer.sub_(1)
    for b in [1, 2]:
        cond = {k:(v[:b] if torch.is_tensor(v) else [{name:t[:b] for name,t in layer.items()} for layer in v]) for k,v in condition.items()}
        for g in [1, 16]:
            root = torch.randn(b, g, 8, 3)
            actual = decode_roots(model, cond, root)
            expected = model.policy.sample_ode(expand_condition(cond, g), root.reshape(b*g, 8, 3)).reshape(b,g,8,3)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert frozen_before == frozen_digest(model)
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    checks.append('Wrapper matches actual source native tiny ActionDiT shift5 ODE for B1/2 G1/16; parameters AND buffers unchanged')
    result = dict(status='PASS', qualification_only=True, checks=checks,
        generator_scope='real native tiny ActionDiT; not full driving model qualification',
        driving_efficacy=False, score_function_max_abs_error=dict(
            mean=float((mean.grad-expected_mu).abs().max()), log_std=float((log_std.grad-expected_std).abs().max())),
        native_time_grid='source shift5 K10; ODE xi is empty', exact_root_dimension=24,
        four_inner_metrics=continuous)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), output=str(output))))


if __name__ == '__main__':
    main()
