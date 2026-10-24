"""Collect true on-policy root samples through the MAIN agent's native runtime."""
import torch
from .gaussian import GaussianParameters, sample_roots, joint_log_prob
from .policy import condition_features
from .training import LatentBehaviorBatch
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


@torch.no_grad()
def collect_behavior(env, policy, tokens, *, split, candidates, generator,
                     proposal_source, policy_version, advantage_callback, training_seed=None):
    """One frozen behavior collection; no old-chain or external search imports.

    Conditions are materialized one scene at a time. Only pooled observations,
    roots and old densities stay in the learner batch, avoiding B full KV sets.
    The environment owns all model/scorer/cost records and canonical semantics.
    """
    if not tokens or candidates <= 0:
        raise ValueError('A nonempty fixed scene collection is required')
    if env.model.training or any(p.requires_grad for p in env.model.parameters()):
        raise ValueError('Latent exploration requires frozen VLM/provider/DiT')
    digest = tensor_digest(policy.state_dict())
    env.set_behavior_identity(dict(experiment=env.experiment_id, version=policy_version,
                                  root_policy_sha256=digest, generator='permanently_frozen_common_base',
                                  deployment='q_sampled_root_then_native_shift5_ODE'))
    features, roots, means, log_stds, old_logps, rewards, records = [], [], [], [], [], [], []
    for token in tokens:
        condition = env.condition(token, split)
        feature = condition_features(condition)
        params = policy(feature).detached()
        root = sample_roots(params, candidates, generator)
        env.ledger.add(latent_root_proposal_attempts=candidates, latent_network_scene_forwards=1)
        old_logp = joint_log_prob(params, root).detach()
        # NativeEnvironment owns chunk4 broadcast. Passing an already-expanded
        # G-scene condition here would repeat it again inside each chunk.
        action = env.sample_ode(condition, root.reshape(candidates, 8, 3))
        metadata = [dict(root=root[0, i].cpu().tolist(), old_mean=params.mean[0].cpu().tolist(),
                         old_log_std=params.log_std[0].cpu().tolist(), old_joint_log_probability=float(old_logp[0, i]),
                         policy_version=policy_version, behavior_digest=digest,
                         innovation_generator_seed=int(generator.initial_seed()),
                         innovation_rng_device=str(generator.device),
                         root_probability='joint24D Gaussian before unmodified source ODE',
                         internal_randomness='empty for ODE') for i in range(candidates)]
        feedback = env.score_actions(token, action, split, proposal_source=proposal_source,
                                     latent_or_proposal_parameters=metadata, seed=training_seed)
        if len(feedback) != candidates or any(row['status'] != 'OK' for row in feedback):
            raise ValueError('Infrastructure failure retained; no zero or missing-candidate fallback')
        reward = torch.tensor([[row['raw_reward'] for row in feedback]], device=feature.device, dtype=torch.float32)
        if not torch.isfinite(reward).all() or (reward < 0).any() or (reward > 1).any():
            raise ValueError('Canonical reward must be finite original0-1')
        features.append(feature.cpu())
        roots.append(root.cpu())
        means.append(params.mean.cpu())
        log_stds.append(params.log_std.cpu())
        old_logps.append(old_logp.cpu())
        rewards.append(reward.cpu())
        records.extend(feedback)
        # Never accumulate full scene memories in a learner buffer. The common
        # environment may independently maintain its explicitly bounded LRU.
        del condition, action
    if digest != tensor_digest(policy.state_dict()):
        raise ValueError('Behavior root policy changed during fresh collection')
    device = next(policy.parameters()).device
    batch = LatentBehaviorBatch.from_collected(policy,
        torch.cat(features).to(device), torch.cat(roots).to(device), torch.cat(rewards).to(device),
        old_parameters=GaussianParameters(torch.cat(means).to(device), torch.cat(log_stds).to(device)),
        old_joint_logp=torch.cat(old_logps).to(device), advantage_callback=advantage_callback)
    env.ledger.add(latent_network_scene_forwards=len(tokens))
    return batch, records


@torch.no_grad()
def single_deployment(env, policy, token, *, split, generator, readout='sample',
                      proposal_source, policy_version, evaluation_seed):
    """One output, no teacher/scorer selection; mean-root is a distinct scheme."""
    if readout not in ('sample', 'mean'):
        raise ValueError('Lock sampled single and mean-root single separately')
    env.set_behavior_identity(dict(experiment=env.experiment_id, version=policy_version,
                                  root_policy_sha256=tensor_digest(policy.state_dict()),
                                  generator='permanently_frozen_common_base', readout=readout))
    condition = env.condition(token, split)
    params = policy(condition_features(condition)).detached()
    roots = (sample_roots(params, 1, generator)[:, 0] if readout == 'sample' else params.mean)
    actions = env.sample_ode(condition, roots)
    metadata = [dict(root=roots[0].cpu().tolist(), mean=params.mean[0].cpu().tolist(),
                     log_std=params.log_std[0].cpu().tolist(), policy_version=policy_version,
                     readout='q_sampled_single' if readout == 'sample' else 'q_mean_root_single',
                     single_candidate=True)]
    return env.score_actions(token, actions, split, proposal_source=proposal_source,
                             latent_or_proposal_parameters=metadata, seed=evaluation_seed)
