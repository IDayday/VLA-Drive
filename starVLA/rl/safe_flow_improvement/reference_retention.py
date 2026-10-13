"""Original SFT replay and fixed-safe-path velocity retention (not ODE KL)."""
import torch

from .contracts import seed_for
from .policy_adapter import encode_condition, original_fm_loss
from .weighted_flow_objective import interpolation, reference_velocity_loss
from starVLA.rl.flow_grpo.rollout import velocity


def fm_draws(head, target, seed, *, repeats=8, nft=False):
    """Independent interpolation draws; candidate generation noise is absent."""
    if repeats != 8 or target.ndim != 3:
        raise ValueError('Keep the registered original eight FM repeats')
    repeated = target.detach().float().repeat(repeats, 1, 1)
    generator = torch.Generator(device=target.device).manual_seed(int(seed))
    noise = torch.randn(repeated.shape, device=target.device, dtype=torch.float32, generator=generator)
    if nft:
        # NFT uses shuffled actual sampler times. The original DDP Euler
        # velocity evaluations are s=0,.1,...,.9, mapped from image t=1-s.
        indices = torch.randperm(10, device=target.device, generator=generator)[:repeats]
        time = (indices.float()/10).repeat_interleave(len(target))
    else:
        with torch.random.fork_rng(devices=[target.device.index]):
            torch.manual_seed(seed_for('original_sft_beta_time', seed, 'fm'))
            time = head.sample_time(len(repeated), target.device, torch.float32)
    return repeated, noise, time


def original_replay_loss(policy, raw, target, seed):
    y, noise, time = fm_draws(policy.action_model, target, seed)
    return original_fm_loss(policy, raw.repeat(8, 1, 1), y, noise, time)


def safe_path_retention(policy, reference, raw, reference_target, seed):
    y, noise, time = fm_draws(policy.action_model, reference_target, seed)
    x, _ = interpolation(y, noise, time)
    bucket = (time*policy.action_model.num_timestep_buckets).long()
    repeated = raw.detach().repeat(8, 1, 1)
    with torch.no_grad():
        fixed = velocity(reference, x, bucket, encode_condition(reference, repeated), use_graph=False)
    current = velocity(policy, x, bucket, encode_condition(policy, repeated), use_graph=False)
    return reference_velocity_loss(current, fixed, scenes=len(raw), repeats=8)[0]
