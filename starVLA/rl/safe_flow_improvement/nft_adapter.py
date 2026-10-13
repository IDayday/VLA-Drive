"""DDP time/sign adaptation of the official DiffusionNFT forward objective.

Source: NVlabs/DiffusionNFT cbb14f84b8312b620390dfcbe2fab69c1383b104,
scripts/train_nft_sd3.py, advantage transform and lines 934--960.
The official image path uses x_t=(1-t)*clean+t*noise and v=noise-clean.
DDP uses s=1-t and v_DDP=-v_image. Its reconstructed clean sample is
x_s + (1-s)*v_DDP. Frozen old velocity must be from the sampling behavior.
"""

import torch


UPSTREAM_COMMIT = "cbb14f84b8312b620390dfcbe2fab69c1383b104"


def nft_forward_loss(current_velocity, old_velocity, noisy_action, clean_action,
                     time_to_data, advantages, *, beta=1.0, advantage_clip=5.0):
    if beta <= 0 or advantage_clip <= 0:
        raise ValueError("positive NFT beta and advantage clip required")
    shape = clean_action.shape
    if any(x.shape != shape for x in (current_velocity, old_velocity, noisy_action)):
        raise ValueError("NFT tensor shapes differ")
    if time_to_data.shape != (len(clean_action),) or advantages.shape != (len(clean_action),):
        raise ValueError("one NFT time/advantage per FM draw required")
    if ((time_to_data < 0) | (time_to_data > 1)).any():
        raise ValueError("NFT mapped time outside 0..1")
    if any(not torch.isfinite(x).all() for x in (current_velocity, old_velocity, noisy_action, clean_action, time_to_data, advantages)):
        raise ValueError("nonfinite NFT input")
    old = old_velocity.detach()
    clean = clean_action.detach()
    noisy = noisy_action.detach()
    positive = beta * current_velocity + (1 - beta) * old
    negative = (1 + beta) * old - beta * current_velocity
    noise_fraction = (1 - time_to_data).reshape(-1, *([1] * (clean.ndim - 1)))
    predicted_positive = noisy + noise_fraction * positive
    predicted_negative = noisy + noise_fraction * negative
    dimensions = tuple(range(1, clean.ndim))
    with torch.no_grad():
        positive_scale = (predicted_positive.double() - clean.double()).abs().mean(dimensions, keepdim=True).clamp(min=1e-5)
        negative_scale = (predicted_negative.double() - clean.double()).abs().mean(dimensions, keepdim=True).clamp(min=1e-5)
    positive_loss = ((predicted_positive - clean).square() / positive_scale).mean(dimensions)
    negative_loss = ((predicted_negative - clean).square() / negative_scale).mean(dimensions)
    fraction = (advantages.detach().clamp(-advantage_clip, advantage_clip) / advantage_clip / 2 + 0.5).clamp(0, 1)
    per_draw = (fraction * positive_loss + (1 - fraction) * negative_loss) * advantage_clip / beta
    return per_draw.mean(), {"per_draw": per_draw, "positive": positive_loss, "negative": negative_loss, "positive_fraction": fraction}
