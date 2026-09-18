"""Explicit denoising-step weighting of the GRPO surrogate, not new rewards.

Time increases noise -> action in DDP. DPPO/SimWAM/ReCogDrive motivate later-step
credit. Raw discount matches their temporal weighting; an explicit mean-one
normalization is available for diagnostics. This is a weighted surrogate, not
an unbiased full-chain PG claim. Reference KL and SFT replay remain separate.
"""
import math
import torch


def validate_discount(gamma):
    if (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
            or not math.isfinite(gamma) or not 0 < gamma <= 1):
        raise ValueError("denoising_discount must be finite in (0,1]")


def transition_credit_weights(valid, gamma=1.0, *, dtype=torch.float32, normalization="raw_discount"):
    """Return B,G,K nonnegative weights with an explicit normalization.

    Masked steps keep their original time index; neither candidate rewards nor
    old probabilities are changed. gamma=1 exactly preserves the original mean.
    """
    validate_discount(gamma)
    if normalization not in ("raw_discount", "scene_mean_one"):
        raise ValueError("unknown denoising credit normalization")
    if valid.ndim != 3 or valid.dtype != torch.bool or valid.shape[-1] == 0:
        raise ValueError("credit requires a boolean [scene,candidate,step] mask")
    count = valid.sum((1, 2), keepdim=True)
    if (count == 0).any():
        raise ValueError("scene has no valid transitions")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("credit weights require probability precision")
    weights = valid.to(dtype=dtype)
    if gamma == 1:
        return weights
    exponent = torch.arange(valid.shape[-1]-1, -1, -1, device=valid.device)
    discount = torch.pow(torch.tensor(gamma, device=valid.device, dtype=dtype), exponent)
    if not torch.isfinite(discount).all() or (discount <= 0).any():
        raise ValueError("denoising_discount underflows; no step may silently disappear")
    weights = weights * discount
    if normalization == "scene_mean_one":
        weights = weights * (count / weights.sum((1, 2), keepdim=True))
    return weights
