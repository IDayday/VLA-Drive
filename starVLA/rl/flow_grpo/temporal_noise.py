"""Fixed full-rank AR(1) covariance across trajectory waypoints, not flow steps.

Q[i,j] = rho**abs(i-j), diag(Q)=1. Initial flow noise stays N(0,I).
Only transition innovations use Q; score correction, density and KL must use it
as well. This is an experimental adaptation motivated by correlated exploration,
not a claim to reproduce Colored Noise PPO/Lattice or their likelihood surrogate.
"""
from functools import lru_cache
import math
import torch


def validate_correlation(rho):
    if not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("temporal_noise_correlation must be finite in [0,1)")


@lru_cache(maxsize=32)
def _matrices(horizon, rho, dtype, device):
    validate_correlation(rho)
    if horizon < 1:
        raise ValueError("positive action horizon required")
    positions = torch.arange(horizon, device=device)
    covariance = rho ** (positions[:, None] - positions[None, :]).abs().to(dtype)
    with torch.autocast(device.type, enabled=False):
        factor = torch.linalg.cholesky(covariance)
    return covariance, factor


def matrices(like, rho):
    return _matrices(like.shape[-2], rho, like.dtype, like.device)


def correlate(values, rho, *, covariance=False):
    if rho == 0:
        return values
    q, factor = matrices(values, rho)
    # Never let an enclosing BF16 autocast round probability/noise transforms.
    with torch.autocast(values.device.type, enabled=False):
        return (q if covariance else factor) @ values


def whiten(values, rho):
    if rho == 0:
        return values
    _, factor = matrices(values, rho)
    with torch.autocast(values.device.type, enabled=False):
        return torch.linalg.solve_triangular(factor, values, upper=False)


def shared_horizon_std(std, mean):
    expanded = torch.broadcast_to(std, mean.shape)
    if not torch.equal(expanded, expanded[..., :1, :].expand_as(expanded)):
        raise ValueError("correlated transitions require waypoint-independent std")
    return expanded
