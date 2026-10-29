"""Directional P/A/R visibility.  Boolean True means *allowed* in SDPA."""
import torch


def reasoning_mask(p_tokens=32, action_tokens=8, r_tokens=16, *, device=None):
    """Return [P,A,R] self-attention visibility; C uses scene cross-attention.

    P sees P; A sees P/A; R sees P/A/R.  Each role is bidirectional within
    itself.  There is no dependence on candidate values or flow sigma.
    """
    if min(p_tokens, action_tokens, r_tokens) < 0 or p_tokens == 0:
        raise ValueError('Reasoning lengths must be nonnegative with nonempty P')
    roles = torch.cat((torch.zeros(p_tokens, dtype=torch.long, device=device),
                       torch.ones(action_tokens, dtype=torch.long, device=device),
                       torch.full((r_tokens,), 2, dtype=torch.long, device=device)))
    return roles[:, None] >= roles[None, :]


def expand_key_valid(allowed, key_valid):
    """Combine a query-specific [Q,K] allow mask and [B,K] key validity."""
    if allowed.ndim != 2 or allowed.dtype != torch.bool:
        raise ValueError('Expected a boolean [Q,K] allowed mask')
    if key_valid.ndim != 2 or key_valid.dtype != torch.bool or key_valid.shape[1] != allowed.shape[1]:
        raise ValueError('Expected aligned boolean [B,K] validity')
    return allowed[None] & key_valid[:, None]
