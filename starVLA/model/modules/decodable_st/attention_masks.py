"""Qwen3 tail groups: preserve prefix, prohibit action-to-world edges."""
import torch


def world_attention_mask(valid_prefix, action_positions, spatial_count, temporal_count, dtype):
    b, n = valid_prefix.shape
    total = n + spatial_count + temporal_count
    causal = torch.ones(total, total, dtype=torch.bool, device=valid_prefix.device).tril()
    allowed = causal.unsqueeze(0).expand(b, -1, -1).clone()
    valid = torch.cat((valid_prefix.bool(), torch.ones(b, total-n, device=valid_prefix.device, dtype=torch.bool)), -1)
    allowed &= valid[:, None, :]
    # Padded query rows cannot affect any valid row; give them their own diagonal
    # to avoid all-masked NaNs while keeping padded keys invisible elsewhere.
    diag = torch.arange(total, device=valid_prefix.device)
    allowed[:, diag, diag] |= ~valid
    if total > n:
        rows = torch.arange(b, device=valid_prefix.device)[:, None]
        for q in range(n, total):
            allowed[rows, q, action_positions] = False
    additive = torch.zeros(b, 1, total, total, dtype=dtype, device=valid_prefix.device)
    return additive.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def append_mrope_positions(position_ids, valid_prefix, count):
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("Expected Qwen3 mRoPE [3,B,L] positions")
    # Text tail positions follow the maximum valid multimodal position.
    maximum = position_ids.masked_fill(~valid_prefix.bool()[None], -1).amax((0, 2))
    tail = maximum[:, None] + 1 + torch.arange(count, device=position_ids.device)[None]
    return torch.cat((position_ids, tail[None].expand(3, -1, -1)), -1)
