"""Source MoT action-only execution with external observation KV.

The external provider replaces the video expert. No dummy video model exists.
Unpadded inputs call the original source cache method verbatim. The padded path
only changes mask shape/indexing; all action attention/post-block functions are
inherited from the source MoT.
"""
import torch
from torch import nn
from .vendor.mot import MoT


class CachedActionRunner(MoT):
    def __init__(self, action_expert, checkpoint_attention=False):
        nn.Module.__init__(self)
        self.mixtures = nn.ModuleDict({'action': action_expert})
        self.expert_order = ['action']
        self.num_layers = len(action_expert.blocks)
        self.num_heads = action_expert.num_heads
        self.attn_head_dim = action_expert.attn_head_dim
        self.mot_checkpoint_mixed_attn = checkpoint_attention
        self.detach_video_kv_for_action = False

    @property
    def action_expert(self):
        return self.mixtures['action']

    def forward(self, actions, timestep, condition):
        expert = self.action_expert
        pre = expert.pre_dit(actions, timestep, condition['context'], condition['context_mask'])
        kv = condition['visual_kv']
        valid = condition['visual_mask']
        if valid.ndim != 2 or len(kv) != self.num_layers:
            raise ValueError('External visual KV/mask contract')
        b, sv = valid.shape
        sa = actions.shape[1]
        if b != actions.shape[0]:
            raise ValueError('Condition/action batch mismatch')
        context = dict(context=pre['context'], mask=pre['context_mask'])
        if bool(valid.all()):
            mask = torch.ones((sv + sa, sv + sa), device=actions.device, dtype=torch.bool)
            x = self.forward_action_with_video_cache(pre['tokens'], pre['freqs'], pre['t_mod'],
                context, kv, mask, sv)
        else:
            # SDPA boolean mask is [B,1,Sa,Sv+Sa]; only action query rows run.
            mask = torch.cat([valid, torch.ones((b, sa), device=valid.device, dtype=torch.bool)], -1)
            mask = mask[:, None, None, :].expand(-1, 1, sa, -1)
            x = pre['tokens']
            for block, cache in zip(expert.blocks, kv):
                q, k, v, residual, g1, shift, scale, g2, cp = self._build_expert_attention_io(
                    expert, block, x, pre['freqs'], pre['t_mod'])
                if cache['k'].shape != (b, sv, self.num_heads * self.attn_head_dim):
                    raise ValueError('External K shape mismatch')
                mixed = self._mixed_attention(q, torch.cat([cache['k'], k], 1),
                    torch.cat([cache['v'], v], 1), mask)
                x = self._apply_post_with_optional_checkpoint(block, residual, g1, shift,
                    scale, g2, cp, mixed, context)
        return expert.post_dit(x, pre)


def expand_condition(condition, repeats):
    """Scene-major broadcast, used only within a frozen/current condition graph."""
    def expand(x):
        return x.repeat_interleave(repeats, dim=0)
    return dict(context=expand(condition['context']), context_mask=expand(condition['context_mask']),
                visual_mask=expand(condition['visual_mask']),
                visual_kv=[{k: expand(v) for k, v in layer.items()} for layer in condition['visual_kv']])


def condition_fp32(condition):
    """Explicit frozen-BF16-prefill -> FP32 action boundary, masks unchanged.

    This is outside IL autocast and never recomputes or truncates Qwen tokens.
    Casting once per scene also avoids repeated KV promotion in ten ODE steps.
    """
    return dict(context=condition['context'].float(),context_mask=condition['context_mask'],
        visual_mask=condition['visual_mask'],
        visual_kv=[{k:v.float() for k,v in layer.items()} for layer in condition['visual_kv']])
