"""Attention-only language LoRA and causal internal task queries (no vocabulary change)."""
import math
import torch
from torch import nn


def configure_language_attention(language_model, backend='eager'):
    """Use the installed Qwen2 native dispatcher; preserve RoPE, GQA and causal padding.

    No global monkey-patch, extra dependency, parameter replacement or silent fallback.
    SDPA chooses a supported fused CUDA kernel for the actual mask/dtype.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM
    if backend not in ('eager', 'sdpa'):
        raise ValueError('Language attention backend must be eager or sdpa')
    if not isinstance(language_model, Qwen2ForCausalLM):
        raise TypeError('Language backend switch requires the inspected native Qwen2ForCausalLM')
    layers = language_model.model.layers
    if not layers or any(not isinstance(layer.self_attn, Qwen2Attention) for layer in layers):
        raise TypeError('Unexpected Qwen2 attention topology; refusing an unverified backend switch')
    language_model.set_attn_implementation(backend)
    configs = [language_model.config, language_model.model.config] + [layer.self_attn.config for layer in layers]
    if any(config._attn_implementation != backend for config in configs):
        raise RuntimeError('Language attention backend was not applied to every actual decoder layer')
    result = dict(backend=backend, model_class=type(language_model).__name__,
                  attention_class=type(layers[0].self_attn).__name__, layers=len(layers))
    print('V2 actual language attention:', result)
    return result


class LanguageAttentionLoRA(nn.Module):
    def __init__(self, base, rank=32, alpha=64):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Language attention projection must be Linear")
        self.base_layer = base
        base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        base = self.base_layer(x)
        residual = self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype)))
        return base + residual.to(base.dtype) * self.scale


def inject_language_lora(language_model, rank=32, alpha=64):
    matched = []
    # Scope is exclusively the actual LLM, never the enclosing VLM/vision tower.
    for name, parent in list(language_model.named_modules()):
        if not all(hasattr(parent, field) for field in ('q_proj', 'k_proj', 'v_proj', 'o_proj')):
            continue
        for field in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            base = getattr(parent, field)
            if isinstance(base, LanguageAttentionLoRA):
                raise RuntimeError("Refusing duplicate language adapter topology")
            setattr(parent, field, LanguageAttentionLoRA(base, rank, alpha))
            matched.append(name + '.' + field)
    if not matched:
        raise RuntimeError("No actual LLM attention q/k/v/o projection matched")
    print('V2 language LoRA modules:', matched)
    return matched


def append_task_queries(prefix, valid_mask, queries):
    """Compact each valid prefix, append queries, then right pad. Never attend padding."""
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask True means a real prefix token")
    lengths = valid_mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("Empty LLM prefix")
    q = queries.shape[0]
    sequences = [torch.cat((row[mask], queries.to(row.dtype)), 0) for row, mask in zip(prefix, valid_mask)]
    embeds = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    positions = torch.arange(embeds.shape[1], device=prefix.device)[None].expand(len(prefix), -1)
    mask = positions < (lengths + q)[:, None]
    indices = lengths[:, None] + torch.arange(q, device=prefix.device)[None]
    return embeds, mask, positions.masked_fill(~mask, 0), indices


class SoftTaskQueries(nn.Module):
    def __init__(self, hidden_size, dim=256):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(16, hidden_size))
        nn.init.normal_(self.queries, std=.02)
        self.projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))

    def forward(self, language_model, prefix, valid_mask):
        embeds, mask, positions, indices = append_task_queries(prefix, valid_mask, self.queries)
        result = language_model.model(inputs_embeds=embeds, attention_mask=mask,
            position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
        hidden = result.last_hidden_state
        selected = hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))
        return self.projection(selected.float())
