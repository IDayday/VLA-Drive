"""Native Qwen3 prefix and tail execution with a forward-local differentiable KV.

The prefix retains its original tensor shapes, SDPA implementation, DeepStack
and mRoPE. Only the appended groups read the cache; action/post-action columns
remain masked. No cache survives this observation or an optimizer update.
"""
import torch


def prefix_cached_forward(base, ids, embeddings, valid, positions, visual, deepstack,
                          tail, tail_positions, tail_mask):
    image_mask = ids.eq(base.qwen_vl_interface.model.config.image_token_id)
    embeddings = embeddings.masked_scatter(image_mask[..., None].expand_as(embeddings),
                                           visual.to(embeddings))
    language = base.qwen_vl_interface.model.model.language_model
    if language.training and getattr(language, "gradient_checkpointing", False):
        raise RuntimeError("prefix_cached_v1 requires native language gradient checkpointing disabled; "
                           "do not allow a checkpoint recomputation to mutate the forward-local KV")
    prefix = language(input_ids=None, inputs_embeds=embeddings, attention_mask=valid,
                      position_ids=positions, visual_pos_masks=image_mask,
                      deepstack_visual_embeds=[x.to(embeddings) for x in deepstack], use_cache=True)
    if prefix.past_key_values is None:
        raise RuntimeError("Native Qwen did not return the requested forward-local KV")
    suffix = language(input_ids=None, inputs_embeds=tail, attention_mask=tail_mask,
                      position_ids=tail_positions, past_key_values=prefix.past_key_values,
                      cache_position=torch.arange(ids.shape[1], ids.shape[1]+tail.shape[1], device=ids.device),
                      use_cache=True)
    # No detached copy: representation loss can update selected LoRA in prefix.
    return torch.cat((prefix.last_hidden_state, suffix.last_hidden_state), dim=1)
