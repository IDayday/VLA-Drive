"""Explicit S0 W48 -> W96 migration; not a function-preserving expansion.

Original token names and vocabulary rows are retained. Within each view the old
48 memory slots precede 48 new slots. New slots share the original view/position
initialization prior but have independent learned queries. W coordinates are a
memory identity, not a claim of image-patch or BEV correspondence.
"""
from __future__ import annotations

import torch
from torch import nn

from starVLA.model.modules.vehicle_joint.initialization import (
    add_random_driving_tokens, initialization_seed, tensor_hash,
)


class ExpandedMemoryEncoding(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.positions.shape != (48, 2) or original.view_ids.numel() != 3:
            raise ValueError("Only the audited S0 three-view 48-memory-slot parent is supported")
        self.view = original.view
        self.position = original.position
        self.register_buffer("positions", original.positions.detach().repeat(2, 1))
        self.register_buffer("view_ids", original.view_ids.detach().clone())

    def forward(self, queries):
        hidden = self.view.embedding_dim
        if queries.shape != (288, hidden):
            raise ValueError("Expanded memory requires 96 slots per original view")
        pos = self.position(self.positions.to(dtype=self.position.weight.dtype))
        offsets = self.view(self.view_ids)[:, None] + pos[None]
        return queries + offsets.reshape_as(queries)


def expand_world_queries(model, queries_per_view: int, *, seed: int) -> dict:
    """Mutate a newly owned, strictly loaded S0 instance once; never a live run."""
    if queries_per_view not in (48, 96):
        raise ValueError("The registered capacity comparison is W48 versus W96")
    if (len(model.foresight_tokens) != 144 or model.foresight_queries.shape[0] != 144
            or model.query_geometry.positions.shape != (48, 2)):
        raise ValueError("Original S0 W/token inventory differs; no implicit remapping")
    old_queries = model.foresight_queries
    old_tokens = list(model.foresight_tokens)
    old_rows = list(model._special_token_ids["foresight"])
    common = {"schema": "st_world_memory_migration_v1", "source_queries_per_view": 48,
              "target_queries_per_view": queries_per_view, "seed": seed,
              "original_query_sha256": tensor_hash(old_queries),
              "original_token_names": old_tokens, "original_vocabulary_rows": old_rows}
    if queries_per_view == 48:
        return {**common, "changed": False, "new_vocabulary_rows": [],
                "old_to_new_query_rows": list(range(144)),
                "initial_function_equivalence": "requires fixed-input real audit"}

    tokenizer = model.qwen_vl_interface.processor.tokenizer
    vlm = model.qwen_vl_interface.model
    added = [f"<st_world_extra_v{v}_{i}>" for v in range(3) for i in range(48)]
    token_record = add_random_driving_tokens(vlm, tokenizer, added, seed)
    old_to_new = torch.cat([torch.arange(48) + 96 * v for v in range(3)])
    new_rows = torch.cat([torch.arange(48) + 96 * v + 48 for v in range(3)])
    with initialization_seed(seed + 1), torch.no_grad():
        expanded = torch.empty((288, old_queries.shape[1]),
                               device=old_queries.device, dtype=old_queries.dtype)
        expanded[old_to_new.to(expanded.device)] = old_queries
        expanded[new_rows.to(expanded.device)] = torch.randn(
            (144, old_queries.shape[1]), device=old_queries.device,
            dtype=old_queries.dtype) * 0.02
    model.foresight_queries = nn.Parameter(expanded, requires_grad=old_queries.requires_grad)
    model.query_geometry = ExpandedMemoryEncoding(model.query_geometry)
    ordered = []
    for view in range(3):
        ordered.extend(old_tokens[view * 48:(view + 1) * 48])
        ordered.extend(added[view * 48:(view + 1) * 48])
    model.foresight_tokens = ordered
    model._special_token_ids["foresight"] = tuple(tokenizer.convert_tokens_to_ids(ordered))
    if [tokenizer.convert_tokens_to_ids(t) for t in old_tokens] != old_rows:
        raise ValueError("Original vocabulary IDs changed during expansion")
    if not torch.equal(model.foresight_queries[old_to_new.to(expanded.device)], old_queries):
        raise RuntimeError("Original learned W values were not retained exactly")
    return {**common, "changed": True, "new_token_initialization": token_record,
            "new_vocabulary_rows": list(token_record["tokens"].values()),
            "old_to_new_query_rows": old_to_new.tolist(),
            "new_query_rows": new_rows.tolist(), "runtime_token_order": ordered,
            "expanded_query_sha256": tensor_hash(expanded),
            "initial_function_equivalence": False,
            "reason": "Longer causal W sequence changes action conditioning"}
