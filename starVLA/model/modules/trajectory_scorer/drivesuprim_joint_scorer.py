# SPDX-License-Identifier: Apache-2.0
# Adapted from William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02, files
# navsim/agents/drivesuprim/drivesuprim_model.py and drivesuprim_config.py.
# Project adaptations: one shared scorer handles a fixed static vocabulary and
# detached Flow-DiT candidates; donor same-width/spatial attention is replaced
# by layer-local asymmetric candidate-to-Qwen-scene attention.  The official
# heads, aggregate formula, global Top-K, and one-stage refinement are retained.

"""DriveSuprim unified static/dynamic coarse scorer and fine refiner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from starVLA.model.modules.planning.types import CandidateMetadata

from .attention import AsymmetricDecoder
from .losses import SUPRIM_METRICS, aggregate_drivesuprim_score


def _score_head(
    model_dim: int, ffn_dim: int, *, imitation: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(model_dim, ffn_dim), nn.ReLU()]
    if imitation:
        layers.extend((nn.Linear(ffn_dim, ffn_dim), nn.ReLU()))
    layers.append(nn.Linear(ffn_dim, 1))
    return nn.Sequential(*layers)


def _score_heads(model_dim: int, ffn_dim: int, use_imitation: bool) -> nn.ModuleDict:
    heads = nn.ModuleDict(
        {name: _score_head(model_dim, ffn_dim) for name in SUPRIM_METRICS}
    )
    if use_imitation:
        heads["imi"] = _score_head(model_dim, ffn_dim, imitation=True)
    return heads


def _gather_candidates(candidates: Tensor, indices: Tensor) -> Tensor:
    if candidates.ndim != 4 or indices.ndim != 2:
        raise ValueError("candidate gather expects [B,N,T,3] and [B,K]")
    return torch.gather(
        candidates,
        1,
        indices[..., None, None].expand(
            -1, -1, candidates.shape[-2], candidates.shape[-1]
        ),
    )


def _gather_states(states: Tensor, indices: Tensor) -> Tensor:
    if states.ndim != 3 or indices.ndim != 2:
        raise ValueError("state gather expects [B,N,D] and [B,K]")
    return torch.gather(states, 1, indices[..., None].expand(-1, -1, states.shape[-1]))


def _gather_logits(logits: Dict[str, Tensor], indices: Tensor) -> Dict[str, Tensor]:
    return {name: torch.gather(value, 1, indices) for name, value in logits.items()}


@dataclass
class DriveSuprimCoarseOutput:
    """All unified-pool outputs plus globally selected Top-K tensors."""

    metric_logits: Dict[str, Tensor]
    aggregate_score: Tensor
    joint_candidates_40: Tensor
    candidate_states: Tensor
    metadata: CandidateMetadata
    topk_indices: Tensor
    topk_metric_logits: Dict[str, Tensor]
    topk_trajectories_40: Tensor
    topk_candidate_states: Tensor
    topk_metadata: CandidateMetadata


@dataclass
class DriveSuprimFineOutput:
    """Intermediate fine-layer predictions and final selected original candidate."""

    layer_metric_logits: List[Dict[str, Tensor]]
    layer_candidate_states: List[Tensor]
    aggregate_score: Tensor
    selected_topk_index: Tensor
    selected_trajectory_40: Tensor
    selected_absolute_index: Tensor
    selected_source: Tensor
    selected_source_index: Tensor


class DriveSuprimCoarseScorer(nn.Module):
    """Score all static and dynamic candidates with one shared coarse model.

    Candidate geometry is physical ``[x,y,heading]`` at 0.1-second intervals.
    Dynamic geometry is detached before concatenation.  Candidate provenance is
    returned solely as integer metadata and is never passed to a learned layer.
    """

    def __init__(
        self,
        *,
        vocab_path: Optional[str] = None,
        static_vocab: Optional[Tensor] = None,
        vocab_size: int = 8192,
        num_poses: int = 40,
        scene_dim: int = 2048,
        ego_state_dim: int = 4,
        model_dim: int = 256,
        ffn_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        coarse_topk: int = 256,
        dropout: float = 0.0,
        normalize_vocab_pos: bool = False,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or num_poses <= 0 or coarse_topk <= 0:
            raise ValueError("vocabulary and Top-K sizes must be positive")
        self.vocab_size = int(vocab_size)
        self.num_poses = int(num_poses)
        self.scene_dim = int(scene_dim)
        self.ego_state_dim = int(ego_state_dim)
        self.model_dim = int(model_dim)
        self.coarse_topk = int(coarse_topk)
        self.normalize_vocab_pos = bool(normalize_vocab_pos)
        self.debug_validate_finite = bool(debug_validate_finite)
        self.register_buffer(
            "static_vocab",
            self._load_static_vocab(vocab_path, static_vocab),
            persistent=False,
        )
        self.candidate_embedding = nn.Sequential(
            nn.Linear(num_poses * 3, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim),
        )
        # Donor normalize_vocab_pos is an optional candidate interaction block.
        # It is intentionally isolated and disabled in the production config.
        self.vocab_normalizer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            if normalize_vocab_pos
            else None
        )
        self.coarse_decoder = AsymmetricDecoder(
            num_layers=num_layers,
            query_dim=model_dim,
            memory_dim=scene_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            return_intermediate=False,
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim),
        )
        self.metric_heads = _score_heads(model_dim, ffn_dim, use_imitation=True)

    def _load_static_vocab(
        self, vocab_path: Optional[str], static_vocab: Optional[Tensor]
    ) -> Tensor:
        if static_vocab is None:
            if not vocab_path:
                raise FileNotFoundError(
                    "DriveSuprim hierarchical scorer requires joint.vocab_path"
                )
            path = Path(vocab_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"DriveSuprim vocabulary not found: {path}")
            if path.suffix == ".npy":
                static_vocab = torch.from_numpy(np.load(path, allow_pickle=False))
            else:
                loaded = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(loaded, dict):
                    for key in ("vocab", "trajectory_vocab", "static_vocab"):
                        if key in loaded:
                            loaded = loaded[key]
                            break
                static_vocab = torch.as_tensor(loaded)
        if not torch.is_tensor(static_vocab):
            raise TypeError("static vocabulary must be a tensor or supported file")
        expected = (self.vocab_size, self.num_poses, 3)
        if tuple(static_vocab.shape) != expected:
            raise ValueError(
                f"static vocabulary shape is {tuple(static_vocab.shape)}, expected {expected}"
            )
        vocab = static_vocab.detach().to(dtype=torch.float32).contiguous()
        if not torch.isfinite(vocab).all():
            raise ValueError("static vocabulary contains NaN or Inf")
        return vocab

    def _ego_token(self, ego_state: Tensor, reference: Tensor) -> Tensor:
        if ego_state is None:
            raise ValueError("DriveSuprim scorer requires ego_state")
        if ego_state.ndim == 3 and ego_state.shape[1] == 1:
            ego_state = ego_state[:, 0]
        expected = (reference.shape[0], self.ego_state_dim)
        if ego_state.ndim != 2 or tuple(ego_state.shape) != expected:
            raise ValueError(
                f"ego_state must have shape {expected} or [B,1,{self.ego_state_dim}], "
                f"got {tuple(ego_state.shape)}"
            )
        return self.ego_encoder(
            ego_state.to(device=reference.device, dtype=reference.dtype)
        )[:, None]

    def build_joint_pool(
        self,
        batch_size: int,
        reference: Tensor,
        dynamic_trajectories_40: Optional[Tensor] = None,
        dynamic_candidate_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, CandidateMetadata]:
        """Return an expanded static vocabulary plus optional detached dynamics."""

        static = self.static_vocab.to(
            device=reference.device, dtype=reference.dtype
        )[None].expand(batch_size, -1, -1, -1)
        static_ids = torch.arange(
            self.vocab_size, device=reference.device, dtype=torch.long
        )[None].expand(batch_size, -1)
        sources = torch.zeros_like(static_ids)
        if dynamic_trajectories_40 is None:
            metadata = CandidateMetadata(sources, static_ids, static_ids)
            metadata.validate(batch_size, self.vocab_size)
            return static, metadata

        if (
            dynamic_trajectories_40.ndim != 4
            or dynamic_trajectories_40.shape[0] != batch_size
            or tuple(dynamic_trajectories_40.shape[-2:]) != (self.num_poses, 3)
        ):
            raise ValueError(
                f"dynamic trajectories must have shape [B,M,{self.num_poses},3]"
            )
        dynamic = dynamic_trajectories_40.detach().to(
            device=reference.device, dtype=reference.dtype
        )
        dynamic_count = dynamic.shape[1]
        local_ids = torch.arange(
            dynamic_count, device=reference.device, dtype=torch.long
        )[None].expand(batch_size, -1)
        if dynamic_candidate_ids is None:
            dynamic_candidate_ids = local_ids
        else:
            if tuple(dynamic_candidate_ids.shape) != (batch_size, dynamic_count):
                raise ValueError("dynamic_candidate_ids must have shape [B,M]")
            dynamic_candidate_ids = dynamic_candidate_ids.to(
                device=reference.device, dtype=torch.long
            )
        absolute_ids = self.vocab_size + local_ids
        metadata = CandidateMetadata(
            source=torch.cat((sources, torch.ones_like(local_ids)), dim=1),
            source_index=torch.cat((static_ids, dynamic_candidate_ids), dim=1),
            absolute_index=torch.cat((static_ids, absolute_ids), dim=1),
        )
        joint = torch.cat((static, dynamic), dim=1)
        metadata.validate(batch_size, joint.shape[1])
        return joint, metadata

    def forward(
        self,
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        *,
        dynamic_trajectories_40: Optional[Tensor] = None,
        dynamic_candidate_ids: Optional[Tensor] = None,
        coarse_topk: Optional[int] = None,
    ) -> DriveSuprimCoarseOutput:
        """Run one global Top-K over the complete static/dynamic candidate pool."""

        if global_scene_tokens.ndim != 3 or global_scene_tokens.shape[-1] != self.scene_dim:
            raise ValueError(
                f"global_scene_tokens must have shape [B,S,{self.scene_dim}]"
            )
        batch_size = global_scene_tokens.shape[0]
        candidates, metadata = self.build_joint_pool(
            batch_size,
            global_scene_tokens,
            dynamic_trajectories_40,
            dynamic_candidate_ids,
        )
        topk = self.coarse_topk if coarse_topk is None else int(coarse_topk)
        if topk <= 0 or topk > candidates.shape[1]:
            raise ValueError(
                f"coarse_topk={topk} exceeds joint candidate count {candidates.shape[1]}"
            )
        states = self.candidate_embedding(
            candidates.reshape(batch_size, candidates.shape[1], self.num_poses * 3)
        )
        if self.vocab_normalizer is not None:
            states = self.vocab_normalizer(states)
        states = self.coarse_decoder(states, global_scene_tokens)
        states = states + self._ego_token(ego_state, states)
        metric_logits = {
            name: head(states).squeeze(-1) for name, head in self.metric_heads.items()
        }
        aggregate = aggregate_drivesuprim_score(metric_logits, include_imitation=True)
        if self.debug_validate_finite and (
            not torch.isfinite(states).all() or not torch.isfinite(aggregate).all()
        ):
            raise ValueError("DriveSuprim coarse scorer produced NaN or Inf")
        topk_indices = torch.topk(aggregate, k=topk, dim=1).indices
        return DriveSuprimCoarseOutput(
            metric_logits=metric_logits,
            aggregate_score=aggregate,
            joint_candidates_40=candidates,
            candidate_states=states,
            metadata=metadata,
            topk_indices=topk_indices,
            topk_metric_logits=_gather_logits(metric_logits, topk_indices),
            topk_trajectories_40=_gather_candidates(candidates, topk_indices),
            topk_candidate_states=_gather_states(states, topk_indices),
            topk_metadata=metadata.gather(topk_indices),
        )


class DriveSuprimFineRefiner(nn.Module):
    """One DriveSuprim refinement stage with per-layer auxiliary predictions."""

    def __init__(
        self,
        *,
        scene_dim: int = 2048,
        model_dim: int = 256,
        ffn_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_mid_output: bool = True,
        use_imitation: bool = True,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if not use_mid_output:
            raise ValueError(
                "joint training requires use_mid_output=true so every fine layer is supervised"
            )
        self.scene_dim = int(scene_dim)
        self.model_dim = int(model_dim)
        self.use_imitation = bool(use_imitation)
        self.debug_validate_finite = bool(debug_validate_finite)
        self.fine_decoder = AsymmetricDecoder(
            num_layers=num_layers,
            query_dim=model_dim,
            memory_dim=scene_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            return_intermediate=True,
        )
        # DriveSuprim shares one refinement-stage head set across its layers.
        self.metric_heads = _score_heads(model_dim, ffn_dim, use_imitation)

    def forward(
        self,
        coarse_output: DriveSuprimCoarseOutput,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
    ) -> DriveSuprimFineOutput:
        """Refine only global Top-K candidate states against dense Qwen memory."""

        if dense_scene_memory.ndim != 3 or dense_scene_memory.shape[-1] != self.scene_dim:
            raise ValueError(
                f"dense_scene_memory must have shape [B,L,{self.scene_dim}]"
            )
        if memory_key_padding_mask is not None:
            if tuple(memory_key_padding_mask.shape) != tuple(dense_scene_memory.shape[:2]):
                raise ValueError("fine memory padding mask must have shape [B,L]")
            if memory_key_padding_mask.dtype is not torch.bool:
                raise TypeError("fine memory padding mask must be boolean")
        layer_states = self.fine_decoder(
            coarse_output.topk_candidate_states,
            dense_scene_memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        layer_logits = [
            {name: head(states).squeeze(-1) for name, head in self.metric_heads.items()}
            for states in layer_states
        ]
        aggregate = aggregate_drivesuprim_score(
            layer_logits[-1], include_imitation=self.use_imitation
        )
        if self.debug_validate_finite and (
            any(not torch.isfinite(states).all() for states in layer_states)
            or not torch.isfinite(aggregate).all()
        ):
            raise ValueError("DriveSuprim fine refiner produced NaN or Inf")
        selected_topk_index = aggregate.argmax(dim=1)
        rows = torch.arange(
            aggregate.shape[0], device=aggregate.device, dtype=torch.long
        )
        return DriveSuprimFineOutput(
            layer_metric_logits=layer_logits,
            layer_candidate_states=layer_states,
            aggregate_score=aggregate,
            selected_topk_index=selected_topk_index,
            selected_trajectory_40=coarse_output.topk_trajectories_40[
                rows, selected_topk_index
            ],
            selected_absolute_index=coarse_output.topk_metadata.absolute_index[
                rows, selected_topk_index
            ],
            selected_source=coarse_output.topk_metadata.source[
                rows, selected_topk_index
            ],
            selected_source_index=coarse_output.topk_metadata.source_index[
                rows, selected_topk_index
            ],
        )
