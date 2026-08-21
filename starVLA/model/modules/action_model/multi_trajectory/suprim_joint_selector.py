# SPDX-License-Identifier: Apache-2.0
# Adapted from William-Yao-2000/DriveSuprim, commit
# 80fe792d7654a596d92e20d030d1650f6f605c02:
#   navsim/agents/drivesuprim/drivesuprim_model.py
#   navsim/agents/drivesuprim/drivesuprim_config.py
# Compatibility changes: appends detached DDP candidates to the official
# static vocabulary, propagates candidate provenance, and replaces donor
# same-width/spatial attention with layer-local 256-query/2048-memory
# asymmetric attention.  Score heads, aggregate formulae, global Top-256, and
# the one-stage three-layer intermediate refinement contract are retained.

"""DriveSuprim static-dynamic joint selector for DDP-DRS."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .asymmetric_decoder import AsymmetricTransformerDecoder
from .candidate_types import CandidateMetadata, JointSelectorOutput
from .config import PlanningConfig, SuprimConfig
from .scene_context import SceneContext
from .trajectory_resampler import STATIC_SAMPLE_INDICES


COARSE_SCORE_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
    "imi",
)
FINE_SCORE_NAMES = COARSE_SCORE_NAMES[:-1]


def _score_head(
    planning_dim: int, ffn_dim: int, imitation: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(planning_dim, ffn_dim), nn.ReLU()]
    if imitation:
        layers.extend((nn.Linear(ffn_dim, ffn_dim), nn.ReLU()))
    layers.append(nn.Linear(ffn_dim, 1))
    return nn.Sequential(*layers)


def _make_score_heads(
    planning_dim: int, ffn_dim: int, include_imitation: bool
) -> nn.ModuleDict:
    heads = nn.ModuleDict(
        {
            name: _score_head(planning_dim, ffn_dim)
            for name in FINE_SCORE_NAMES
        }
    )
    if include_imitation:
        heads["imi"] = _score_head(planning_dim, ffn_dim, imitation=True)
    return heads


def _stable_log(values: Tensor) -> Tensor:
    return torch.log(values.clamp_min(torch.finfo(values.dtype).tiny))


def drivesuprim_coarse_aggregate(scores: Dict[str, Tensor]) -> Tensor:
    """Numerically stable expression of DriveSuprim's coarse formula."""

    missing = set(COARSE_SCORE_NAMES).difference(scores)
    if missing:
        raise KeyError(f"DriveSuprim coarse scores missing keys: {sorted(missing)}")
    weighted = (
        5.0 * torch.sigmoid(scores["time_to_collision_within_bound"])
        + 5.0 * torch.sigmoid(scores["ego_progress"])
        + 2.0 * torch.sigmoid(scores["lane_keeping"])
        + torch.sigmoid(scores["history_comfort"])
    )
    return (
        0.02 * F.log_softmax(scores["imi"], dim=-1)
        + 0.1 * F.logsigmoid(scores["traffic_light_compliance"])
        + 0.5 * F.logsigmoid(scores["no_at_fault_collisions"])
        + 0.5 * F.logsigmoid(scores["drivable_area_compliance"])
        + 0.3 * F.logsigmoid(scores["driving_direction_compliance"])
        + 6.0 * _stable_log(weighted)
    )


def drivesuprim_fine_aggregate(scores: Dict[str, Tensor]) -> Tensor:
    """Numerically stable expression of DriveSuprim's fine formula."""

    missing = set(FINE_SCORE_NAMES).difference(scores)
    if missing:
        raise KeyError(f"DriveSuprim fine scores missing keys: {sorted(missing)}")
    weighted = (
        5.0 * torch.sigmoid(scores["time_to_collision_within_bound"])
        + 5.0 * torch.sigmoid(scores["ego_progress"])
        + 2.0 * torch.sigmoid(scores["lane_keeping"])
        + torch.sigmoid(scores["history_comfort"])
    )
    aggregate = (
        0.1 * F.logsigmoid(scores["traffic_light_compliance"])
        + 0.5 * F.logsigmoid(scores["no_at_fault_collisions"])
        + 0.5 * F.logsigmoid(scores["drivable_area_compliance"])
        + 0.3 * F.logsigmoid(scores["driving_direction_compliance"])
        + 6.0 * _stable_log(weighted)
    )
    if "imi" in scores:
        aggregate = aggregate + 0.02 * F.log_softmax(scores["imi"], dim=-1)
    return aggregate


class HydraTrajHead(nn.Module):
    """DriveSuprim candidate embedding, coarse decoder, and metric heads."""

    def __init__(
        self,
        num_poses: int,
        planning_dim: int,
        memory_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._num_poses = num_poses
        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, planning_dim),
        )
        self.transformer = AsymmetricTransformerDecoder(
            num_layers=num_layers,
            planning_dim=planning_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            return_intermediate=False,
        )
        self.heads = _make_score_heads(
            planning_dim, ffn_dim, include_imitation=True
        )

    def forward(
        self,
        candidates: Tensor,
        scene_memory: Tensor,
        status_encoding: Tensor,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        batch_size, candidate_count, horizon, _ = candidates.shape
        candidate_states = self.pos_embed(
            candidates.reshape(batch_size, candidate_count, horizon * 3)
        )
        coarse_states = self.transformer(
            tgt=candidate_states,
            memory=scene_memory,
            memory_key_padding_mask=None,
        )
        trajectory_status = coarse_states + status_encoding[:, None, :]
        scores = {
            name: head(trajectory_status).squeeze(-1)
            for name, head in self.heads.items()
        }
        return scores, drivesuprim_coarse_aggregate(scores), trajectory_status


class RefineTrajHead(nn.Module):
    """One DriveSuprim refinement stage with three asymmetric layers."""

    def __init__(
        self,
        planning_dim: int,
        memory_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
        use_mid_output: bool,
        use_imitation: bool,
    ) -> None:
        super().__init__()
        self.use_mid_output = use_mid_output
        self.transformer = AsymmetricTransformerDecoder(
            num_layers=num_layers,
            planning_dim=planning_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            return_intermediate=True,
        )
        self.heads = _make_score_heads(
            planning_dim, ffn_dim, include_imitation=use_imitation
        )

    def forward(
        self,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        trajectory_status: Tensor,
    ) -> Tuple[List[Dict[str, Tensor]], Tensor, Tensor]:
        layer_features = self.transformer(
            tgt=trajectory_status,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        layer_results = [
            {
                name: head(features).squeeze(-1)
                for name, head in self.heads.items()
            }
            for features in layer_features
        ]
        if not self.use_mid_output:
            layer_results = layer_results[-1:]
        final_aggregate = drivesuprim_fine_aggregate(layer_results[-1])
        return layer_results, final_aggregate, final_aggregate.argmax(dim=1)


class DriveSuprimJointSelector(nn.Module):
    """Run one global static+dynamic coarse selection and fine refinement."""

    def __init__(
        self,
        config: SuprimConfig,
        *,
        planning_config: Optional[PlanningConfig] = None,
        scene_dim: int = 2048,
        ego_status_dim: Optional[int] = None,
        static_vocab: Optional[Tensor] = None,
        enforce_fidelity: bool = True,
    ) -> None:
        super().__init__()
        if enforce_fidelity:
            config.validate()
        planning = planning_config or PlanningConfig()
        self.enforce_fidelity = enforce_fidelity
        self.config = config
        self.planning_config = planning
        self.scene_dim = int(scene_dim)
        self.ego_status_dim = ego_status_dim

        self.register_buffer(
            "static_vocab", self._load_vocab(config, static_vocab), persistent=True
        )
        self.status_encoding = (
            nn.Linear(ego_status_dim, planning.planning_dim)
            if ego_status_dim is not None
            else None
        )
        self._trajectory_head = HydraTrajHead(
            num_poses=config.num_trajectory_points,
            planning_dim=planning.planning_dim,
            memory_dim=scene_dim,
            num_heads=planning.num_heads,
            ffn_dim=planning.ffn_dim,
            num_layers=config.coarse_layers,
            dropout=planning.dropout,
        )
        self._trajectory_offset_head = RefineTrajHead(
            planning_dim=planning.planning_dim,
            memory_dim=scene_dim,
            num_heads=planning.num_heads,
            ffn_dim=planning.ffn_dim,
            num_layers=config.refinement_layers,
            dropout=planning.dropout,
            use_mid_output=config.use_mid_output,
            use_imitation=config.use_imitation_head,
        )
        self.last_latency_coarse: Optional[float] = None
        self.last_latency_refinement: Optional[float] = None
        self.profile_latency = False

    def _timing_start(self, reference: Tensor) -> float:
        if self.profile_latency and reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter()

    def _timing_end(self, reference: Tensor, started: float) -> float:
        if self.profile_latency and reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter() - started

    @staticmethod
    def _load_vocab(config: SuprimConfig, static_vocab: Optional[Tensor]) -> Tensor:
        if static_vocab is None:
            if config.vocab_path is None:
                raise FileNotFoundError(
                    "DriveSuprim static vocabulary path is required when enabled"
                )
            path = Path(config.vocab_path)
            if not path.is_file():
                raise FileNotFoundError(f"DriveSuprim vocabulary not found: {path}")
            static_vocab = torch.from_numpy(np.load(path, allow_pickle=False))
        if not torch.is_tensor(static_vocab):
            raise TypeError("DriveSuprim vocabulary must be a tensor or .npy array")
        expected = (config.vocab_size, config.num_trajectory_points, 3)
        if tuple(static_vocab.shape) != expected:
            raise ValueError(
                f"DriveSuprim vocabulary has shape {tuple(static_vocab.shape)}, "
                f"expected {expected}"
            )
        vocab = static_vocab.detach().to(dtype=torch.float32).contiguous()
        if not torch.isfinite(vocab).all():
            raise ValueError("DriveSuprim vocabulary contains NaN or Inf")
        return vocab

    def build_joint_candidates(
        self,
        batch_size: int,
        dynamic_traj40: Optional[Tensor] = None,
        dynamic_candidate_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, CandidateMetadata]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        static = self.static_vocab[None].expand(batch_size, -1, -1, -1)
        static_count = static.shape[1]
        static_indices = torch.arange(
            static_count, device=static.device, dtype=torch.long
        )[None].expand(batch_size, -1)
        static_source = torch.zeros_like(static_indices)
        if dynamic_traj40 is None:
            metadata = CandidateMetadata(static_source, static_indices, None)
            metadata.validate(torch.Size((batch_size, static_count)))
            return static, metadata

        if (
            dynamic_traj40.ndim != 4
            or dynamic_traj40.shape[0] != batch_size
            or dynamic_traj40.shape[-2:]
            != (self.config.num_trajectory_points, 3)
        ):
            raise ValueError("dynamic_traj40 must have shape [B, M, 40, 3]")
        dynamic = dynamic_traj40.detach().to(
            device=static.device, dtype=static.dtype
        )
        dynamic_count = dynamic.shape[1]
        dynamic_indices = torch.arange(
            dynamic_count, device=static.device, dtype=torch.long
        )[None].expand(batch_size, -1)
        if dynamic_candidate_ids is None:
            dynamic_candidate_ids = dynamic_indices
        elif tuple(dynamic_candidate_ids.shape) != (batch_size, dynamic_count):
            raise ValueError("dynamic_candidate_ids must have shape [B, M]")
        else:
            dynamic_candidate_ids = dynamic_candidate_ids.to(
                device=static.device, dtype=torch.long
            )
        metadata = CandidateMetadata(
            source=torch.cat((static_source, torch.ones_like(dynamic_indices)), 1),
            source_index=torch.cat((static_indices, dynamic_indices), 1),
            dynamic_candidate_id=torch.cat(
                (torch.full_like(static_indices, -1), dynamic_candidate_ids), 1
            ),
        )
        joint = torch.cat((static, dynamic), dim=1)
        metadata.validate(joint.shape[:2])
        return joint, metadata

    def _encode_status(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        ego_status: Optional[Tensor],
    ) -> Tensor:
        if self.status_encoding is None:
            return torch.zeros(
                batch_size,
                self.planning_config.planning_dim,
                device=device,
                dtype=dtype,
            )
        if ego_status is None:
            raise ValueError("DriveSuprim selector requires ego_status")
        current = (
            ego_status
            if ego_status.ndim == 2
            else ego_status[:, -1].flatten(start_dim=1)
            if ego_status.ndim >= 3
            else None
        )
        if current is None or tuple(current.shape) != (
            batch_size,
            self.ego_status_dim,
        ):
            raise ValueError("ego_status shape does not match configured status input")
        return self.status_encoding(current.to(device=device, dtype=dtype))

    @staticmethod
    def resolve_selected_trajectory_8(
        selected_trajectory_40: Tensor,
        selected_source: Tensor,
        selected_source_index: Tensor,
        dynamic_traj8: Optional[Tensor],
    ) -> Tensor:
        if selected_trajectory_40.ndim != 3 or selected_trajectory_40.shape[-2:] != (
            40,
            3,
        ):
            raise ValueError("selected_trajectory_40 must have shape [B, 40, 3]")
        batch_size = selected_trajectory_40.shape[0]
        result = selected_trajectory_40[:, list(STATIC_SAMPLE_INDICES), :].clone()
        dynamic_mask = selected_source == 1
        if dynamic_mask.any():
            if dynamic_traj8 is None:
                raise ValueError("dynamic metadata has no original dynamic_traj8")
            if dynamic_traj8.ndim != 4 or dynamic_traj8.shape[0] != batch_size:
                raise ValueError("dynamic_traj8 must have shape [B, M, 8, 3]")
            rows = torch.arange(batch_size, device=selected_source.device)[dynamic_mask]
            columns = selected_source_index[dynamic_mask]
            if (columns >= dynamic_traj8.shape[1]).any():
                raise ValueError("dynamic source_index exceeds candidate count")
            dynamic_source = dynamic_traj8.detach().to(
                device=result.device, dtype=result.dtype
            )
            result[dynamic_mask] = dynamic_source[rows, columns]
        return result

    def _fine_memory(
        self, scene_context: SceneContext
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.config.fine_memory_source == "dense_qwen_memory":
            return (
                scene_context.dense_scene_memory,
                scene_context.memory_key_padding_mask,
            )
        if self.config.fine_memory_source == "global_scene_tokens":
            return scene_context.global_scene_tokens, None
        raise ValueError(
            f"unsupported fine_memory_source={self.config.fine_memory_source!r}"
        )

    def forward(
        self,
        scene_context: SceneContext,
        dynamic_traj8: Optional[Tensor] = None,
        dynamic_traj40: Optional[Tensor] = None,
        dynamic_candidate_ids: Optional[Tensor] = None,
        ego_status: Optional[Tensor] = None,
    ) -> JointSelectorOutput:
        if self.enforce_fidelity:
            scene_context.validate(
                expected_num_queries=16,
                expected_scene_dim=self.scene_dim,
                check_finite=self.config.debug_validate_finite,
            )
        global_memory = scene_context.global_scene_tokens
        if global_memory.ndim != 3 or global_memory.shape[-1] != self.scene_dim:
            raise ValueError(
                f"global scene memory must have shape [B, S, {self.scene_dim}]"
            )
        if scene_context.dense_scene_memory.shape[-1] != self.scene_dim:
            raise ValueError("dense scene memory width does not match selector")
        if (dynamic_traj8 is None) != (dynamic_traj40 is None):
            raise ValueError("dynamic_traj8 and dynamic_traj40 must be supplied together")
        batch_size = global_memory.shape[0]
        if dynamic_traj8 is not None:
            if dynamic_traj8.ndim != 4 or dynamic_traj8.shape != (
                batch_size,
                dynamic_traj8.shape[1],
                8,
                3,
            ):
                raise ValueError("dynamic_traj8 must have shape [B, M, 8, 3]")
            if dynamic_traj8.shape[:2] != dynamic_traj40.shape[:2]:
                raise ValueError("dynamic 8/40 candidate counts differ")
            try:
                torch.testing.assert_close(
                    dynamic_traj40[..., list(STATIC_SAMPLE_INDICES), :],
                    dynamic_traj8,
                    rtol=1e-4,
                    atol=1e-5,
                )
            except AssertionError as error:
                raise ValueError("dynamic 8/40 time convention mismatch") from error

        joint_candidates, metadata = self.build_joint_candidates(
            batch_size,
            dynamic_traj40=dynamic_traj40,
            dynamic_candidate_ids=dynamic_candidate_ids,
        )
        candidate_count = joint_candidates.shape[1]
        if self.config.coarse_topk > candidate_count:
            raise ValueError(
                f"DriveSuprim Top-k {self.config.coarse_topk} exceeds candidate "
                f"count {candidate_count}"
            )
        joint_candidates = joint_candidates.to(
            device=global_memory.device, dtype=global_memory.dtype
        )
        metadata = CandidateMetadata(
            source=metadata.source.to(global_memory.device),
            source_index=metadata.source_index.to(global_memory.device),
            dynamic_candidate_id=(
                None
                if metadata.dynamic_candidate_id is None
                else metadata.dynamic_candidate_id.to(global_memory.device)
            ),
        )
        metadata.validate(joint_candidates.shape[:2])
        status = self._encode_status(
            batch_size, global_memory.device, global_memory.dtype, ego_status
        )

        coarse_start = self._timing_start(global_memory)
        coarse_scores, aggregate_coarse, candidate_status = self._trajectory_head(
            joint_candidates,
            global_memory,
            status,
        )
        if not torch.isfinite(aggregate_coarse).all():
            raise ValueError("DriveSuprim coarse aggregate contains NaN or Inf")
        top_indices = torch.topk(
            aggregate_coarse,
            k=self.config.coarse_topk,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        rows = torch.arange(batch_size, device=top_indices.device)[:, None].expand_as(
            top_indices
        )
        top_trajectories = joint_candidates[rows, top_indices]
        top_status = candidate_status[rows, top_indices]
        top_metadata = metadata.gather(top_indices)
        self.last_latency_coarse = self._timing_end(global_memory, coarse_start)

        fine_memory, fine_padding_mask = self._fine_memory(scene_context)
        refinement_start = self._timing_start(fine_memory)
        layer_results, aggregate_fine, selected_position = (
            self._trajectory_offset_head(
                fine_memory,
                fine_padding_mask,
                top_status,
            )
        )
        if not torch.isfinite(aggregate_fine).all():
            raise ValueError("DriveSuprim fine aggregate contains NaN or Inf")
        batch_rows = torch.arange(batch_size, device=selected_position.device)
        selected_trajectory_40 = top_trajectories[batch_rows, selected_position]
        selected_source = top_metadata.source[batch_rows, selected_position]
        selected_source_index = top_metadata.source_index[
            batch_rows, selected_position
        ]
        selected_trajectory_8 = self.resolve_selected_trajectory_8(
            selected_trajectory_40,
            selected_source,
            selected_source_index,
            dynamic_traj8,
        )
        self.last_latency_refinement = self._timing_end(
            fine_memory, refinement_start
        )
        coarse_output = dict(coarse_scores)
        coarse_output["aggregate_score"] = aggregate_coarse
        return JointSelectorOutput(
            selected_trajectory_40=selected_trajectory_40,
            selected_trajectory_8=selected_trajectory_8,
            selected_source=selected_source,
            selected_source_index=selected_source_index,
            coarse_scores=coarse_output,
            fine_scores={
                "layer_results": layer_results,
                "aggregate_score": aggregate_fine,
            },
            top256_indices=top_indices,
            top256_metadata=top_metadata,
        )
