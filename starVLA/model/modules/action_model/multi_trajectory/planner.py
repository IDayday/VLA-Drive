"""End-to-end DDP with DrivoR scoring and DriveSuprim refinement planner."""

from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .candidate_types import DynamicScorerOutput, JointSelectorOutput, PlannerDiagnostics
from .config import MultiTrajectoryConfig
from .ddp_multi_sampler import DDPMultiSampler
from .drivor_dynamic_scorer import DrivoRDynamicScorer
from .global_scene_qformer import GlobalSceneQFormer
from .losses import DriveSuprimLoss, DrivoRSubScoreLoss
from .scene_context import SceneContext
from .suprim_joint_selector import DriveSuprimJointSelector
from .trajectory_codec import normalized_deltas_to_poses
from .trajectory_resampler import trajectory_8_to_40


def _set_requires_grad(module: Optional[nn.Module], value: bool) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad = value


def _load_checkpoint_file(
    path: str,
    *,
    reject_training_initializer: bool = False,
    expected_scene_dim: int = 2048,
    expected_planning_dim: int = 256,
) -> Mapping[str, Tensor]:
    """Load one new component checkpoint after validating architecture metadata."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"component checkpoint not found: {checkpoint_path}")
    try:
        checkpoint: Any = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint {checkpoint_path} must contain a mapping")
    metadata = checkpoint.get("ddp_drs_checkpoint")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "DDP-DRS component checkpoint requires ddp_drs_checkpoint metadata "
            "with scene_dim and planning_dim"
        )
    scene_dim = metadata.get("scene_dim")
    planning_dim = metadata.get("planning_dim")
    if scene_dim != expected_scene_dim:
        raise RuntimeError(
            f"checkpoint scene_dim={scene_dim!r} is incompatible; expected "
            f"scene_dim={expected_scene_dim}. Old 256-wide scene checkpoints "
            "cannot initialize the 2048-wide Q-Former."
        )
    if planning_dim != expected_planning_dim:
        raise RuntimeError(
            f"checkpoint planning_dim={planning_dim!r} is incompatible; expected "
            f"planning_dim={expected_planning_dim}"
        )
    inference_ready = bool(metadata.get("inference_ready", False))
    requires_training = list(metadata.get("requires_training", ()))
    requires_training_preview = requires_training[:10]
    print(
        "DDP-DRS component checkpoint metadata: "
        f"component={metadata.get('component')} scene_dim={scene_dim} "
        f"planning_dim={planning_dim} inference_ready={inference_ready} "
        f"requires_training_count={len(requires_training)} "
        f"requires_training_preview={requires_training_preview}"
    )
    if reject_training_initializer and not inference_ready:
        raise RuntimeError(
            "strict DDP-DRS inference rejects a donor warm-start checkpoint; "
            f"train these parameters first: {requires_training}"
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise TypeError(
            f"checkpoint {checkpoint_path} must contain a tensor state_dict"
        )
    return state


def _strict_load(module: nn.Module, state: Mapping[str, Tensor], label: str) -> None:
    expected_state = module.state_dict()
    expected = set(expected_state)
    supplied = set(state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    shape_mismatch = sorted(
        key
        for key in expected & supplied
        if tuple(expected_state[key].shape) != tuple(state[key].shape)
    )
    print(f"{label} missing_keys: {missing}")
    print(f"{label} unexpected_keys: {unexpected}")
    print(f"{label} shape_mismatch_keys: {shape_mismatch}")
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(f"{label} checkpoint state_dict does not strictly match")
    module.load_state_dict(dict(state), strict=True)


def _strip_optional_prefix(
    state: Mapping[str, Tensor], prefix: str
) -> Mapping[str, Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def _resolve_coarse_score_targets(
    coarse_targets: Mapping[str, Any],
    dynamic_output: Optional[DynamicScorerOutput],
) -> Mapping[str, Tensor]:
    """Align cached K-dynamic labels with the scorer-selected Top-M order."""

    if "static" not in coarse_targets and "dynamic" not in coarse_targets:
        return coarse_targets  # static-only donor contract
    static_targets = coarse_targets.get("static")
    dynamic_targets = coarse_targets.get("dynamic")
    if not isinstance(static_targets, Mapping):
        raise TypeError("structured coarse targets require a static mapping")
    if dynamic_output is None:
        return dict(static_targets)
    if not isinstance(dynamic_targets, Mapping):
        raise TypeError("joint coarse targets require a dynamic metric mapping")
    topk_indices = dynamic_output.topk_indices
    rows = torch.arange(topk_indices.shape[0], device=topk_indices.device)[
        :, None
    ].expand_as(topk_indices)
    resolved_targets: Dict[str, Tensor] = {}
    for name, static_values in static_targets.items():
        if name not in dynamic_targets:
            raise KeyError(f"dynamic coarse targets are missing {name!r}")
        if not torch.is_tensor(static_values) or not torch.is_tensor(
            dynamic_targets[name]
        ):
            raise TypeError("coarse score targets must be tensors after collation")
        dynamic_values = dynamic_targets[name].to(device=topk_indices.device)
        gathered_dynamic = dynamic_values[rows, topk_indices]
        resolved_targets[name] = torch.cat(
            (
                static_values.to(
                    device=gathered_dynamic.device,
                    dtype=gathered_dynamic.dtype,
                ),
                gathered_dynamic,
            ),
            dim=1,
        )
    return resolved_targets


class DDPDrivoRSuprimPlanner(nn.Module):
    """Qwen-once/Q-Former-once DDP-DRS planner."""

    def __init__(
        self,
        action_head: nn.Module,
        config: MultiTrajectoryConfig,
        qwen_hidden_dim: int,
        scene_compressor: Optional[GlobalSceneQFormer] = None,
        dynamic_scorer: Optional[DrivoRDynamicScorer] = None,
        suprim_selector: Optional[DriveSuprimJointSelector] = None,
    ) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("DDPDrivoRSuprimPlanner is only built when enabled=true")
        if qwen_hidden_dim <= 0:
            raise ValueError("qwen_hidden_dim must be positive")
        config.validate()
        self.config = config
        self.multi_sampler = DDPMultiSampler(
            action_head=action_head,
            num_candidates=config.num_dynamic_candidates,
        )
        self._validate_action_head(action_head)
        ego_status_dim = self._resolve_ego_status_dim(action_head, config)
        stage = config.training_stage
        requires_scene = stage != "cache_candidates"
        requires_scorer = stage in {
            "train_drivor",
            "train_suprim_joint",
            "joint_finetune",
            "inference",
        }
        requires_selector = stage in {
            "train_suprim_static",
            "train_suprim_joint",
            "joint_finetune",
            "inference",
        }
        scene_was_injected = scene_compressor is not None
        scorer_was_injected = dynamic_scorer is not None
        selector_was_injected = suprim_selector is not None

        if requires_selector and not selector_was_injected and config.suprim.vocab_path is None:
            raise FileNotFoundError(
                "DriveSuprim requires multi_trajectory.suprim.vocab_path"
            )
        if stage in {"train_suprim_static", "train_suprim_joint"}:
            if (
                not scene_was_injected
                and config.scene_compressor.checkpoint_path is None
            ):
                raise FileNotFoundError(
                    f"{stage} requires a trained scene-compressor checkpoint"
                )
        if stage == "train_suprim_joint" and (
            not scorer_was_injected and config.drivor.checkpoint_path is None
        ):
            raise FileNotFoundError(
                "train_suprim_joint requires a trained DrivoR scorer checkpoint"
            )
        if stage == "inference" and config.strict_inference:
            required = (
                (scene_was_injected, config.scene_compressor.checkpoint_path, "scene compressor"),
                (scorer_was_injected, config.drivor.checkpoint_path, "DrivoR scorer"),
                (selector_was_injected, config.suprim.checkpoint_path, "DriveSuprim selector"),
            )
            missing = [label for injected, path, label in required if not injected and path is None]
            if missing:
                raise FileNotFoundError(
                    "strict DDP-DRS inference requires checkpoints for: "
                    + ", ".join(missing)
                )

        if requires_scene:
            scene_cfg = config.scene_compressor
            self.scene_compressor = scene_compressor or GlobalSceneQFormer(
                input_dim=qwen_hidden_dim,
                scene_dim=scene_cfg.scene_dim,
                num_queries=scene_cfg.num_queries,
                num_layers=scene_cfg.num_layers,
                num_heads=scene_cfg.num_heads,
                ffn_dim=scene_cfg.ffn_dim,
                dropout=scene_cfg.dropout,
                query_init_std=scene_cfg.query_init_std,
                debug_validate_finite=scene_cfg.debug_validate_finite,
            )
        if requires_scorer:
            self.dynamic_scorer = dynamic_scorer or DrivoRDynamicScorer(
                config.drivor,
                ego_status_dim=ego_status_dim,
                planning_config=config.planning,
                scene_dim=config.scene_compressor.scene_dim,
            )
        if requires_selector:
            self.suprim_selector = suprim_selector or DriveSuprimJointSelector(
                config.suprim,
                planning_config=config.planning,
                scene_dim=config.scene_compressor.scene_dim,
                ego_status_dim=ego_status_dim,
            )
            self.suprim_selector.profile_latency = config.diagnostics_enabled

        self.last_diagnostics: Optional[PlannerDiagnostics] = None
        self._configure_training_stage()
        self._load_configured_checkpoints(
            scene_was_injected=scene_was_injected,
            scorer_was_injected=scorer_was_injected,
            selector_was_injected=selector_was_injected,
        )
        self._print_parameter_counts()

    @staticmethod
    def _validate_action_head(action_head: nn.Module) -> None:
        horizon = getattr(action_head, "action_horizon", None)
        action_dim = getattr(action_head, "action_dim", None)
        if (horizon, action_dim) != (8, 3):
            raise ValueError(
                "DDP-DRS requires unchanged DDP output [B,8,3]; got "
                f"horizon={horizon}, action_dim={action_dim}"
            )
        if not callable(getattr(action_head, "predict_action", None)):
            raise TypeError("action_head must expose predict_action")

    @staticmethod
    def _resolve_ego_status_dim(
        action_head: nn.Module, config: MultiTrajectoryConfig
    ) -> int:
        if config.drivor.ego_status_dim is not None:
            return config.drivor.ego_status_dim
        action_config = getattr(action_head, "config", None)
        state_dim = getattr(action_config, "state_dim", None)
        if state_dim:
            return int(state_dim)
        state_encoder = getattr(action_head, "state_encoder", None)
        if state_encoder is not None:
            for module in state_encoder.modules():
                if isinstance(module, nn.Linear):
                    return int(module.in_features)
        raise ValueError(
            "cannot infer ego-status dimension; set multi_trajectory.drivor.ego_status_dim"
        )

    def _configure_training_stage(self) -> None:
        stage = self.config.training_stage
        _set_requires_grad(self.multi_sampler.action_head, False)
        _set_requires_grad(
            getattr(self, "scene_compressor", None),
            stage in {"train_drivor", "joint_finetune"},
        )
        _set_requires_grad(
            getattr(self, "dynamic_scorer", None),
            stage in {"train_drivor", "joint_finetune"},
        )
        _set_requires_grad(
            getattr(self, "suprim_selector", None),
            stage in {"train_suprim_static", "train_suprim_joint", "joint_finetune"},
        )
        if stage in {"cache_candidates", "inference"}:
            _set_requires_grad(getattr(self, "scene_compressor", None), False)
            _set_requires_grad(getattr(self, "dynamic_scorer", None), False)
            _set_requires_grad(getattr(self, "suprim_selector", None), False)
        if stage == "inference":
            self.eval()

    def train(self, mode: bool = True) -> "DDPDrivoRSuprimPlanner":
        if self.config.training_stage == "inference":
            return super().train(False)
        super().train(mode)
        stage = self.config.training_stage
        for name, train_stages in (
            ("scene_compressor", {"train_drivor", "joint_finetune"}),
            ("dynamic_scorer", {"train_drivor", "joint_finetune"}),
            (
                "suprim_selector",
                {"train_suprim_static", "train_suprim_joint", "joint_finetune"},
            ),
        ):
            module = getattr(self, name, None)
            if module is not None and stage not in train_stages:
                module.eval()
        self.multi_sampler.action_head.eval()
        return self

    def _load_configured_checkpoints(
        self,
        *,
        scene_was_injected: bool,
        scorer_was_injected: bool,
        selector_was_injected: bool,
    ) -> None:
        reject = self.config.training_stage == "inference" and self.config.strict_inference
        expected_scene = self.config.scene_compressor.scene_dim
        expected_planning = self.config.planning.planning_dim
        specifications = (
            (
                "scene_compressor",
                self.config.scene_compressor.checkpoint_path,
                scene_was_injected,
                "scene_compressor.",
                "Global scene compressor",
            ),
            (
                "dynamic_scorer",
                self.config.drivor.checkpoint_path,
                scorer_was_injected,
                "dynamic_scorer.",
                "DrivoR dynamic scorer",
            ),
            (
                "suprim_selector",
                self.config.suprim.checkpoint_path,
                selector_was_injected,
                "suprim_selector.",
                "DriveSuprim selector",
            ),
        )
        for attribute, path, was_injected, prefix, label in specifications:
            module = getattr(self, attribute, None)
            if module is None or path is None:
                continue
            if was_injected:
                raise ValueError(f"cannot both inject {attribute} and configure its checkpoint")
            state = _load_checkpoint_file(
                path,
                reject_training_initializer=reject,
                expected_scene_dim=expected_scene,
                expected_planning_dim=expected_planning,
            )
            _strict_load(module, _strip_optional_prefix(state, prefix), label)

    @staticmethod
    def _parameter_count(module: Optional[nn.Module]) -> int:
        return 0 if module is None else sum(p.numel() for p in module.parameters())

    def _print_parameter_counts(self) -> None:
        total = self._parameter_count(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        scene = self._parameter_count(getattr(self, "scene_compressor", None))
        scorer = self._parameter_count(getattr(self, "dynamic_scorer", None))
        selector = self._parameter_count(getattr(self, "suprim_selector", None))
        print(
            "DDP-DRS parameters: "
            f"total={total} trainable={trainable} scene_compressor={scene} "
            f"drivor_scorer={scorer} suprim_selector={selector}"
        )

    def component_checkpoint_payload(
        self, component: str, *, inference_ready: bool
    ) -> Dict[str, Any]:
        """Build a portable, dimension-tagged component checkpoint payload."""

        module = getattr(self, component, None)
        if module is None or not isinstance(module, nn.Module):
            raise ValueError(f"planner component {component!r} is not instantiated")
        return {
            "state_dict": module.state_dict(),
            "ddp_drs_checkpoint": {
                "component": component,
                "scene_dim": self.config.scene_compressor.scene_dim,
                "planning_dim": self.config.planning.planning_dim,
                "inference_ready": bool(inference_ready),
                "requires_training": [] if inference_ready else [component],
            },
        }

    def sample_candidates(
        self, vl_embs_list: List[Tensor], state: Optional[Tensor]
    ) -> Tensor:
        return self.multi_sampler.sample(
            vl_embs_list=vl_embs_list,
            state=state,
            num_candidates=self.config.num_dynamic_candidates,
            seed=self.config.deterministic_seed,
        )

    def sample_physical_candidates(
        self, vl_embs_list: List[Tensor], state: Optional[Tensor]
    ) -> Tensor:
        return normalized_deltas_to_poses(self.sample_candidates(vl_embs_list, state))

    def encode_scene(
        self, full_hidden_state: Tensor, attention_mask: Tensor
    ) -> SceneContext:
        if not hasattr(self, "scene_compressor"):
            raise RuntimeError("scene compressor is disabled for this stage")
        return self.scene_compressor(
            full_hidden_state=full_hidden_state,
            attention_mask=attention_mask,
            detach_input=self.config.scene_compressor.detach_qwen_memory,
        )

    def score_dynamic_candidates(
        self,
        dynamic_trajectories: Tensor,
        scene_context: SceneContext,
        state: Tensor,
    ) -> DynamicScorerOutput:
        if not hasattr(self, "dynamic_scorer"):
            raise RuntimeError("DrivoR scorer is disabled for this stage")
        return self.dynamic_scorer(
            proposals=dynamic_trajectories,
            global_scene_tokens=scene_context.global_scene_tokens,
            ego_status=state,
            topk=self.config.drivor.dynamic_topk,
        )

    def select_candidates(
        self,
        scene_context: SceneContext,
        state: Tensor,
        dynamic_output: Optional[DynamicScorerOutput],
    ) -> JointSelectorOutput:
        if not hasattr(self, "suprim_selector"):
            raise RuntimeError("DriveSuprim selector is disabled for this stage")
        if dynamic_output is None:
            dynamic_traj8 = dynamic_traj40 = dynamic_ids = None
        else:
            dynamic_traj8 = dynamic_output.topk_trajectories
            dynamic_traj40 = trajectory_8_to_40(dynamic_traj8)
            dynamic_ids = dynamic_output.topk_indices
        return self.suprim_selector(
            scene_context=scene_context,
            dynamic_traj8=dynamic_traj8,
            dynamic_traj40=dynamic_traj40,
            dynamic_candidate_ids=dynamic_ids,
            ego_status=state,
        )

    def forward_with_outputs(
        self,
        vl_embs_list: List[Tensor],
        state: Tensor,
        full_hidden_state: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[JointSelectorOutput, Optional[DynamicScorerOutput]]:
        if self.config.training_stage == "cache_candidates":
            raise RuntimeError("cache_candidates uses sample_candidates(), not planner forward")
        def timing_start(reference: Tensor) -> float:
            if self.config.diagnostics_enabled and reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            return time.perf_counter()

        def timing_end(reference: Tensor, started: float) -> float:
            if self.config.diagnostics_enabled and reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            return time.perf_counter() - started

        total_start = timing_start(full_hidden_state)
        sampling_latency = scorer_latency = 0.0
        dynamic_trajectories: Optional[Tensor] = None
        dynamic_output: Optional[DynamicScorerOutput] = None

        start = timing_start(full_hidden_state)
        scene_context = self.encode_scene(full_hidden_state, attention_mask)
        scene_latency = timing_end(scene_context.global_scene_tokens, start)
        if self.config.dynamic_candidates_enabled:
            start = timing_start(full_hidden_state)
            dynamic_trajectories = self.sample_physical_candidates(vl_embs_list, state)
            sampling_latency = timing_end(full_hidden_state, start)
        if dynamic_trajectories is not None:
            start = timing_start(scene_context.global_scene_tokens)
            dynamic_output = self.score_dynamic_candidates(
                dynamic_trajectories, scene_context, state
            )
            scorer_latency = timing_end(
                scene_context.global_scene_tokens, start
            )
        joint_output = self.select_candidates(scene_context, state, dynamic_output)

        if self.config.diagnostics_enabled:
            dynamic_in_top = (joint_output.top256_metadata.source == 1).sum(dim=1)
            dynamic_denominator = (
                1 if dynamic_output is None else dynamic_output.topk_trajectories.shape[1]
            )
            self.last_diagnostics = PlannerDiagnostics(
                drivor_selected_index=(
                    None if dynamic_output is None else dynamic_output.topk_indices[:, 0]
                ),
                drivor_sub_scores=(
                    None if dynamic_output is None else dynamic_output.sub_scores
                ),
                drivor_aggregate_score=(
                    None if dynamic_output is None else dynamic_output.aggregate_score
                ),
                dynamic_top16_indices=(
                    None if dynamic_output is None else dynamic_output.topk_indices
                ),
                suprim_top256_indices=joint_output.top256_indices,
                final_candidate_source=joint_output.selected_source,
                final_candidate_source_index=joint_output.selected_source_index,
                dynamic_selected_ratio=(joint_output.selected_source == 1).float().mean(),
                dynamic_enter_top256_ratio=(
                    dynamic_in_top.float() / float(dynamic_denominator)
                ).mean(),
                latency_ddp_sampling=sampling_latency,
                latency_scene_compressor=scene_latency,
                latency_drivor_scorer=scorer_latency,
                latency_suprim_coarse=self.suprim_selector.last_latency_coarse,
                latency_suprim_refinement=self.suprim_selector.last_latency_refinement,
                latency_total_inference=timing_end(
                    scene_context.global_scene_tokens, total_start
                ),
                global_scene_tokens_bytes=(
                    scene_context.global_scene_tokens.numel()
                    * scene_context.global_scene_tokens.element_size()
                ),
                dense_scene_memory_bytes=(
                    scene_context.dense_scene_memory.numel()
                    * scene_context.dense_scene_memory.element_size()
                ),
                peak_memory=(
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0
                ),
            )
        else:
            self.last_diagnostics = None
        return joint_output, dynamic_output

    def compute_training_loss(
        self,
        vl_embs_list: List[Tensor],
        state: Tensor,
        full_hidden_state: Tensor,
        attention_mask: Tensor,
        targets: Mapping[str, Any],
        cached_dynamic_trajectories: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute losses only from explicit offline labels supplied by training."""

        stage = self.config.training_stage
        if stage in {"inference", "cache_candidates"}:
            raise RuntimeError(f"stage {stage!r} does not define a training loss")
        scene_context = self.encode_scene(full_hidden_state, attention_mask)
        dynamic_stage = stage in {
            "train_drivor",
            "train_suprim_joint",
            "joint_finetune",
        }
        dynamic_trajectories = None
        if dynamic_stage:
            if cached_dynamic_trajectories is None:
                # Direct module tests may deliberately exercise the sampler.
                # Production datasets always supply the validated offline cache.
                dynamic_trajectories = self.sample_physical_candidates(
                    vl_embs_list, state
                )
            else:
                expected = (
                    full_hidden_state.shape[0],
                    self.config.num_dynamic_candidates,
                    8,
                    3,
                )
                if tuple(cached_dynamic_trajectories.shape) != expected:
                    raise ValueError(
                        "cached DDP trajectories must have shape "
                        f"{expected}, got {tuple(cached_dynamic_trajectories.shape)}"
                    )
                if not torch.isfinite(cached_dynamic_trajectories).all():
                    raise ValueError("cached DDP trajectories contain NaN or Inf")
                dynamic_trajectories = cached_dynamic_trajectories.detach()
        dynamic_output = (
            self.score_dynamic_candidates(dynamic_trajectories, scene_context, state)
            if dynamic_trajectories is not None
            else None
        )
        drivor_loss: Optional[Tensor] = None
        drivor_components: Optional[Dict[str, Tensor]] = None
        if stage in {"train_drivor", "joint_finetune"}:
            score_targets = targets.get("drivor_scores")
            if not isinstance(score_targets, Mapping):
                raise KeyError(f"{stage} requires targets['drivor_scores']")
            drivor_loss, drivor_components = DrivoRSubScoreLoss()(
                dynamic_output.sub_scores, score_targets
            )
            if stage == "train_drivor":
                return {
                    "loss": drivor_loss,
                    "drivor_loss": drivor_loss,
                    "loss_components": drivor_components,
                    "dynamic_output": dynamic_output,
                }

        joint_output = self.select_candidates(scene_context, state, dynamic_output)
        coarse_targets = targets.get("coarse_scores")
        target_trajectory = targets.get("trajectory")
        if not isinstance(coarse_targets, Mapping) or not torch.is_tensor(target_trajectory):
            raise KeyError(
                "DriveSuprim training requires targets['coarse_scores'] and "
                "targets['trajectory']"
            )
        coarse_targets = _resolve_coarse_score_targets(
            coarse_targets, dynamic_output
        )
        dynamic_40 = (
            None
            if dynamic_output is None
            else trajectory_8_to_40(dynamic_output.topk_trajectories)
        )
        dynamic_ids = None if dynamic_output is None else dynamic_output.topk_indices
        candidates, _ = self.suprim_selector.build_joint_candidates(
            target_trajectory.shape[0], dynamic_40, dynamic_ids
        )
        candidates = candidates.to(
            device=target_trajectory.device, dtype=target_trajectory.dtype
        )
        loss_module = DriveSuprimLoss(sigma=self.config.suprim.imitation_sigma)
        coarse_predictions = {
            name: value
            for name, value in joint_output.coarse_scores.items()
            if name != "aggregate_score"
        }
        coarse_loss, coarse_components = loss_module.coarse_loss(
            coarse_predictions, coarse_targets, candidates, target_trajectory
        )
        top_indices = joint_output.top256_indices
        rows = torch.arange(top_indices.shape[0], device=top_indices.device)[:, None]
        rows = rows.expand_as(top_indices)
        top_candidates = candidates.to(top_indices.device)[rows, top_indices]
        refinement_targets = targets.get("refinement_scores")
        if refinement_targets is None:
            refinement_targets = {
                name: values.to(top_indices.device)[rows, top_indices]
                for name, values in coarse_targets.items()
            }
        if not isinstance(refinement_targets, Mapping):
            raise TypeError("targets['refinement_scores'] must be a mapping")
        refinement_loss, refinement_components = loss_module.refinement_loss(
            joint_output.fine_scores["layer_results"],
            refinement_targets,
            top_candidates,
            target_trajectory.to(top_indices.device),
        )
        suprim_loss = coarse_loss + refinement_loss
        total = suprim_loss if drivor_loss is None else suprim_loss + drivor_loss
        return {
            "loss": total,
            "drivor_loss": drivor_loss,
            "drivor_loss_components": drivor_components,
            "suprim_coarse_loss": coarse_loss,
            "suprim_refinement_loss": refinement_loss,
            "coarse_loss_components": coarse_components,
            "refinement_loss_components": refinement_components,
            "joint_output": joint_output,
        }

    def forward(
        self,
        vl_embs_list: List[Tensor],
        state: Tensor,
        full_hidden_state: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        context = (
            torch.inference_mode()
            if self.config.training_stage == "inference"
            else nullcontext()
        )
        try:
            with context:
                joint_output, _ = self.forward_with_outputs(
                    vl_embs_list=vl_embs_list,
                    state=state,
                    full_hidden_state=full_hidden_state,
                    attention_mask=attention_mask,
                )
                return joint_output.selected_trajectory_8
        except Exception:
            if not self.config.smoke_test_fallback_to_single_ddp:
                raise
            warnings.warn(
                "DDP-DRS smoke-test failure; using explicitly enabled single-DDP fallback",
                RuntimeWarning,
                stacklevel=2,
            )
            return normalized_deltas_to_poses(
                self.multi_sampler.action_head.predict_action(vl_embs_list, state)
            )
