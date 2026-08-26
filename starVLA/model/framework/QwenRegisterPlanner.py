"""Integrated learned-only Qwen Register64 planner inference framework."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from starVLA.model.framework.QwenRegisterGenerator import (
    QwenRegisterGenerator,
    _autocast_context,
)
from starVLA.model.modules.register_planner.checkpoint import (
    load_register_generator_checkpoint,
    load_stage_component_checkpoint,
    sha256_file,
)
from starVLA.model.modules.register_planner.selectors import (
    DynamicDriveSuprimSelector,
    HybridDriveSuprimSelector,
)
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.drivesuprim_joint_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


_REGISTER_METRIC_SCHEMA = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
    "aggregate_score",
)


def _build_drivor(config) -> DrivoRDynamicScorer:
    return DrivoRDynamicScorer(
        scene_dim=int(config.get("scene_dim", 256)),
        ego_state_dim=int(config.get("ego_state_dim", 4)),
        model_dim=int(config.get("model_dim", 256)),
        ffn_dim=int(config.get("ffn_dim", 1024)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 1)),
        dropout=float(config.get("dropout", 0.0)),
        decoder_style=str(config.get("decoder_style", "donor_register")),
        proj_drop=float(config.get("proj_drop", 0.1)),
        drop_path=float(config.get("drop_path", 0.2)),
        layer_scale_init=float(config.get("layer_scale_init", 0.0)),
    )


def _build_fine(config) -> DriveSuprimFineRefiner:
    return DriveSuprimFineRefiner(
        scene_dim=int(config.get("scene_dim", 256)),
        model_dim=int(config.get("model_dim", 256)),
        ffn_dim=int(config.get("ffn_dim", 1024)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("refinement_layers", 3)),
        dropout=float(config.get("dropout", 0.0)),
        use_mid_output=bool(config.get("use_mid_output", True)),
        use_imitation=bool(config.get("use_imitation", True)),
    )


@FRAMEWORK_REGISTRY.register("QwenRegisterPlanner")
class QwenRegisterPlanner(QwenRegisterGenerator):
    """Qwen -> Q-Former -> Register64 -> learned selector -> final action."""

    def __init__(
        self,
        config,
        *,
        drivor_scorer: Optional[DrivoRDynamicScorer] = None,
        suprim_selector: Optional[nn.Module] = None,
        load_checkpoints: bool = True,
        **generator_dependencies: Any,
    ) -> None:
        super().__init__(config, **generator_dependencies)
        inference = config.framework.inference
        self.selector_type = str(inference.get("selector_type", "drivor"))
        valid = {
            "none",
            "drivor",
            "drivor_suprim_dynamic",
            "drivor_suprim_hybrid",
        }
        if self.selector_type not in valid:
            raise ValueError(f"unknown Register planner selector {self.selector_type!r}")
        self.return_all_proposals = bool(
            inference.get("return_all_proposals", False)
        )
        self.dynamic_topm = int(inference.get("dynamic_topm", 32))
        self.fine_memory_source = str(
            inference.get("fine_memory_source", "global_scene_tokens")
        )
        if self.fine_memory_source not in {
            "global_scene_tokens",
            "dense_scene_memory",
        }:
            raise ValueError("unsupported fine memory source")
        generator_checkpoint = inference.get("generator_checkpoint")
        if load_checkpoints:
            if not generator_checkpoint:
                raise FileNotFoundError("integrated inference requires generator_checkpoint")
            load_register_generator_checkpoint(
                str(generator_checkpoint),
                qwen_vl_interface=self.qwen_vl_interface,
                action_input_model=self.action_input_model,
                scene_encoder=self.scene_encoder,
                register_generator=self.register_generator,
                expected_metadata={
                    "qwen_base_model": str(config.framework.qwenvl.base_vlm),
                    "proposal_num": self.register_generator.proposal_num,
                    "num_poses": 8,
                    "state_dim": 3,
                    "scene_queries": self.scene_encoder.num_queries,
                    "scene_dim": self.scene_encoder.output_dim,
                    "decoder_layers": self.register_generator.num_layers,
                    "decoder_heads": self.register_generator.num_heads,
                    "proposal_head_style": "donor_mlp_v1",
                    "stage_loss_mode": str(
                        self.register_generator.stage_loss_mode
                    ),
                    "proposal_head_count": int(
                        self.register_generator.proposal_head_count
                    ),
                },
            )

        self.drivor_scorer: Optional[DrivoRDynamicScorer] = None
        self.suprim_selector: Optional[nn.Module] = None
        if self.selector_type == "none":
            if self.register_generator.proposal_num != 1:
                raise ValueError("selector_type=none requires proposal_num=1")
        else:
            self.drivor_scorer = drivor_scorer or _build_drivor(
                config.framework.drivor_scorer
            )
            drivor_checkpoint = inference.get("drivor_checkpoint")
            if load_checkpoints:
                if not drivor_checkpoint:
                    raise FileNotFoundError("selected inference requires drivor_checkpoint")
                load_stage_component_checkpoint(
                    str(drivor_checkpoint),
                    stage="drivor_scorer",
                    module=self.drivor_scorer,
                    expected_metadata={
                        "generator_checkpoint_sha256": sha256_file(
                            str(generator_checkpoint)
                        ),
                        "proposal_num": self.register_generator.proposal_num,
                        "scene_dim": self.scene_encoder.output_dim,
                        "model_dim": int(config.framework.drivor_scorer.model_dim),
                        "decoder_layers": int(
                            config.framework.drivor_scorer.num_layers
                        ),
                        "decoder_heads": int(
                            config.framework.drivor_scorer.num_heads
                        ),
                        "metric_schema": list(_REGISTER_METRIC_SCHEMA),
                        "training_profile": "drivor_offline_bank_v1",
                    },
                )
            if self.selector_type == "drivor_suprim_dynamic":
                self.suprim_selector = suprim_selector or DynamicDriveSuprimSelector(
                    _build_fine(config.framework.suprim.fine)
                )
                if load_checkpoints:
                    checkpoint = inference.get("suprim_checkpoint")
                    if not checkpoint:
                        raise FileNotFoundError("dynamic Suprim checkpoint is required")
                    load_stage_component_checkpoint(
                        str(checkpoint),
                        stage="suprim_dynamic",
                        module=self.suprim_selector,
                        expected_metadata={
                            "drivor_checkpoint_sha256": sha256_file(
                                str(drivor_checkpoint)
                            ),
                            "selector_type": "dynamic",
                            "dynamic_topm": self.dynamic_topm,
                            "training_profile": "drivesuprim_dynamic_bank_v1",
                        },
                    )
            elif self.selector_type == "drivor_suprim_hybrid":
                hybrid = config.framework.suprim
                if suprim_selector is not None:
                    # Explicit injection is reserved for reduced-dimension logic tests.
                    self.suprim_selector = suprim_selector
                else:
                    coarse_cfg = hybrid.coarse
                    coarse = DriveSuprimCoarseScorer(
                        vocab_path=str(hybrid.static_vocab_path),
                        vocab_size=int(coarse_cfg.get("static_vocab_size", 8192)),
                        num_poses=40,
                        scene_dim=int(coarse_cfg.get("scene_dim", 256)),
                        model_dim=int(coarse_cfg.get("model_dim", 256)),
                        ffn_dim=int(coarse_cfg.get("ffn_dim", 1024)),
                        num_heads=int(coarse_cfg.get("num_heads", 8)),
                        num_layers=int(coarse_cfg.get("coarse_layers", 3)),
                        coarse_topk=int(coarse_cfg.get("coarse_topk", 256)),
                        dropout=float(coarse_cfg.get("dropout", 0.0)),
                        normalize_vocab_pos=False,
                    )
                    if coarse.vocab_size + self.dynamic_topm != 8224:
                        raise ValueError("hybrid inference requires 8192 static + 32 dynamic")
                    self.suprim_selector = HybridDriveSuprimSelector(
                        coarse, _build_fine(hybrid.fine)
                    )
                if load_checkpoints:
                    checkpoint = inference.get("suprim_checkpoint")
                    if not checkpoint:
                        raise FileNotFoundError("hybrid Suprim checkpoint is required")
                    load_stage_component_checkpoint(
                        str(checkpoint),
                        stage="suprim_hybrid",
                        module=self.suprim_selector,
                        expected_metadata={
                            "drivor_checkpoint_sha256": sha256_file(
                                str(drivor_checkpoint)
                            ),
                            "selector_type": "hybrid",
                            "dynamic_topm": self.dynamic_topm,
                            "training_profile": "drivesuprim_hybrid_bank_v1",
                            "static_vocabulary_sha256": sha256_file(
                                str(hybrid.static_vocab_path)
                            ),
                        },
                    )

        # This framework is an inference artifact.  Stage G owns trainability.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Avoid stochastic decoder dropout even if a generic evaluator calls train().
        return super().train(False)

    def _fine_memory(self, scene) -> tuple[Tensor, Optional[Tensor]]:
        if self.fine_memory_source == "global_scene_tokens":
            return scene.global_tokens, None
        return scene.dense_memory, scene.memory_key_padding_mask

    @torch.inference_mode()
    def forward(self, examples: Sequence[dict], **_: Any) -> dict[str, Any]:
        scene, generated, ego_state = self.encode_scene_and_generate(examples)
        proposals = generated.proposals
        batch_size = proposals.shape[0]
        device = proposals.device
        drivor_score = suprim_score = None
        if self.selector_type == "none":
            selected = proposals[:, 0]
            selected_index = torch.zeros(batch_size, device=device, dtype=torch.long)
            selected_source = torch.ones_like(selected_index)
        else:
            assert self.drivor_scorer is not None
            with _autocast_context(proposals):
                drivor = self.drivor_scorer(
                    proposals.detach(),
                    scene.global_tokens,
                    ego_state,
                    topm=self.dynamic_topm,
                )
            rows = torch.arange(batch_size, device=device)
            drivor_score = drivor.aggregate_score[
                rows, drivor.topm_indices[:, 0]
            ]
            if self.selector_type == "drivor":
                selected = drivor.topm_trajectories_8[:, 0]
                selected_index = drivor.topm_indices[:, 0]
                selected_source = torch.ones_like(selected_index)
            elif self.selector_type == "drivor_suprim_dynamic":
                assert isinstance(self.suprim_selector, DynamicDriveSuprimSelector)
                memory, mask = self._fine_memory(scene)
                with _autocast_context(proposals):
                    fine = self.suprim_selector(drivor, memory, mask)
                selected = self.trajectory_codec.downsample_40_to_8(
                    fine.selected_trajectory_40
                )
                selected_index = fine.selected_source_index
                selected_source = fine.selected_source
                suprim_score = fine.aggregate_score[
                    rows, fine.selected_topk_index
                ]
            else:
                assert isinstance(self.suprim_selector, HybridDriveSuprimSelector)
                memory, mask = self._fine_memory(scene)
                with _autocast_context(proposals):
                    hybrid = self.suprim_selector(
                        drivor,
                        scene.global_tokens,
                        ego_state,
                        memory,
                        mask,
                    )
                selected = self.trajectory_codec.downsample_40_to_8(
                    hybrid.fine.selected_trajectory_40
                )
                selected_index = hybrid.fine.selected_absolute_index
                selected_source = hybrid.fine.selected_source
                suprim_score = hybrid.fine.aggregate_score[
                    rows, hybrid.fine.selected_topk_index
                ]
        result: dict[str, Any] = {
            "trajectory_navsim_8": selected,
            "normalized_actions": self.trajectory_codec.navsim_to_flow(selected),
            "all_proposals": proposals if self.return_all_proposals else None,
            "selected_index": selected_index,
            "selected_source": selected_source,
            "drivor_score": drivor_score,
            "suprim_score": suprim_score,
        }
        return result

    predict_action = forward
