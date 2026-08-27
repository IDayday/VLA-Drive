"""Qwen Register64 with CLOVER Stage-1 proposal/value co-training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from starVLA.model.framework.QwenRegisterGenerator import (
    QwenRegisterGenerator,
    _autocast_context,
)
from starVLA.model.modules.register_planner.clover_losses import (
    CloverStage1TrajectoryLoss,
)
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import PDMSValueLoss
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _build_clover_scorer(config) -> DrivoRDynamicScorer:
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
        noc=float(config.get("noc", 1.0)),
        dac=float(config.get("dac", 1.0)),
        ddc=float(config.get("ddc", 0.0)),
        ttc=float(config.get("ttc", 5.0)),
        ep=float(config.get("ep", 5.0)),
        comfort=float(config.get("comfort", 2.0)),
        aggregate_head=bool(config.get("aggregate_head", True)),
        selection_mode=str(config.get("selection_mode", "learned_aggregate")),
        aggregate_temperature=float(config.get("aggregate_temperature", 1.0)),
        selection_alpha=float(config.get("selection_alpha", 0.0)),
        debug_validate_finite=bool(config.get("debug_validate_finite", False)),
    )


@FRAMEWORK_REGISTRY.register("QwenRegisterClover")
class QwenRegisterClover(QwenRegisterGenerator):
    """Stage-1 CLOVER adaptation with a shared metric-aware scene encoder.

    Proposal geometry is detached inside ``DrivoRDynamicScorer``.  Therefore
    score loss updates the scorer and shared Qwen/Q-Former scene features, but
    cannot leak through waypoints into the Register64 generator.  This is the
    exact generator/scorer gradient boundary used by DrivoR and CLOVER.
    """

    def __init__(self, config, *, drivor_scorer=None, **dependencies: Any) -> None:
        super().__init__(config, **dependencies)
        scorer_config = config.framework.drivor_scorer
        self.drivor_scorer = drivor_scorer or _build_clover_scorer(scorer_config)
        loss_config = config.framework.clover_loss
        self.clover_trajectory_loss = CloverStage1TrajectoryLoss(
            gt_weight=float(loss_config.get("gt_weight", 1.0)),
            pseudo_expert_weight=float(
                loss_config.get("pseudo_expert_weight", 0.5)
            ),
        )
        self.clover_value_loss = PDMSValueLoss(
            submetric_weight=float(loss_config.get("submetric_weight", 1.0)),
            aggregate_weight=float(loss_config.get("aggregate_weight", 1.0)),
            listwise_weight=float(loss_config.get("listwise_weight", 1.0)),
            pairwise_weight=float(loss_config.get("pairwise_weight", 0.5)),
            listwise_temperature=float(
                loss_config.get("listwise_temperature", 0.1)
            ),
            pairwise_temperature=float(
                loss_config.get("pairwise_temperature", 0.1)
            ),
            pairwise_margin=float(loss_config.get("pairwise_margin", 0.05)),
        )
        self.scorer_loss_weight = float(loss_config.get("scorer_weight", 1.0))

    def forward(
        self,
        examples: Sequence[dict],
        *,
        clover_supervisor=None,
        pseudo_experts: Tensor | None = None,
        pseudo_expert_mask: Tensor | None = None,
        compute_loss: bool = True,
        generate_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if clover_supervisor is None:
            return super().forward(
                examples, generate_only=generate_only, **kwargs
            )
        if generate_only:
            raise ValueError("CLOVER scored forward and generate_only are exclusive")
        _, action_values = self._extract_examples(examples, require_actions=True)
        scene, generated, ego_state = self.encode_scene_and_generate(examples)
        actions = torch.as_tensor(
            np.asarray(action_values),
            device=generated.proposals.device,
            dtype=torch.float32,
        )
        ground_truth = self.trajectory_codec.flow_to_navsim(actions)
        tokens = [str(example["token"]) for example in examples]

        # Submit the expensive CPU evaluator before the GPU scorer so both can
        # overlap. The supervisor receives the complete K=64 pool once.
        metric_future = clover_supervisor.score_async(
            tokens, generated.proposals.float()
        )
        with _autocast_context(generated.proposals):
            scorer_output = self.drivor_scorer(
                generated.proposals,
                scene.global_tokens,
                ego_state,
                topm=generated.proposals.shape[1],
            )
        metric_targets = metric_future.result()
        rows = torch.arange(generated.proposals.shape[0], device=generated.proposals.device)
        selected_indices = scorer_output.aggregate_score.argmax(dim=1)
        oracle_indices = metric_targets["aggregate_score"].argmax(dim=1)
        selected_true = metric_targets["aggregate_score"][rows, selected_indices]
        oracle_true = metric_targets["aggregate_score"][rows, oracle_indices]
        output: dict[str, Any] = {
            "scene_context": scene,
            "generator_output": generated,
            "ego_state": ego_state,
            "ground_truth": ground_truth,
            "scorer_output": scorer_output,
            "metric_targets": metric_targets,
            "metrics": {
                "selected_true_pdms": selected_true.mean(),
                "oracle_pdms_64": oracle_true.mean(),
                "scorer_regret": (oracle_true - selected_true).mean(),
            },
        }
        if not compute_loss:
            return output
        if pseudo_experts is None or pseudo_expert_mask is None:
            raise ValueError("CLOVER Stage-1 training requires pseudo experts")
        trajectory = self.clover_trajectory_loss(
            generated.proposals,
            ground_truth,
            pseudo_experts,
            pseudo_expert_mask,
        )
        value_loss, value_details = self.clover_value_loss(
            scorer_output.metric_logits,
            scorer_output.aggregate_logit,
            metric_targets,
        )
        total = trajectory.loss + self.scorer_loss_weight * value_loss
        output.update(
            loss=total,
            losses={
                "trajectory_gt": trajectory.gt_loss,
                "pseudo_expert_coverage": trajectory.pseudo_expert_loss,
                "scorer": value_loss,
                **{f"scorer_{name}": value for name, value in value_details.items()},
            },
        )
        return output

    def log_architecture_summary(self, logger) -> None:
        super().log_architecture_summary(logger)
        logger.info(
            "CLOVER PDMS scorer: params=%d aggregate_head=%s selection=%s "
            "proposal_detach=true",
            sum(parameter.numel() for parameter in self.drivor_scorer.parameters()),
            self.drivor_scorer.aggregate_head_enabled,
            self.drivor_scorer.selection_mode,
        )
