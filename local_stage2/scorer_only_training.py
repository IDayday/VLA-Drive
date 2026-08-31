"""Strict scorer-only continuation for the Stage-2 reproduction audit.

The callback deliberately keeps every Stage-2 parameter in the optimizer so a
Lightning training checkpoint can restore the original AdamW moments and LR
scheduler without changing parameter-group topology.  After DDP has reduced
the gradients, all non-scorer gradients are set to ``None``.  AdamW therefore
skips both the parameter update and decoupled weight decay for the frozen
proposal path.

The proposal-producing modules are also kept in evaluation mode during
training.  Freezing parameters alone would not freeze proposals because the
released Q-Former and trajectory decoder contain dropout / drop-path layers.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.EpisodeDrive.episodedrive_agent import EpisodeDriveAgent


SCORER_PARAMETER_PREFIXES: Tuple[str, ...] = (
    "action_head.pos_embed.",
    "action_head.scorer_attention.",
    "action_head.scorer.",
    # Alternative scorer architectures wrap the immutable released decoder
    # under ``base_decoder`` and keep every trainable scoring tensor here.
    "action_head.scorer_branch.",
)


def is_scorer_parameter(name: str) -> bool:
    """Return whether ``name`` belongs to the strictly trainable scorer path."""

    return name.startswith(SCORER_PARAMETER_PREFIXES)


def scorer_and_frozen_parameter_names(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Partition optimizer-visible parameters without changing their topology."""

    scorer = []
    frozen = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        (scorer if is_scorer_parameter(name) else frozen).append(name)
    return tuple(scorer), tuple(frozen)


def hard_freeze_non_scorer_parameters(agent) -> None:
    """Remove the immutable proposal/VLM path from fresh scorer optimizers.

    The legacy ``grad=None`` mode exists only so a *resumed* checkpoint can
    keep its original optimizer topology.  The matched scorer experiments load
    epoch-3 weights into a fresh optimizer, so retaining trainable flags on the
    2B VLM, checkpoint LoRA tensors, and proposal decoder has no benefit and
    makes DDP treat them as potentially used parameters.
    """

    for name, parameter in agent.named_parameters():
        parameter.requires_grad_(is_scorer_parameter(name))

    scorer_branch = getattr(agent.action_head, "scorer_branch", None)
    if scorer_branch is not None and hasattr(
        scorer_branch, "freeze_inactive_parameters"
    ):
        scorer_branch.freeze_inactive_parameters()


class ScorerOnlyFreezeCallback(pl.Callback):
    """Preserve a checkpoint's proposal bank while continuing scorer training."""

    def __init__(self) -> None:
        super().__init__()
        self._scorer_names: Tuple[str, ...] = ()
        self._frozen_names: Tuple[str, ...] = ()

    @staticmethod
    def _agent(pl_module):
        if not hasattr(pl_module, "agent"):
            raise TypeError("ScorerOnlyFreezeCallback requires AgentLightningModule")
        return pl_module.agent

    @staticmethod
    def _set_training_modes(agent) -> None:
        """Make proposals deterministic while leaving scorer dropout trainable."""

        if agent.backbone is not None:
            agent.backbone.eval()

        action_head = agent.action_head
        if hasattr(action_head, "set_scorer_only_train_mode"):
            action_head.set_scorer_only_train_mode()
            return
        action_head.q_former.eval()
        action_head.hist_encoding.eval()
        action_head.init_feature.eval()
        action_head.trajectory_decoder.eval()
        action_head.traj_head.eval()

        action_head.pos_embed.train()
        action_head.scorer_attention.train()
        action_head.scorer.train()

    def on_fit_start(self, trainer, pl_module) -> None:
        agent = self._agent(pl_module)
        hard_freeze = os.getenv("DRIVEVLA_HARD_FREEZE_SCORER_ONLY", "0") == "1"
        if hard_freeze:
            unexpected_trainable = tuple(
                name
                for name, parameter in agent.named_parameters()
                if parameter.requires_grad and not is_scorer_parameter(name)
            )
            if unexpected_trainable:
                raise RuntimeError(
                    "Hard scorer freeze left non-scorer parameters trainable: "
                    + ", ".join(unexpected_trainable[:10])
                )
            self._scorer_names = tuple(
                name
                for name, parameter in agent.named_parameters()
                if parameter.requires_grad and is_scorer_parameter(name)
            )
            self._frozen_names = tuple(
                name
                for name, parameter in agent.named_parameters()
                if name.startswith("action_head.") and not parameter.requires_grad
            )
        else:
            self._scorer_names, self._frozen_names = (
                scorer_and_frozen_parameter_names(agent.named_parameters())
            )
        if not self._scorer_names:
            raise RuntimeError("Scorer-only continuation found no scorer parameters")
        if not self._frozen_names:
            raise RuntimeError("Scorer-only continuation found no proposal parameters")
        if float(agent.loss.trajectory_weight) != 0.0:
            raise RuntimeError(
                "Strict scorer-only continuation requires loss.trajectory_weight=0"
            )
        self._set_training_modes(agent)
        scorer_values = sum(
            parameter.numel()
            for name, parameter in agent.named_parameters()
            if name in self._scorer_names
        )
        frozen_values = sum(
            parameter.numel()
            for name, parameter in agent.named_parameters()
            if name in self._frozen_names
        )
        if trainer.is_global_zero:
            print(
                "SCORER_ONLY_FREEZE active: "
                f"scorer={len(self._scorer_names)} tensors/{scorer_values:,} values, "
                f"frozen={len(self._frozen_names)} tensors/{frozen_values:,} values"
            )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._set_training_modes(self._agent(pl_module))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Lightning normally calls ``train()`` only at the epoch boundary.  The
        # cheap assertion/reset here protects the invariant from future hooks.
        agent = self._agent(pl_module)
        if hasattr(agent.action_head, "proposal_path_is_training"):
            proposal_path_is_training = agent.action_head.proposal_path_is_training()
        else:
            proposal_path_is_training = (
                agent.action_head.q_former.training
                or agent.action_head.trajectory_decoder.training
            )
        if proposal_path_is_training:
            self._set_training_modes(agent)

    def on_after_backward(self, trainer, pl_module) -> None:
        agent = self._agent(pl_module)
        scorer_gradient_count = 0
        for name, parameter in agent.named_parameters():
            if is_scorer_parameter(name):
                scorer_gradient_count += int(parameter.grad is not None)
            else:
                # ``None`` is essential: AdamW skips both its moment update and
                # decoupled weight decay only when no gradient is present.
                parameter.grad = None
        if scorer_gradient_count == 0:
            raise RuntimeError("Scorer-only backward produced no scorer gradients")


class ScorerOnlyEpisodeDriveAgent(EpisodeDriveAgent):
    """EpisodeDrive agent that appends the strict scorer-only callback."""

    def name(self) -> str:
        # AgentLightningModule's released validation dispatch is name-based:
        # only names containing ``episode`` receive selected/best PDM scoring
        # and the val/score_epoch checkpoint metric.  Keep that contract for
        # every architecture subclass, regardless of its Python class name.
        return f"EpisodeDriveScorer::{self.__class__.__name__}"

    def initialize(self) -> None:
        super().initialize()
        if os.getenv("DRIVEVLA_HARD_FREEZE_SCORER_ONLY", "0") == "1":
            hard_freeze_non_scorer_parameters(self)

    def get_training_callbacks(self):
        callbacks = list(super().get_training_callbacks())
        # Tiny architecture smokes should prove that a real checkpoint can
        # load, score and backpropagate without writing another multi-GB copy
        # of the frozen VLM.  Production/pilot runs leave this unset and retain
        # the normal best/latest checkpoint policy.
        if os.getenv("DRIVEVLA_DISABLE_SCORER_CHECKPOINTS", "0") == "1":
            callbacks = [
                callback
                for callback in callbacks
                if not isinstance(callback, ModelCheckpoint)
            ]
        return [ScorerOnlyFreezeCallback(), *callbacks]


__all__ = [
    "SCORER_PARAMETER_PREFIXES",
    "ScorerOnlyEpisodeDriveAgent",
    "ScorerOnlyFreezeCallback",
    "hard_freeze_non_scorer_parameters",
    "is_scorer_parameter",
    "scorer_and_frozen_parameter_names",
]
