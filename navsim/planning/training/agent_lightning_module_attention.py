from typing import Dict, Tuple

import torch
from torch import Tensor

from navsim.planning.training.agent_lightning_module import AgentLightningModule


class AgentLightningModuleAttention(AgentLightningModule):
    """Lightning adapter that forwards sample tokens to the memory bank."""

    @staticmethod
    def _attach_tokens(
        features: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        if "tokens" in features:
            return features
        tokens = targets.get("token")
        if tokens is None:
            raise KeyError(
                "Attention training requires targets['token'] or features['tokens']"
            )
        output = dict(features)
        output["tokens"] = tokens
        return output

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ) -> Tensor:
        if getattr(self.agent._config, "memory_mode", "") != "attention":
            return super()._step(batch, logging_prefix)

        features, targets = batch
        features = self._attach_tokens(features, targets)
        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                )
            return loss_dict["loss"]
        return loss_dict

    def validation_step(self, batch, batch_idx):
        if getattr(self.agent._config, "memory_mode", "") == "attention":
            return self._step(batch, "val")
        return super().validation_step(batch, batch_idx)
