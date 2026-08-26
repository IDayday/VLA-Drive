# SPDX-License-Identifier: Apache-2.0
# Proposal-head topology adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a and ZebinX/DriveVLA-M0 commit
# 7fabe160fc9bb41f9278845b36d457bf871f697a.

"""Deterministic one-register-per-trajectory proposal generator."""

from __future__ import annotations

from typing import List

from torch import Tensor, nn

from .decoder import RegisterTrajectoryDecoder
from .outputs import RegisterGeneratorOutput


class ProposalHead(nn.Module):
    """Donor MLP mapping one register to one complete physical trajectory."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class RegisterTrajectoryGenerator(nn.Module):
    """Generate K trajectories in one four-layer decoder forward.

    There is no random noise, Flow time, Euler solver, chunking, stochastic
    sampling, source embedding, or scorer-to-generator feedback in this class.
    """

    def __init__(
        self,
        *,
        proposal_num: int = 64,
        num_poses: int = 8,
        state_dim: int = 3,
        model_dim: int = 256,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 1,
        one_token_per_trajectory: bool = True,
        proj_drop: float = 0.1,
        drop_path: float = 0.2,
        layer_scale_init: float = 0.0,
        ego_state_dim: int = 4,
        stage_loss_mode: str = "final_only",
    ) -> None:
        super().__init__()
        if not one_token_per_trajectory:
            raise ValueError("Register generator requires one token per trajectory")
        if stage_loss_mode not in {"final_only", "all_layers"}:
            raise ValueError(
                "stage_loss_mode must be 'final_only' or 'all_layers'"
            )
        for name, value in (
            ("proposal_num", proposal_num),
            ("num_poses", num_poses),
            ("state_dim", state_dim),
            ("model_dim", model_dim),
            ("ffn_dim", ffn_dim),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("ego_state_dim", ego_state_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        self.proposal_num = int(proposal_num)
        self.num_poses = int(num_poses)
        self.state_dim = int(state_dim)
        self.model_dim = int(model_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ego_state_dim = int(ego_state_dim)
        self.one_token_per_trajectory = True
        self.proposal_head_style = "donor_mlp_v1"
        self.stage_loss_mode = str(stage_loss_mode)

        self.trajectory_registers = nn.Embedding(self.proposal_num, self.model_dim)
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_state_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.model_dim),
        )
        self.trajectory_decoder = RegisterTrajectoryDecoder(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale_init=layer_scale_init,
            return_intermediate=True,
        )
        output_dim = self.num_poses * self.state_dim
        if self.stage_loss_mode == "final_only":
            self.final_proposal_head = ProposalHead(
                self.model_dim, self.ffn_dim, output_dim
            )
            self.proposal_heads = None
        else:
            self.final_proposal_head = None
            self.proposal_heads = nn.ModuleList(
                [
                    ProposalHead(self.model_dim, self.ffn_dim, output_dim)
                    for _ in range(self.num_layers + 1)
                ]
            )

    @property
    def proposal_head_count(self) -> int:
        return 1 if self.stage_loss_mode == "final_only" else self.num_layers + 1

    def _validate_inputs(self, scene_tokens: Tensor, ego_state: Tensor) -> Tensor:
        if scene_tokens.ndim != 3 or scene_tokens.shape[-1] != self.model_dim:
            raise ValueError(
                f"scene_tokens must have shape [B,S,{self.model_dim}]"
            )
        if ego_state.ndim == 3 and ego_state.shape[1] == 1:
            ego_state = ego_state[:, 0]
        expected = (scene_tokens.shape[0], self.ego_state_dim)
        if ego_state.ndim != 2 or tuple(ego_state.shape) != expected:
            raise ValueError(
                f"ego_state must have shape {expected} or [B,1,{self.ego_state_dim}]"
            )
        return ego_state.to(device=scene_tokens.device, dtype=scene_tokens.dtype)

    def _decode_head(self, head: nn.Module, tokens: Tensor) -> Tensor:
        batch_size = tokens.shape[0]
        return head(tokens).reshape(
            batch_size,
            self.proposal_num,
            self.num_poses,
            self.state_dim,
        )

    def forward(
        self, scene_tokens: Tensor, ego_state: Tensor
    ) -> RegisterGeneratorOutput:
        ego_state = self._validate_inputs(scene_tokens, ego_state)
        batch_size = scene_tokens.shape[0]
        ego_token = self.ego_encoder(ego_state)[:, None]
        register_tokens = self.trajectory_registers.weight[None].expand(
            batch_size, -1, -1
        ) + ego_token

        token_outputs = self.trajectory_decoder(register_tokens, scene_tokens)
        if not isinstance(token_outputs, list) or len(token_outputs) != self.num_layers:
            raise RuntimeError("Register decoder violated its intermediate-output contract")
        if self.stage_loss_mode == "final_only":
            if self.final_proposal_head is None or self.proposal_heads is not None:
                raise RuntimeError("final-only proposal-head topology is inconsistent")
            proposal_list: List[Tensor] = [
                self._decode_head(self.final_proposal_head, token_outputs[-1])
            ]
        else:
            if self.proposal_heads is None or self.final_proposal_head is not None:
                raise RuntimeError("all-layers proposal-head topology is inconsistent")
            proposal_list = [
                self._decode_head(self.proposal_heads[0], register_tokens)
            ]
            for layer_index, layer_tokens in enumerate(token_outputs):
                proposal_list.append(
                    self._decode_head(
                        self.proposal_heads[layer_index + 1], layer_tokens
                    )
                )
        return RegisterGeneratorOutput(
            proposals=proposal_list[-1],
            proposal_list=proposal_list,
            final_tokens=token_outputs[-1],
            token_list=token_outputs,
        )

    def architecture_metadata(self) -> dict:
        return {
            "proposal_num": self.proposal_num,
            "num_poses": self.num_poses,
            "state_dim": self.state_dim,
            "model_dim": self.model_dim,
            "ffn_dim": self.ffn_dim,
            "decoder_layers": self.num_layers,
            "decoder_heads": self.num_heads,
            "one_token_per_trajectory": self.one_token_per_trajectory,
            "proposal_head_style": self.proposal_head_style,
            "stage_loss_mode": self.stage_loss_mode,
            "proposal_head_count": self.proposal_head_count,
        }
