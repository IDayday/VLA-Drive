"""GT-trajectory-conditioned future planning-register predictor."""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import nn


class FutureRegisterPredictor(nn.Module):
    """Predict ``[B,K,H,R,D]`` future registers from current registers/actions."""

    def __init__(
        self,
        hidden_dim: int = 256,
        predictor_layers: int = 2,
        num_heads: int = 8,
        trajectory_points: int = 8,
        horizons_sec: Sequence[float] = (0.5, 1.5, 3.0),
        dt: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if predictor_layers != 2:
            raise ValueError(
                f"PlanReg-WM-V1 requires two predictor layers, got {predictor_layers}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.trajectory_points = int(trajectory_points)
        self.dt = float(dt)
        self.horizons_sec = tuple(float(value) for value in horizons_sec)
        self.horizon_indices = tuple(
            int(round(value / self.dt)) - 1 for value in self.horizons_sec
        )
        if any(index < 0 or index >= self.trajectory_points for index in self.horizon_indices):
            raise ValueError(
                f"Horizons {self.horizons_sec} do not fit {self.trajectory_points} points"
            )

        self.point_mlp = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        trajectory_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_transformer = nn.TransformerEncoder(
            trajectory_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.horizon_embeddings = nn.Embedding(
            len(self.horizons_sec), self.hidden_dim
        )
        register_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.register_transformer = nn.TransformerEncoder(
            register_layer,
            num_layers=predictor_layers,
            enable_nested_tensor=False,
        )
        self.residual_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def trajectory_point_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (
            self.trajectory_points,
            3,
        ):
            raise ValueError(
                "trajectories must have shape [B,K,8,3], got "
                f"{tuple(trajectories.shape)}"
            )
        xy = trajectories[..., :2]
        heading = trajectories[..., 2]
        previous_xy = torch.cat((torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), dim=-2)
        speed = torch.linalg.vector_norm(xy - previous_xy, dim=-1) / self.dt
        previous_speed = torch.cat(
            (torch.zeros_like(speed[..., :1]), speed[..., :-1]), dim=-1
        )
        acceleration = (speed - previous_speed) / self.dt
        return torch.stack(
            (
                xy[..., 0],
                xy[..., 1],
                torch.sin(heading),
                torch.cos(heading),
                speed,
                acceleration,
            ),
            dim=-1,
        )

    def _validate_horizons(
        self,
        horizons: Union[torch.Tensor, Sequence[float]],
    ) -> None:
        horizon_tensor = torch.as_tensor(horizons, dtype=torch.float32).cpu()
        expected = torch.tensor(self.horizons_sec, dtype=torch.float32)
        if horizon_tensor.shape != expected.shape or not torch.allclose(
            horizon_tensor, expected, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"horizons must be {list(self.horizons_sec)}, got {horizon_tensor.tolist()}"
            )

    def forward(
        self,
        current_registers: torch.Tensor,
        trajectories: torch.Tensor,
        horizons: Union[torch.Tensor, Sequence[float]],
        *,
        use_action_condition: bool = True,
    ) -> torch.Tensor:
        self._validate_horizons(horizons)
        if current_registers.ndim != 3 or current_registers.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"current_registers must be [B,R,{self.hidden_dim}], got "
                f"{tuple(current_registers.shape)}"
            )
        if trajectories.shape[0] != current_registers.shape[0]:
            raise ValueError("Trajectory and register batch dimensions differ")

        batch_size, candidate_count = trajectories.shape[:2]
        point_features = self.trajectory_point_features(trajectories)
        trajectory_tokens = self.point_mlp(point_features).reshape(
            batch_size * candidate_count,
            self.trajectory_points,
            self.hidden_dim,
        )
        causal_mask = torch.triu(
            torch.ones(
                self.trajectory_points,
                self.trajectory_points,
                dtype=torch.bool,
                device=trajectory_tokens.device,
            ),
            diagonal=1,
        )
        trajectory_tokens = self.trajectory_transformer(
            trajectory_tokens,
            mask=causal_mask,
            is_causal=True,
        )
        trajectory_tokens = trajectory_tokens[:, self.horizon_indices]
        if not use_action_condition:
            trajectory_tokens = trajectory_tokens * 0.0
        horizon_ids = torch.arange(
            len(self.horizons_sec), device=trajectory_tokens.device
        )
        trajectory_tokens = trajectory_tokens + self.horizon_embeddings(
            horizon_ids
        ).unsqueeze(0)

        horizon_count = len(self.horizons_sec)
        register_count = current_registers.shape[1]
        current = current_registers[:, None, None].expand(
            batch_size,
            candidate_count,
            horizon_count,
            register_count,
            self.hidden_dim,
        )
        conditions = trajectory_tokens.reshape(
            batch_size,
            candidate_count,
            horizon_count,
            self.hidden_dim,
        )
        flat_current = current.reshape(
            batch_size * candidate_count * horizon_count,
            register_count,
            self.hidden_dim,
        )
        flat_conditions = conditions.reshape(
            batch_size * candidate_count * horizon_count,
            1,
            self.hidden_dim,
        )
        predictor_tokens = torch.cat((flat_conditions, flat_current), dim=1)
        predicted_tokens = self.register_transformer(predictor_tokens)[:, 1:]
        residual = self.residual_output(predicted_tokens)
        predicted = self.output_norm(flat_current + residual)
        return predicted.reshape(
            batch_size,
            candidate_count,
            horizon_count,
            register_count,
            self.hidden_dim,
        )
