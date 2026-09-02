"""GT-trajectory-conditioned future planning-register predictor."""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn.functional as F
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
        normalize_state_space: bool = True,
        x_scale: float = 30.0,
        y_scale: float = 10.0,
        speed_scale: float = 15.0,
        acceleration_scale: float = 8.0,
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
        self.normalize_state_space = bool(normalize_state_space)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.speed_scale = float(speed_scale)
        self.acceleration_scale = float(acceleration_scale)
        for name in (
            "x_scale",
            "y_scale",
            "speed_scale",
            "acceleration_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
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

    def normalize_register_state(self, registers: torch.Tensor) -> torch.Tensor:
        if not self.normalize_state_space:
            return registers
        return F.layer_norm(
            registers,
            normalized_shape=(self.hidden_dim,),
            weight=None,
            bias=None,
        )

    def _resolve_current_speed(
        self,
        trajectories: torch.Tensor,
        current_speed: torch.Tensor = None,
        current_ego_motion: torch.Tensor = None,
    ) -> torch.Tensor:
        if current_speed is not None and current_ego_motion is not None:
            raise ValueError("Provide current_speed or current_ego_motion, not both")
        batch_size, candidate_count = trajectories.shape[:2]
        if current_ego_motion is not None:
            motion = torch.as_tensor(
                current_ego_motion,
                device=trajectories.device,
                dtype=trajectories.dtype,
            )
            if motion.shape != (batch_size, 2):
                raise ValueError(
                    "current_ego_motion must be [B,2] velocity, got "
                    f"{tuple(motion.shape)}"
                )
            current_speed = torch.linalg.vector_norm(motion, dim=-1)
        if current_speed is None:
            return trajectories.new_zeros(batch_size, candidate_count)
        speed = torch.as_tensor(
            current_speed,
            device=trajectories.device,
            dtype=trajectories.dtype,
        )
        if speed.ndim == 2 and speed.shape == (batch_size, 1):
            speed = speed[:, 0]
        if speed.ndim == 1 and speed.shape[0] == batch_size:
            return speed[:, None].expand(-1, candidate_count)
        if speed.shape == (batch_size, candidate_count):
            return speed
        raise ValueError(
            "current_speed must be [B], [B,1], or [B,K], got "
            f"{tuple(speed.shape)}"
        )

    def trajectory_point_features(
        self,
        trajectories: torch.Tensor,
        *,
        current_speed: torch.Tensor = None,
        current_ego_motion: torch.Tensor = None,
    ) -> torch.Tensor:
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
        initial_speed = self._resolve_current_speed(
            trajectories,
            current_speed=current_speed,
            current_ego_motion=current_ego_motion,
        )
        previous_speed = torch.cat(
            (initial_speed[..., None], speed[..., :-1]), dim=-1
        )
        acceleration = (speed - previous_speed) / self.dt
        return torch.stack(
            (
                xy[..., 0] / self.x_scale,
                xy[..., 1] / self.y_scale,
                torch.sin(heading),
                torch.cos(heading),
                speed / self.speed_scale,
                acceleration / self.acceleration_scale,
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
        current_speed: torch.Tensor = None,
        current_ego_motion: torch.Tensor = None,
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
        point_features = self.trajectory_point_features(
            trajectories,
            current_speed=current_speed,
            current_ego_motion=current_ego_motion,
        )
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
        current_n = self.normalize_register_state(current_registers)
        current = current_n[:, None, None].expand(
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
        predicted = flat_current + residual
        return predicted.reshape(
            batch_size,
            candidate_count,
            horizon_count,
            register_count,
            self.hidden_dim,
        )
