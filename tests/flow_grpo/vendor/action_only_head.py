# Source: IDayday/VLA-Drive f9449d55bea6895a7a0bd86d09d7ab85fd353f26
# starVLA/model/modules/action_model/GR00T_ActionHeader.py
# Copyright 2025 NVIDIA Corp. and affiliates. Modified by Junqiu YU/Fudan, 2025.
# Test-only AST extraction; repository license Apache-2.0.
def merge_action_context(vl_embs: torch.Tensor, extra_context: torch.Tensor | None) -> torch.Tensor:
    """Append optional planner context to VLM queries.

    Args:
        vl_embs: Action-query features ``[B, A, H]``.
        extra_context: Optional context features ``[B, Q, H]``.

    Returning ``vl_embs`` itself for ``None`` preserves the baseline path
    exactly, including storage identity and numerical behavior.
    """
    assert vl_embs.ndim == 3, 'vl_embs must be [B,A,H]'
    if extra_context is None:
        return vl_embs
    assert extra_context.ndim == 3, 'extra_context must be [B,Q,H]'
    assert extra_context.shape[0] == vl_embs.shape[0], 'action context batch mismatch'
    assert extra_context.shape[2] == vl_embs.shape[2], 'action context hidden dim mismatch'
    assert extra_context.device == vl_embs.device, 'action context device mismatch'
    return torch.cat((vl_embs, extra_context.to(dtype=vl_embs.dtype)), dim=1)

class FlowmatchingActionHead:

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, video_token=None, state: torch.Tensor=None, extra_context: torch.Tensor | None=None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if state is not None else None
        vl_embs = self.qwen_proj(merge_action_context(vl_embs, extra_context))
        if video_token is not None:
            vl_embs = torch.cat((vl_embs, video_token), dim=1)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = action_features
        model_output = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t_discretized, return_all_hidden_states=False)
        pred = self.action_decoder(model_output)
        pred_actions = pred
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor=None, extra_context: torch.Tensor | None=None) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(size=(batch_size, self.config.action_horizon, self.config.action_dim), dtype=vl_embs.dtype, device=device)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None
        vl_embs = self.qwen_proj(merge_action_context(vl_embs, extra_context))
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = action_features
            model_output = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timesteps_tensor)
            pred = self.action_decoder(model_output)
            pred_velocity = pred
            actions = actions + dt * pred_velocity
        return actions
