"""Current visual world → all-masked joint rollout → original ego DiT bridge."""
import torch
from torch import nn
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter
from .flow import JointTrajectoryFlow


class JointTrajectoryPolicy(nn.Module):
    def __init__(self, baseline, world_config, graph_config):
        super().__init__()
        if world_config.get('token_layout') != 'append_tail':
            raise ValueError('Native policy preservation requires append_tail')
        world_config = dict(world_config, enabled=True, action_adapter=False, return_world_features=True)
        self.world = StructuredWorldPolicy(baseline, world_config)
        self.graph_config = dict(graph_config)
        dim = baseline.qwen_vl_interface.model.get_input_embeddings().embedding_dim
        horizon = baseline.config.framework.action_model.action_horizon
        self.graph = JointTrajectoryFlow(dim, steps=horizon, **{
            key: value for key, value in graph_config.items() if key in ('dim', 'heads', 'layers', 'scale_m')})
        self.graph_to_world = nn.Linear(graph_config.get('dim', 256), dim)
        self.adapter = WorldToActionAdapter(dim)

    def encode_current(self, examples, model_inputs=None):
        # Explicit whitelist strips actions, WorldTargets and any future metadata.
        current = [{key: e[key] for key in ('image', 'lang', 'state', 'token') if key in e} for e in examples]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=next(self.parameters()).is_cuda):
            actions, pred = self.world.encode_conditions(current, model_inputs)
        b = actions.shape[0]
        actor = torch.cat([actions.mean(1, keepdim=True), pred['agent_features']], 1).float()
        context = torch.cat([actions, pred['scene_features']], 1).float()
        xy = torch.cat([pred['boxes'].new_zeros(b, 1, 2), pred['boxes'][..., :2]], 1).detach().float()
        confidence = 1 - pred['logits'].softmax(-1)[..., -1]
        existence = torch.cat([confidence.new_ones(b, 1), confidence], 1).detach().float()
        return actions, pred, dict(actor_features=actor, context=context, current_xy=xy, existence=existence)

    def rollout_condition(self, native_actions, current, noise):
        with torch.autocast('cuda', enabled=False):
            trajectories, features = self.graph.sample(noise.float(), **current,
                sampling_steps=self.graph_config.get('sampling_steps', 10))
            # No imputation loss/GT-conditioned hidden states can enter this method.
            conditions = self.adapter(native_actions.float(), self.graph_to_world(features.float()))
        # Qwen hidden features may be BF16; the released DiT consumes FP32.
        # Retain the adapter residual in FP32 rather than quantizing it back to BF16.
        return conditions.float(), trajectories, features

    @torch.no_grad()
    def predict_action(self, examples, model_inputs=None, graph_noise=None):
        native, prediction, current = self.encode_current(examples, model_inputs)
        if graph_noise is None:
            # Separate RNG stream: adding the world graph does not shift ego FM noise.
            generator = torch.Generator(device=native.device).manual_seed(self.graph_config.get('sampling_seed', 2037))
            graph_noise = torch.randn(len(examples), current['actor_features'].shape[1], self.graph.steps, 2,
                                      device=native.device, generator=generator)
        condition, trajectories, _ = self.rollout_condition(native, current, graph_noise)
        head = self.world.baseline.action_model
        # Match the original BF16 random draw even with an FP32 learned residual.
        ego_noise = torch.randn(len(examples), head.config.action_horizon, head.config.action_dim,
                                device=native.device, dtype=native.dtype)
        with torch.autocast('cuda', dtype=torch.float32):
            actions = head.predict_action(condition, initial_noise=ego_noise)
        return {'normalized_actions': actions.cpu().numpy(), 'joint_trajectories_xy': trajectories,
                'world_prediction': prediction}
