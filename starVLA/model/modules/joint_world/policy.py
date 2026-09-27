"""Current visual world → all-masked joint rollout → original ego DiT bridge."""
import torch
from torch import nn
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter
from .flow import JointTrajectoryFlow


class JointTrajectoryPolicy(nn.Module):
    def __init__(self, baseline, world_config, graph_config, bev_config=None):
        super().__init__()
        if world_config.get('token_layout') != 'append_tail':
            raise ValueError('Native policy preservation requires append_tail')
        world_config = dict(world_config, enabled=True, action_adapter=False, return_world_features=True)
        self.world = StructuredWorldPolicy(baseline, world_config)
        self.graph_config = dict(graph_config)
        dim = baseline.qwen_vl_interface.model.get_input_embeddings().embedding_dim
        horizon = baseline.config.framework.action_model.action_horizon
        self.graph = JointTrajectoryFlow(dim, steps=horizon, **{
            key: value for key, value in graph_config.items() if key in ('dim', 'heads', 'layers', 'scale_m', 'trajectory_mode', 'agent_scale_m', 'edge_feature_dim')})
        self.graph_to_world = nn.Linear(graph_config.get('dim', 256), dim)
        self.adapter = WorldToActionAdapter(dim)
        self.bev_config = dict(bev_config or {})
        self.bev_provider = self.bev_encoder = self.bev_fusion = self.interaction_head = None
        if self.bev_config.get('enabled', False):
            import os
            from starVLA.model.modules.structured_world.pretrained_bev import PretrainedVisualBEVProvider
            from .bev_tasks import TaskBEVEncoder, BEVGraphFusion, InteractionGeometryHead
            if self.world.provider is not None:
                raise ValueError('BEV side memory currently combines with the original Qwen image path')
            self.bev_provider = PretrainedVisualBEVProvider(os.path.expandvars(self.bev_config['weights']),
                                                          self.bev_config['weights_sha256'])
            self.bev_encoder = TaskBEVEncoder(1024, 128, self.bev_provider.grid_shape, horizon)
            self.bev_fusion = BEVGraphFusion(dim, 128)
            self.interaction_head = InteractionGeometryHead(128)

    def fuse_bev(self, current, features, coordinates, support):
        if self.bev_encoder is None:
            raise ValueError('BEV feature path is disabled')
        with torch.autocast('cuda', enabled=False):
            prediction = self.bev_encoder(features, coordinates, support)
            actors, bev_actors = self.bev_fusion(current['actor_features'], current['current_xy'],
                                                 prediction['memory'], support)
            prediction['pair_separation'] = self.interaction_head(bev_actors)
        return dict(current, actor_features=actors), prediction

    def encode_current(self, examples, model_inputs=None, return_bev_aux=False, local_observations=None):
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
        current = dict(actor_features=actor, context=context, current_xy=xy, existence=existence)
        if self.graph_config.get('local_selector') is not None:
            if local_observations is None or len(local_observations)!=b:raise ValueError('Local policy needs current calibrated observations')
            from .local_graph import build_local_graph,stack_graphs,gather_current
            graphs=[build_local_graph(pred['boxes'][i].detach().cpu().numpy(),pred['logits'][i].detach().cpu().numpy(),
                                      local_observations[i],self.graph_config['local_selector']) for i in range(b)]
            current=gather_current(current,stack_graphs(graphs,actor.device))
        bev_aux = None
        if self.bev_provider is not None:
            if model_inputs is None:
                raise ValueError('BEV requires calibrated current-only ModelInputs')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features, coordinates, support, metadata = self.bev_provider(model_inputs)
            current, bev_aux = self.fuse_bev(current, features, coordinates, support)
        if return_bev_aux:
            return actions, pred, current, bev_aux
        return actions, pred, current

    def rollout_condition(self, native_actions, current, noise):
        with torch.autocast('cuda', enabled=False):
            trajectories, features = self.graph.sample(noise.float(), **current,
                sampling_steps=self.graph_config.get('sampling_steps', 10))
            # No imputation loss/GT-conditioned hidden states can enter this method.
            graph=current.get('local_graph');mask=graph.active_actor_mask if graph is not None else None
            safe=features.float() if mask is None else torch.where(mask[...,None],features.float(),torch.zeros_like(features.float()))
            conditions = self.adapter(native_actions.float(), self.graph_to_world(safe),mask)
        # Qwen hidden features may be BF16; the released DiT consumes FP32.
        # Retain the adapter residual in FP32 rather than quantizing it back to BF16.
        return conditions.float(), trajectories, features

    @torch.no_grad()
    def predict_action(self, examples, model_inputs=None, graph_noise=None, local_observations=None):
        native, prediction, current = self.encode_current(examples, model_inputs,local_observations=local_observations)
        if graph_noise is None:
            # Separate RNG stream: adding the world graph does not shift ego FM noise.
            if 'local_graph' in current:
                from .local_masks import stable_noise
                graph_noise=stable_noise([e['token'] for e in examples],current['local_graph'].source_slot_ids,self.graph.steps,
                                         self.graph_config.get('sampling_seed',2037),device=native.device)
            else:
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
