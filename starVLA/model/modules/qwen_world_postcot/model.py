"""One current Qwen pass; full representation and post-action auxiliary losses."""
import copy
import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import record_function
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from .qwen_encoder import SelectedQwenEncoder
from .memory import SceneMemoryProvider, PredictiveMemory
from .actor import CleanXActor
from .flow import CleanXFlowPolicy
from .readouts import DINOReadout, FutureReadout, CurrentObjectReadout, valid_scene_mse, current_object_loss
from .post_action import PostActionHead, post_action_loss

TRAIN_ONLY_PREFIXES = ('dino_readout.', 'future_readout.', 'object_readout.', 'post_head.')
DEFAULT_WEIGHTS = dict(x=10., dino=1., future=.5, object=.1, post=.2)


class QwenWorldPostCoT(nn.Module):
    def __init__(self, qwen_path, *, config=None, checkpoint_config=None, train_heads=True,
                 qwen_encoder=None):
        super().__init__()
        self.config = copy.deepcopy(config or {})
        self.qwen = qwen_encoder if qwen_encoder is not None else SelectedQwenEncoder(
            qwen_path, checkpoint_config=checkpoint_config)
        self.scene_provider = SceneMemoryProvider(**self.config.get('memory', {}))
        self.predictive_provider = PredictiveMemory(**self.config.get('predictive', {}))
        actor = CleanXActor(**self.config.get('action', {}))
        self.policy = CleanXFlowPolicy(actor, shift=5., inference_steps=10)
        self.train_heads = bool(train_heads)
        if self.train_heads:
            self.dino_readout = DINOReadout(**self.config.get('dino_readout', {}))
            self.future_readout = FutureReadout(**self.config.get('future_readout', {}))
            self.object_readout = CurrentObjectReadout(**self.config.get('object_readout', {}))
            self.post_head = PostActionHead(**self.config.get('post_head', {}))
        self.qwen_frozen_for_interface = False

    @property
    def actor(self):
        return self.policy.actor

    def configure_stage(self, interface):
        # All parameters remain in the original optimizer partition. During
        # interface updates only the current Qwen forward is no_grad/eval.
        self.qwen_frozen_for_interface = bool(interface)
        self.qwen.train(self.training and not interface)

    def train(self, mode=True):
        super().train(mode)
        if self.qwen_frozen_for_interface:
            self.qwen.eval()
        return self

    def encode_condition(self, current_inputs, ego):
        with record_function('world/current_qwen'):
            if self.qwen_frozen_for_interface:
                with torch.no_grad():
                    features = self.qwen(current_inputs)
            else:
                features = self.qwen(current_inputs)
        with record_function('world/spatial_global_memory'):
            memory = self.scene_provider(features, ego)
        with record_function('world/predictive_memory'):
            return self.predictive_provider(memory)

    def forward(self, batch, *, weights=None, denominators=None):
        if not self.train_heads:
            raise RuntimeError('Deployment model cannot execute auxiliary training')
        weights = DEFAULT_WEIGHTS if weights is None else weights
        if set(weights) != set(DEFAULT_WEIGHTS):
            raise ValueError('All five complete-model loss weights must be explicit')
        memory = self.encode_condition(batch['current_inputs'], batch['ego'])
        with record_function('world/context_KV_once'):
            context = self.actor.prepare_context(memory)
        with record_function('world/main_clean_x'):
            loss_x, x_terms = self.policy.loss(context, batch['target'], valid=batch['action_valid'],
                noise=batch.get('noise'), sigma=batch.get('sigma'))
            if denominators is not None:
                world = dist.get_world_size() if dist.is_initialized() else 1
                # Global counts may be accumulated inFP64; the supervised loss
                # and its division retain the declared FP32 reduction contract.
                denom=denominators['action_elements'].clamp_min(1).to(x_terms['numerator'].dtype)
                loss_x = x_terms['numerator'] * world / denom
        with record_function('world/current_DINO_readout'):
            dino = self.dino_readout(memory)
            loss_dino = valid_scene_mse(dino, batch['dino_target'], batch['dino_valid'],
                scene_denominator=None if denominators is None else denominators['dino'])
        with record_function('world/future_readout'):
            future = self.future_readout(memory, batch['future_queries'])
            loss_future = valid_scene_mse(future, batch['future_target'], batch['future_valid'],
                scene_denominator=None if denominators is None else denominators['future'])
        with record_function('world/current_object_readout'):
            objects = self.object_readout(memory)
            object_terms = current_object_loss(objects, batch['objects'],
                scene_denominators=None if denominators is None else denominators['object'])
            loss_object = object_terms['loss'] if isinstance(object_terms, dict) else object_terms
        with record_function('world/auxiliary_exact_candidate_actor'):
            selected = batch['aux_scene_indices'].detach().long()
            candidates = batch['candidate_trajectory'].detach().float()
            if candidates.shape != (len(selected), 2, 8, 3):
                raise ValueError('Two exact physical candidates per selected current scene required')
            candidate_to_scene = selected.repeat_interleave(2)
            normalized = ActionCodec.norm_odo(candidates).reshape(-1, 8, 3)
            replay = self.actor.forward_features(normalized,
                torch.zeros(len(normalized), device=normalized.device, dtype=torch.float32),
                context.index_select(candidate_to_scene))
        with record_function('world/post_action_head'):
            logits = self.post_head(replay).reshape(len(selected), 2, 6)
            post = post_action_loss(logits, batch['candidate_labels'].detach(),
                batch['candidate_valid'].detach(), batch['candidate_pair_mask'].detach(),
                scene_denominators=None if denominators is None else denominators['post'])
        losses = dict(x=loss_x, dino=loss_dino, future=loss_future, object=loss_object, post=post['loss'])
        total = sum(float(weights[key])*value for key, value in losses.items())
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite complete-model loss')
        metrics = {key: value.detach() for key, value in losses.items()}
        metrics.update(post_factor=post['factor'].detach(), post_pair=post['pair'].detach(),
                       sigma_mean=x_terms['sigma'].mean().detach())
        return dict(loss=total, losses=metrics)

    @torch.no_grad()
    def predict(self, current_inputs, ego, initial_noise):
        # No train-only object, teacher/cache/candidate/score input is accepted.
        with torch.autocast(initial_noise.device.type, dtype=torch.bfloat16,
                            enabled=initial_noise.device.type in ('cuda', 'cpu')):
            memory = self.encode_condition(current_inputs, ego)
            context = self.actor.prepare_context(memory)
            raw = self.policy.sample_ode(context, initial_noise.float())
        return dict(normalized_action=raw.float(), requested_trajectory=ActionCodec.denorm_odo(raw.float()))

    def deployment_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith(TRAIN_ONLY_PREFIXES)}
