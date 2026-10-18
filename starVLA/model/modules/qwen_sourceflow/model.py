"""Native Qwen observation expert + source MoT action strategy, no DDP dependencies."""
import torch
from torch import nn
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenEncoder
from starVLA.model.modules.qwen_sourceflow.condition_provider import QwenConditionProvider
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


class QwenSourceFlowVLA(nn.Module):
    def __init__(self, qwen_path, action_expert, *, checkpoint_config=None):
        super().__init__()
        self.qwen = QwenEncoder(qwen_path, checkpoint_config=checkpoint_config)
        self.provider = QwenConditionProvider(action_expert, self.qwen.hidden_dim, self.qwen.num_layers)
        self.policy = SourceFlowPolicy(action_expert)

    def encode_condition(self, observation_batch, proprio):
        return self.provider(self.qwen(observation_batch), proprio)

    def forward(self, observation_batch, proprio, normalized_action, **fm_randomness):
        condition = self.encode_condition(observation_batch, proprio)
        return self.policy(condition, normalized_action, **fm_randomness)

    @torch.no_grad()
    def predict(self, observation_batch, proprio, initial_noise):
        with torch.autocast(initial_noise.device.type,dtype=torch.bfloat16):
            condition = self.encode_condition(observation_batch, proprio)
        with torch.autocast(initial_noise.device.type,enabled=False):
            raw = self.policy.sample_ode(condition, initial_noise)
        return dict(normalized_action=raw, requested_trajectory=ActionCodec.denorm_odo(raw))
