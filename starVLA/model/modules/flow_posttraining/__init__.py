"""Direct original-DiT post-training with detached canonical outcome feedback."""
from .fpo_objective import (FPOConfig, CFMSamples, sample_cfm, cfm_coordinate_mean,
                            fpo_ratio, group_centered_advantages, global_action_count, fpo_surrogate)
from .policy_snapshot import (DiTFlowPolicy, FrozenPolicySnapshot, PolicyIdentity,
                              FlowRandomStreams, module_hash, tensor_hash, policy_eval_mode)
from .rollout_contract import (RolloutBatch, FeedbackBatch, FeedbackReuseGuard, sample_rollout_group,
                               CoupledFPOBatch, prepare_fpo_batch)

__all__ = ["FPOConfig", "CFMSamples", "sample_cfm", "cfm_coordinate_mean", "fpo_ratio",
           "group_centered_advantages", "global_action_count", "fpo_surrogate", "DiTFlowPolicy",
           "FrozenPolicySnapshot", "PolicyIdentity", "FlowRandomStreams", "module_hash", "tensor_hash",
           "policy_eval_mode", "RolloutBatch", "FeedbackBatch", "FeedbackReuseGuard", "sample_rollout_group",
           "CoupledFPOBatch", "prepare_fpo_batch"]
