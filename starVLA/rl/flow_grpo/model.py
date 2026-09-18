"""Wrapped-forward training modes; full SFT replay and independent reference."""
import copy
import torch
from torch import nn
from .rollout import sample_chain, evaluate_transitions
from .math import reduce_dimensions, conditional_kl, clipped_surrogate


def make_reference(policy):
    """Clone the FULL action dependency closure, including Qwen and world queries.

    Pure auxiliary decoders and their GRU/readouts are absent. No actor storage is
    shared. This object is not a child of the trainable actor/DeepSpeed engine.
    """
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT

    reference = Qwenvl_OFT.__new__(Qwenvl_OFT)
    nn.Module.__init__(reference)
    for name in (
        "config",
        "_special_token_ids",
        "act_tok",
        "w_depth",
        "robot_history_token",
        "rgb_query_tokens",
        "gs_query_tokens",
        "act_query_tokens",
        "reward_query_tokens",
        "action_prompt_mode",
    ):
        setattr(reference, name, copy.deepcopy(getattr(policy, name)))
    for name in (
        "qwen_vl_interface",
        "action_input_model",
        "action_model",
        "rgb_query",
        "gs_query",
    ):
        if hasattr(policy, name):
            setattr(reference, name, copy.deepcopy(getattr(policy, name)))
    reference.requires_grad_(False).eval()
    actor_ptrs = {p.data_ptr() for p in policy.parameters()}
    if any(p.data_ptr() in actor_ptrs for p in reference.parameters()):
        raise RuntimeError("reference shares actor parameter storage")
    return reference


class FlowGRPOActor(nn.Module):
    def __init__(self, policy, rl_config):
        super().__init__()
        self.policy = policy
        self.rl_config = rl_config

    def forward(self, mode, **kwargs):
        if mode == "rollout":
            with torch.no_grad():
                return sample_chain(self.policy, **kwargs)
        if mode == "transitions":
            return evaluate_transitions(self.policy, **kwargs)
        if mode == "sft":
            return self.policy.compute_sft_losses(**kwargs)
        if mode != "update":
            raise ValueError(f"unknown Flow-GRPO forward mode {mode}")
        rollout = kwargs["rollout"]
        cfg = self.rl_config
        stats = evaluate_transitions(
            self.policy,
            rollout.observation,
            rollout,
            checkpoint=cfg["runtime"]["activation_checkpointing"],
        )
        current = reduce_dimensions(
            stats["elementwise_logprob"], rollout.dimension_mask, rollout.spec.reduction
        )
        pg, ratio = clipped_surrogate(
            current,
            rollout.old_logprob,
            rollout.advantages[:, :, None],
            cfg["algorithm"]["ppo_clip_range"],
        )
        if rollout.reference_mean is None:
            raise ValueError("missing full SFT reference statistics")
        kl = reduce_dimensions(
            conditional_kl(
                stats["mean"],
                stats["std"],
                rollout.reference_mean,
                rollout.reference_std,
                temporal_correlation=getattr(rollout.spec, "temporal_noise_correlation", 0.0),
            ),
            rollout.dimension_mask,
            rollout.spec.reduction,
        )
        valid = rollout.transition_mask
        per_scene_count = valid.sum((1, 2))
        if (per_scene_count == 0).any():
            raise ValueError("scene has no valid transitions")
        grpo = (pg * valid).sum((1, 2)) / per_scene_count
        reference = (kl * valid).sum((1, 2)) / per_scene_count
        # Each replay scene exactly once. Original losses retain internal masks
        # and weights (including depth*0.1), independent of candidate/time counts.
        replay = kwargs["replay"]
        components = {}
        for sample in replay:
            if cfg["runtime"].get("noise_seed_schedule") == "global_scene_v1":
                # Same original SFT random draws for the same scene key across
                # microbatch/rank layouts. These keys never enter observations.
                device = next(self.policy.parameters()).device
                devices = [device.index] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    seed = int(sample["_flow_sample_seed"])
                    torch.random.default_generator.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.default_generators[device.index].manual_seed(seed)
                    values = self.policy.compute_sft_losses([sample])
            else:
                values = self.policy.compute_sft_losses([sample])
            for key, value in values.items():
                components[key] = components.get(key, 0) + value / len(replay)
        sft = sum(components.values())
        total = (
            grpo.mean()
            + cfg["algorithm"]["reference_kl_coefficient"] * reference.mean()
            + cfg["retention"]["original_sft_coefficient"] * sft
        )
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite joint loss")
        result = dict(
            loss=total,
            grpo=grpo.mean(),
            reference=reference.mean(),
            sft=sft,
            components=components,
            ratio=ratio.detach(),
            logratio=(current - rollout.old_logprob).detach(),
        )
        if kwargs.get("diagnostic_outputs", False):
            result["diagnostics"] = {
                **stats,
                "current_logprob": current,
                "per_transition_pg": pg,
                "per_transition_kl": kl,
            }
        return result
