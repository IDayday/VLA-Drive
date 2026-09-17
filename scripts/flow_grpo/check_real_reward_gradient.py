"""RL-only backward using the first actual training group and its official rewards.

No seed/candidate selection, diagnostic advantages, resampling, or optimizer step.
Only locally generated trusted rollout files may be passed here.
"""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import apply_rl_freezes, grad_summary
from starVLA.rl.flow_grpo.loading import load_policy, enable_checkpointing
from starVLA.rl.flow_grpo.rollout import evaluate_transitions
from starVLA.rl.flow_grpo.math import (
    reduce_dimensions,
    clipped_surrogate,
    group_advantages,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, sft = resolve_config(args.config)
    # The first stored group from the first update, never selected by its score.
    rollout = torch.load(args.rollout, map_location="cuda", weights_only=False)[0]
    assert rollout.policy_version == 0
    rollout.validate(0)
    assert rollout.provenance["sft_sha256"] == cfg["checkpoint_contract"]["sha256"]
    algorithm = cfg["algorithm"]
    advantage = group_advantages(
        rollout.rewards,
        epsilon=algorithm["advantage_epsilon"],
        clip=algorithm["advantage_clip"],
    )
    torch.testing.assert_close(advantage, rollout.advantages, atol=0, rtol=0)
    if not advantage.count_nonzero():
        raise ValueError(
            "first actual group is all-equal; no nonzero RL gradient may be claimed"
        )
    policy = load_policy(cfg, sft).cuda().eval().bfloat16()
    apply_rl_freezes(policy, cfg)
    enable_checkpointing(policy, True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        stats = evaluate_transitions(
            policy, rollout.observation, rollout, checkpoint=True
        )
        current = reduce_dimensions(
            stats["elementwise_logprob"], rollout.dimension_mask, rollout.spec.reduction
        )
        logratio = current - rollout.old_logprob
        torch.testing.assert_close(
            logratio, torch.zeros_like(logratio), atol=1e-5, rtol=0
        )
        pg, ratio = clipped_surrogate(
            current,
            rollout.old_logprob,
            advantage[:, :, None],
            algorithm["ppo_clip_range"],
        )
        loss = (pg * rollout.transition_mask).sum() / rollout.transition_mask.sum()
    loss.backward()
    nonzero, missing, invalid = [], [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
        elif parameter.grad is None or not parameter.grad.count_nonzero():
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            invalid.append(name)
        else:
            nonzero.append(name)
    result = dict(
        status="TESTED" if not missing and not invalid else "FAILED",
        scope="RL-only; first real training group, unchanged official NAVSIM rewards/advantages; no optimizer step",
        rollout=str(args.rollout),
        tokens=rollout.observation.tokens,
        checkpoint_sha256=policy._flow_source_sha256,
        rewards=rollout.rewards.tolist(),
        advantages=advantage.tolist(),
        max_abs_logratio=float(logratio.detach().abs().max()),
        loss=float(loss.detach()),
        modules=grad_summary(policy),
        nonzero_parameter_names=nonzero,
        missing_or_zero_gradients=missing,
        nonfinite_gradients=invalid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(result["status"], "nonzero gradient tensors", len(nonzero), flush=True)
    assert not missing and not invalid


if __name__ == "__main__":
    main()
