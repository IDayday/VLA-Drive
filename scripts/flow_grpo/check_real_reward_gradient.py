"""RL-only backward using the fixed actual training groups and their official rewards.

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
    parser.add_argument("--rollout", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Use every group in every supplied rank file, including equal-reward groups",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, sft = resolve_config(args.config)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.rollout) != 1 and not args.all_groups:
        raise ValueError("multiple rank files require explicit --all-groups")
    buffers = []
    for path in args.rollout:
        saved = torch.load(path, map_location="cuda", weights_only=False)
        buffers.extend(saved if args.all_groups else saved[:1])
    if not buffers:
        raise ValueError("empty behavior batch")
    policy = load_policy(cfg, sft).cuda().eval().bfloat16()
    apply_rl_freezes(policy, cfg)
    enable_checkpointing(policy, True)
    algorithm, rows = cfg["algorithm"], []
    for rollout in buffers:
        assert rollout.policy_version == 0
        rollout.validate(0)
        assert rollout.provenance["sft_sha256"] == cfg["checkpoint_contract"]["sha256"]
        advantage = group_advantages(
            rollout.rewards,
            epsilon=algorithm["advantage_epsilon"],
            clip=algorithm["advantage_clip"],
        )
        torch.testing.assert_close(advantage, rollout.advantages, atol=0, rtol=0)
        # Every supplied scene contributes with equal weight, even if all G rewards
        # are equal. This is graph coverage, NOT ZeRO optimizer/accumulation evidence.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stats = evaluate_transitions(
                policy, rollout.observation, rollout, checkpoint=True
            )
            current = reduce_dimensions(
                stats["elementwise_logprob"],
                rollout.dimension_mask,
                rollout.spec.reduction,
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
            loss = (
                (pg * rollout.transition_mask).sum()
                / rollout.transition_mask.sum()
                / len(buffers)
            )
        loss.backward()
        rows.append(
            dict(
                tokens=rollout.observation.tokens,
                rewards=rollout.rewards.tolist(),
                advantages=advantage.tolist(),
                nonzero_advantage=bool(advantage.count_nonzero()),
                loss=float(loss.detach()),
                max_abs_logratio=float(logratio.detach().abs().max()),
            )
        )
        del stats, current, pg, ratio, loss, logratio
    effective = sum(row["nonzero_advantage"] for row in rows)
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
        status="TESTED" if effective and not missing and not invalid else "FAILED",
        scope="RL-only CUDA BF16 graph coverage of all supplied fixed first behavior groups; official rewards unchanged; no optimizer, no ZeRO accumulation claim",
        rollouts=list(map(str, args.rollout)),
        groups=rows,
        official_nonzero_advantage_groups=effective,
        checkpoint_sha256=policy._flow_source_sha256,
        max_abs_logratio=max(row["max_abs_logratio"] for row in rows),
        modules=grad_summary(policy),
        nonzero_parameter_names=nonzero,
        missing_or_zero_gradients=missing,
        nonfinite_gradients=invalid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(result["status"], "nonzero gradient tensors", len(nonzero), flush=True)
    assert effective and not missing and not invalid


if __name__ == "__main__":
    main()
