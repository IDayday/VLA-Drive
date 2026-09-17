"""Real-checkpoint gate: perturb Qwen/action weights on a fixed saved SDE chain."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.contracts import apply_rl_freezes
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.rollout import (
    SamplingSpec,
    sample_chain,
    evaluate_transitions,
)
from starVLA.rl.flow_grpo.math import reduce_dimensions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, sft = resolve_config(args.config)
    tokens, _ = split_tokens(cfg)
    sample = KeyedDataset(sft)[(tokens[0], cfg["runtime"]["seed"])]
    obs = prepare_policy_observation([sample])
    policy = load_policy(cfg, sft).cuda().eval().bfloat16()
    apply_rl_freezes(policy, cfg)
    spec = SamplingSpec(
        group_size=cfg["sampling"]["group_size"],
        num_steps=cfg["sampling"]["num_steps"],
        noise_level=cfg["sampling"]["noise_level"],
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        chain = sample_chain(policy, obs, spec, 0, 42, {"diagnostic": True})
        before = evaluate_transitions(policy, obs, chain)
    base = reduce_dimensions(before["elementwise_logprob"]) - chain.old_logprob
    torch.testing.assert_close(base, torch.zeros_like(base), atol=1e-5, rtol=0)
    result = {
        "status": "TESTED",
        "scene": tokens[0],
        "checkpoint": cfg["sft_checkpoint"],
        "checkpoint_sha256": policy._flow_source_sha256,
        "unperturbed_max_abs_logratio": float(base.abs().max()),
        "perturbations": {},
    }
    for name, parameter in [
        (
            "qwen",
            policy.qwen_vl_interface.model.model.language_model.layers[
                0
            ].self_attn.q_proj.weight,
        ),
        ("action", policy.action_model.action_decoder.layer2.weight),
    ]:
        original = parameter.detach().clone()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                parameter.add_(0.01)
                changed = evaluate_transitions(policy, obs, chain)
            delta = (
                reduce_dimensions(changed["elementwise_logprob"]) - chain.old_logprob
            )
            ratio = delta.exp()
            assert torch.isfinite(ratio).all() and float(delta.abs().max()) > 1e-5
            result["perturbations"][name] = dict(
                max_abs_logratio=float(delta.abs().max()),
                ratio_min=float(ratio.min()),
                ratio_max=float(ratio.max()),
            )
        finally:
            with torch.no_grad():
                parameter.copy_(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
