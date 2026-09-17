"""Full real action-only model: fixed chains, exhaustive gradients and Adam updates.

CPU FP32 math is independently executable when all GPUs are occupied. It does
not certify the CUDA/ZeRO production profile; that gate must run on free GPUs.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from contextlib import nullcontext
import argparse
import json
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.data import KeyedDataset, to_device
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.loading import load_policy, enable_checkpointing
from starVLA.rl.flow_grpo.contracts import (
    apply_rl_freezes,
    optimizer_groups,
    tensor_hashes,
)
from starVLA.rl.flow_grpo.rollout import (
    SamplingSpec,
    sample_chain,
    evaluate_transitions,
)
from starVLA.rl.flow_grpo.model import FlowGRPOActor, make_reference
from starVLA.rl.flow_grpo.math import group_advantages, reduce_dimensions
from starVLA.rl.flow_grpo.reward import RewardService
from starVLA.rl.flow_grpo.comparison import compare_named
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, dependency_versions
from starVLA.rl.flow_grpo.checkpoint import capture_rng, restore_rng
from starVLA.rl.flow_grpo.audit import capture_source_environment
from starVLA.rl.flow_grpo.trainer import behavior_digest


def snapshot(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--behavior")
    parser.add_argument(
        "--advantage-source",
        choices=["official", "diagnostic_signed"],
        default="official",
    )
    parser.add_argument("--groups", nargs="+", type=int, default=[2, 8])
    args = parser.parse_args()
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("production BF16 layout probe requires CUDA")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    configure_numerics()
    torch.manual_seed(42)
    cfg, sft = resolve_config(args.config)
    cfg["runtime"]["activation_checkpointing"] = True
    sft.framework.qwenvl.load_device = args.device
    sft.framework.qwenvl.load_dtype = (
        "float32" if args.precision == "fp32" else "bfloat16"
    )
    if args.precision == "fp32":
        sft.framework.qwenvl.attn_implementation = "sdpa"
    capture_source_environment(out, cfg)
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    device = torch.device(args.device)
    policy = (
        load_policy(cfg, sft, SimpleNamespace(process_index=0, device=device))
        .to(device)
        .eval()
    )
    # Preserve exact production BF16 forward weight VALUES when lifting to FP32.
    policy.bfloat16()
    if args.precision == "fp32":
        policy.float()
    apply_rl_freezes(policy, cfg)
    reference = make_reference(policy).to(device)
    enable_checkpointing(policy, True)
    actor = FlowGRPOActor(policy, cfg)
    immutable = {
        "visual": tensor_hashes(policy.get_submodule("qwen_vl_interface.model.visual")),
        "reference": tensor_hashes(reference),
    }
    train, _ = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    scene, replay = dataset[(train[0], 42)], dataset[(train[1], 104771)]
    observation = prepare_policy_observation([scene])
    supplied_behavior = (
        torch.load(args.behavior, map_location=device, weights_only=False)
        if args.behavior
        else None
    )
    if supplied_behavior is not None:
        observation = supplied_behavior.observation
    context = (
        sdpa_kernel(SDPBackend.MATH) if args.precision == "fp32" else nullcontext()
    )
    with context:
        with torch.no_grad():
            buffer = (
                supplied_behavior
                if supplied_behavior is not None
                else sample_chain(
                    policy, observation, SamplingSpec(group_size=8), 0, 42, {}
                )
            )
            ref = evaluate_transitions(reference, observation, buffer)
            buffer.reference_mean, buffer.reference_std = ref["mean"], ref["std"]
        from infer import deal_action_1225

        trajectories = deal_action_1225(
            buffer.raw_final_action.cpu().numpy(),
            act_norm=int(sft.datasets.vla_data.act_norm),
        )
        service = RewardService(
            Path("navsim").resolve(),
            cfg["paths"]["metric_cache"],
            "train",
            train,
            workers=0,
        )
        try:
            buffer.score_records = service.score(observation.tokens, trajectories)
        finally:
            service.close()
        buffer.rewards = torch.tensor(
            [[x.score for x in row] for row in buffer.score_records], device=device
        )
        buffer.advantages = group_advantages(
            buffer.rewards,
            epsilon=cfg["algorithm"]["advantage_epsilon"],
            clip=cfg["algorithm"]["advantage_clip"],
        )
        torch.save(buffer, out / "fixed_behavior.pt")
        official_advantages = buffer.advantages.clone()
        if args.advantage_source == "diagnostic_signed":
            # Same scene/chain even for equal-reward groups. This is only a signed
            # gradient diagnostic, NEVER a formal rollout or reward replacement.
            buffer.advantages = torch.tensor([[-1.0, 1.0] * 4], device=device)
            buffer.provenance["diagnostic_only_signed_advantages"] = True
        del reference, ref
        initial = snapshot(policy)
        rng = capture_rng(policy)
        ode_noise = torch.randn(
            (1, 8, 4), generator=torch.Generator().manual_seed(4242)
        ).to(device)
        report = {
            "status": "NOT_READY",
            "precision": args.precision,
            "device": args.device,
            "scope": "real model FP32 Adam mathematical diagnostic; not production ZeRO certification",
            "weight_sha256": cfg["checkpoint_contract"]["sha256"],
            "dependencies": dependency_versions(),
            "fixed_behavior_sha256": behavior_digest(buffer),
            "token": observation.tokens[0],
            "real_rewards": buffer.rewards.tolist(),
            "official_advantages": official_advantages.tolist(),
            "advantage_source": args.advantage_source,
            "groups": {},
        }
        for g in args.groups:
            records = []
            for chunk in (1, 2):
                start = time.monotonic()
                for name, parameter in policy.named_parameters():
                    if name in initial:
                        parameter.data.copy_(initial[name].to(device))
                policy.zero_grad(set_to_none=True)
                restore_rng(rng, policy)
                small = replace(
                    buffer,
                    spec=replace(buffer.spec, group_size=g, candidate_chunk_size=chunk),
                    chain=buffer.chain[:, :g],
                    old_elementwise_logprob=buffer.old_elementwise_logprob[:, :g],
                    dimension_mask=buffer.dimension_mask[:, :g],
                    transition_mask=buffer.transition_mask[:, :g],
                    candidate_ids=buffer.candidate_ids[:g],
                    advantages=buffer.advantages[:, :g],
                    reference_mean=buffer.reference_mean[:, :g],
                    reference_std=buffer.reference_std[:, :g],
                )
                projected = []

                def hook(module, inputs, value, captured=projected):
                    if torch.is_grad_enabled():
                        for tensor in (inputs[0], value):
                            if tensor.requires_grad:
                                tensor.retain_grad()
                        captured.append((inputs[0], value))

                handle = policy.action_model.qwen_proj.register_forward_hook(hook)
                result = actor(
                    mode="update",
                    rollout=small,
                    replay=to_device([replay], device),
                    diagnostic_outputs=True,
                )
                losses = {
                    k: float(result[k].detach())
                    for k in ("loss", "grpo", "reference", "sft")
                }
                result["loss"].backward()
                handle.remove()
                gradients = {
                    n: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for n, p in policy.named_parameters()
                    if p.requires_grad
                }
                layers = {
                    k: v.detach().cpu().clone()
                    for k, v in result["diagnostics"].items()
                }
                for i, (raw, projection) in enumerate(projected):
                    layers[f"condition_raw_{i}"] = raw.detach().cpu()
                    layers[f"condition_projected_{i}"] = projection.detach().cpu()
                    layers[f"condition_raw_gradient_{i}"] = (
                        None if raw.grad is None else raw.grad.detach().cpu()
                    )
                    layers[f"condition_projected_gradient_{i}"] = (
                        None
                        if projection.grad is None
                        else projection.grad.detach().cpu()
                    )
                # Parameter gradients remain in their original accumulation dtype.
                # The Adam masters are FP32 here by construction, explicitly separate
                # from installed DeepSpeed optimizer/master acceptance.
                groups = optimizer_groups(
                    policy, sft, cfg["optimizer"]["initial_lr_multiplier"]
                )
                optcfg = sft.trainer.optimizer
                optimizer = torch.optim.AdamW(
                    groups,
                    betas=tuple(optcfg.betas),
                    eps=optcfg.eps,
                    weight_decay=optcfg.weight_decay,
                )
                pre_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg["optimizer"]["max_grad_norm"]
                )
                received = {
                    n: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for n, p in policy.named_parameters()
                    if p.requires_grad
                }
                optimizer.step()
                moments = {
                    n + "/" + key: value.detach().cpu().clone()
                    for n, p in policy.named_parameters()
                    for key, value in optimizer.state.get(p, {}).items()
                    if isinstance(value, torch.Tensor)
                }
                weights = snapshot(policy)
                with torch.no_grad():
                    post = evaluate_transitions(policy, observation, small)
                    layers["post_logprob"] = reduce_dimensions(
                        post["elementwise_logprob"]
                    ).cpu()
                    condition = policy.encode_policy_condition(observation)
                    layers["post_fixed_noise_ode"] = policy.action_model._euler_sample(
                        ode_noise.clone(), condition
                    ).cpu()
                record = {
                    "gradients": gradients,
                    "optimizer_received": received,
                    "moments": moments,
                    "weights": weights,
                    "updates": {n: weights[n] - initial[n] for n in weights},
                    "layers": layers,
                }
                torch.save(record, out / f"g{g}_chunk{chunk}.pt")
                if records:
                    comparison = {
                        key: compare_named(records[0][key], record[key])
                        for key in record
                    }
                    report["groups"][str(g)] = {
                        "status": "PASS"
                        if all(v["status"] == "PASS" for v in comparison.values())
                        else "FAIL",
                        "comparisons": comparison,
                        "losses": losses,
                        "pre_clip_norm": float(pre_norm),
                    }
                    (out / "numerical_report.json").write_text(
                        json.dumps(report, indent=2)
                    )
                else:
                    records.append(record)
                print(
                    json.dumps(
                        {
                            "group": g,
                            "chunk": chunk,
                            "seconds": time.monotonic() - start,
                            "losses": losses,
                        }
                    ),
                    flush=True,
                )
                del (
                    optimizer,
                    result,
                    projected,
                    received,
                    gradients,
                    moments,
                    weights,
                    layers,
                    record,
                )
            del records
        report["historical_tolerances"] = {"atol": 2e-6, "rtol": 2e-3}
        report["production_gate"] = "NOT_RUN"
        report["visual_unchanged"] = (
            tensor_hashes(policy.get_submodule("qwen_vl_interface.model.visual"))
            == immutable["visual"]
        )
        (out / "numerical_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
