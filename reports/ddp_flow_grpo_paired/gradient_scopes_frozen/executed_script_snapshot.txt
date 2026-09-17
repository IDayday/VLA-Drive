"""Real full-policy CPU FP32 graph coverage; explicitly not ZeRO certification.

Uses a previously saved G=8,K=10 chain. Signed diagnostic advantages and a
reference-only perturbation are never admitted to the training data path.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import argparse
import json
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.loading import load_policy, enable_checkpointing
from starVLA.rl.flow_grpo.contracts import (
    apply_rl_freezes,
    tensor_hashes,
    optimizer_groups,
    parameter_manifest,
    check_optimizer,
)
from starVLA.rl.flow_grpo.model import make_reference, FlowGRPOActor
from starVLA.rl.flow_grpo.rollout import evaluate_transitions
from starVLA.rl.flow_grpo.math import (
    reduce_dimensions,
    conditional_kl,
    clipped_surrogate,
)
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.audit import capture_source_environment
from starVLA.rl.flow_grpo.comparison import module_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.set_num_threads(16)
    torch.manual_seed(42)
    cfg, sft = resolve_config(args.config)
    sft.framework.qwenvl.load_device = "cpu"
    sft.framework.qwenvl.load_dtype = "float32"
    sft.framework.qwenvl.attn_implementation = "sdpa"
    capture_source_environment(out, cfg)
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    policy = (
        load_policy(
            cfg, sft, SimpleNamespace(process_index=0, device=torch.device("cpu"))
        )
        .bfloat16()
        .float()
        .eval()
    )
    source = parameter_manifest(policy, optimizer_groups(policy, sft, 0.1))
    exceptions = apply_rl_freezes(policy, cfg)
    groups = optimizer_groups(policy, sft, 0.1)
    actor = FlowGRPOActor(policy, cfg)
    check_optimizer(actor, groups)
    manifest = parameter_manifest(policy, groups)
    (out / "source_manifest.json").write_text(json.dumps(source, indent=2))
    (out / "actor_manifest.json").write_text(json.dumps(manifest, indent=2))
    reference = make_reference(policy)
    reference_hash = tensor_hashes(reference)
    frozen_hash = tensor_hashes(policy, lambda n, p: not p.requires_grad)
    enable_checkpointing(policy, True)
    train, _ = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    replay = dataset[(train[1], 104771)]
    buffer = torch.load(args.behavior, map_location="cpu", weights_only=False)
    observation = buffer.observation
    # Defense against an accidental future-observation substitution.
    assert (
        observation.tokens
        == prepare_policy_observation([dataset[(train[0], 42)]]).tokens
    )
    advantages = torch.tensor([[-1.0, 1.0] * 4])[:, :, None]
    report = {
        "scope": "real full model CPU FP32 graph coverage, synthetic diagnostic advantages only, no optimizer updates",
        "production_gate": "NOT_RUN",
        "checkpoint_sha256": cfg["checkpoint_contract"]["sha256"],
        "authorized_freezes": exceptions,
        "groups": {},
        "token": observation.tokens[0],
    }
    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            ref = evaluate_transitions(reference, observation, buffer)
            cur = evaluate_transitions(policy, observation, buffer)
            report["initial_kl_max"] = float(
                conditional_kl(cur["mean"], cur["std"], ref["mean"], ref["std"])
                .abs()
                .max()
            )
            ratio = (
                reduce_dimensions(cur["elementwise_logprob"]) - buffer.old_logprob
            ).exp()
            report["initial_ratio_minmax"] = [float(ratio.min()), float(ratio.max())]
            torch.save({k: v for k, v in cur.items()}, out / "transition_layers.pt")
        del cur
        for scope in ("rl_only", "sft_only", "reference_only", "total"):
            start = time.monotonic()
            policy.zero_grad(set_to_none=True)
            changed = None
            if scope == "reference_only":
                parameter = policy.qwen_vl_interface.model.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight
                changed = parameter.detach().clone()
                with torch.no_grad():
                    parameter.add_(0.01)
            torch.manual_seed(104771)
            if scope != "sft_only":
                cur = evaluate_transitions(policy, observation, buffer, checkpoint=True)
                current = reduce_dimensions(cur["elementwise_logprob"])
                pg = clipped_surrogate(current, buffer.old_logprob, advantages, 0.02)[
                    0
                ].mean()
                kl = reduce_dimensions(
                    conditional_kl(cur["mean"], cur["std"], ref["mean"], ref["std"])
                ).mean()
            if scope in ("sft_only", "total"):
                losses = policy.compute_sft_losses([replay])
                sf = sum(losses.values())
            loss = {
                "rl_only": lambda: pg,
                "sft_only": lambda: sf,
                "reference_only": lambda: kl,
                "total": lambda: pg + 0.01 * kl + 0.1 * sf,
            }[scope]()
            loss.backward()
            entries, modules = {}, defaultdict(
                lambda: {
                    "trainable": 0,
                    "present": 0,
                    "nonzero": 0,
                    "norm_squared": 0.0,
                }
            )
            for name, p in policy.named_parameters():
                if p.requires_grad:
                    grad = p.grad
                    norm = (
                        float(grad.norm(dtype=torch.float64))
                        if grad is not None
                        else None
                    )
                    finite = (
                        bool(torch.isfinite(grad).all()) if grad is not None else True
                    )
                    entries[name] = {
                        "present": grad is not None,
                        "l2": norm,
                        "finite": finite,
                        "dtype": str(grad.dtype) if grad is not None else None,
                    }
                    group = modules[module_group(name)]
                    group["trainable"] += 1
                    group["present"] += int(grad is not None)
                    group["nonzero"] += int(norm is not None and norm > 0)
                    group["norm_squared"] += (norm or 0.0) ** 2
            failures = [
                name
                for name, entry in entries.items()
                if not entry["present"] or not entry["finite"]
            ]
            for module in ("language", "history", "projector", "action"):
                if modules[module]["nonzero"] == 0:
                    failures.append("no nonzero " + module)
            if any(p.grad is not None for p in reference.parameters()):
                failures.append("reference received gradient")
            if any(
                p.grad is not None for p in policy.parameters() if not p.requires_grad
            ):
                failures.append("frozen received gradient")
            if changed is not None:
                with torch.no_grad():
                    parameter.copy_(changed)
            if (
                tensor_hashes(reference) != reference_hash
                or tensor_hashes(policy, lambda n, p: not p.requires_grad)
                != frozen_hash
            ):
                failures.append("immutable weights changed")
            report["groups"][scope] = {
                "status": "FAIL" if failures else "PASS",
                "failures": failures,
                "loss": float(loss.detach()),
                "parameters": entries,
                "modules": dict(modules),
                "seconds": time.monotonic() - start,
            }
            (out / "gradient_scopes.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        scope: {
                            k: v
                            for k, v in report["groups"][scope].items()
                            if k != "parameters"
                        }
                    }
                ),
                flush=True,
            )
            del loss
            if scope != "sft_only":
                del cur, pg, kl, current
            if scope in ("sft_only", "total"):
                del losses, sf
        report["status"] = (
            "PASS"
            if all(x["status"] == "PASS" for x in report["groups"].values())
            else "FAIL"
        )
        (out / "gradient_scopes.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("gradient scope failures; full report preserved")


if __name__ == "__main__":
    main()
