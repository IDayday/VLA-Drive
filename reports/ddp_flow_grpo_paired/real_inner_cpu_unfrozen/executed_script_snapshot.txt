"""Real full-policy two-inner-epoch + exact-resume test on CPU FP32.

This uses installed Accelerate, actual official advantages/replay, full saved
chains, AdamW and production checkpoint functions. It does not release ZeRO/CUDA
or the formal global scene batch. No synthetic training advantages are used.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import hashlib
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from accelerate import Accelerator
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.loading import load_policy, enable_checkpointing, file_sha
from starVLA.rl.flow_grpo.contracts import (
    apply_rl_freezes,
    optimizer_groups,
    parameter_manifest,
    tensor_hashes,
    digest,
)
from starVLA.rl.flow_grpo.model import FlowGRPOActor, make_reference
from starVLA.rl.flow_grpo.rollout import evaluate_transitions
from starVLA.rl.flow_grpo.math import reduce_dimensions
from starVLA.rl.flow_grpo.data import KeyedDataset, SceneStream
from starVLA.rl.flow_grpo.checkpoint import save_boundary, resume_boundary, capture_rng
from starVLA.rl.flow_grpo.reproducibility import (
    configure_numerics,
    processor_identity,
    dependency_versions,
)
from starVLA.rl.flow_grpo.audit import capture_source_environment, source_fingerprints
from starVLA.rl.flow_grpo.trainer import behavior_digest
from scripts.flow_grpo.compare_boundaries import flatten


def state_hashes(value):
    result = {}
    for name, entry in flatten(value):
        if isinstance(entry, torch.Tensor):
            result[name] = hashlib.sha256(
                entry.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
        else:
            result[name] = repr(entry)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--behavior", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.set_num_threads(16)
    torch.manual_seed(42)
    cfg, sft = resolve_config(args.config)
    cfg["runtime"].update(
        output_dir=str(root / "continuous"),
        run_mode="diagnostic",
        max_updates=2,
        accumulation_steps=1,
        deepspeed_stage=0,
        numerical_profile="cpu_fp32_real_model_diagnostic",
    )
    sft.framework.qwenvl.load_device = "cpu"
    sft.framework.qwenvl.load_dtype = "float32"
    sft.framework.qwenvl.attn_implementation = "sdpa"
    capture_source_environment(root, cfg)
    (root / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    acc = Accelerator(cpu=True, mixed_precision="no")
    policy = (
        load_policy(
            cfg, sft, SimpleNamespace(process_index=0, device=torch.device("cpu"))
        )
        .bfloat16()
        .float()
        .eval()
    )
    apply_rl_freezes(policy, cfg)
    groups = optimizer_groups(policy, sft, 0.1)
    manifest = parameter_manifest(policy, groups)
    reference = make_reference(policy)
    immutable = {
        "reference": tensor_hashes(reference),
        "frozen": tensor_hashes(policy, lambda n, p: not p.requires_grad),
    }
    enable_checkpointing(policy, True)
    actor = FlowGRPOActor(policy, cfg)
    optcfg = sft.trainer.optimizer
    opt = torch.optim.AdamW(
        groups,
        betas=tuple(optcfg.betas),
        eps=optcfg.eps,
        weight_decay=optcfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    actor, opt, scheduler = acc.prepare(actor, opt, scheduler)
    train, _ = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    replay = dataset[(train[1], 104771)]
    buffer = torch.load(args.behavior, map_location="cpu", weights_only=False)
    if buffer.provenance.get("diagnostic_only_signed_advantages"):
        raise ValueError("official saved advantages required")
    original_digest = behavior_digest(buffer)
    streams = [SceneStream(dataset, train, seed, cursor=1) for seed in (42, 104771)]
    provenance = {
        "processor": processor_identity(cfg["paths"]["base_vlm"]),
        "numerics": {
            "model": "float32",
            "attention": "sdpa_math",
            "tf32": False,
            "accumulation": "float32",
            "world_size": 1,
            "global_scenes": 1,
        },
        "dependencies": dependency_versions(),
        "source": digest(source_fingerprints()),
        "data_manifest": file_sha(cfg["paths"]["asset_manifest"]),
        "checkpoint": cfg["checkpoint_contract"]["sha256"],
    }
    report = {
        "status": "RUNNING",
        "production_gate": "NOT_RUN",
        "scope": "real CPU FP32 Accelerate, global scene batch1, G8K10, official advantages; not ZeRO-2 validation",
        "official_rewards": buffer.rewards.tolist(),
        "official_advantages": buffer.advantages.tolist(),
        "old_behavior_sha256": original_digest,
        "steps": [],
    }
    noise = torch.randn((1, 8, 4), generator=torch.Generator().manual_seed(4242))

    def probe():
        with torch.no_grad():
            current = evaluate_transitions(policy, buffer.observation, buffer)
            ratio = (
                reduce_dimensions(current["elementwise_logprob"]) - buffer.old_logprob
            ).exp()
            condition = policy.encode_policy_condition(buffer.observation)
            ode = policy.action_model._euler_sample(noise.clone(), condition)
        return {
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "max_logratio": float(ratio.log().abs().max()),
            "ode": ode.tolist(),
        }

    def step(label):
        start = time.monotonic()
        pre = probe()
        opt.zero_grad(set_to_none=True)
        result = actor(mode="update", rollout=buffer, replay=[replay])
        losses = {
            k: float(result[k].detach()) for k in ("loss", "grpo", "reference", "sft")
        }
        acc.backward(result["loss"])
        norm = float(acc.clip_grad_norm_(actor.parameters(), 1.0))
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        post = probe()
        if behavior_digest(buffer) != original_digest:
            raise AssertionError("old chain/logprob/advantages changed")
        row = {
            "label": label,
            "pre": pre,
            "post": post,
            "losses": losses,
            "pre_clip_grad_norm": norm,
            "seconds": time.monotonic() - start,
            "behavior_sha256": behavior_digest(buffer),
        }
        report["steps"].append(row)
        (root / "inner_resume.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        del result

    def full_state():
        return {
            "model": tensor_hashes(policy),
            "optimizer": state_hashes(opt.state_dict()),
            "scheduler": state_hashes(scheduler.state_dict()),
            "rng": state_hashes(capture_rng(policy)),
        }

    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            ref = evaluate_transitions(reference, buffer.observation, buffer)
            buffer.reference_mean, buffer.reference_std = ref["mean"], ref["std"]
        step("inner0")
        saved = save_boundary(
            acc,
            actor,
            cfg,
            sft,
            manifest,
            provenance,
            1,
            0,
            streams,
            {"buffers": [buffer], "replay": [replay], "next_inner": 1},
        )
        step("inner1_continuous")
        expected = full_state()
        continuous = save_boundary(
            acc, actor, cfg, sft, manifest, provenance, 2, 1, streams
        )
        update, version, pending = resume_boundary(
            saved, acc, actor, cfg, provenance, streams
        )
        if (update, version, pending["next_inner"]) != (1, 0, 1):
            raise AssertionError("policy/inner version wrong")
        buffer = pending["buffers"][0]
        replay = pending["replay"][0]
        if behavior_digest(buffer) != original_digest:
            raise AssertionError("resume replaced behavior")
        step("inner1_resumed")
        actual = full_state()
        changes = {
            scope: [
                name
                for name, value in expected[scope].items()
                if actual[scope].get(name) != value
            ]
            for scope in expected
        }
        after = {
            "reference": tensor_hashes(reference),
            "frozen": tensor_hashes(policy, lambda n, p: not p.requires_grad),
        }
        report["exact_state_differences"] = changes
        report["immutable_equal"] = immutable == after
        report["old_behavior_unchanged"] = behavior_digest(buffer) == original_digest
        report["resumed_post_probe_equal"] = (
            report["steps"][1]["post"] == report["steps"][2]["post"]
        )
        report["resume_policy_version"] = version
        report["continuous_checkpoint"] = str(continuous)
        cfg["runtime"]["output_dir"] = str(root / "resumed")
        resumed = save_boundary(
            acc, actor, cfg, sft, manifest, provenance, 2, 1, streams
        )
        report["resumed_checkpoint"] = str(resumed)
        report["status"] = (
            "PASS"
            if not any(changes.values())
            and report["immutable_equal"]
            and report["resumed_post_probe_equal"]
            else "FAIL"
        )
        (root / "inner_resume.json").write_text(json.dumps(report, indent=2))
        (root / "exact_state_hashes.json").write_text(
            json.dumps({"continuous": expected, "resumed": actual}, indent=2)
        )
    if report["status"] != "PASS":
        raise AssertionError("real CPU inner resume mismatch; all evidence retained")
    print(
        json.dumps({"status": report["status"], "production_gate": "NOT_RUN"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
