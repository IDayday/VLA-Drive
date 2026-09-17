"""Real QwenOFT checkpoint integration gates, never replaced by toy models."""
from contextlib import contextmanager
from pathlib import Path
import ast
import importlib
import json
import os
import subprocess
import types
import time
import torch
import numpy as np
from .config import split_tokens
from .loading import load_policy, enable_checkpointing
from .contracts import (
    optimizer_groups,
    parameter_manifest,
    grad_summary,
    apply_rl_freezes,
    check_manifest,
)
from .data import KeyedDataset, to_device
from .observation import prepare_policy_observation
from .rollout import SamplingSpec, sample_chain, evaluate_transitions
from .math import reduce_dimensions, conditional_kl, clipped_surrogate
from .model import make_reference
from .reward import RewardService
from .checkpoint import capture_rng, restore_rng
from .source_oracle import action_only_source

BASE_SHA = "9fe1459b8f6ab69a15274450ec301d541209bedd"


def original_method(file, class_name, method, official=False):
    source = (
        Path("tests/flow_grpo/vendor/official_sft_forward.py").read_text()
        if official
        else subprocess.check_output(["git", "show", f"{BASE_SHA}:{file}"], text=True)
    )
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    imported = importlib.import_module(file[:-3].replace("/", "."))
    scope = dict(vars(imported))
    exec(compile(module, file, "exec"), scope)
    return scope[method]


@contextmanager
def original_head(policy):
    head = policy.action_model
    current = head.forward
    current_euler = head._euler_sample
    path = "starVLA/model/modules/action_model/GR00T_ActionHeader.py"
    head.forward = types.MethodType(
        original_method(path, "FlowmatchingActionHead", "forward"), head
    )
    head._euler_sample = types.MethodType(
        original_method(path, "FlowmatchingActionHead", "_euler_sample"), head
    )
    try:
        yield
    finally:
        head.forward = current
        head._euler_sample = current_euler


def tensor_hashes(model, predicate=lambda n, p: True):
    import hashlib

    hashes = {}
    for name, p in model.named_parameters():
        if predicate(name, p):
            hashes[name] = hashlib.sha256(
                p.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
    return hashes


def run_diagnostics(cfg, sft, output, gradients=True):
    # Use the same deterministic backward kernels as the production trainer.
    # The environment flag alone configures FlashAttention, not PyTorch SDPA.
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .audit import capture_source_environment

    capture_source_environment(output, cfg)
    report = {
        "source_sha": cfg["checkpoint_contract"].get("code_sha", BASE_SHA),
        "tests": {},
        "real_checkpoint": cfg["sft_checkpoint"],
        "failures": [],
    }

    def record(name, fn):
        start = time.monotonic()
        try:
            value = fn()
            report["tests"][name] = {
                "status": "TESTED",
                "seconds": time.monotonic() - start,
                "result": value,
            }
            print("PASS", name, value, flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            report["tests"][name] = {"status": "FAILED", "error": repr(exc)}
            report["failures"].append(name)
        (output / "real_integration.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

    train, val = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    sample = dataset[(train[0], 42)]
    batch = [to_device(sample, "cuda")]
    observation = prepare_policy_observation([sample])
    policy = load_policy(cfg, sft).cuda().eval()
    groups = optimizer_groups(policy, sft)
    original_manifest = parameter_manifest(policy, groups)
    allowed_freezes = apply_rl_freezes(policy, cfg)
    groups = optimizer_groups(policy, sft)
    manifest = parameter_manifest(policy, groups)
    check_manifest(original_manifest, manifest, allowed_freezes)
    (output / "source_sft_parameter_manifest.json").write_text(
        json.dumps(original_manifest, indent=2)
    )
    (output / "sft_parameter_manifest.json").write_text(json.dumps(manifest, indent=2))
    # The official SFT forward assumes DeepSpeed's BF16 module conversion.
    # Retain the pre-wrapper manifest above, then match that real runtime dtype.
    policy.to(dtype=torch.bfloat16)
    policy._inference_qwen_forward_mode = (
        "optimized"
        if cfg["checkpoint_contract"]["policy_feature_output"] == "normalized"
        else "legacy"
    )
    report["checkpoint_sha256"] = policy._flow_source_sha256
    report["scene"] = sample["token"]
    report["views"] = [im.size for im in sample["image"]]
    report["numel"] = sum(p.numel() for p in policy.parameters())
    report["trainable_numel"] = manifest["trainable_numel"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Original head and original (unrefactored) SFT forward are AST-loaded from
    # the recorded workspace commit, not a second call to the shared new kernel.
    def ode_equivalence():
        state = capture_rng(policy)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if cfg["checkpoint_contract"]["origin"] == "action_only_release":
                with action_only_source(policy, cfg["checkpoint_contract"]):
                    old = policy.predict_action_infer_1d([sample])["normalized_actions"]
            else:
                with original_head(policy):
                    old = policy.predict_action_infer_1d_legacy([sample])[
                        "normalized_actions"
                    ]
            restore_rng(state, policy)
            new = policy.predict_action_infer_1d([sample])["normalized_actions"]
        np.testing.assert_allclose(new, old, atol=2e-5, rtol=2e-5)
        np.savez(output / "ode_comparison.npz", old=old, new=new, gt=sample["action"])
        return {"max_abs": float(np.max(np.abs(new - old))), "atol": 2e-5, "rtol": 2e-5}

    record("01_original_ode_equivalence", ode_equivalence)

    def sft_equivalence():
        state = capture_rng(policy)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if cfg["checkpoint_contract"]["origin"] == "action_only_release":
                with action_only_source(policy, cfg["checkpoint_contract"]):
                    old = policy(batch)
            else:
                oldforward = original_method(
                    "starVLA/model/framework/QwenOFT.py",
                    "Qwenvl_OFT",
                    "forward",
                    official=cfg["checkpoint_contract"]["origin"] == "official_release",
                )
                with original_head(policy):
                    old = oldforward(policy, batch)
            restore_rng(state, policy)
            new = policy.compute_sft_losses(batch)
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=2e-6, atol=2e-6)
        return {
            "original": {k: float(v) for k, v in old.items()},
            "shared": {k: float(v) for k, v in new.items()},
        }

    record("02_original_sft_equivalence", sft_equivalence)

    def projection_once():
        calls = []
        handle = policy.action_model.qwen_proj.register_forward_hook(
            lambda *args: calls.append(1)
        )
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                policy.predict_action_infer_1d([sample])
        finally:
            handle.remove()
        assert len(calls) == 1
        return {"projection_calls": len(calls)}

    record("03_projection_once", projection_once)
    spec = SamplingSpec(
        group_size=cfg["sampling"]["group_size"],
        num_steps=cfg["sampling"]["num_steps"],
        candidate_chunk_size=cfg["sampling"]["candidate_chunk_size"],
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = sample_chain(policy, observation, spec, 0, 42, {"diagnostic": True})

    def ratio_gate():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stats = evaluate_transitions(policy, observation, rollout)
        logratio = reduce_dimensions(stats["elementwise_logprob"]) - rollout.old_logprob
        torch.testing.assert_close(
            logratio, torch.zeros_like(logratio), rtol=0, atol=1e-5
        )
        return {"max_abs_logratio": float(logratio.abs().max()), "atol": 1e-5}

    record("09_behavior_recompute_ratio", ratio_gate)
    from .numerical_checks import precision_gate

    record(
        "23_real_bf16_network_fp32_probability",
        lambda: precision_gate(policy, observation, rollout),
    )
    reference = make_reference(policy)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = evaluate_transitions(reference, observation, rollout)

    def reference_gate():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            current = evaluate_transitions(policy, observation, rollout)
        kl = conditional_kl(current["mean"], current["std"], ref["mean"], ref["std"])
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0)
        return {"max_kl": float(kl.max())}

    record("10_equal_full_reference_kl", reference_gate)
    from infer import deal_action_1225

    physical = deal_action_1225(
        rollout.raw_final_action.cpu().numpy(),
        act_norm=int(sft.datasets.vla_data.act_norm),
    )

    def reward_gate():
        scores = []
        for workers in [0, 2]:
            service = RewardService(
                Path("navsim").resolve(),
                cfg["paths"]["metric_cache"],
                "train",
                [sample["token"]],
                workers=workers,
            )
            try:
                rows = service.score([sample["token"]], physical)
                reversed_rows = service.score(
                    [sample["token"]], physical[:, ::-1].copy()
                )
                single_rows = service.score([sample["token"]], physical[:, :1])
                values = [r.score for r in rows[0]]
                np.testing.assert_array_equal(
                    values, [r.score for r in reversed_rows[0]][::-1]
                )
                assert values[0] == single_rows[0][0].score
                scores.append(values)
            finally:
                service.close()
        np.testing.assert_array_equal(*scores)
        report["reward_scores"] = scores[0]
        return {"scores": scores[0], "serial_parallel_identical": True}

    record("17_official_reward_pool_order_batch", reward_gate)

    def perturbation_and_negative_gates():
        original_encode = policy.encode_policy_condition
        policy.encode_policy_condition = lambda *a, **kw: original_encode(
            *a, **kw
        ).detach()
        try:
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    evaluate_transitions(policy, observation, rollout)
                except RuntimeError as exc:
                    assert "detached" in str(exc)
                else:
                    raise AssertionError("detached current condition was accepted")
        finally:
            policy.encode_policy_condition = original_encode
        changes = {}
        for label, parameter in [
            (
                "qwen",
                policy.qwen_vl_interface.model.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight,
            ),
            ("action", policy.action_model.action_decoder.layer2.weight),
        ]:
            saved = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(0.01)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                changed = evaluate_transitions(policy, observation, rollout)
                independent = evaluate_transitions(reference, observation, rollout)
            torch.testing.assert_close(independent["mean"], ref["mean"], atol=0, rtol=0)
            kl = conditional_kl(
                changed["mean"], changed["std"], ref["mean"], ref["std"]
            ).mean()
            assert (
                kl > 0
            ), f"{label} perturbation did not change full policy distribution"
            changes[label] = float(kl)
            if label == "qwen":
                own_encode = reference.encode_policy_condition
                reference.encode_policy_condition = policy.encode_policy_condition
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        bad = evaluate_transitions(reference, observation, rollout)
                    assert not torch.equal(
                        bad["mean"], ref["mean"]
                    ), "negative reference feature-injection test did not detect contamination"
                finally:
                    reference.encode_policy_condition = own_encode
            with torch.no_grad():
                parameter.copy_(saved)
        return {
            "perturbed_kl": changes,
            "detached_condition_rejected": True,
            "current_features_in_reference_detected": True,
        }

    record(
        "10_13_full_reference_perturbation_and_negative",
        perturbation_and_negative_gates,
    )

    def future_isolation():
        changed = dict(
            sample,
            action=np.full_like(sample["action"], 999),
            depth_data={"depth": "forbidden"},
            future_images=["forbidden"],
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = policy.encode_policy_condition(observation)
            b = policy.encode_policy_condition(prepare_policy_observation([changed]))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        return {"max_abs": float((a - b).abs().max())}

    record("20_real_future_label_isolation", future_isolation)
    if gradients:
        enable_checkpointing(policy, True)
        source_map = {}
        graph_source_map = {}
        gradient_reports = {}
        ref_hash = tensor_hashes(reference)
        frozen_hash = tensor_hashes(policy, lambda n, p: not p.requires_grad)
        # A signed diagnostic advantage tests graph connectivity even if a real
        # on-policy group is all-equal. It is NEVER used for training/updating.
        diagnostic_adv = torch.linspace(-1, 1, spec.group_size, device="cuda")[
            None, :, None
        ]
        report[
            "gradient_probe_advantage"
        ] = "fixed signed diagnostic; no optimizer update; real reward kept separately"
        auxiliary_modules = [
            name
            for name in ("rgb_model", "gs_model", "rgb_act_pre", "gs_act_pre")
            if hasattr(policy, name)
        ]
        sources = ["rl", "reference", "sft", "total"]
        if auxiliary_modules:
            sources.insert(2, "auxiliary")
        else:
            report["tests"]["12_gradient_auxiliary"] = {
                "status": "TESTED",
                "result": "not applicable: checkpoint has no auxiliary branches; no auxiliary loss was introduced",
            }
        for source in sources:

            def gradient_gate(source=source):
                policy.zero_grad(set_to_none=True)
                perturbed = None
                if source == "reference":
                    parameter = (
                        policy.qwen_vl_interface.model.model.language_model.layers[
                            0
                        ].self_attn.q_proj.weight
                    )
                    perturbed = parameter.detach().clone()
                    with torch.no_grad():
                        parameter.add_(0.01)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source in ("rl", "reference", "total"):
                        cur = evaluate_transitions(
                            policy, observation, rollout, checkpoint=True
                        )
                        current = reduce_dimensions(cur["elementwise_logprob"])
                        pg = clipped_surrogate(
                            current, rollout.old_logprob, diagnostic_adv
                        )[0].mean()
                        kl = reduce_dimensions(
                            conditional_kl(
                                cur["mean"], cur["std"], ref["mean"], ref["std"]
                            )
                        ).mean()
                    if source in ("auxiliary", "sft", "total"):
                        losses = policy.compute_sft_losses(batch)
                        sf = sum(losses.values())
                        aux = losses["rgb_loss"] + losses["gs_loss"]
                    value = {
                        "rl": lambda: pg,
                        "reference": lambda: kl,
                        "auxiliary": lambda: aux,
                        "sft": lambda: sf,
                        "total": lambda: pg + 0.01 * kl + 0.1 * sf,
                    }[source]()
                value.backward()
                summary = grad_summary(policy)
                gradient_reports[source] = summary
                for name, p in policy.named_parameters():
                    if p.grad is not None:
                        graph_source_map.setdefault(name, []).append(source)
                        if torch.count_nonzero(p.grad).item():
                            source_map.setdefault(name, []).append(source)
                if source in ("rl", "reference"):
                    for module in [
                        name
                        for name in [
                            "qwen_vl_interface",
                            "action_input_model",
                            "rgb_query",
                            "gs_query",
                            "action_model",
                        ]
                        if hasattr(policy, name)
                    ]:
                        assert (
                            summary[module]["nonzero_tensors"] > 0
                        ), f"no RL gradient in {module}"
                    assert all(
                        summary[name]["grad_tensors"] == 0 for name in auxiliary_modules
                    )
                if source == "auxiliary":
                    for module in auxiliary_modules:
                        assert (
                            summary[module]["nonzero_tensors"] > 0
                        ), f"no auxiliary gradient in {module}"
                assert all(p.grad is None for p in reference.parameters())
                if perturbed is not None:
                    with torch.no_grad():
                        parameter.copy_(perturbed)
                policy.zero_grad(set_to_none=True)
                (output / "gradient_coverage.json").write_text(
                    json.dumps(gradient_reports, indent=2)
                )
                return {"loss": float(value.detach()), "modules": summary}

            record("12_gradient_" + source, gradient_gate)

        def checkpoint_gradient_gate():
            from .rollout import velocity
            from .math import transition

            snapshots = []
            losses = []
            # Compare one fixed transition block with the full original VLM
            # input. Full G*K without recomputation exceeds 80 GB on this model;
            # the production path checkpoints every block and is tested above.
            reference.to("cpu")
            torch.cuda.empty_cache()
            for enabled in (False, True):
                enable_checkpointing(policy, enabled)
                policy.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cond = policy.encode_policy_condition(observation)
                    xt = rollout.chain[:, 0, 0]
                    xn = rollout.chain[:, 0, 1]
                    bucket = torch.zeros(1, device="cuda", dtype=torch.long)
                    v = velocity(policy, xt, bucket, cond, checkpoint=enabled)
                    dist = transition(
                        xt, v, 0.0, 0.1, noise_level=spec.noise_level, first_dt=0.1
                    )
                    loss = -dist.logprob(xn).mean()
                loss.backward()
                losses.append(float(loss))
                if not snapshots:
                    snapshots.append(
                        {
                            n: p.grad.detach().cpu().clone()
                            for n, p in policy.named_parameters()
                            if p.grad is not None
                        }
                    )
                else:
                    for n, p in policy.named_parameters():
                        if n in snapshots[0]:
                            torch.testing.assert_close(
                                p.grad.cpu(),
                                snapshots[0][n],
                                atol=2e-6,
                                rtol=2e-3,
                                msg=lambda message: f"{n}: {message}",
                            )
                policy.zero_grad(set_to_none=True)
                del cond, v, dist, loss
            return {
                "scope": "one fixed transition block; full VLM input",
                "losses": losses,
                "gradient_tensors": len(snapshots[0]),
                "bf16_gradient_rtol": 2e-3,
                "gradient_atol": 2e-6,
            }

        record("22_checkpointing_gradient_equivalence", checkpoint_gradient_gate)

        def candidate_microbatch_gate():
            from dataclasses import replace

            # Two candidates from the SAME stored ten-step real rollout. Chunking
            # changes compute grouping only; no fresh trajectories or rewards.
            enable_checkpointing(policy, True)
            snapshots = None
            values = []
            max_absolute = 0.0
            for chunk in (1, 2):
                small = replace(
                    rollout,
                    spec=replace(
                        rollout.spec, group_size=2, candidate_chunk_size=chunk
                    ),
                    chain=rollout.chain[:, :2],
                    old_elementwise_logprob=rollout.old_elementwise_logprob[:, :2],
                    dimension_mask=rollout.dimension_mask[:, :2],
                    transition_mask=rollout.transition_mask[:, :2],
                    candidate_ids=rollout.candidate_ids[:2],
                )
                policy.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    stats = evaluate_transitions(
                        policy, observation, small, checkpoint=True
                    )
                    current = reduce_dimensions(stats["elementwise_logprob"])
                    signed = torch.tensor([-1.0, 1.0], device=current.device)[
                        None, :, None
                    ]
                    loss = clipped_surrogate(current, small.old_logprob, signed)[
                        0
                    ].mean()
                loss.backward()
                values.append(float(loss.detach()))
                if snapshots is None:
                    snapshots = {
                        n: p.grad.detach().cpu().clone()
                        for n, p in policy.named_parameters()
                        if p.grad is not None
                    }
                else:
                    actual_names = {
                        n for n, p in policy.named_parameters() if p.grad is not None
                    }
                    assert actual_names == set(snapshots)
                    for name, parameter in policy.named_parameters():
                        if name in snapshots:
                            grad = parameter.grad.detach().cpu()
                            max_absolute = max(
                                max_absolute,
                                float(
                                    (grad.float() - snapshots[name].float()).abs().max()
                                ),
                            )
                            torch.testing.assert_close(
                                grad,
                                snapshots[name],
                                atol=2e-6,
                                rtol=2e-3,
                                msg=lambda message: name + ": " + message,
                            )
                del stats, current, loss
                policy.zero_grad(set_to_none=True)
            return {
                "scope": "same real scene and two saved candidates, all ten transitions",
                "chunks": [1, 2],
                "losses": values,
                "gradient_tensors": len(snapshots),
                "gradient_max_abs": max_absolute,
                "atol": 2e-6,
                "rtol": 2e-3,
            }

        record(
            "22_candidate_microbatch_gradient_equivalence", candidate_microbatch_gate
        )

        def frozen_gate():
            assert ref_hash == tensor_hashes(reference)
            assert frozen_hash == tensor_hashes(
                policy, lambda n, p: not p.requires_grad
            )
            (output / "frozen_reference_hashes.json").write_text(
                json.dumps({"reference": ref_hash, "frozen": frozen_hash}, indent=2)
            )
            return {
                "reference_parameters": len(ref_hash),
                "frozen_parameters": len(frozen_hash),
            }

        record("25_reference_frozen_immutable", frozen_gate)
        (output / "sft_parameter_manifest.json").write_text(
            json.dumps(
                parameter_manifest(policy, groups, source_map, graph_source_map),
                indent=2,
            )
        )
    report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (output / "real_integration.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    if report["failures"]:
        raise RuntimeError(f'real integration failures: {report["failures"]}')
