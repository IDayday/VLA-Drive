"""Bounded, isolated expert/Scorer exact-resume qualification on bound features.

Real probes use existing F_plan/W/A, audited GT and the two-candidate bank. They
never write or supply main experiment weights. A separately labelled tiny CPU
fixture checks terminal replay without shortening either real probe's LR clock.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
import yaml

from iqe.contracts import require
from iqe.data.manifests import load_scenes
from iqe.data.sampler import ConsumedSampler
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from iqe.losses import LossTerm, expert_il_terms
from iqe.training.checkpoint import capture_rng, restore_rng
from iqe.training.resume import inspect_checkpoint
from iqe.training.trainer import train
from .binding import binding, load_head
from .config import Config
from .features import JointCache, planner_bundle
from .objectives import comparison_weights, denominators, terms
from .partition import training_ids
from .resume_audit import equal_tree
from .sampling import TwoStreamSampler
from .scorer import construct, initialization_rng
from .training import load_bank, score_context, training_config, training_device


@contextmanager
def deterministic_probe(device):
    """Configure before CUDA startup and restore flags/RNG for library callers."""
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "bounded resume audit uses one independently authorized device")
    if str(device).startswith("cuda"):
        require(not torch.cuda.is_initialized(), "run CUDA resume qualification in a fresh process")
    before = {"rng": capture_rng(), "threads": torch.get_num_threads(),
              "deterministic": torch.are_deterministic_algorithms_enabled(),
              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    try:
        yield training_device(device)
    finally:
        restore_rng(before["rng"])
        torch.set_num_threads(before["threads"])
        torch.use_deterministic_algorithms(before["deterministic"], warn_only=before["warn_only"])
        torch.backends.cudnn.benchmark = before["benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        if before["cublas"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = before["cublas"]


def _gradient_evidence(module):
    result = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            result[name] = {"present": False}
        else:
            gradient = gradient.detach().cpu().contiguous()
            result[name] = {"present": True, "finite": bool(torch.isfinite(gradient).all()),
                            "shape": list(gradient.shape), "dtype": str(gradient.dtype),
                            "norm": float(gradient.float().norm()),
                            "sha256": hashlib.sha256(gradient.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    require(any(v.get("norm", 0) > 0 for v in result.values()), "probe has no effective gradient")
    require(all(v.get("finite", True) for v in result.values()), "nonfinite probe gradient")
    return result


def _run_to(folder, target, factory, fetch, loss, cfg, dependencies, options):
    """Resume only complete compatible receipts; never silently replay a stale run."""
    paths = sorted(folder.glob("step_*.pt"))
    inspected = [inspect_checkpoint(p, dependencies, cfg) for p in paths]
    latest = max((p["optimizer_step"] for p in inspected), default=0)
    receipt = folder/f"AUDIT_STEP_{target:06d}.json"
    if latest >= target:
        require(receipt.exists(), "audit checkpoint complete but gradient evidence missing; use a fresh --audit-id, preserve this interrupted proof")
        result = read_json(receipt)
        require(result["checkpoint_hash"] == file_hash(folder/f"step_{target:06d}.pt"), "audit receipt checkpoint changed")
        return result
    module, sampler = factory()
    initial_hash = module_hash(module)
    history = train(module, loss, fetch, sampler, cfg, dependencies, folder,
                    resume="auto" if latest else None, stop_after=target, **options)
    require(history and history[-1]["optimizer_step"] == target, "probe interrupted before requested boundary; rerun the same audit")
    saved = torch.load(folder/f"step_{target:06d}.pt", map_location="cpu", weights_only=False)
    result = {"checkpoint": str(folder/f"step_{target:06d}.pt"),
              "checkpoint_hash": file_hash(folder/f"step_{target:06d}.pt"),
              "optimizer_step": target, "model_hash": module_hash(module), "gradients": _gradient_evidence(module),
              "fresh_factory_model_hash": initial_hash, "differs_from_fresh_factory": module_hash(module) != initial_hash,
              "sampler_matches_checkpoint": equal_tree(sampler.state_dict(), saved["sampler"]),
              "next_entries": sampler.peek(cfg["global_batch_size"])}
    atomic_json(receipt, result, immutable=True)
    return result


def _run_pair(root, factory, fetch, loss, cfg, dependencies, options):
    """Original training recipe, four real updates; pause exactly after update 2."""
    atomic_json(root/"PROBE_PROTOCOL.json", {"cfg": cfg, "dependencies": dependencies,
                "continuous_steps": 4, "split_steps": [2, 2], "uses_stop_after_not_shortened_LR_clock": True,
                "main_training_weights_initialized_from_probe": False}, immutable=True)
    _run_to(root/"continuous", 4, factory, fetch, loss, cfg, dependencies, options)
    _run_to(root/"split", 2, factory, fetch, loss, cfg, dependencies, options)
    _run_to(root/"split", 4, factory, fetch, loss, cfg, dependencies, options)
    a, b = (torch.load(root/folder/"step_000004.pt", map_location="cpu", weights_only=False)
            for folder in ("continuous", "split"))
    ga, gb = (read_json(root/folder/"AUDIT_STEP_000004.json") for folder in ("continuous", "split"))
    logs = [[json.loads(line) for line in (root/folder/"steps.jsonl").read_text().splitlines()]
            for folder in ("continuous", "split")]
    require(all([row["optimizer_step"] for row in rows] == [1, 2, 3, 4] for rows in logs), "missing/duplicate resume probe updates")
    checks = {key: equal_tree(a[key], b[key]) for key in (
        "module", "optimizer", "scheduler", "rng_by_rank", "sampler", "consumed_sample_ids",
        "optimizer_step", "scaler", "global_batch", "accumulation", "resolved_config", "dependencies", "environment", "trainer_state")}
    for key in ("loss", "losses", "gradient_norm", "sample_ids"):
        checks[key] = equal_tree([r[key] for r in logs[0]], [r[key] for r in logs[1]])
    checks["full_final_gradient_tensors"] = ga["gradients"] == gb["gradients"]
    checks["next_batch_entries"] = ga["next_entries"] == gb["next_entries"]
    checks["live_sampler_matches_checkpoint"] = ga["sampler_matches_checkpoint"] and gb["sampler_matches_checkpoint"]
    checks["actual_parameter_updates"] = ga["differs_from_fresh_factory"] and gb["differs_from_fresh_factory"]
    actual_resume = read_json(root/"split/RESUME_RESOLUTION.json")
    checks["actual_restored_next_batch"] = actual_resume["next_sample_ids"] == logs[0][2]["sample_ids"]
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "continuous_checkpoint": ga["checkpoint"], "split_checkpoint": gb["checkpoint"],
              "continuous_checkpoint_hash": ga["checkpoint_hash"], "split_checkpoint_hash": gb["checkpoint_hash"],
              "next_sample_ids": [x["scene_id"] for x in ga["next_entries"]],
              "sample_ids_step_3": logs[0][2]["sample_ids"],
              "losses": [r["loss"] for r in logs[0]], "gradient_norms": [r["gradient_norm"] for r in logs[0]],
              "gradient_evidence": ga["gradients"], "environment": a["environment"],
              "continuous_steps": 4, "split_steps": [2, 2], "total_optimizer_updates": 8,
              "global_batch": cfg["global_batch_size"], "main_LR_timeline_updates": cfg["max_optimizer_steps"],
              "VLM_calls": 0, "main_weights_initialized_from_probe": False}
    atomic_json(root/"RESULT.json", result)
    require(all(checks.values()), "real same-runtime exact resume mismatch; full evidence preserved")
    return result


def _expert_probe(config, root, scenes, device):
    manifest = read_json(config.root/"EXPERT_MANIFEST.json")
    require(manifest["hash"] == digest({k: v for k, v in manifest.items() if k != "hash"}), "expert manifest changed")
    allowed = training_ids(config, list(scenes.values()), "expert_fit")
    require(set(manifest["hard"]) <= allowed, "resume probe hard set contains D_S or heldout data")
    ids = sorted(manifest["hard"], key=lambda x: digest([config.selection_seed, "expert_resume_probe", x]))[:32]
    require(len(ids) == 32, "real expert resume qualification requires 32 audited D_E hard scenes")
    frozen = JointCache(config).batch([scenes[i] for i in ids], device)
    require(all(not value.requires_grad and torch.isfinite(value).all() for value in frozen.values()), "invalid frozen probe features/GT")
    base = load_head(config, device).eval().requires_grad_(False)
    before = module_hash(base)
    features = planner_bundle(config, frozen, ids)
    with torch.no_grad():
        original = base(features)
    rng = np.random.default_rng(config.natural_seed)
    entries = [{"scene_id": ids[i], "bucket": "audited_hard_resume_only"}
               for i in rng.integers(len(ids), size=5*128)]
    plan = {"entries": entries, "plan_hash": digest(entries)}
    position = {sid: i for i, sid in enumerate(ids)}
    contract = read_json(config.root/"OFFLINE_BASE_CONTRACT.json")
    stochastic_layers = {name: {"class": module.__class__.__name__, "p": getattr(module, "p", None),
                               "drop_prob": getattr(module, "drop_prob", None)}
                         for name, module in base.named_modules()
                         if isinstance(module, torch.nn.Dropout) or module.__class__.__name__ == "DropPath"}

    def factory():
        expert = deepcopy(base).requires_grad_(True)
        require(all(a.data_ptr() != b.data_ptr() for a, b in zip(base.parameters(), expert.parameters())), "expert aliases frozen Base")
        return expert, ConsumedSampler(plan)

    def fetch(batch, _device):
        ix = [position[e["scene_id"]] for e in batch]
        values = {k: value[ix] for k, value in frozen.items()}
        return {"features": planner_bundle(config, values, [e["scene_id"] for e in batch]),
                "target": values["target_raw"], "mask": torch.ones(values["target_raw"].shape[:2], device=_device, dtype=torch.bool)}

    def loss(module, batch):
        return expert_il_terms(module(batch["features"]), batch["target"], masks=batch["mask"],
                               prev_weight=contract["query_architecture"]["prev_weight"])

    cfg = training_config(config, "expert")
    require(cfg["max_optimizer_steps"] == 1000 and cfg["schedule"] == "warmup_constant", "probe requires original 1000-update expert timeline")
    deps = {"purpose": "two_candidate_real_expert_resume_only", "base_binding": file_hash(config.root/"BASE_BINDING.json"),
            "partition": file_hash(config.root/"DATA_PARTITION.json"), "expert_manifest": file_hash(config.root/"EXPERT_MANIFEST.json"),
            "scene_ids": ids, "plan_hash": plan["plan_hash"]}
    atomic_json(root/"INPUTS.json", {"scene_ids": ids, "role": "D_E audited H only",
                "diagnostic_sampling": "32 hard scenes with replacement; no change to main 80/20 sampler",
                "stochastic_layers": stochastic_layers, "dropout_and_DropPath_disabled": False}, immutable=True)
    result = _run_pair(root, factory, fetch, loss, cfg, deps,
                       {"device": device, "seed": config.training_rng_seed,
                        "denominator_function": lambda batch: {"il": batch["mask"].any(-1).sum().float()}})
    with torch.no_grad():
        after = base(features)
    retention = {"parameter_buffer_hash_equal": module_hash(base) == before,
                 "raw_exact": torch.equal(original.raw, after.raw), "physical_exact": torch.equal(original.physical, after.physical)}
    result.update(real_scenes=len(ids), frozen_base_retention=retention, stochastic_layers=stochastic_layers)
    atomic_json(root/"RESULT.json", result)
    require(all(retention.values()), "frozen Base changed during resume probe")
    return result


def _scorer_probe(config, root, scenes, device):
    bank, arrays = load_bank(config)
    allowed = training_ids(config, list(scenes.values()), "scorer_fit")
    ix = [i for i, r in enumerate(bank["rows"]) if r["scene_id"] in allowed
          and r["role"] == "incremental_fit" and abs(arrays["scores"][i, 1]-arrays["scores"][i, 0]) > .01]
    ix.sort(key=lambda i: digest([config.selection_seed, "scorer_resume_probe", bank["rows"][i]["scene_id"]]))
    ix = ix[:64]
    require(ix, "SCORER_COMPARISON_SIGNAL_ABSENT: no real D_S comparison pairs for resume qualification")
    ids = [bank["rows"][i]["scene_id"] for i in ix]
    position = {sid: i for i, sid in enumerate(ids)}
    frozen = JointCache(config).batch([scenes[i] for i in ids], device)
    scores = torch.as_tensor(arrays["scores"][ix], device=device)
    components = torch.as_tensor(arrays["components"][ix], device=device)
    trajectories = torch.as_tensor(arrays["trajectories"][ix], device=device)
    weights = comparison_weights(scores, torch.isfinite(scores)).cpu().numpy()
    require((weights > 0).all(), "resume comparison population contains invalid/tied labels")
    contract = load_head(config, "cpu").trajectory_contract
    action_dim = frozen["planner_scene" if config.score_context == "planner_ego" else "action"].shape[-1]

    def factory():
        module = construct(config, contract, frozen["world"].shape[-1], action_dim, initialization_receipt=False)
        sampler = TwoStreamSampler(ids, weights, natural_seed=config.natural_seed, comparison_seed=config.comparison_seed,
                                   pointwise_updates=2, bank_hash=file_hash(config.root/"PAIR_BANK.json"))
        return module, sampler

    def fetch(batch, _device):
        indices = [position[e["scene_id"]] for e in batch]
        values = {k: value[indices] for k, value in frozen.items()}
        return {"context": score_context(config, values), "scores": scores[indices],
                "components": components[indices], "component_valid": torch.isfinite(components[indices]),
                "trajectories": trajectories[indices], "valid": torch.isfinite(scores[indices]),
                "natural": torch.tensor([e["natural"] for e in batch], device=_device),
                "comparison": torch.tensor([e["comparison"] for e in batch], device=_device)}

    def loss(module, batch):
        return terms(module(batch["context"], batch["trajectories"]), batch)

    cfg = training_config(config, "scorer")
    deps = {"purpose": "two_candidate_real_scorer_resume_only", "base_binding": file_hash(config.root/"BASE_BINDING.json"),
            "partition": file_hash(config.root/"DATA_PARTITION.json"), "bank": file_hash(config.root/"PAIR_BANK.json"),
            "scene_ids": ids, "comparison_weights": weights.tolist(), "probe_pointwise_updates": 2}
    atomic_json(root/"INPUTS.json", {"scene_ids": ids, "role": "D_S only", "q": (weights/weights.sum()).tolist(),
                "main_pointwise_updates_unchanged": config.pointwise_updates, "probe_pointwise_updates": 2,
                "probe_stage_change_after_pause": True, "external_scorer_weights_loaded": False,
                "initialization_seed": config.module_init_seed, "scorer_dropout": config.scorer_dropout}, immutable=True)
    result = _run_pair(root, factory, fetch, loss, cfg, deps,
                       {"device": device, "seed": config.training_rng_seed, "denominator_function": denominators,
                        "weights": {"value": 1., "components": .2, "ranking": .2}})
    state = torch.load(root/"split/step_000004.pt", map_location="cpu", weights_only=False)
    require(state["sampler"]["natural_consumed"] == 384 and state["sampler"]["comparison_consumed"] == 128,
            "resume did not preserve the natural/comparison stage transition")
    result.update(real_scenes=len(ids), fit_subrole="scorer_fit", phase_transition_verified=True,
                  natural_consumed=384, comparison_consumed=128, probe_pointwise_updates=2,
                  main_pointwise_updates_unchanged=config.pointwise_updates)
    atomic_json(root/"RESULT.json", result)
    return result


def _terminal_cpu_fixture(root):
    """Synthetic control-flow check, explicitly separate from real-data claims."""
    entries = [{"scene_id": str(i % 4)} for i in range(3*128)]
    plan = {"entries": entries, "plan_hash": digest(entries)}
    cfg = {"lr": 1e-4, "weight_decay": .01, "global_batch_size": 128, "micro_batch_size": 128,
           "max_optimizer_steps": 2, "warmup_steps": 1, "lr_min_ratio": .1, "schedule": "warmup_constant",
           "grad_clip_norm": 1., "validation_every_steps": 2, "checkpoint_every_steps": 2, "early_stop_patience": 10**9}
    deps = {"purpose": "synthetic_CPU_terminal_control_flow_only", "schema": 1}
    def new():
        with initialization_rng(54321):
            return torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    def fetch(batch, device):
        x = torch.tensor([[int(e["scene_id"])/4, 1.] for e in batch], device=device)
        return {"x": x, "target": x[:, :1]*.5}
    def loss(module, batch):
        return {"il": LossTerm((module(batch["x"])-batch["target"]).square().sum(), batch["target"].new_tensor(len(batch["x"])))}
    train(new(), loss, fetch, ConsumedSampler(plan), cfg, deps, root, device="cpu", seed=2468,
          resume="auto" if list(root.glob("step_*.pt")) else None)
    path = root/"step_000002.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    before = file_hash(path)
    restored, sampler = new(), ConsumedSampler(plan)
    def forbidden(*args):
        raise AssertionError("terminal resume attempted to read a training batch")
    history = train(restored, loss, forbidden, sampler, cfg, deps, root, device="cpu", seed=2468, resume="auto")
    checks = {"zero_new_updates": history == [], "checkpoint_and_optimizer_unchanged": file_hash(path) == before,
              "model": equal_tree(restored.state_dict(), saved["module"]),
              "sampler": equal_tree(sampler.state_dict(), saved["sampler"]),
              "RNG": equal_tree(capture_rng(), saved["rng_by_rank"][0]),
              "terminal": saved["trainer_state"]["termination"]["termination_reason"] == "MAX_STEPS"}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "new_optimizer_steps_on_terminal_resume": len(history), "real_data_evidence": False,
              "existing_early_stop_crash_window_test": "tests/iqe/test_closeout_terminal.py::test_terminal_crash_does_not_consume_or_update"}
    atomic_json(root/"AUDIT_RESULT.json", result)
    require(all(checks.values()), "terminal CPU control-flow qualification failed")
    return result


def audit(config, device="cuda:0", *, audit_id="deterministic_v1", kind="both"):
    require(config.research_protocol == "two_candidate_v1", "two-candidate data isolation protocol required")
    require(kind in {"both", "expert", "scorer"} and re.fullmatch(r"[A-Za-z0-9_-]+", audit_id), "invalid audit kind/identity")
    root = config.root/"qualification"/"two_candidate_resume_v1"/audit_id
    with lock(root/"RUN.lock"), deterministic_probe(device) as resolved_device:
        bound = binding(config)
        scenes = {s.scene_id: s for s in load_scenes(config.root/"scenes.json")}
        require(not {s.split_role for s in scenes.values()} & {"final_test", "navtest"}, "forbidden probe role")
        identity = {"schema": "two_candidate_resume_v1", "config_hash": config.identity,
                    "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"), "base_optimizer_step": bound["optimizer_step"],
                    "base_checkpoint_hash": bound["checkpoint_hash"], "audit_code_hash": file_hash(__file__),
                    "partition_hash": file_hash(config.root/"DATA_PARTITION.json"), "device": resolved_device,
                    "main_weights_initialized_from_probe": False, "main_runtime_changed": False,
                    "separate_deterministic_backend": True, "TF32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        atomic_json(root/"IDENTITY.json", identity, immutable=True)
        atomic_json(root/"RESOLVED_CONFIG.json", asdict(config), immutable=True)
        results = {}
        if kind in {"both", "expert"}:
            results["expert"] = _expert_probe(config, root/"expert", scenes, resolved_device)
        if kind in {"both", "scorer"}:
            results["scorer"] = _scorer_probe(config, root/"scorer", scenes, resolved_device)
        results["terminal_cpu_fixture"] = _terminal_cpu_fixture(root/"terminal_cpu_fixture")
        result = {"status": "PASS", "scope": kind, "base": identity, "results": results,
                  "audit_root": str(root), "main_weights_initialized_from_probe": False,
                  "main_training_outputs_written": False, "scientific_gain": "UNTESTED"}
        atomic_json(root/f"RESULT_{kind}.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0", help="one independently authorized GPU, or CPU")
    parser.add_argument("--audit-id", default="deterministic_v1", help="isolated immutable proof identity")
    parser.add_argument("--kind", choices=("expert", "scorer", "both"), default="both")
    args = parser.parse_args()
    # Read the existing configuration without rewriting its main-run receipt.
    with open(args.config) as stream:
        config = Config(**yaml.safe_load(stream))
    print(json.dumps(audit(config, args.device, audit_id=args.audit_id, kind=args.kind), allow_nan=False))


if __name__ == "__main__":
    main()
