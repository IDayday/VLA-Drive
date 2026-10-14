"""Bounded, post-training checks of the selected research bundle on real inputs.

This does not train a new model or change the experiment's selection rule. All
outputs belong to the new experiment; the source campaign stays read-only.
"""
from pathlib import Path
import argparse
import gc
import json
import time

import numpy as np
import torch

from iqe.contracts import FeatureBundle, require
from iqe.data.sources import load_observation
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, atomic_torch, file_hash, read_json
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_batch_layout import OriginalBatchReplay
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.scorer_training import CONTEXT_KEYS
from iqe.specialist_study.training import runtime
from .bundle import load as load_bundle
from .config import load


TOLERANCES = {"context": 1e-5, "raw": 1e-5, "physical": 1e-4, "score": 1e-5}


def real_resume(config, device):
    """Eight diagnostic updates total, including the two-flow/dropout boundary."""
    from iqe.specialist_study.bank import load_bank
    from iqe.specialist_study.objectives import StudySampler, pair_population
    from iqe.specialist_study.qualifications import equal_tree
    from iqe.specialist_study.scorer_training import batch_fetch
    from iqe.specialist_study.training import preload
    from iqe.training.trainer import train
    from iqe.io import digest
    from .runner import create
    from .objectives import terms, denominators
    folder = config.root/'qualification/real_resume_R11'
    path = folder/'REPORT.json'
    source = config.source()
    meta, full = load_bank(source, 'deploy', 'scorer_fit')
    pairs, _ = pair_population(full['scores'], full['comparison_valid'])
    indices = list(dict.fromkeys(int(pair[0]) for pair in pairs))[:32]
    require(indices, 'no genuine comparisons for the resume audit')
    available = scenes_by_role(source)['scorer_fit']
    scenes = [available[i] for i in indices]
    data = {k: v[indices] for k, v in full.items()}
    dependencies = {'kind': 'disposable_hurdle_dropout_real_resume', 'experiment': config.identity,
                    'bank_hash': meta['hash'], 'scene_ids': digest([s.scene_id for s in scenes]),
                    'diagnostic_phase_boundary': 2, 'production_phase_boundary': config.phase_switch}
    if path.exists():
        report = read_json(path)
        require(report['dependencies'] == dependencies, 'resume diagnostic dependencies changed')
        return report
    cache = preload(source, scenes)
    fetch = batch_fetch(source, scenes, cache, None, data, data, False)
    cfg = {'lr': config.lr, 'weight_decay': config.weight_decay, 'global_batch_size': 128,
           'micro_batch_size': config.micro_batch, 'max_optimizer_steps': 4, 'warmup_steps': 1,
           'lr_min_ratio': 1., 'grad_clip_norm': 1., 'schedule': 'warmup_constant',
           'validation_steps': [], 'checkpoint_every_steps': 2, 'validation_every_steps': 1000,
           'early_stop_patience': 10**9}
    def execute(name, stop=None, resume=None):
        model, _ = create(config, 'R11')
        if resume is None and any((folder/name).glob('step_*.pt.COMPLETE.json')):
            resume = 'auto'
        sampler = StudySampler([s.scene_id for s in scenes], data['scores'], data['comparison_valid'],
                               seed=config.sampler_seed, bank_hash=meta['hash'], phase_steps=2)
        train(model, lambda m, b: terms(m(b['context'], b['trajectories']), b, 'zero_hurdle'),
              fetch, sampler, cfg, dependencies, folder/name, device=device,
              weights={'value': 1., 'nonzero': 1., 'components': .2, 'ranking': .2},
              resume=resume, stop_after=stop, seed=config.training_seed,
              denominator_function=lambda b: denominators(b, 'zero_hurdle'))
        return sampler.peek(128)
    next_full = execute('continuous')
    execute('interrupted', stop=2)
    next_resumed = execute('interrupted', resume='auto')
    checkpoints = [folder/name/'step_000004.pt' for name in ('continuous', 'interrupted')]
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in checkpoints]
    equality = {k: equal_tree(a[k], b[k]) for k in
                ('module', 'optimizer', 'scheduler', 'sampler', 'rng_by_rank', 'consumed_sample_ids')}
    equality['next_two_flow_sample_ids'] = next_full == next_resumed
    logs = [[json.loads(line) for line in (folder/name/'steps.jsonl').read_text().splitlines()]
            for name in ('continuous', 'interrupted')]
    equality['loss_sequence'] = [row['loss'] for row in logs[0]] == [row['loss'] for row in logs[1]]
    require(all(equality.values()), 'real-data dropout/two-flow exact resume mismatch')
    report = {'status': 'PASS', 'dependencies': dependencies, 'equality': equality,
              'real_scene_count': len(scenes), 'diagnostic_optimizer_updates_total': 8,
              'production_models_modified': False, 'diagnostic_weights_used_for_selection': False,
              'checkpoint_hashes': [file_hash(p) for p in checkpoints], 'device': device,
              'hardware': torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU'}
    atomic_json(path, report, immutable=True)
    return report


def compare_outputs(actual, expected):
    differences = {}
    for name, tolerance in (("trajectory", TOLERANCES["physical"]),
                            ("candidates", TOLERANCES["physical"]),
                            ("predicted_scores", TOLERANCES["score"])):
        a, b = actual[name].detach().cpu(), expected[name].detach().cpu()
        require(a.shape == b.shape and a.dtype == b.dtype, "output schema changed: " + name)
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), "nonfinite inference output")
        differences[name] = float((a-b).abs().max())
        require(differences[name] <= tolerance, "inference output drift: " + name)
    require(torch.equal(actual["selected"].cpu(), expected["selected"].cpu()), "selected candidate changed")
    require(actual["calls"] == {"visual": 1, "language": 1}, "second VLA forward in inference")
    return differences


def _capture(system, observations):
    captured = {}
    def context(_module, arguments):
        require("context" not in captured, "repeated Scorer forward")
        captured["context"] = {k: v.detach().cpu().clone() for k, v in arguments[0].items()}
    def base(_module, _arguments, result):
        require("raw" not in captured, "repeated Base decoder")
        captured["raw"] = result.raw.detach().cpu().clone()
    handles = [system.scorer.register_forward_pre_hook(context),
               system.adapter.framework.action_model.expert.register_forward_hook(base)]
    try:
        result = system.shared_forward(observations)
    finally:
        for handle in handles:
            handle.remove()
    return result, captured


@torch.no_grad()
def online_cache(config, system, device):
    source = config.source()
    scenes = scenes_by_role(source)["stage_val"][:4]
    replay = OriginalBatchReplay(source)
    groups = list(dict.fromkeys(replay.batch_for(s.scene_id) for s in scenes))
    wanted = {s.scene_id for s in scenes}
    cache = ReusedContext(source, capacity=16)
    rows = []
    for group in groups:
        batch = [replay.scenes[sid] for sid in group]
        require(all(s.split_role != "final_test" for s in batch), "forbidden qualification role")
        observations = [load_observation(s, system.adapter.contract) for s in batch]
        online, captured = _capture(system, observations)
        indices = [i for i, s in enumerate(batch) if s.scene_id in wanted]
        selected = [batch[i] for i in indices]
        x = cache.batch(selected, device)
        features = FeatureBundle(x["planner_scene"], x["planner_ego"], x["planner_mask"],
                                 system.adapter.contract_hash, tuple(s.scene_id for s in selected))
        base = system.adapter.predict_s0(features)
        private = {k: x[k] for k in ("planner_scene", "planner_ego", "planner_mask", "world", "action", "current_ego")}
        trajectories = torch.stack([base.physical]+[m(private).physical for m in system.experts], 1)
        context = {k: x[k].detach() for k in CONTEXT_KEYS}
        scores = system.scorer(context, trajectories)["values"]
        differences = {}
        for name, value in context.items():
            a, b = value.cpu(), captured["context"][name][indices]
            require(a.shape == b.shape and a.dtype == b.dtype, "cache context schema changed")
            if a.dtype == torch.bool:
                require(torch.equal(a, b), "cache mask changed")
                differences[name] = 0.
            else:
                differences[name] = float((a-b).abs().max())
                require(differences[name] <= TOLERANCES["context"], "cache context drift: " + name)
        differences["Base_raw"] = float((base.raw.cpu()-captured["raw"][indices]).abs().max())
        differences["candidate_physical"] = float((trajectories.cpu()-online["candidates"][indices].cpu()).abs().max())
        differences["predicted_scores"] = float((scores.cpu()-online["predicted_scores"][indices].cpu()).abs().max())
        require(differences["Base_raw"] <= TOLERANCES["raw"] and
                differences["candidate_physical"] <= TOLERANCES["physical"] and
                differences["predicted_scores"] <= TOLERANCES["score"], "online/cache numerical drift")
        # The registered pool order is Base then stable expert ID, so argmax's
        # first-index tie rule matches the deployed wrapper here.
        require(system.expert_ids == tuple(sorted(system.expert_ids)), "unexpected pool order")
        require(torch.equal(scores.argmax(1).cpu(), online["selected"][indices].cpu()), "cache selection changed")
        # Deliberately unusable training labels: the deployment whitelist must
        # discard these, not attempt to encode them as current observations.
        polluted = [row | {"GT": torch.tensor(float("nan")), "future_actors": object(),
                           "score": object(), "target_trajectory": object()} for row in observations]
        changed, captured_changed = _capture(system, polluted)
        label_errors = compare_outputs(changed, online)
        require(all(torch.equal(v, captured_changed["context"][k]) for k, v in captured["context"].items()),
                "training labels changed score context")
        require(torch.equal(captured["raw"], captured_changed["raw"]), "training labels changed Base raw output")
        rows.append({"scene_ids": [s.scene_id for s in selected], "canonical_batch": list(group),
                     "observation_hashes": [s.observation_hash for s in selected], "differences": differences,
                     "label_pollution_output_errors": label_errors, "label_independence": True,
                     "online_calls": online["calls"], "branch_calls": online["branch_calls"],
                     "cache_head_VLA_calls": 0})
    return {"status": "PASS", "real_scenes": len(scenes), "batches": rows,
            "original_batch_plan_hash": replay.plan_hash,
            "original_batch_layout_provenance": replay.provenance, "tolerances": TOLERANCES,
            "batch_shape_invariance_claimed": False, "no_training_or_selection_use": True}


@torch.no_grad()
def benchmark(config, system, device):
    scenes = scenes_by_role(config.source())["stage_val"][:32]
    count = len(system.experts)
    rows = []
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    for n, score in ((0, False), (count, False), (count, True)):
        for i, expert in enumerate(system.experts):
            expert.to(device if i < n else "cpu")
        system.scorer.to(device if score else "cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elapsed, head_ms = [], []
        for iteration in range(35):
            events = []
            if score and device.startswith("cuda"):
                def before(*_):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(); events.append(event)
                def after(*_):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(); events.append(event)
                handles = [system.scorer.register_forward_pre_hook(before), system.scorer.register_forward_hook(after)]
            else:
                handles = []
            try:
                sync(); started = time.perf_counter()
                obs = load_observation(scenes[iteration % len(scenes)], system.adapter.contract)
                result = system.shared_forward([obs], expert_count=n, score=score)
                sync(); seconds = time.perf_counter()-started
            finally:
                for handle in handles:
                    handle.remove()
            require(result["calls"] == {"visual": 1, "language": 1}, "VLA call count changed")
            require(result["branch_calls"] == {"Base_decoder": 1, "expert_decoders": [1]*n,
                    "score_context_projections": int(score), "spatial_reader": 0}, "private/head calls changed")
            if iteration >= 3:
                elapsed.append(seconds)
                if events:
                    require(len(events) == 2, "Scorer timer call count changed")
                    head_ms.append(events[0].elapsed_time(events[1]))
        rows.append({"K": n+1, "Scorer": score, "batch": 1, "warmup": 3, "measurements": len(elapsed),
                     "mean_s": float(np.mean(elapsed)), "p50_s": float(np.median(elapsed)),
                     "p95_s": float(np.quantile(elapsed, .95)), "scorer_GPU_mean_ms": float(np.mean(head_ms)) if head_ms else None,
                     "scorer_GPU_p50_ms": float(np.median(head_ms)) if head_ms else None,
                     "scorer_GPU_p95_ms": float(np.quantile(head_ms, .95)) if head_ms else None,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None,
                     "real_image_decode_and_identity_checks_included": True, "inactive_weights_on_CPU": True,
                     "CUDA_synchronization": "before/after end-to-end; events around Scorer", "calls": result["calls"],
                     "branch_calls": result["branch_calls"]})
    return {"rows": rows, "hardware": torch.cuda.get_device_name() if device.startswith("cuda") else "CPU",
            "runtime": "original Base inference contract; deterministic FP32 Scorer",
            "VLM_parameters": sum(p.numel() for p in system.adapter.framework.qwen_vl_interface.parameters()),
            "expert_parameters": [sum(p.numel() for p in m.parameters()) for m in system.experts],
            "Scorer_parameters": sum(p.numel() for p in system.scorer.parameters()),
            "external_scorer_weights_loaded": False, "exclusive_device_assumed": False}


def run(config, device, command):
    from iqe.specialist_study.resource_guard import check
    check(config, 'benchmark-system')
    device = runtime(device)
    if command == 'resume-audit':
        return real_resume(config, device)
    bundle = config.root/"research_bundle/bundle.pt"
    dependencies = {"bundle": file_hash(bundle), "choice": file_hash(config.root/"MODEL_CHOICE.json")}
    folder = config.root/"qualification"
    report = folder/("ROUNDTRIP.json" if command == "roundtrip" else "ONLINE_AND_LATENCY.json")
    if report.exists():
        result = read_json(report)
        require(result["dependencies"] == dependencies, "stale qualification")
        return result
    system = load_bundle(bundle, device)
    scenes = scenes_by_role(config.source())["stage_val"]
    if command == "roundtrip":
        receipt = read_json(folder/"REFERENCE.json")
        require(receipt["dependencies"] == dependencies and receipt["observation_hash"] == scenes[0].observation_hash,
                "roundtrip observation/dependency mismatch")
        require(file_hash(folder/"REFERENCE.pt") == receipt["checksum"], "roundtrip reference changed")
        expected = torch.load(folder/"REFERENCE.pt", map_location="cpu", weights_only=True)
        actual = system.shared_forward([load_observation(scenes[0], system.adapter.contract)])
        result = {"status": "PASS", "dependencies": dependencies, "errors": compare_outputs(actual, expected),
                  "separate_CLI_process": True, "calls": actual["calls"], "branch_calls": actual["branch_calls"]}
    else:
        before = module_hash(system)
        context = online_cache(config, system, device)
        timing = benchmark(config, system, device)
        actual = system.shared_forward([load_observation(scenes[0], system.adapter.contract)])
        atomic_torch(folder/"REFERENCE.pt", {k: actual[k].detach().cpu() for k in
                      ("trajectory", "candidates", "predicted_scores", "selected")}, immutable=True)
        atomic_json(folder/"REFERENCE.json", {"dependencies": dependencies, "scene_id": scenes[0].scene_id,
                     "observation_hash": scenes[0].observation_hash, "checksum": file_hash(folder/"REFERENCE.pt")}, immutable=True)
        require(module_hash(system) == before, "qualification changed model parameters/buffers")
        result = {"status": "PASS", "dependencies": dependencies, "online_cache": context, "latency": timing,
                  "parameters_and_buffers_unchanged": True, "ACTIVE_updated": False,
                  "test_or_navtest_access": False, "roundtrip_command_still_required": True}
    atomic_json(report, result, immutable=True)
    del system
    gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("online", "roundtrip", "resume-audit"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(json.dumps(run(load(args.config), args.device, args.command)))


if __name__ == "__main__":
    main()
