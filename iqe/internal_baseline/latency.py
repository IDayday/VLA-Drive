"""Real image deployment timing of the selected experiment bundle, not cache timing."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from iqe.io import atomic_json, atomic_bytes, read_json, file_hash
from iqe.contracts import require
from iqe.data.manifests import load_scenes
from iqe.data.sources import load_observation
from .inference import load_bundle
from .features import encode_joint, JointCache
from .training import score_context, make_scorer


def benchmark(config, device="cuda:0", warmup=5, repetitions=30):
    require(warmup >= 5 and repetitions >= 30, "deployment latency needs a meaningful timing window")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    algorithm = load_bundle(config.root/"bundle", device)
    language = algorithm.adapter.framework.qwen_vl_interface.model.model.language_model
    calls, projections = [], []
    h = language.register_forward_hook(lambda *args: calls.append(1))
    ph = algorithm.scorer.world_projection.register_forward_hook(lambda *args: projections.append(1))
    scenes = [s for s in load_scenes(config.root/"scenes.json") if s.split_role == "stage_val"][:max(32, repetitions)]
    records = []
    sync = lambda: torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i in range(warmup+repetitions):
            scene = scenes[i % len(scenes)]
            sync(); t0 = time.perf_counter()
            obs = [load_observation(scene, algorithm.adapter.contract)]
            t1 = time.perf_counter(); calls.clear(); projections.clear()
            planner, context = encode_joint(algorithm.adapter, obs)
            sync(); t2 = time.perf_counter()
            base = algorithm.adapter.predict_s0(planner)
            sync(); t3 = time.perf_counter()
            expert = algorithm.expert(planner)
            sync(); t4 = time.perf_counter()
            candidates = torch.stack((base.physical, expert.physical), 1)
            scoring_input = score_context(config, context.tensors() | {"planner_scene": planner.scene, "planner_mask": planner.valid_tokens})
            prediction = algorithm.scorer(scoring_input, candidates)
            sync(); t5 = time.perf_counter()
            selected = (prediction["values"][:, 1] > prediction["values"][:, 0]).long()
            final = candidates[torch.arange(len(obs), device=device), selected].cpu()
            sync(); t6 = time.perf_counter()
            require(len(calls) == 1 and len(projections) == 1, "deployment call count changed")
            require(torch.isfinite(final).all().item(), "deployment output invalid")
            if i >= warmup:
                records.append({"preprocessing_ms": (t1-t0)*1000, "shared_VLM_Qformer_ego_ms": (t2-t1)*1000,
                                "base_decoder_ms": (t3-t2)*1000, "expert_decoder_ms": (t4-t3)*1000,
                                "scorer_ms": (t5-t4)*1000, "selection_cpu_copy_ms": (t6-t5)*1000,
                                "image_to_trajectory_ms": (t6-t0)*1000})
    h.remove(); ph.remove()
    stats = {key: {"mean": float(np.mean([r[key] for r in records])), "p50": float(np.median([r[key] for r in records])),
                   "p95": float(np.quantile([r[key] for r in records], .95))} for key in records[0]}
    # Own bundle state roundtrip and cached-vs-online scorer connection, separately
    # from all timing windows. No environment labels are inputs to the algorithm.
    scene = scenes[0]; observations = [load_observation(scene, algorithm.adapter.contract)]
    with torch.no_grad():
        direct = algorithm(observations, return_debug=True)
    cached = JointCache(config).batch([scene], device)
    state = torch.load(config.root/"bundle/algorithm.pt", weights_only=True, map_location="cpu")
    clone = make_scorer(config).to(device).eval()
    clone.load_state_dict(state["internal_scorer"], strict=True)
    with torch.no_grad():
        a = algorithm.scorer(score_context(config, cached), direct["candidates"])["values"]
        b = clone(score_context(config, cached), direct["candidates"])["values"]
    require(torch.equal(a, b), "own scorer bundle roundtrip changes predictions")
    result = {"status": "PASS", "hardware": torch.cuda.get_device_name(device), "batch": 1, "K": 2,
              "warmup": warmup, "repetitions": repetitions, "source": "real_images_online",
              "synchronization": "CUDA synchronize at each measured boundary, single process",
              "statistics_ms": stats, "samples": records,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "VLM_calls_per_scene": 1, "score_projection_calls_per_scene": 1,
              "scorer_roundtrip_bitwise": True, "cached_online_prediction_max_abs": float((a-direct["predicted_values"]).abs().max()),
              "bundle_hash": file_hash(config.root/"bundle/BUNDLE.json"), "external_scorer_weights_loaded": False,
              "parameters": read_json(config.root/"SCORE_CONTEXT_AUDIT.json")}
    atomic_json(config.root/"LATENCY_AND_PARAMETERS.json", result)
    atomic_bytes(config.root/"LATENCY_AND_PARAMETER_REPORT.md", ("# Deployment cost\n\nReal images, selected own weights, one original VLM forward. All values below are measured; the frozen-feature training path is not used for timing.\n\n```json\n"+json.dumps(result, indent=2)+"\n```\n").encode())
    return result
