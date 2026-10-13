"""Cached head training using the existing exact-resume, DDP-aware IQE trainer."""
from copy import deepcopy
from pathlib import Path
import io
import os
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, atomic_bytes, file_hash, digest
from iqe.data.manifests import load_scenes
from iqe.losses import expert_il_terms
from iqe.training.trainer import train
from iqe.evaluation.retention import module_hash, module_hash_parameters, module_hash_buffers
from .binding import load_head
from .features import JointCache, planner_bundle
from .sampling import ExpertSampler, TwoStreamSampler
from .scorer import construct
from . import objectives
from .bank import expert_candidates, cache_candidates, score_candidates, COMPONENTS
from .evaluation import selection_metrics, bank_statistics, argmax_base_tie


def training_device(requested):
    if str(requested).startswith("cuda"):
        index = int(os.environ.get("LOCAL_RANK", 0)) if int(os.environ.get("WORLD_SIZE", 1)) > 1 else (torch.device(requested).index or 0)
        os.environ["LOCAL_RANK"] = str(index)
        torch.cuda.set_device(index)
        return f"cuda:{index}"
    require(str(requested) == "cpu", "unsupported training device")
    return "cpu"


def prepare_head_runtime(config):
    if config.research_protocol != "two_candidate_v1":
        return
    # Fixed before either main head is trained. No precision/architecture change;
    # deterministic FP32 head execution makes same-runtime resume auditable.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "head CUBLAS contract mismatch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    atomic_json(config.root/"HEAD_TRAINING_RUNTIME.json", {"precision": "float32_parameters_gradients_adam_and_head_compute",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True, "tf32": False,
                "input_context": "unaltered_locked_Base_cache", "expert_dropout": "inherit_Base_enabled",
                "scorer_dropout": config.scorer_dropout, "shared_VLM_calls": 0}, immutable=True)


def training_config(config, kind):
    return {"lr": 1e-4, "weight_decay": .01, "global_batch_size": 128,
            "micro_batch_size": config.expert_micro_batch if kind == "expert" else config.scorer_micro_batch,
            "max_optimizer_steps": config.expert_updates if kind == "expert" else config.scorer_updates,
            "warmup_steps": 50 if kind == "expert" else 100, "lr_min_ratio": .1,
            "grad_clip_norm": 1., "validation_every_steps": config.validation_interval,
            "checkpoint_every_steps": config.validation_interval, "early_stop_patience": 10**9,
            "schedule": "warmup_constant" if config.research_protocol == "two_candidate_v1" else "warmup_cosine"}


def train_expert(config, device="cuda:0", resume=None, stop_after=None):
    prepare_head_runtime(config)
    device = training_device(device)
    scenes = load_scenes(config.root/"scenes.json")
    lookup = {s.scene_id: s for s in scenes}
    manifest = read_json(config.root/"EXPERT_MANIFEST.json")
    require(all(lookup[i].split_role == "incremental_fit" for i in manifest["hard"]+manifest["ordinary"]), "expert role leakage")
    from .partition import training_ids
    allowed = training_ids(config, scenes, "expert_fit")
    require(set(manifest["hard"]+manifest["ordinary"]) <= allowed, "D_S entered expert training")
    from .cache_execution import training_cache
    cache = training_cache(config, [lookup[i] for i in dict.fromkeys(manifest["hard"]+manifest["ordinary"])])
    base = load_head(config, device).eval().requires_grad_(False)
    expert = deepcopy(base).requires_grad_(True)
    old_hash = {"parameters": module_hash_parameters(base), "buffers": module_hash_buffers(base)}
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(base.parameters(), expert.parameters())), "expert aliases Base storage")
    probe = [lookup[manifest["hard"][0]]]
    probe_features = planner_bundle(config, cache.batch(probe, device), [s.scene_id for s in probe])
    with torch.no_grad():
        old_output = base(probe_features)
        require(torch.equal(old_output.raw, expert.eval()(probe_features).raw), "clone output mismatch")
    sampler = ExpertSampler(manifest["hard"], manifest["ordinary"], config.natural_seed, manifest["hash"], manifest["sampling"]["hard"])
    def fetch(entries, device):
        rows = [lookup[e["scene_id"]] for e in entries]
        x = cache.batch(rows, device)
        return {"features": planner_bundle(config, x, [s.scene_id for s in rows]), "target": x["target_raw"],
                "mask": torch.tensor([not e.get("_padding", False) for e in entries], device=device)[:, None].expand(-1, x["target_raw"].shape[1])}
    contract = read_json(config.root/"OFFLINE_BASE_CONTRACT.json")
    def loss(m, batch):
        return expert_il_terms(m(batch["features"]), batch["target"], masks=batch["mask"], prev_weight=contract["query_architecture"]["prev_weight"])
    val = [s for s in scenes if s.split_role == "stage_val"]
    old_scores = read_json(config.root/"BASE_FIT_VAL_SCORES.json")
    def validate(m, step):
        weight_hash = module_hash(m)
        candidates = expert_candidates(config, m, val, weight_hash, device)
        labels = score_candidates(config, val, candidates, f"expert_val_{step}")
        scores = np.array([[old_scores[s.scene_id]["total_score_01"], l.total_score_01] for s, l in zip(val, labels)])
        comp = np.array([[list(old_scores[s.scene_id]["named_components"][c] for c in COMPONENTS),
                          list(l.named_components[c] for c in COMPONENTS)] for s, l in zip(val, labels)])
        metrics = selection_metrics(scores, comp, scores.argmax(1), [s.source_log_id for s in val],
                                    bootstrap_repetitions=config.bootstrap_repetitions, bootstrap_seed=config.bootstrap_seed)
        hard = np.array([s.scene_id in set(manifest["hard_val"]) for s in val])
        metrics["hard_val_expert_minus_base"] = float((scores[hard, 1]-scores[hard, 0]).mean()) if hard.any() else None
        metrics["hard_val_count"] = int(hard.sum())
        metrics["normal_val_expert_minus_base"] = float((scores[~hard, 1]-scores[~hard, 0]).mean()) if (~hard).any() else None
        metrics["weight_hash"] = weight_hash
        metrics["selection_key"] = [metrics["oracle_gain"], metrics["hard_val_expert_minus_base"] or 0.]
        return metrics
    deps = {"experiment_id": config.experiment_id, "base": file_hash(config.root/"BASE_BINDING.json"),
            "manifest": file_hash(config.root/"EXPERT_MANIFEST.json"), "kind": "own_expert_1"}
    result = train(expert, loss, fetch, sampler, training_config(config, "expert"), deps, config.root/"expert",
                   device=device, resume=resume, stop_after=stop_after, validate=validate, seed=config.training_rng_seed,
                   denominator_function=lambda b: {"il": b["mask"].any(-1).sum().float()})
    with torch.no_grad():
        after = base(probe_features)
    audit = {"base_hash_before": old_hash, "base_hash_after": {"parameters": module_hash_parameters(base), "buffers": module_hash_buffers(base)},
             "raw_max_abs": float((after.raw-old_output.raw).abs().max()), "physical_max_abs": float((after.physical-old_output.physical).abs().max()),
             "shared_model_present_in_optimizer": False, "vlm_calls_during_training": 0}
    require(audit["base_hash_before"] == audit["base_hash_after"] and audit["raw_max_abs"] == audit["physical_max_abs"] == 0., "Base drift")
    atomic_json(config.root/"EXPERT_FREEZE_AUDIT.json", audit)
    return read_json(config.root/"expert/result.json")


def build_pair_bank(config, device="cuda:0"):
    result = read_json(config.root/"expert/result.json")
    require(result["status"] == "COMPLETE", "expert final/best checkpoint is not selected")
    best = config.root/"expert/best.pt"
    require(file_hash(best) == result["best_checkpoint_hash"], "expert best changed")
    expert = load_head(config, device)
    expert.load_state_dict(torch.load(best, weights_only=True, map_location=device), strict=True)
    scenes = load_scenes(config.root/"scenes.json")
    if config.research_protocol == "two_candidate_v1":
        from .partition import training_ids
        ds = training_ids(config, scenes, "scorer_fit")
        expert_manifest = read_json(config.root/"EXPERT_MANIFEST.json")
        hard_sets = read_json(config.root/"HARD_SETS_BEFORE_EXPERT.json")
        diagnostic_ids = set(expert_manifest["hard"]+expert_manifest["ordinary"]+hard_sets["F_train"])
        scenes = [s for s in scenes if s.scene_id in ds or s.scene_id in diagnostic_ids or s.split_role != "incremental_fit"]
        atomic_json(config.root/"PAIR_BANK_SCENES.json", {"scene_ids": [s.scene_id for s in scenes],
                    "purpose": "all D_S/val/cal/dev plus fixed D_E H/N/F diagnostics; D_E never used for Scorer fit",
                    "partition_hash": file_hash(config.root/"DATA_PARTITION.json"),
                    "expert_manifest_hash": file_hash(config.root/"EXPERT_MANIFEST.json")}, immutable=True)
    # Fixed development set is generated once here, after expert checkpoint selection.
    base_candidates = cache_candidates(config, scenes, "base")
    base_labels = score_candidates(config, scenes, base_candidates, "bank_base_all_roles")
    new_candidates = expert_candidates(config, expert, scenes, module_hash(expert), device)
    new_labels = score_candidates(config, scenes, new_candidates, "bank_expert_all_roles")
    # Preserve the official labels exactly in the bank. Loss arithmetic explicitly
    # uses FP32; selection/statistical evaluation uses the original float64 labels.
    scores = np.array([[a.total_score_01, b.total_score_01] for a, b in zip(base_labels, new_labels)], dtype=np.float64)
    components = np.array([[[r.named_components[c] for c in COMPONENTS] for r in (a, b)] for a, b in zip(base_labels, new_labels)], dtype=np.float64)
    trajectories = np.stack([[np.load(a.trajectory_ref)["trajectory"], np.load(b.trajectory_ref)["trajectory"]] for a, b in zip(base_candidates, new_candidates)])
    from .partition import training_ids
    all_scenes = load_scenes(config.root/"scenes.json")
    ds = training_ids(config, all_scenes, "scorer_fit")
    de = training_ids(config, all_scenes, "expert_fit")
    identities = [{"scene_id": s.scene_id, "role": s.split_role, "training_subrole": "scorer_fit" if s.scene_id in ds else "expert_fit" if s.scene_id in de else None,
                   "log": s.source_log_id, "base_key": a.key, "expert_key": b.key,
                   "base_score_hash": digest(asdict_score(x)), "expert_score_hash": digest(asdict_score(y))}
                  for s, a, b, x, y in zip(scenes, base_candidates, new_candidates, base_labels, new_labels)]
    data = io.BytesIO(); np.savez(data, scores=scores, components=components, trajectories=trajectories)
    atomic_bytes(config.root/"PAIR_BANK.npz", data.getvalue(), immutable=True)
    atomic_json(config.root/"PAIR_BANK.json", {"rows": identities, "array_hash": file_hash(config.root/"PAIR_BANK.npz"),
                "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"), "expert_best_hash": file_hash(best),
                "score_protocol_hash": file_hash(config.root/"SCORE_PROTOCOL.json"), "components": list(COMPONENTS)}, immutable=True)
    stats = {}
    for role in sorted({s.split_role for s in scenes}):
        mask = np.array([s.split_role == role for s in scenes])
        stats[role] = bank_statistics(scores[mask], components[mask], [s.source_log_id for s, yes in zip(scenes, mask) if yes])
    if config.research_protocol == "two_candidate_v1":
        for role, ids in (("expert_fit", de), ("scorer_fit", ds)):
            mask = np.array([s.scene_id in ids for s in scenes])
            stats[role] = bank_statistics(scores[mask], components[mask], [s.source_log_id for s, yes in zip(scenes, mask) if yes])
    atomic_json(config.root/"PAIR_BANK_STATISTICS.json", stats, immutable=True)
    atomic_json(config.root/"PAIR_BANK.COMPLETE.json", {"status": "COMPLETE", "bank_hash": file_hash(config.root/"PAIR_BANK.json"),
                "array_hash": file_hash(config.root/"PAIR_BANK.npz"), "scene_manifest_hash": file_hash(config.root/"scenes.json"),
                "rows": len(scenes), "candidate_ids": ["expert_0", "expert_1"]}, immutable=True)
    return stats


def asdict_score(x):
    from dataclasses import asdict
    return asdict(x)


def load_bank(config):
    manifest = read_json(config.root/"PAIR_BANK.json")
    receipt = read_json(config.root/"PAIR_BANK.COMPLETE.json")
    require(receipt["status"] == "COMPLETE" and receipt["bank_hash"] == file_hash(config.root/"PAIR_BANK.json")
            and receipt["scene_manifest_hash"] == file_hash(config.root/"scenes.json")
            and receipt["candidate_ids"] == ["expert_0", "expert_1"], "bank incomplete/corrupt/incorrect candidate identities")
    require(file_hash(config.root/"PAIR_BANK.npz") == manifest["array_hash"], "candidate bank array changed")
    require(manifest["base_binding_hash"] == file_hash(config.root/"BASE_BINDING.json")
            and manifest["expert_best_hash"] == file_hash(config.root/"expert/best.pt")
            and manifest["score_protocol_hash"] == file_hash(config.root/"SCORE_PROTOCOL.json"), "stale bank dependencies")
    with np.load(config.root/"PAIR_BANK.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    require(arrays["scores"].shape == (receipt["rows"], 2) and np.isfinite(arrays["scores"]).all(), "two-candidate score bank shape/validity")
    expected_ids = [s.scene_id for s in load_scenes(config.root/"scenes.json")]
    if config.research_protocol == "two_candidate_v1":
        scope = read_json(config.root/"PAIR_BANK_SCENES.json")
        require(scope["partition_hash"] == file_hash(config.root/"DATA_PARTITION.json") and
                scope["expert_manifest_hash"] == file_hash(config.root/"EXPERT_MANIFEST.json"), "stale pair bank subroles")
        require(set(scope["scene_ids"]) <= set(expected_ids), "unknown bank scenes")
        expected_ids = scope["scene_ids"]
    require([r["scene_id"] for r in manifest["rows"]] == expected_ids, "bank scene ID/order join")
    return manifest, arrays


def score_context(config, values):
    if config.score_context == "planner_ego":
        return {"world": values["world"][:, :0], "action": values["planner_scene"],
                "current_ego": values["current_ego"], "world_valid": values["world_valid"][:, :0], "action_valid": values["planner_mask"]}
    return {k: values[k] for k in ("world", "action", "current_ego", "world_valid", "action_valid")}


def make_scorer(config):
    from iqe.query_base import source_trajectory_contract
    scenes = load_scenes(config.root/"scenes.json")
    example = JointCache(config).read(scenes[0])
    tc = read_json(config.root/"OFFLINE_BASE_CONTRACT.json")["trajectory"]
    return construct(config, source_trajectory_contract(tc["horizon"], tc["dt"]), example["world"].shape[-1],
                     example["planner_scene" if config.score_context == "planner_ego" else "action"].shape[-1])


@torch.no_grad()
def predict_scores(config, module, scenes, trajectories, device):
    cache = JointCache(config)
    module.eval()
    values, components = [], []
    for at in range(0, len(scenes), config.scorer_micro_batch):
        subset = scenes[at:at+config.scorer_micro_batch]
        x = cache.batch(subset, device)
        p = module(score_context(config, x), torch.as_tensor(trajectories[at:at+len(subset)], device=device))
        require(torch.isfinite(p["values"]).all().item() and torch.isfinite(p["component_logits"]).all().item(),
                "nonfinite Scorer output on a valid frozen candidate bank")
        values.append(p["values"].cpu().numpy()); components.append(p["component_logits"].sigmoid().cpu().numpy())
    return np.concatenate(values), np.concatenate(components)


def train_scorer(config, device="cuda:0", resume=None, stop_after=None):
    prepare_head_runtime(config)
    device = training_device(device)
    manifest, data = load_bank(config)
    scenes = load_scenes(config.root/"scenes.json")
    lookup = {s.scene_id: s for s in scenes}
    indices = {r["scene_id"]: i for i, r in enumerate(manifest["rows"])}
    from .partition import training_ids
    allowed = training_ids(config, scenes, "scorer_fit")
    fit = [r["scene_id"] for r in manifest["rows"] if r["scene_id"] in allowed]
    fit_scores = torch.as_tensor(data["scores"][[indices[i] for i in fit]])
    w = objectives.comparison_weights(fit_scores, torch.ones_like(fit_scores, dtype=torch.bool)).numpy()
    sampler = TwoStreamSampler(fit, w, natural_seed=config.natural_seed, comparison_seed=config.comparison_seed,
                               pointwise_updates=config.pointwise_updates, bank_hash=file_hash(config.root/"PAIR_BANK.json"))
    atomic_json(config.root/"COMPARISON_DISTRIBUTION.json", {"q_hash": sampler.q_hash, "bank_hash": sampler.bank_hash,
                "scene_ids": fit, "weights": w.tolist(), "unique_valid": int((w > 0).sum()), "replacement": True,
                "fit_subrole": "scorer_fit" if config.research_protocol == "two_candidate_v1" else "incremental_fit",
                "valid_group_count": len({lookup[i].source_log_id for i, x in zip(fit, w) if x > 0}),
                "effective_weighted_population": float(w.sum()**2/(w*w).sum()) if w.sum() else 0.,
                "status": "PRESENT" if w.sum() else "SCORER_COMPARISON_SIGNAL_ABSENT",
                "objective": "q=abs(gap)/sum(abs(gap)); sampled ranking has no second gap factor"}, immutable=True)
    from .cache_execution import training_cache
    cache = training_cache(config, [lookup[name] for name in fit])
    module = make_scorer(config)
    def fetch(entries, device):
        rows = [lookup[e["scene_id"]] for e in entries]
        ix = [indices[s.scene_id] for s in rows]
        x = cache.batch(rows, device)
        scores = torch.as_tensor(data["scores"][ix], device=device)
        return {"context": score_context(config, x), "trajectories": torch.as_tensor(data["trajectories"][ix], device=device),
                "scores": scores, "valid": torch.ones_like(scores, dtype=torch.bool),
                "components": torch.as_tensor(data["components"][ix], device=device),
                "component_valid": torch.ones(len(ix), 2, len(COMPONENTS), device=device, dtype=torch.bool),
                "natural": torch.tensor([e["natural"] and not e.get("_padding", False) for e in entries], device=device),
                "comparison": torch.tensor([e["comparison"] and not e.get("_padding", False) for e in entries], device=device)}
    def loss(m, b):
        return objectives.terms(m(b["context"], b["trajectories"]), b)
    val = [lookup[r["scene_id"]] for r in manifest["rows"] if r["role"] == "stage_val"]
    vi = [indices[s.scene_id] for s in val]
    def validate(m, step):
        p, _ = predict_scores(config, m, val, data["trajectories"][vi], device)
        metrics = selection_metrics(data["scores"][vi], data["components"][vi], argmax_base_tie(p), [s.source_log_id for s in val],
                                    bootstrap_repetitions=config.bootstrap_repetitions, bootstrap_seed=config.bootstrap_seed)
        metrics["selection_key"] = [metrics["selected"], -sum(c["new_violations"] for c in metrics["safety"].values())]
        return metrics
    deps = {"experiment_id": config.experiment_id, "base": file_hash(config.root/"BASE_BINDING.json"),
            "bank": file_hash(config.root/"PAIR_BANK.json"), "q_hash": sampler.q_hash,
            "initialization": file_hash(config.root/"SCORER_INITIALIZATION.json"), "kind": "own_random_internal_scorer"}
    train(module, loss, fetch, sampler, training_config(config, "scorer"), deps, config.root/"scorer", device=device,
          weights={"value": 1., "components": .2, "ranking": .2}, resume=resume, stop_after=stop_after,
          validate=validate, seed=config.training_rng_seed, denominator_function=objectives.denominators)
    return read_json(config.root/"scorer/result.json")
