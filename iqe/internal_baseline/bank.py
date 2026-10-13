from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
import subprocess
import time
import numpy as np
import torch
from iqe.contracts import require, CandidateRecord, ScoreRecord, strict_record
from iqe.io import atomic_bytes, atomic_json, read_json, file_hash, digest
from iqe.data.manifests import load_scenes
from iqe.scoring.reference import COMPONENTS, python_source_hash
from .features import JointCache, planner_bundle


SCORING_CONFIG = {"proposal_sampling": {"num_poses": 40, "interval_length": .1},
                  "scorer": {"progress_weight": 5., "ttc_weight": 5., "comfortable_weight": 2.,
                  "driving_direction_weight": 0., "driving_direction_horizon": 1.,
                  "driving_direction_compliance_threshold": 2., "driving_direction_violation_threshold": 6.,
                  "stopped_speed_threshold": .005, "progress_distance_threshold": 5.}}


def protocol(config):
    path = config.root / "SCORE_PROTOCOL.json"
    if path.exists():
        p = read_json(path)
        require(p["navsim_source_hash"] == python_source_hash(Path(p["navsim_root"])/"navsim")
                and p["nuplan_source_hash"] == python_source_hash(Path(p["nuplan_root"])/"nuplan")
                and p["config"] == SCORING_CONFIG, "official protocol source/config changed")
        return p
    nuplan_root = str(Path(config.navsim_root).parent / "nuplan-devkit")
    p = {"metric": "NAVSIM_v1.1_PDMS", "navsim_root": config.navsim_root, "nuplan_root": nuplan_root,
         "navsim_source_hash": python_source_hash(Path(config.navsim_root)/"navsim"),
         "nuplan_source_hash": python_source_hash(Path(nuplan_root)/"nuplan"),
         "reference": "metric_cache.trajectory", "traffic": "official_nonreactive",
         "backend": "official_single_reference_fresh_state", "score_scale": "zero_one",
         "config": SCORING_CONFIG, "eval_seed": 42, "repetitions": 1,
         "components": list(COMPONENTS)}
    p["protocol_hash"] = digest(p)
    atomic_json(path, p, immutable=True)
    return p


def save_candidate(config, scene, expert_id, checkpoint_hash, physical):
    p = np.asarray(physical, dtype=np.float32)
    contract = read_json(config.root / "OFFLINE_BASE_CONTRACT.json")["trajectory"]
    require(p.shape == (contract["horizon"], 3) and np.isfinite(p).all(), "invalid physical candidate, not a normal zero score")
    folder = config.root / "candidates" / expert_id / checkpoint_hash
    path = folder / (scene.scene_id + ".npz")
    if path.exists():
        with np.load(path) as old:
            require(np.array_equal(old["trajectory"], p), "immutable candidate output changed")
    else:
        data = io.BytesIO(); np.savez(data, trajectory=p)
        atomic_bytes(path, data.getvalue(), immutable=True)
    b = read_json(config.root / "BASE_BINDING.json")
    record = CandidateRecord(1, scene.scene_id, expert_id, checkpoint_hash, b["contract_hash"], file_hash(path), str(path),
                             contract["raw_representation"], contract["frame"], contract["horizon"], contract["dt"], True,
                             file_hash(__file__), config.identity, b["bound_at_utc"])
    if path.with_suffix(".json").exists():
        previous = strict_record(CandidateRecord, read_json(path.with_suffix(".json")))
        require(previous.key == record.key and previous.horizon == record.horizon and previous.dt == record.dt
                and previous.raw_action_representation == record.raw_action_representation
                and previous.physical_frame == record.physical_frame
                and previous.generation_config_hash == record.generation_config_hash, "candidate identity changed")
        # Actual bytes were compared above. Preserve first-generation provenance
        # when orchestration code changes without changing this candidate function.
        return previous
    atomic_json(path.with_suffix(".json"), asdict(record), immutable=True)
    return record


def cache_candidates(config, scenes, kind):
    require(kind in {"base", "gt"}, "only Base/GT can be exported from frozen cache")
    b = read_json(config.root / "BASE_BINDING.json")
    def one(s):
        # Bounded independent I/O, ordered results. No sampler or GPU ordering is
        # changed, and no mutable LRU is shared between reader threads.
        cache = JointCache(config, capacity=0)
        return save_candidate(config, s, "expert_0" if kind == "base" else "audit_gt", b["checkpoint_hash"] if kind == "base" else s.target_id,
                              cache.read(s)["base_physical" if kind == "base" else "target_physical"].numpy())
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, scenes))


@torch.no_grad()
def expert_candidates(config, module, scenes, weight_hash, device, batch_size=64):
    cache = JointCache(config)
    module.eval()
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for at in range(0, len(scenes), batch_size):
            subset = scenes[at:at+batch_size]
            values = cache.batch(subset, device)
            pred = module(planner_bundle(config, values, [s.scene_id for s in subset]))
            def one(item):
                s, p = item
                return save_candidate(config, s, "expert_1", weight_hash, p)
            rows.extend(pool.map(one, zip(subset, pred.physical.cpu().numpy())))
    return rows


def score_candidates(config, scenes, candidates, job_name):
    require(len(scenes) == len(candidates), "candidate scene join length")
    proto = protocol(config)
    items = []
    for scene, c in zip(scenes, candidates):
        require(c.scene_id == scene.scene_id, "candidate ID join")
        seed = int(digest({"scene_id": scene.scene_id, "seed": 42, "repetition": 0})[:8], 16)
        request = {"candidate": asdict(c), "scene": asdict(scene), "protocol_hash": proto["protocol_hash"],
                   "navsim_root": proto["navsim_root"], "source_hash": proto["navsim_source_hash"], "config": proto["config"],
                   "seed": seed, "eval_seed": 42, "repetition": 0}
        response = config.root / "scores" / (digest([c.key, scene.metric_context_hash, proto["protocol_hash"], seed]) + ".json")
        items.append({"request": request, "response": str(response)})
    identity = digest(items)
    folder = config.root / "scoring_jobs" / (job_name + "_" + identity[:12])
    job = {"items": items, "navsim_root": config.navsim_root, "nuplan_root": proto["nuplan_root"],
           "attempts_per_invocation": config.scoring_attempts}
    path = folder / "JOB.json"
    atomic_json(path, job, immutable=True)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    def run(shard):
        with open(folder / f"worker_{shard}.log", "a") as log:
            return subprocess.run([config.scoring_python, "-m", "iqe.internal_baseline.score_worker", str(path),
                                   "--shard", str(shard), "--shards", str(config.scoring_workers)],
                                  env=env, cwd=repo, stdout=log, stderr=log).returncode
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.scoring_workers) as pool:
        codes = list(pool.map(run, range(config.scoring_workers)))
    atomic_json(folder/"STATUS.json", {"status": "COMPLETE" if not any(codes) else "PARTIAL", "codes": codes,
                "wall_seconds": time.perf_counter()-started, "candidate_count": len(candidates), "workers": config.scoring_workers})
    require(not any(codes), "official label job incomplete; retry same command, inspect " + str(folder))
    records = []
    for item, candidate in zip(items, candidates):
        receipt = read_json(Path(item["response"]).with_suffix(".COMPLETE.json"))
        require(receipt["checksum"] == file_hash(item["response"]) and receipt["request_hash"] == digest(item["request"]), "score receipt identity")
        score = strict_record(ScoreRecord, read_json(item["response"]))
        require(score.score_valid and score.candidate_key == candidate.key, "invalid official label/join")
        records.append(score)
    return records


def build_initial(config):
    if config.research_protocol == "two_candidate_v1":
        return build_threshold_hard_set(config)
    scenes = load_scenes(config.root / "scenes.json")
    # Do not inspect dev labels while defining H_train/H_val or choosing weights.
    selected = [s for s in scenes if s.split_role in {"incremental_fit", "stage_val"}]
    candidates = cache_candidates(config, selected, "base")
    scores = score_candidates(config, selected, candidates, "initial_base")
    base = {s.scene_id: asdict(r) for s, r in zip(selected, scores)}
    atomic_json(config.root / "BASE_FIT_VAL_SCORES.json", base, immutable=True)
    hard = {}
    for role in ("incremental_fit", "stage_val"):
        rows = [s for s in selected if s.split_role == role]
        rows.sort(key=lambda s: (base[s.scene_id]["total_score_01"], digest([config.selection_seed, s.scene_id])))
        hard[role] = [s.scene_id for s in rows[:int(np.ceil(.2*len(rows)))]]
    atomic_json(config.root / "HARD_SETS_BEFORE_EXPERT.json", hard, immutable=True)
    lookup = {s.scene_id: s for s in scenes}
    target_scenes = [lookup[i] for i in hard["incremental_fit"]]
    targets = score_candidates(config, target_scenes, cache_candidates(config, target_scenes, "gt"), "hard_gt_audit")
    audit, eligible = {}, []
    for s, target in zip(target_scenes, targets):
        reasons = []
        if target.total_score_01 < .8: reasons.append("GT_PDMS_BELOW_0_8")
        if target.total_score_01-base[s.scene_id]["total_score_01"] <= 1e-6: reasons.append("GT_NOT_BETTER")
        if any(not target.component_valid_masks[c] or target.named_components[c] != 1. for c in COMPONENTS[:2]): reasons.append("GT_NC_DAC_NOT_FULL")
        audit[s.scene_id] = {"eligible": not reasons, "reasons": reasons, "GT": asdict(target), "Base": base[s.scene_id]}
        if not reasons: eligible.append(s.scene_id)
    ordinary = sorted((s for s in scenes if s.split_role == "incremental_fit" and s.scene_id not in set(hard["incremental_fit"])),
                      key=lambda s: digest([config.selection_seed, "ordinary", s.scene_id]))[:2048]
    # All ordinary targets were validated against their real immutable target IDs
    # during cache generation. No teacher or low-score resampling is introduced.
    manifest = {"schema": "expert_90_10_v1", "hard": eligible, "ordinary": [s.scene_id for s in ordinary],
                "hard_val": hard["stage_val"], "gt_audit": audit, "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
                "sampling": {"hard": .9, "ordinary": .1}, "seed": config.natural_seed, "source": "resampling_only"}
    manifest["hash"] = digest(manifest)
    atomic_json(config.root / "EXPERT_MANIFEST.json", manifest, immutable=True)
    require(eligible, "NO_ELIGIBLE_HARD_GT: report target audit, do not fabricate specialization")
    return {"hard_candidates": len(target_scenes), "eligible_hard": len(eligible), "ordinary": len(ordinary), "hard_val": len(hard["stage_val"])}


def failure_and_target_audit(base, target=None, *, epsilon=1e-8):
    """Fixed protocol; missing components are unavailable, never implicit failure."""
    protected = COMPONENTS[:2]
    missing = [c for c in protected if not base["component_valid_masks"].get(c, False)]
    if not base["score_valid"] or missing:
        return {"failure": False, "eligible": False, "reasons": ["BASE_LABEL_OR_PROTECTED_COMPONENT_UNAVAILABLE"]}
    failed = base["total_score_01"] < .8 or any(base["named_components"][c] < 1-epsilon for c in protected)
    if target is None:
        return {"failure": failed, "eligible": False, "reasons": []}
    reasons = []
    if not target["score_valid"]: reasons.append("TARGET_SCORE_UNAVAILABLE")
    elif target["total_score_01"] < .8: reasons.append("GT_PDMS_BELOW_0_8")
    if target["score_valid"] and target["total_score_01"]-base["total_score_01"] < .05-1e-12:
        reasons.append("GT_GAIN_BELOW_0_05")
    if any(not target["component_valid_masks"].get(c, False) or target["named_components"].get(c, 0) < 1-epsilon for c in protected):
        reasons.append("GT_NC_DAC_NOT_FULL")
    if not failed: reasons.append("NOT_BASE_FAILURE")
    return {"failure": failed, "eligible": not reasons, "reasons": reasons}


def build_threshold_hard_set(config):
    from .partition import training_ids
    scenes = load_scenes(config.root/"scenes.json")
    expert_ids = training_ids(config, scenes, "expert_fit")
    selected = [s for s in scenes if s.scene_id in expert_ids or s.split_role == "stage_val"]
    labels = score_candidates(config, selected, cache_candidates(config, selected, "base"), "base_expert_fit_stage_val")
    base = {s.scene_id: asdict(r) for s, r in zip(selected, labels)}
    atomic_json(config.root/"BASE_FIT_VAL_SCORES.json", base, immutable=True)
    failures = {role: [s.scene_id for s in selected if (s.scene_id in expert_ids if role == "expert_fit" else s.split_role == "stage_val")
                       and failure_and_target_audit(base[s.scene_id])["failure"]] for role in ("expert_fit", "stage_val")}
    lookup = {s.scene_id: s for s in scenes}
    # Choose ordinary candidates before target scores. Safe/high Base rows have
    # priority; score failures/target gaps cannot silently enter the ordinary set.
    failed_set = set(failures["expert_fit"])
    ordinary_candidates = sorted((s for s in selected if s.scene_id in expert_ids and s.scene_id not in failed_set
                                   and not failure_and_target_audit(base[s.scene_id])["reasons"]),
                                 key=lambda s: digest([config.selection_seed, "ordinary", s.scene_id]))[:2048]
    target_ids = list(dict.fromkeys(failures["expert_fit"]+failures["stage_val"]+[s.scene_id for s in ordinary_candidates]))
    target_scenes = [lookup[i] for i in target_ids]
    targets = score_candidates(config, target_scenes, cache_candidates(config, target_scenes, "gt"), "gt_hard_val_ordinary_audit")
    target = {s.scene_id: asdict(r) for s, r in zip(target_scenes, targets)}
    audits, hard = {}, {}
    for role in failures:
        hard[role] = []
        for sid in failures[role]:
            a = failure_and_target_audit(base[sid], target[sid])
            a.update(Base=base[sid], GT=target[sid], input_consistency=lookup[sid].input_consistency_status,
                     target_hash=lookup[sid].target_id)
            audits[sid] = a
            if a["eligible"]: hard[role].append(sid)
    ordinary = []
    for s in ordinary_candidates:
        t = target[s.scene_id]
        legal = t["score_valid"] and all(t["component_valid_masks"].get(c, False) and t["named_components"][c] >= 1-1e-8 for c in COMPONENTS[:2])
        audits[s.scene_id] = {"ordinary_eligible": legal, "GT": t, "Base": base[s.scene_id],
                            "reasons": [] if legal else ["ORDINARY_GT_PROTECTED_SAFETY_UNAVAILABLE_OR_FAILED"]}
        if legal: ordinary.append(s.scene_id)
    fixed = {"F_train": failures["expert_fit"], "H_train": hard["expert_fit"], "F_val": failures["stage_val"],
             "H_val": hard["stage_val"], "rules": {"high_quality": .8, "target_gain": .05, "safety_epsilon": 1e-8},
             "defined_before_expert_training": True, "expert_fit_only": True}
    atomic_json(config.root/"HARD_SETS_BEFORE_EXPERT.json", fixed, immutable=True)
    manifest = {"schema": "two_candidate_expert_80_20_v1", "hard": hard["expert_fit"], "ordinary": ordinary,
                "hard_val": hard["stage_val"], "failure_val": failures["stage_val"], "gt_audit": audits,
                "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
                "partition_hash": file_hash(config.root/"DATA_PARTITION.json"),
                "sampling": {"hard": .8, "ordinary": .2}, "seed": config.natural_seed, "source": "resampling_only",
                "ordinary_rule": "fixed hash max2048 non-failure Base with valid target and full GT NC/DAC; no D_S"}
    manifest["hash"] = digest(manifest)
    atomic_json(config.root/"EXPERT_MANIFEST.json", manifest, immutable=True)
    atomic_json(config.root/"EXPERT_TRAIN_MANIFEST.json", manifest, immutable=True)
    atomic_json(config.root/"TARGET_AUDIT_QUEUE.json", {i:a for i,a in audits.items() if a["reasons"]}, immutable=True)
    require(manifest["hard"], "NO_ELIGIBLE_HARD_SET")
    require(ordinary, "NO_ELIGIBLE_ORDINARY_SET: preserve fixed 80/20 recipe")
    return {"failure_train": len(failures["expert_fit"]), "eligible_hard": len(hard["expert_fit"]),
            "ordinary": len(ordinary), "failure_val": len(failures["stage_val"]), "hard_val": len(hard["stage_val"])}
