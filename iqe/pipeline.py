"""Real artifact pipeline used by every CLI stage, including both incremental rounds."""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from .contracts import FeatureBundle, FeatureRecord, CandidateRecord, ScoreRecord, TrajectoryContract, require
from .io import atomic_json, atomic_torch, read_json, digest, file_hash, complete, reusable, BlockedError
from .config import resolve
from .data.manifests import load_scenes, save_scenes, freeze_targets
from .data.sources import load_observation, load_target, import_current, validate_incoming
from .data.splits import split_training_logs, validate_isolation
from .data.feature_cache import FeatureCache
from .data.candidate_bank import CandidateBank, save_trajectory
from .data.mining import MiningThresholds, diagnose_scene, safe_quality
from .data.sampler import sampling_plan, ConsumedSampler
from .data.support_retrieval import retrieve
from .registry import ExpertRegistry
from .s0_adapter import S0Adapter
from .model import IQEModel, CandidateBatch
from .scorer import TrajectoryScorer
from .router import SceneRouter
from .scoring.reference import ReferenceBackend, COMPONENTS
from .selector import SelectionRule, RouterRule
from .losses import scorer_terms, router_terms, scorer_denominators
from .training.trainer import train
from .evaluation.retention import frozen_snapshot, audit_frozen, module_hash
from .evaluation.oracle import oracle_report
from .evaluation.selection import selection_report
from .evaluation.calibration import calibrate, calibrate_router


def stack_features(features):
    require(features and len({f.contract_hash for f in features}) == 1, "feature batch contract mismatch")
    keys = set(features[0].conditions or {})
    require(all(set(f.conditions or {}) == keys for f in features), "frozen condition schema mismatch")
    # Preserve additional frozen Decoder inputs; zero padding only adds masked memory.
    length = max(f.scene.shape[1] for f in features)
    pad = torch.nn.functional.pad
    return FeatureBundle(torch.cat([pad(f.scene, (0, 0, 0, length - f.scene.shape[1])) for f in features]), torch.cat([f.ego for f in features]),
                         torch.cat([pad(f.valid_tokens, (0, length - f.scene.shape[1]), value=False) for f in features]), features[0].contract_hash,
                         tuple(s for f in features for s in f.scene_ids),
                         {k: torch.cat([f.conditions[k] for f in features]) for k in keys})


class Pipeline:
    def __init__(self, config, *, mode, max_samples, device=None):
        require(mode in {"smoke", "full", "dry-run", "profile"}, "explicit execution mode required")
        require(max_samples > 0, "explicit sample budget required")
        require(mode != "smoke" or max_samples <= 128, "smoke at most128 scenes per role")
        self.raw_config = config
        self.mode, self.max_samples = mode, max_samples
        self.root = Path(config["output_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = read_json(config["s0"]["contract"])
        self.config = resolve(config, self.contract, self.root / "resolved_config.json")
        self.device = device or config["execution"]["device"]
        if self.device.startswith("cuda"):
            import os
            # Set the rank device BEFORE constructing/loading any VLM or expert.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
        self.bank = CandidateBank(self.root / "bank")
        self.features = FeatureCache(self.root / "features")
        self.registry = ExpertRegistry(self.root / "registry.json", digest(self.contract))
        self.trajectory_contract = TrajectoryContract(**self.contract["trajectory"])
        self._model = None
        self._backend = None
        self._previous_bundles = {}

    def round_root(self, number):
        p = self.root / "rounds" / f"round_{number:03d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def scenes(self, roles=None):
        path = self.config["data"]["scene_manifest"] or self.root / "scenes.json"
        if (self.root / "OFFICIAL_SCENES.json").exists():
            pointer = read_json(self.root / "OFFICIAL_SCENES.json")
            require(file_hash(pointer["path"]) == pointer["checksum"], "official scene-context manifest changed")
            path = pointer["path"]
        scenes = load_scenes(path)
        admitted = self.root / "admitted_sources.json"
        if admitted.exists():
            scenes.extend(load_scenes(admitted))
        exposure = self.root / "base_exposure.json"
        if exposure.exists():
            seen = set(read_json(exposure)["fit_scene_ids"])
            scenes = [replace(s, s0_seen="yes" if s.scene_id in seen else "no") for s in scenes]
        validate_isolation(scenes)
        require(all(s.split_role != "final_test" for s in scenes), "final_test must use its isolated explicit manifest/entry")
        if roles is None:
            return scenes
        require("final_test" not in roles, "training/evaluation pipeline does not traverse final_test")
        result = []
        for role in roles:
            rows = sorted((s for s in scenes if s.split_role == role), key=lambda s: s.scene_id)
            require(rows, f"missing data role {role}")
            # Natural evaluation prefixes are frozen by ID, never selected using outcomes.
            new = [s for s in rows if s.added_round > 0]
            require(len(new) <= self.max_samples, "admitted observations exceed explicit per-role budget")
            result.extend([s for s in rows if s.added_round == 0][:self.max_samples - len(new)] + new)
        return result

    def backend(self):
        if self._backend is None:
            m = self.config["metric"]
            self._backend = ReferenceBackend(m["navsim_root"], m["python"], {k: m[k] for k in ("proposal_sampling", "scorer")},
                                             timeout=m["timeout_seconds"], eval_seed=m["eval_seed"])
            require(self._backend.source_hash == self.contract["metric"]["python_source_hash"], "official backend differs from locked source")
        return self._backend

    def require_reference_audit(self):
        if self.config["metric"]["require_candidate_set_invariance"]:
            evidence = read_json(self.root / "protocol_audit/SCORE_CONTEXT_EVIDENCE.json")
            require(evidence["passed"] and evidence["original_entry_equivalence"] and evidence["protocol_hash"] == self.backend().protocol_hash,
                    "reference equivalence/invariance evidence missing or stale; run verify-reference")

    def model(self, through_round=0):
        if self._model is None:
            self._model = IQEModel(S0Adapter.load_and_validate_s0(self.contract, self.device)).to(self.device)
        existing = self.registry.latest(self.registry.read())
        for eid, item in existing.items():
            if eid == "expert_0" or item["created_round"] > through_round:
                continue
            require(item["status"] != "training", "expert not frozen yet")
            if eid not in self._model.experts:
                # Historical IDs are never renumbered; rejected slots require explicitly retained weights.
                self._model.append_expert(eid, item["architecture"]["variant"], item["architecture"].get("bottleneck", 32))
                path = self.round_root(item["created_round"]) / "expert" / "best.pt"
                require(file_hash(path) == item["checkpoint_hash"], "expert checkpoint changed")
                self._model.experts[eid].load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)
        self._model.set_trainable_stage("inference"); self._model.eval()
        return self._model

    def validate_expert_dependencies(self, experts):
        latest = self.registry.latest(self.registry.read())
        for eid, expected in experts.items():
            require(eid in latest and latest[eid]["checkpoint_hash"] == expected, "stale candidate pool expert registry/hash")
            path = self.contract["query_checkpoint"] if eid == "expert_0" else self.round_root(latest[eid]["created_round"]) / "expert/best.pt"
            require(file_hash(path) == expected, "candidate dependency checkpoint changed after generation")

    def feature_record(self, scene):
        c = self.contract
        return FeatureRecord(1, scene.scene_id, scene.observation_hash, scene.metric_context_hash, c["query_checkpoint_hash"],
            c["query_checkpoint_hash"], c["tokenizer_hash"], c["prompt_hash"], c["transform_hash"], c["normalizer_hash"],
            c["camera_time_hash"], "float32", "fp32_master_bf16_autocast", "no_augmentation", 0, "", {}, True, None, "")

    def cached(self, scene):
        return self.features.get(self.feature_record(scene))

    def build_splits(self):
        if self.mode == "dry-run":
            return {"status": "DRY_RUN", "writes_scientific_artifacts": False}
        if self.config["data"]["scene_manifest"]:
            scenes = self.scenes()
        else:
            d = self.config["data"]
            index = read_json(Path(d["train_root"]) / "index.json")
            ratios = self.config["splits"]["fallback_fit_val_cal_log_ratios"]
            chosen = {r: [] for r in ("incremental_fit", "stage_val", "selector_cal")}
            for row in sorted(index, key=lambda r: r["token"]):
                value = int(digest({"group": "group:" + row["log"], "seed": self.config["seed"]})[:16], 16) / 2**64
                role = "incremental_fit" if value < ratios[0] else "stage_val" if value < sum(ratios[:2]) else "selector_cal"
                if len(chosen[role]) < self.max_samples:
                    chosen[role].append(row["token"])
            require(all(chosen.values()), "allowed source logs cannot supply all fit/val/cal roles")
            train_scenes = import_current(d["train_root"], d["train_metric_metadata"], "incremental_fit", scene_ids=set(t for ids in chosen.values() for t in ids))
            dev_scenes = import_current(d["dev_root"], d["dev_metric_metadata"], "dev_report", limit=self.max_samples)
            scenes = split_training_logs(train_scenes + dev_scenes, self.config["splits"]["fallback_fit_val_cal_log_ratios"], self.config["seed"])
            save_scenes(self.root / "scenes.json", scenes)
        audit = validate_isolation(scenes)
        atomic_json(self.root / "split_audit.json", audit, immutable=True)
        return audit

    def freeze_base(self, checkpoint):
        require(self.mode != "dry-run", "dry-run cannot lock S0")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        require(state["kind"] == "iqe_query_base" and state["framework_contract_hash"] == self.contract["framework_contract_hash"], "not a Query framework-only base checkpoint")
        require(file_hash(checkpoint) == read_json(str(checkpoint) + ".COMPLETE.json")["checksum"], "incomplete Query base")
        require(not state["learned_driving_weights_loaded"], "learned DiT weights unexpectedly loaded")
        require(state.get("fit_scene_ids") and digest(state["fit_scene_ids"]) == state["dependencies"]["fit_manifest"], "base exposure records unavailable/inconsistent")
        state_mode = state.get("mode") or read_json(Path(checkpoint).parent / "result.json")["mode"]
        require(self.mode == state_mode, "cannot relabel a smoke/profile base checkpoint as formal training")
        from .query_base import bind_source
        bind_source(self.contract["source_root"], self.contract["source_commit"])
        from tools.ddpolicy_vehicle.training_state import epoch_batches
        fit = state["fit_scene_ids"]
        seen = set()
        seed = state.get("seed", int(self.contract["source_config"]["seed"]))
        for epoch in range(state["epoch"] + 1):
            batches = epoch_batches(len(fit), state["dependencies"]["config"]["global_batch_size"], seed, epoch)
            for batch in batches[:state["offset"]] if epoch == state["epoch"] else batches:
                seen.update(fit[i] for i in batch)
        atomic_json(self.root / "base_exposure.json", {"fit_scene_ids": sorted(seen), "allowed_fit_scene_ids":fit,
            "checkpoint_hash": file_hash(checkpoint), "seen_observations":len(seen)}, immutable=True)
        c = dict(self.contract) | {"binding_status": "LOCKED_QUERY_S0", "query_checkpoint": str(Path(checkpoint).resolve()),
                                  "query_checkpoint_hash": file_hash(checkpoint), "base_optimizer_steps": state["optimizer_step"],
                                  "base_mode": self.mode, "base_split_manifest_hash": state["dependencies"]["fit_manifest"]}
        # Contract evolution is a separate immutable lock artifact; framework contract stays immutable.
        atomic_json(self.root / "LOCKED_S0.json", c, immutable=True)
        self.contract = c
        self.config = resolve(self.raw_config, c, self.root / "resolved_config.json")
        self.registry = ExpertRegistry(self.root / "registry.json", digest(c))
        self.registry.append({"expert_id": "expert_0", "created_round": 0, "parent": "generic_framework",
            "architecture": {"variant": "independent", **c["query_architecture"]}, "checkpoint_hash": c["query_checkpoint_hash"],
            "train_manifest_hash": c["base_split_manifest_hash"], "shared_contract_hash": digest(c), "status": "served", "gates": {"base": True}})
        return {"status": "LOCKED_QUERY_S0", "checkpoint": str(checkpoint), "hash": c["query_checkpoint_hash"]}

    def use_locked_base(self):
        require((self.root / "LOCKED_S0.json").exists(), "Query base not locked: run train-base then freeze-base")
        self.contract = read_json(self.root / "LOCKED_S0.json")
        require(self.mode != "full" or self.contract.get("base_mode") == "full", "formal runs cannot use a smoke/profile-trained Query base")
        self.config = resolve(self.raw_config, self.contract, self.root / "resolved_config.json")
        self.registry = ExpertRegistry(self.root / "registry.json", digest(self.contract))

    def cache_features(self, roles, *, shard_index=0, num_shards=1):
        require(0 <= shard_index < num_shards, "invalid feature shard")
        model = self.model()
        scenes = self.scenes(roles)[shard_index::num_shards]
        self.cache_scene_list(model, scenes)
        return {"status": "COMPLETE", "scenes": len(scenes), "roles": roles, "cache": str(self.features.root),
                "shard_index":shard_index,"num_shards":num_shards}

    def cache_scene_list(self, model, scenes):
        from .io import lock
        for scene in scenes:
            record = self.feature_record(scene)
            with lock(self.features.root / (record.cache_key + ".ENCODE.lock")):
                try:
                    self.features.get(record)
                except FileNotFoundError:
                    with torch.no_grad():
                        bundle = model.encode_scene([load_observation(scene, self.contract)])
                    self.features.put(record, bundle)

    def export_candidates(self, number, roles):
        model = self.model(number)
        pool = self.registry.candidate_pool()
        if self.mode == "smoke":
            pool = {eid: e["checkpoint_hash"] for eid, e in self.registry.latest(self.registry.read()).items()
                    if e["status"] != "training" and e["created_round"] <= number}
        pool = {eid: h for eid, h in pool.items() if int(eid.split('_')[-1]) <= number}
        dest = self.round_root(number) / "candidates.json"
        if dest.exists():
            # The recorded evaluation pool remains immutable after gate state transitions.
            pool = read_json(dest)["experts"]
            latest = self.registry.latest(self.registry.read())
            require(all(latest[e]["checkpoint_hash"] == h for e, h in pool.items()), "recorded candidate pool checkpoint changed")
        scenes = self.scenes(roles)
        lookup_path = self.root / "candidate_index.json"
        lookup = read_json(lookup_path) if lookup_path.exists() else {}
        rows = {}
        for scene in scenes:
            rows[scene.scene_id] = {}
            f = self.cached(scene).to(self.device)
            for eid, checkpoint_hash in pool.items():
                key = digest({"scene": scene.scene_id, "expert": eid, "checkpoint": checkpoint_hash, "feature": f.contract_hash})
                pointer = self.root / "candidate_lookup" / key[:2] / (key + ".json")
                if pointer.exists():
                    lookup[key] = read_json(pointer)["candidate_key"]
                if key not in lookup:
                    with torch.no_grad():
                        output = model.adapter.expert_forward(model.experts[eid], f)
                    trajectory = output.physical[0].detach().cpu().numpy()
                    ref = self.root / "bank" / "trajectories" / (key + ".npz")
                    th = save_trajectory(ref, trajectory)
                    record = CandidateRecord(1, scene.scene_id, eid, checkpoint_hash, f.contract_hash, th, str(ref),
                        self.trajectory_contract.raw_representation, "ego_relative_rear_axle", self.trajectory_contract.horizon,
                        self.trajectory_contract.dt, bool(np.isfinite(trajectory).all()), self.contract["iqe_code_hash"],
                        self.config["config_hash"], datetime.now(timezone.utc).isoformat())
                    self.bank.put_candidate(record)
                    lookup[key] = record.key
                    atomic_json(pointer,{"candidate_key":record.key},immutable=True)
                self.bank.candidate(lookup[key])
                rows[scene.scene_id][eid] = {"candidate_key": lookup[key]}
        result = self.bank.pool(dest, rows, pool, self.backend().protocol_hash)
        return {"status": "COMPLETE", "pool_hash": result["pool_hash"], "scenes": len(rows), "experts": list(pool)}

    def score_candidates(self, number):
        from concurrent.futures import ThreadPoolExecutor
        from itertools import islice
        import time
        start = time.perf_counter()
        folder = self.round_root(number)
        manifest = read_json(folder / "candidates.json")
        self.validate_expert_dependencies(manifest["experts"])
        if (folder / "scored.json").exists():
            done = read_json(folder / "scored.json")
            require(done["protocol_hash"] == self.backend().protocol_hash and done["experts"] == manifest["experts"]
                    and set(done["scenes"]) == set(manifest["scenes"]), "completed score stage dependencies changed")
            self.bank.validate_pool(done,self.scenes(),manifest["experts"])
            cost = folder / "scoring_cost.json"
            if not cost.exists():
                atomic_json(cost, {"status":"COMPLETE", "pool_hash":done["pool_hash"],
                    "real_labels":sum(len(v) for v in done["scenes"].values()),
                    "wall_seconds":None, "cost_status":"UNAVAILABLE_INTERRUPTED_AFTER_BANK_COMMIT"})
            return read_json(cost) | {"reused":True}
        scenes = {s.scene_id: s for s in self.scenes()}
        rows = deepcopy(manifest["scenes"])
        legacy = read_json(self.root / "score_index.json") if (self.root / "score_index.json").exists() else {}
        pending, cached, errors = [], 0, []
        def register(sid, eid, repetition, key, candidate, score):
            nonlocal cached
            score_key = self.bank.put_score(score)
            pointer = self.root / "score_lookup" / key[:2] / (key + ".json")
            atomic_json(pointer, {"score_key":score_key})
            keys = rows[sid][eid]
            keys.setdefault("repetition_score_keys", [None] * self.config["metric"]["repetitions"])[repetition] = score_key
            if repetition == 0:
                keys["score_key"] = score_key
            if not score.score_valid:
                errors.append({"scene_id":sid,"expert_id":eid,"error":score.error_reason,"candidate_finite":candidate.finite_valid})
        for sid, experts in rows.items():
            for eid, keys in experts.items():
                candidate = self.bank.candidate(keys["candidate_key"])
                for repetition in range(self.config["metric"]["repetitions"]):
                    key = digest({"candidate":candidate.key,"context":scenes[sid].metric_context_hash,
                        "protocol":self.backend().protocol_hash,"seed":self.config["metric"]["eval_seed"],"repetition":repetition})
                    pointer = self.root / "score_lookup" / key[:2] / (key + ".json")
                    score_key = read_json(pointer)["score_key"] if pointer.exists() else legacy.get(key)
                    score = self.bank.score(score_key,candidate,self.backend().protocol_hash,scenes[sid].metric_context_hash) if score_key else None
                    if score is not None and (score.score_valid or not candidate.finite_valid):
                        register(sid,eid,repetition,key,candidate,score);cached += 1
                    else:
                        pending.append((sid,eid,repetition,key,candidate))
        def evaluate(job):
            sid,eid,repetition,key,candidate = job
            return job,self.backend().score(candidate,scenes[sid],repetition=repetition)
        workers = self.config["execution"]["cpu_score_workers"]
        # Independent fresh official processes; bound queued work as well as workers.
        iterator = iter(pending)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while batch := list(islice(iterator,workers)):
                for job,score in pool.map(evaluate,batch):
                    register(*job,score)
        atomic_json(folder / "score_errors.json",errors)
        if any(e["candidate_finite"] for e in errors):
            atomic_json(folder / "scoring_progress.json",{"schema_version":1,"scenes":rows,"experts":manifest["experts"],
                "protocol_hash":manifest["protocol_hash"],"status":"PARTIAL","errors":errors})
            raise BlockedError("official scoring failed finite-candidate attempts; valid per-candidate results retained for resume")
        require(not any(e["expert_id"] == "expert_0" for e in errors),"Base invalid: explicit failure, no fabricated fallback")
        result = self.bank.pool(folder / "scored.json",rows,manifest["experts"],manifest["protocol_hash"])
        report = {"status":"COMPLETE","pool_hash":result["pool_hash"],"real_labels":sum(len(v) for v in rows.values()),
            "excluded_invalid_candidates":len(errors),"reused_labels":cached,"fresh_scoring_calls":len(pending),
            "CPU_worker_limit":workers,"wall_seconds":time.perf_counter()-start}
        atomic_json(folder / "scoring_cost.json",report)
        return report

    def target_score(self, scene):
        try:
            raw = load_target(scene, self.trajectory_contract)
        except (ValueError, OSError, KeyError) as error:
            invalid = ScoreRecord(1, digest({"scene": scene.scene_id, "target": scene.target_id}), self.backend().protocol_hash,
                scene.metric_context_hash, "unavailable", None, {c:None for c in COMPONENTS}, {c:False for c in COMPONENTS},
                "target_audit", "reference-v1", self.config["metric"]["eval_seed"], 0, False, str(error))
            return invalid, {"finite": False, "coordinates_time_yaw_legal": False, "input_context_consistent": scene.input_consistency_status == "verified"}
        physical = self.trajectory_contract.physical(raw)
        key = digest({"scene": scene.scene_id, "target": scene.target_id, "normalizer": self.contract["normalizer_hash"]})
        ref = self.root / "bank" / "targets" / (key + ".npz")
        trajectory_hash = save_trajectory(ref, physical.numpy())
        candidate = CandidateRecord(1, scene.scene_id, "ground_truth", scene.target_id, digest(self.contract), trajectory_hash, str(ref),
            self.trajectory_contract.raw_representation, "ego_relative_rear_axle", self.trajectory_contract.horizon,
            self.trajectory_contract.dt, bool(torch.isfinite(physical).all()), self.contract["iqe_code_hash"], self.config["config_hash"], "target_audit")
        self.bank.put_candidate(candidate)
        cache = self.root / "target_scores" / (digest({"candidate": candidate.key, "context": scene.metric_context_hash, "protocol": self.backend().protocol_hash}) + ".json")
        score = None
        if cache.exists():
            from .contracts import strict_record
            score = strict_record(ScoreRecord, read_json(cache))
        if score is None or not score.score_valid:
            score = self.backend().score(candidate, scene)
            self.bank.put_score(score)
            atomic_json(cache, asdict(score))
        legal = bool(torch.isfinite(raw).all()) and (self.trajectory_contract.raw_dim != 4 or bool((raw[..., 2:].norm(dim=-1) > 1e-8).all()))
        return score, {"finite": bool(torch.isfinite(physical).all()), "coordinates_time_yaw_legal": legal,
                       "input_context_consistent": scene.input_consistency_status == "verified"}

    def build_round(self, number, steps=None):
        self.require_reference_audit()
        require(number > 0, "incremental round starts at1")
        folder = self.round_root(number)
        old_folder = self.round_root(number - 1)
        old = read_json(old_folder / "scored.json")
        completed = folder / "expert_manifest.json"
        if completed.exists():
            previous = read_json(completed)
            require(previous["old_pool_hash"] == old["pool_hash"] and previous["config_hash"] == self.config["config_hash"], "completed round dependencies changed")
            require(len(previous["sampling_plan"]["entries"]) == (steps or self.config["expert_train"]["max_optimizer_steps"]) * self.config["expert_train"]["global_batch_size"], "resume round optimizer budget changed")
            return {"status": "REUSED", "manifest_hash": previous["manifest_hash"], "probe": previous["probe"], "sampling": previous["sampling_plan"]["metadata"]}
        fit = self.scenes(["incremental_fit"])
        # Diagnose decisions made by the actual previous serving bundle. The
        # last evaluated selector may have failed release and never been served.
        old_ids = sorted(old["experts"], key=lambda e:int(e.split("_")[-1]))
        previous_winners = self.previous_indices(number, "incremental_fit", {"scenes":fit,"expert_ids":old_ids})
        selected_by_scene = {s.scene_id:old_ids[int(i)] for s,i in zip(fit,previous_winners)}
        thresholds = MiningThresholds(**{k: self.config["mining"][k] for k in ("high_quality", "severe_low", "target_gain_margin", "zero_epsilon")})
        diagnoses, audited, hard = [], [], []
        for scene in fit:
            require(scene.scene_id in old["scenes"], "old whole-pool labels missing")
            accepted = self.registry.candidate_pool() if self.mode != "smoke" else old["experts"]
            scores = {eid: self.bank.score(keys["score_key"], self.bank.candidate(keys["candidate_key"]), old["protocol_hash"], scene.metric_context_hash)
                      for eid, keys in old["scenes"][scene.scene_id].items() if eid in accepted}
            require(scores, "no accepted old candidates for failure mining")
            if self.config["mining"]["variant"] == "fixed_s0":
                scores = {"expert_0":scores["expert_0"]}
            target, audit = self.target_score(scene)
            selected = selected_by_scene[scene.scene_id]
            report = diagnose_scene(scene, scores, target, selected, audit, thresholds)
            diagnoses.append(report)
            if safe_quality(target, thresholds) and all(audit.values()):
                audited.append(scene)
            if report["eligible_for_expert_training"]:
                hard.append(scene)
        atomic_json(folder / "gap_diagnoses.json", diagnoses, immutable=True)
        atomic_json(folder / "target_audit_queue.json", [d for d in diagnoses if d.get("target_audit_reasons")], immutable=True)
        new = []
        quarantine = []
        if self.config["data"]["new_scene_manifest"]:
            incoming = load_scenes(self.config["data"]["new_scene_manifest"])
            existing = self.scenes()
            by_group = {s.source_group_id: s.split_role for s in existing}
            admitted_ids = {s.scene_id for s in existing}
            incoming = [s for s in incoming if s.scene_id not in admitted_ids and s.added_round == number]
            for s in incoming:
                require(s.split_role == "incremental_fit", "new training source must inherit an allowed fit role")
                if s.parent_group_id:
                    require(by_group.get(s.parent_group_id) == s.split_role, "unknown/held-out parent group for new source")
            validate_isolation(existing + incoming)
            chosen, quarantine = freeze_targets(incoming, external_targets=self.config["mining"]["external_targets_enabled"])
            for s in chosen:
                try:
                    validate_incoming(s, self.contract | {"external_targets_enabled": self.config["mining"]["external_targets_enabled"]}, self.trajectory_contract)
                except (ValueError, OSError) as error:
                    quarantine.append({"scene_id": s.scene_id, "reason": str(error)})
                    continue
                target, audit = self.target_score(s)
                if safe_quality(target, thresholds) and all(audit.values()):
                    self.cache_scene_list(self.model(number - 1), [s])
                    old_scores = {}
                    f = self.cached(s).to(self.device)
                    for eid, checkpoint in old["experts"].items():
                        if self.mode != "smoke" and eid not in self.registry.candidate_pool():
                            continue
                        with torch.no_grad():
                            out = self._model.adapter.expert_forward(self._model.experts[eid], f)
                        ref = folder / "new_source_old_candidates" / (digest({"scene": s.scene_id, "expert": checkpoint}) + ".npz")
                        th = save_trajectory(ref, out.physical[0].cpu().numpy())
                        candidate = CandidateRecord(1, s.scene_id, eid, checkpoint, f.contract_hash, th, str(ref), self.trajectory_contract.raw_representation,
                            "ego_relative_rear_axle", self.trajectory_contract.horizon, self.trajectory_contract.dt, bool(torch.isfinite(out.physical).all()),
                            self.contract["iqe_code_hash"], self.config["config_hash"], "new_source_old_pool_audit")
                        self.bank.put_candidate(candidate)
                        old_scores[eid] = self.backend().score(candidate, s)
                        self.bank.put_score(old_scores[eid])
                    diagnosis = diagnose_scene(s, old_scores, target, "expert_0", audit, thresholds)
                    if diagnosis["eligible_for_expert_training"]:
                        new.append(s)
                    else:
                        quarantine.append({"scene_id": s.scene_id, "reason": "new_source_not_an_eligible_remaining_gap", "diagnosis": diagnosis})
                else:
                    quarantine.append({"scene_id": s.scene_id, "reason": "new_target_failed_official_audit"})
            if new:
                previous = load_scenes(self.root / "admitted_sources.json") if (self.root / "admitted_sources.json").exists() else []
                save_scenes(self.root / "admitted_sources.json", previous + new, immutable=False)
                save_scenes(folder / "admitted_sources_version.json", previous + new)
        atomic_json(folder / "source_quarantine.json", quarantine, immutable=True)
        probe = False
        if not hard and not new:
            if self.mode == "smoke" and self.config["execution"]["smoke_probe_when_no_gap"] and audited:
                hard, probe = audited, True
            else:
                atomic_json(folder / "NO_ELIGIBLE_COVERAGE_GAP.json", {"status": "NO_ELIGIBLE_COVERAGE_GAP", "diagnoses": len(diagnoses)}, immutable=True)
                raise BlockedError("NO_ELIGIBLE_COVERAGE_GAP: no authorized scientific expert fit; Scorer/evaluation can continue independently")
        hard, _ = freeze_targets(hard, external_targets=self.config["mining"]["external_targets_enabled"])
        support = []; retrieval_reports = []
        if hard and audited:
            vectors = [torch.cat((self.cached(s).scene.mean(1)[0], self.cached(s).ego.flatten())).numpy() for s in audited]
            for s in hard:
                vector = torch.cat((self.cached(s).scene.mean(1)[0], self.cached(s).ego.flatten())).numpy()
                found, report = retrieve(s, vector, audited, vectors, self.config["expert_data"]["support_limit"], digest(self.contract))
                support.extend(found); retrieval_reports.append(report)
        support = list({s.scene_id: s for s in support}.values())
        anchor_cap = self.config["expert_data"]["global_anchor_group_cap"]
        groups = sorted({s.source_group_id for s in audited}, key=lambda g: digest({"group": g, "round": number, "seed": self.config["seed"]}))[:anchor_cap]
        anchors = [s for s in audited if s.source_group_id in groups]
        buckets = {"hard_original": hard, "new_labeled": new, "local_support": support, "global_anchor": anchors}
        chosen_targets, _ = freeze_targets([s for rows in buckets.values() for s in rows],
            external_targets=self.config["mining"]["external_targets_enabled"])
        fixed_targets = {s.observation_hash: s.target_id for s in chosen_targets}
        buckets = {key: [s for s in rows if fixed_targets.get(s.observation_hash) == s.target_id] for key, rows in buckets.items()}
        cfg = self.config["expert_train"]
        updates = steps or cfg["max_optimizer_steps"]
        plan = sampling_plan(buckets, self.config["expert_data"]["mixture"], updates * cfg["global_batch_size"],
            self.config["seed"] + number, self.config["expert_data"]["max_source_group_draw_fraction"],
            self.config["expert_data"]["explicit_anchor_zero"])
        payload = {"round": number, "old_pool_hash": old["pool_hash"], "config_hash": self.config["config_hash"], "buckets": {k: [s.scene_id for s in rows] for k, rows in buckets.items()},
                   "records": [asdict(s) for s in {s.scene_id: s for rows in buckets.values() for s in rows}.values()],
                   "sampling_plan": plan, "probe": probe, "science": "UNTESTED" if probe or self.mode == "smoke" else "PENDING",
                   "target_selection_rule": "audited GT only; conflicting target IDs rejected", "retrieval": retrieval_reports}
        payload["manifest_hash"] = digest(payload)
        atomic_json(folder / "expert_manifest.json", payload, immutable=True)
        return {"status": "COMPLETE", "manifest_hash": payload["manifest_hash"], "probe": probe, "sampling": plan["metadata"]}

    def train_expert(self, number, *, steps=None, resume=None, stop_after=None, non_exact_finetune=False):
        folder = self.round_root(number)
        manifest = read_json(folder / "expert_manifest.json")
        require(manifest["manifest_hash"] == digest({k: v for k, v in manifest.items() if k != "manifest_hash"}), "expert manifest hash")
        result_path = folder / "expert" / "result.json"
        latest = self.registry.latest(self.registry.read()).get(f"expert_{number}")
        if result_path.exists() and read_json(result_path)["status"] == "COMPLETE" and latest and latest["status"] != "training":
            require(latest["train_manifest_hash"] == manifest["manifest_hash"] and latest["checkpoint_hash"] == file_hash(folder / "expert/best.pt"), "completed expert dependencies changed")
            return read_json(result_path) | {"reused": True}
        model = self.model(number - 1)
        eid = f"expert_{number}"
        if eid not in model.experts:
            model.append_expert(eid, self.config["model"]["variant"], self.config["model"]["adapter_bottleneck"])
        fixed_scenes = self.scenes(["stage_val"])[:32]
        f = stack_features([self.cached(s) for s in fixed_scenes]).to(self.device)
        reference = {old: {"raw": out.raw.detach().cpu(), "physical": out.physical.detach().cpu()}
                     for old in model.experts if old != eid for out in [model.adapter.expert_forward(model.experts[old], f)]}
        before = frozen_snapshot(model, eid)
        atomic_json(folder / "frozen_before.json", before, immutable=True)
        atomic_torch(folder / "frozen_reference.pt", {"features": asdict(f.to("cpu")), "outputs": reference})
        event = {"expert_id": eid, "created_round": number, "parent": "expert_0",
                 "architecture": {"variant": self.config["model"]["variant"], "bottleneck": self.config["model"]["adapter_bottleneck"]},
                 "checkpoint_hash": "training", "train_manifest_hash": manifest["manifest_hash"],
                 "shared_contract_hash": digest(self.contract), "status": "training", "gates": {"probe": manifest["probe"]}}
        self.registry.append(event)
        model.set_trainable_stage("expert_train", eid); model.train()
        from .contracts import SceneRecord, strict_record
        lookup = {r["scene_id"]: strict_record(SceneRecord, r) for r in manifest["records"]}
        def fetch(entries, device):
            rows = [lookup[e["scene_id"]] for e in entries]
            return {"features": stack_features([self.cached(s) for s in rows]).to(device),
                    "target": torch.stack([load_target(s, self.trajectory_contract) for s in rows]).to(device),
                    "masks": torch.tensor([not e.get("_padding", False) for e in entries], device=device)[:, None].expand(-1, self.trajectory_contract.horizon)}
        def loss_fn(expert, batch):
            return model.adapter.compute_expert_il_loss(expert(batch["features"]), batch["target"], batch["masks"])
        cfg = deepcopy(self.config["expert_train"])
        cfg["precision"] = self.config["model"]["precision"]
        if steps:
            cfg["max_optimizer_steps"] = steps; cfg["warmup_steps"] = min(cfg["warmup_steps"], steps - 1)
        require(self.mode != "smoke" or cfg["max_optimizer_steps"] <= 32, "expert smoke optimizer-step budget <=32")
        require(len(manifest["sampling_plan"]["entries"]) == cfg["max_optimizer_steps"] * cfg["global_batch_size"], "plan/update budget mismatch")
        dependencies = {"S0": self.contract["query_checkpoint_hash"], "old_experts": before, "manifest": manifest["manifest_hash"]}
        validation = self.expert_validation(number)
        def audit_stage(optimizer):
            model.train()
            model.audit_optimizer(optimizer)
        train(model.experts[eid], loss_fn, fetch, ConsumedSampler(manifest["sampling_plan"]), cfg, dependencies,
              folder / "expert", device=self.device, resume=resume, non_exact_finetune=non_exact_finetune, stop_after=stop_after,
              validate=validation, seed=self.config["seed"] + number, audit=audit_stage,
              denominator_function=lambda data: {"il": data["masks"].any(-1).sum().float()})
        model.set_trainable_stage("audit")
        require(frozen_snapshot(model, eid) == before, "old/shared hashes changed during expert training")
        result = read_json(folder / "expert" / "result.json")
        if result["status"] != "COMPLETE":
            return result
        model.experts[eid].load_state_dict(torch.load(folder / "expert" / "best.pt", map_location=self.device, weights_only=True), strict=True)
        event.update(checkpoint_hash=file_hash(folder / "expert" / "best.pt"), status="frozen")
        self.registry.append(event)
        return read_json(folder / "expert" / "result.json")

    def expert_validation(self, number):
        old = read_json(self.round_root(number - 1) / "scored.json")
        rows = self.scenes(["stage_val"])
        adapter = self._model.adapter
        require(all(s.scene_id in old["scenes"] for s in rows), "fixed stage_val old pool labels missing")
        def validate(expert, step):
            from concurrent.futures import ThreadPoolExecutor
            gains, successes, losses = [], 0, []
            expert_hash = module_hash(expert)
            jobs = []
            thresholds = MiningThresholds(**{k: self.config["mining"][k] for k in ("high_quality","severe_low","target_gain_margin","zero_epsilon")})
            for scene in rows:
                f = self.cached(scene).to(self.device)
                out = expert(f)
                old_scores = [self.bank.score(keys["score_key"], self.bank.candidate(keys["candidate_key"]), old["protocol_hash"], scene.metric_context_hash)
                              for keys in old["scenes"][scene.scene_id].values()]
                require(all(s.score_valid for s in old_scores), "invalid stage_val labels")
                best = max(s.total_score_01 for s in old_scores)
                key = digest({"round": number, "step": step, "scene": scene.scene_id, "expert_state": expert_hash})
                ref = self.round_root(number) / "validation_candidates" / (key + ".npz")
                th = save_trajectory(ref, out.physical[0].cpu().numpy())
                c = CandidateRecord(1, scene.scene_id, f"expert_{number}", key, f.contract_hash, th, str(ref), self.trajectory_contract.raw_representation,
                    "ego_relative_rear_axle", self.trajectory_contract.horizon, self.trajectory_contract.dt, True, self.contract["iqe_code_hash"], self.config["config_hash"], "stage_val")
                jobs.append((c,scene,best,any(safe_quality(s,thresholds) for s in old_scores)))
                terms = adapter.compute_expert_il_loss(out, load_target(scene, self.trajectory_contract)[None].to(self.device))
                losses.append(float(terms["il"].numerator / terms["il"].denominator.clamp_min(1)))
            backend = self.backend()
            def score_job(job):
                candidate, scene, best, solved = job
                return backend.score(candidate, scene), best, solved
            with ThreadPoolExecutor(max_workers=self.config["execution"]["cpu_score_workers"]) as workers:
                for score,best,solved in workers.map(score_job,jobs):
                    require(score.score_valid, "official checkpoint validation failed")
                    gains.append(max(best,score.total_score_01)-best)
                    successes += safe_quality(score,thresholds) and not solved
            return {"role": "stage_val", "fixed_scene_ids": [s.scene_id for s in rows], "oracle_gain_01": float(np.mean(gains)),
                    "new_successes": int(successes), "IL_loss": float(np.mean(losses)),
                    "selection_key": [float(np.mean(gains)), int(successes), -float(np.mean(losses))]}
        return validate

    def frozen_audit(self, number):
        folder = self.round_root(number)
        model = self.model(number)
        before = read_json(folder / "frozen_before.json")
        saved = torch.load(folder / "frozen_reference.pt", map_location=self.device, weights_only=True)
        # Exclude the current new expert from the preserved old function set.
        model.set_trainable_stage("expert_train", f"expert_{number}"); model.eval()
        report = audit_frozen(model, before, saved["outputs"], FeatureBundle(**saved["features"]),
                              self.config["tolerances"]["fp32_cuda" if self.device.startswith("cuda") else "fp32_cpu"])
        old = read_json(self.round_root(number - 1) / "scored.json")
        scene_map = {s.scene_id:s for s in self.scenes(["stage_val"])}
        pdms_checks = []
        # Repeat the actual per-scene generation/scoring path; trajectory tolerance
        # alone does not imply equal simulator decisions near metric boundaries.
        for sid in saved["features"]["scene_ids"]:
            scene = scene_map[sid]
            for eid in saved["outputs"]:
                if eid not in old["scenes"][sid]:
                    continue
                keys = old["scenes"][sid][eid]
                prior = self.bank.candidate(keys["candidate_key"])
                prior_score = self.bank.score(keys["score_key"], prior, old["protocol_hash"], scene.metric_context_hash)
                with torch.no_grad():
                    output = model.adapter.expert_forward(model.experts[eid], self.cached(scene).to(self.device))
                ref = folder / "frozen_score_audit" / f"{sid}_{eid}.npz"
                h = save_trajectory(ref, output.physical[0].cpu().numpy())
                candidate = replace(prior, trajectory_ref=str(ref), trajectory_hash=h)
                score = self.backend().score(candidate, scene)
                require(score.score_valid and prior_score.score_valid, "old candidate real scoring audit failed")
                require(abs(score.total_score_01-prior_score.total_score_01) <= 1e-12 and score.named_components == prior_score.named_components,
                        "old candidate real PDMS/component drift")
                pdms_checks.append({"scene_id":sid,"expert_id":eid,"before":prior_score.total_score_01,"after":score.total_score_01,
                                    "trajectory_hash_before":prior.trajectory_hash,"trajectory_hash_after":h})
        report["official_score_retention"] = pdms_checks
        atomic_json(folder / "frozen_audit.json", report, immutable=True)
        model.set_trainable_stage("inference")
        return report

    def bank_arrays(self, number, role):
        self.require_reference_audit()
        manifest = read_json(self.round_root(number) / "scored.json")
        self.validate_expert_dependencies(manifest["experts"])
        scenes = self.scenes([role]); ids = tuple(sorted(manifest["experts"], key=lambda e:int(e[7:])))
        self.bank.validate_pool(manifest, self.scenes(), manifest["experts"])
        trajectories, scores, valid = [], [], []
        components, masks = {c: [] for c in COMPONENTS}, {c: [] for c in COMPONENTS}
        for scene in scenes:
            require(scene.scene_id in manifest["scenes"], "role absent from candidate bank")
            tau, score_row, valid_row, c_rows, m_rows = [], [], [], {c: [] for c in COMPONENTS}, {c: [] for c in COMPONENTS}
            for eid in ids:
                keys = manifest["scenes"][scene.scene_id][eid]
                candidate = self.bank.candidate(keys["candidate_key"])
                score = self.bank.score(keys["score_key"], candidate, manifest["protocol_hash"], scene.metric_context_hash)
                tau.append(np.load(candidate.trajectory_ref)["trajectory"])
                score_row.append(score.total_score_01 if score.score_valid else np.nan)
                valid_row.append(score.score_valid and candidate.finite_valid)
                for c in COMPONENTS:
                    c_rows[c].append(score.named_components[c] if score.component_valid_masks[c] else np.nan)
                    m_rows[c].append(score.component_valid_masks[c])
            trajectories.append(tau); scores.append(score_row); valid.append(valid_row)
            for c in COMPONENTS:
                components[c].append(c_rows[c]); masks[c].append(m_rows[c])
        return {"manifest": manifest, "scenes": scenes, "expert_ids": ids, "trajectories": torch.tensor(np.asarray(trajectories), dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32), "valid": torch.tensor(valid, dtype=torch.bool),
                "components": {c: torch.tensor(v, dtype=torch.float32) for c, v in components.items()},
                "component_valid": {c: torch.tensor(v, dtype=torch.bool) for c, v in masks.items()}}

    def evaluate_oracle(self, number, role):
        require(role == "stage_val", "expert scientific gate only uses fixed stage_val")
        d = self.bank_arrays(number, role)
        t = MiningThresholds(**{k: self.config["mining"][k] for k in ("high_quality", "severe_low", "target_gain_margin", "zero_epsilon")})
        report = oracle_report(d["scores"].numpy(), d["valid"].numpy(), {k: v.numpy() for k, v in d["components"].items()},
            {k: v.numpy() for k, v in d["component_valid"].items()}, d["expert_ids"], [s.source_group_id for s in d["scenes"]],
            f"expert_{number}", t, self.config["gates"], smoke=self.mode == "smoke")
        frozen = read_json(self.round_root(number) / "frozen_audit.json")
        report["frozen_audit_passed"] = frozen["passed"]
        report["oracle_gate"] = report["oracle_gate"] and frozen["passed"]
        from .evaluation.diversity import candidate_diagnostics
        report["candidate_diagnostics"] = candidate_diagnostics(d["trajectories"].numpy(),d["scores"].numpy(),d["valid"].numpy(),
            {k:v.numpy() for k,v in d["components"].items()},{k:v.numpy() for k,v in d["component_valid"].items()},
            d["expert_ids"],[s.source_log_id for s in d["scenes"]],t.high_quality)
        atomic_json(self.round_root(number) / "oracle_stage_val.json", report, immutable=True)
        event = deepcopy(self.registry.latest(self.registry.read())[f"expert_{number}"])
        event["status"] = "candidate_only" if report["oracle_gate"] or self.mode == "smoke" else "rejected"
        event["gates"] = {"oracle": report, "probe": event["gates"].get("probe", False)}
        self.registry.append(event)
        return report

    def selector_module(self, number, router=False, *, warm_start=False):
        width = self.contract["query_architecture"]["width"]
        ids = tuple(sorted(read_json(self.round_root(number) / "scored.json")["experts"], key=lambda e:int(e[7:])))
        if router:
            module = SceneRouter(width, width, ids, self.config["router_ablation"]["hidden_size"])
            if warm_start and number > 1:
                previous = read_json(self.round_root(number - 1) / "router" / "identity.json")
                old = SceneRouter(width, width, previous["expert_ids"], self.config["router_ablation"]["hidden_size"])
                old.load_state_dict(torch.load(self.round_root(number - 1) / "router" / "best.pt", map_location="cpu", weights_only=True), strict=True)
                module = old.expanded(ids)
        else:
            c = self.config["scorer"]
            module = TrajectoryScorer(width, width, COMPONENTS, c["hidden_size"], c["num_layers"], c["attention_heads"], c["dropout"])
            if warm_start and number > 1 and c["warm_start_previous_scorer"]:
                path = self.round_root(number - 1) / "scorer" / "best.pt"
                require(path.exists(), "previous Scorer required for warm start")
                module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return module.to(self.device)

    def previous_indices(self, number, role, arrays):
        snapshot = self.round_root(number) / "previous_serving.json"
        if not snapshot.exists():
            active = self.root / "ACTIVE.json"
            # Smoke compares to the preceding temporary bundle, never a formal active pointer.
            temporary = self.round_root(number - 1) / "bundle" / "BUNDLE.json"
            if self.mode == "smoke" and temporary.exists():
                prior = {"path": str(temporary.parent), "bundle_hash": read_json(temporary)["bundle_hash"]}
            else:
                prior = read_json(active) if active.exists() else {"policy": "locked_query_S0"}
            atomic_json(snapshot, prior, immutable=True)
        previous = read_json(snapshot)
        if previous.get("policy") == "locked_query_S0":
            return np.zeros(len(arrays["scenes"]), int)
        from .export import load_bundle
        key = previous["bundle_hash"]
        if key not in self._previous_bundles:
            adapter = self._model.adapter if self._model is not None else self.model(number).adapter
            self._previous_bundles[key] = load_bundle(previous["path"], self.device, shared_adapter=adapter)
        model, rule, metadata = self._previous_bundles[key]
        require(metadata["bundle_hash"] == previous["bundle_hash"] and digest(model.adapter.contract) == digest(self.contract), "previous serving/S0 version conflict")
        winners = []
        for s in arrays["scenes"]:
            f = self.cached(s).to(self.device)
            with torch.no_grad():
                if metadata["selector_type"] == "scene_router":
                    _, ids = model.forward_prerouted(None, rule, features=f)
                else:
                    _, ids, _ = model.predict(None, rule, features=f)
            require(ids[0] in arrays["expert_ids"], "current pool omits an expert served by previous bundle")
            winners.append(arrays["expert_ids"].index(ids[0]))
        return np.array(winners)

    def prediction_arrays(self, module, arrays, router=False):
        from .scorer import ScorePrediction
        # Evaluation is independent per scene/candidate, so chunking bounds GPU
        # activations without changing the selector function or candidate pool.
        module.eval()
        chunks, feature_chunks = [], []
        batch = self.config["scorer"]["micro_batch_size"]
        with torch.no_grad():
            for at in range(0, len(arrays["scenes"]), batch):
                f = stack_features([self.cached(s) for s in arrays["scenes"][at:at+batch]])
                feature_chunks.append(f)
                if router:
                    chunks.append(module(f.to(self.device)).cpu())
                else:
                    p = module(f.to(self.device), arrays["trajectories"][at:at+batch].to(self.device), arrays["valid"][at:at+batch].to(self.device))
                    chunks.append(ScorePrediction(p.values.cpu(), {c:v.cpu() for c,v in p.component_logits.items()}, p.valid.cpu()))
        features = stack_features(feature_chunks)
        candidates = CandidateBatch(arrays["trajectories"], torch.empty(0), arrays["expert_ids"], arrays["valid"], features)
        prediction = torch.cat(chunks) if router else ScorePrediction(torch.cat([p.values for p in chunks]),
            {c:torch.cat([p.component_logits[c] for p in chunks]) for c in chunks[0].component_logits}, torch.cat([p.valid for p in chunks]))
        return candidates, prediction

    def train_selector(self, number, *, router=False, steps=None, resume=None, stop_after=None, non_exact_finetune=False):
        arrays = self.bank_arrays(number, "incremental_fit")
        if not self.config["scorer"]["replay_old_candidates"]:
            keep = arrays["expert_ids"].index(f"expert_{number}")
            arrays["valid"][:, :keep] = False
            arrays["valid"][:, keep+1:] = False
        val = self.bank_arrays(number, "stage_val")
        module = self.selector_module(number, router, warm_start=not bool(resume))
        model = self.model(number)
        before = frozen_snapshot(model)
        if router:
            model.router = module; model.set_trainable_stage("router_train")
        else:
            model.scorer = module; model.set_trainable_stage("scorer_train")
        kind = "router" if router else "scorer"
        folder = self.round_root(number) / kind
        identity = {"kind": kind, "round": number, "pool_hash": arrays["manifest"]["pool_hash"], "expert_ids": list(arrays["expert_ids"]),
                    "training_role": "incremental_fit", "validation_role": "stage_val", "old_new_replay": self.config["scorer"]["replay_old_candidates"],
                    "warm_started": number > 1, "frozen_generator_hashes": before}
        atomic_json(folder / "identity.json", identity, immutable=True)
        cfg = deepcopy(self.config["scorer"])
        cfg["precision"] = self.config["model"]["precision"]
        if steps:
            cfg["max_optimizer_steps"] = steps; cfg["warmup_steps"] = min(cfg["warmup_steps"], steps - 1)
        require(self.mode != "smoke" or cfg["max_optimizer_steps"] <= 32, "selector smoke optimizer-step budget <=32")
        batch = cfg["global_scene_batch_size"]
        rng = np.random.default_rng(self.config["seed"] + number)
        plan_rows = []
        count = batch * cfg["max_optimizer_steps"]
        # Uniform scene permutations, all old/new candidates per selected scene.
        while len(plan_rows) < count:
            for i in rng.permutation(len(arrays["scenes"])):
                if len(plan_rows) == count:
                    break
                s = arrays["scenes"][i]
                plan_rows.append({"scene_id": s.scene_id, "source_group_id": s.source_group_id, "bucket": "scene_uniform", "index": int(i)})
        plan = {"entries": plan_rows, "plan_hash": digest(plan_rows)}
        atomic_json(folder / "sampling_plan.json", plan, immutable=True)
        def fetch(entries, device):
            ii = [e["index"] for e in entries]
            return {"features": stack_features([self.cached(arrays["scenes"][i]) for i in ii]).to(device),
                "trajectories": arrays["trajectories"][ii].to(device), "scores": arrays["scores"][ii].to(device),
                "valid": arrays["valid"][ii].to(device) & torch.tensor([not e.get("_padding", False) for e in entries], device=device)[:, None], "components": {c: v[ii].to(device) for c, v in arrays["components"].items()},
                "component_valid": {c: v[ii].to(device) for c, v in arrays["component_valid"].items()}}
        def loss_fn(network, data):
            if router:
                return router_terms(network(data["features"]), data["scores"], data["valid"], self.config["router_ablation"]["label_temperature"])
            return scorer_terms(network(data["features"], data["trajectories"], data["valid"]), data["scores"], data["valid"],
                data["components"], data["component_valid"], huber_delta=cfg["huber_delta"], temperature=cfg["rank_temperature"], tie_epsilon=cfg["rank_tie_epsilon"])
        def validate(network, step):
            candidates, predicted = self.prediction_arrays(network, val, router)
            rule = RouterRule() if router else SelectionRule(0, {c: 0. for c in self.config["selection"]["protected_metrics"]},
                                               {c: .02 for c in self.config["selection"]["protected_metrics"]})
            if router:
                indices = rule(predicted, val["expert_ids"]).cpu().numpy()
            else:
                _, winners, _ = rule(candidates, predicted)
                indices = np.array([val["expert_ids"].index(w) for w in winners])
            report = selection_report(val["scores"].numpy(), val["valid"].numpy(), indices, self.previous_indices(number, "stage_val", val),
                {c: v.numpy() for c, v in val["components"].items()}, {c: v.numpy() for c, v in val["component_valid"].items()},
                [s.source_group_id for s in val["scenes"]])
            violations = sum(report["vs_" + baseline][c]["new_violations"] for baseline in ("s0","previous") for c in self.config["selection"]["protected_metrics"])
            report.update(role="stage_val", fixed_scene_ids=[s.scene_id for s in val["scenes"]],
                          selection_key=[-violations, -report["selection_regret"], report["selected_mean_01"]])
            return report
        def audit_stage(optimizer):
            model.train()
            model.audit_optimizer(optimizer)
        train(module, loss_fn, fetch, ConsumedSampler(plan), cfg, identity, folder, device=self.device,
              weights=None if router else cfg["loss_weights"], resume=resume, non_exact_finetune=non_exact_finetune,
              stop_after=stop_after, validate=validate, seed=self.config["seed"] + number, audit=audit_stage,
              denominator_function=(lambda data: {"router": (data["valid"] & torch.isfinite(data["scores"])).any(-1).sum().float()})
                  if router else (lambda data: scorer_denominators(data, cfg["rank_tie_epsilon"])))
        require(frozen_snapshot(model) == before, "Scorer/Router training changed a candidate generator")
        result = read_json(folder / "result.json")
        result["frozen_audit_passed"] = True
        atomic_json(folder / "result.json", result)
        if not router:
            from .scorer import geometry_diagnostics
            module.load_state_dict(torch.load(folder / "best.pt", map_location=self.device, weights_only=True), strict=True)
            candidates, _ = self.prediction_arrays(module, val)
            limit = min(len(candidates.trajectories), cfg["micro_batch_size"])
            atomic_json(folder / "geometry_diagnostics.json", geometry_diagnostics(module,
                candidates.features.subset(list(range(limit))).to(self.device), candidates.trajectories[:limit].to(self.device)))
        return result

    def calibrated_selector(self, number, *, router=False):
        kind = "router" if router else "scorer"
        module = self.selector_module(number, router)
        path = self.round_root(number) / kind / "best.pt"
        module.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)
        module.eval()
        rule_path = self.round_root(number) / ("router_calibration.json" if router else "calibration.json")
        saved = read_json(rule_path)
        require(saved["selector_checkpoint_hash"] == file_hash(path), "calibration is stale for this selector")
        require(saved["pool_hash"] == read_json(self.round_root(number) / "scored.json")["pool_hash"], "calibration bank mismatch")
        return module, RouterRule(**saved["rule"]) if router else SelectionRule(**saved["rule"]), saved

    def calibrate(self, number, role, *, router=False):
        require(role == "selector_cal", "only selector_cal may calibrate")
        d = self.bank_arrays(number, role)
        module = self.selector_module(number, router)
        path = self.round_root(number) / ("router" if router else "scorer") / "best.pt"
        module.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)
        candidates, prediction = self.prediction_arrays(module, d, router)
        previous = self.previous_indices(number, role, d)
        c, m = {c: v.numpy() for c, v in d["components"].items()}, {c: v.numpy() for c, v in d["component_valid"].items()}
        if router:
            options = self.config["router_ablation"] | {"budgets": self.config["selection"]["budgets"]}
            rule, report = calibrate_router(prediction, d["expert_ids"], d["scores"].numpy(), c, m, previous, options, role=role, dependency_hash=d["manifest"]["pool_hash"])
        else:
            rule, report = calibrate(candidates, prediction, d["scores"].numpy(), c, m, previous, self.config["selection"],
                                     role=role, dependency_hash=d["manifest"]["pool_hash"])
        payload = {"rule": asdict(rule), "report": report, "selector_checkpoint_hash": file_hash(path), "pool_hash": d["manifest"]["pool_hash"]}
        atomic_json(self.round_root(number) / ("router_calibration.json" if router else "calibration.json"), payload, immutable=True)
        return payload

    def evaluate(self, number, role, *, router=False):
        require(role in {"incremental_fit", "stage_val", "dev_report", "selector_cal"}, "final_test has a separate locked-bundle entry")
        d = self.bank_arrays(number, role)
        module, rule, calibration = self.calibrated_selector(number, router=router)
        candidates, prediction = self.prediction_arrays(module, d, router)
        if router:
            indices = rule(prediction, d["expert_ids"]).cpu().numpy()
        else:
            _, winners, reasons = rule(candidates, prediction)
            indices = np.array([d["expert_ids"].index(w) for w in winners])
        report = selection_report(d["scores"].numpy(), d["valid"].numpy(), indices, self.previous_indices(number, role, d),
            {c: v.numpy() for c, v in d["components"].items()}, {c: v.numpy() for c, v in d["component_valid"].items()},
            [s.source_group_id for s in d["scenes"]])
        report.update(role=role, pool_hash=d["manifest"]["pool_hash"], selector_checkpoint_hash=calibration["selector_checkpoint_hash"],
                      smoke=self.mode == "smoke", science="UNTESTED" if self.mode == "smoke" else "PENDING_RELEASE_GATE",
                      frozen_output_drift=read_json(self.round_root(number) / "frozen_audit.json"))
        cases = []
        previous = self.previous_indices(number, role, d)
        for j, (scene, winner) in enumerate(zip(d["scenes"], indices)):
            keys = d["manifest"]["scenes"][scene.scene_id][d["expert_ids"][winner]]
            candidate = self.bank.candidate(keys["candidate_key"])
            cached_score = self.bank.score(keys["score_key"], candidate, d["manifest"]["protocol_hash"], scene.metric_context_hash)
            direct = self.backend().score(candidate, scene)
            require(direct.score_valid and asdict(direct) == asdict(cached_score), "selected official trajectory score differs from candidate bank")
            old_winner = int(previous[j])
            cases.append({"scene_id":scene.scene_id,"selected":d["expert_ids"][winner],"previous":d["expert_ids"][old_winner],
                "trajectory_ref":candidate.trajectory_ref,"trajectory_hash":candidate.trajectory_hash,
                "true_score_01":direct.total_score_01,"previous_score_01":float(d["scores"][j,old_winner]),
                "true_gain_vs_previous":direct.total_score_01-float(d["scores"][j,old_winner]),
                "components":direct.named_components,"direct_official_score_matches_bank":True})
        report["selection_cases"] = cases
        name = ("router_" if router else "") + "evaluation_" + role + ".json"
        atomic_json(self.round_root(number) / name, report, immutable=True)
        atomic_json(self.round_root(number) / (("router_" if router else "") + "selection_" + role + ".json"),
                    {s.scene_id: d["expert_ids"][i] for s, i in zip(d["scenes"], indices)}, immutable=True)
        return report

    def export_bundle(self, number, *, require_gates, activate=False, router=False):
        from .export import release_gate, export_bundle, activate_bundle
        folder = self.round_root(number)
        module, rule, calibrated = self.calibrated_selector(number, router=router)
        model = self.model(number)
        serving_ids = tuple(sorted(read_json(folder / "scored.json")["experts"], key=lambda e:int(e[7:])))
        require(all(e in model.experts for e in serving_ids), "serving pool expert missing")
        if router:
            model.router = module
        else:
            model.scorer = module
        evaluation = read_json(folder / (("router_" if router else "") + "evaluation_dev_report.json"))
        gate = release_gate(evaluation, read_json(folder / "oracle_stage_val.json"), read_json(folder / "frozen_audit.json"),
                            calibrated["report"], self.config["gates"]["minimum_logs_for_ci_gate"])
        atomic_json(folder / "release_gate.json", gate, immutable=True)
        if require_gates or self.mode == "full":
            require(gate["passed"], "CANDIDATE_GAIN_NOT_REALIZED: no formal bundle release; retain previous serving version")
        require(not activate or self.mode == "full", "smoke cannot activate a formal bundle")
        result = export_bundle(model, rule, self.registry, folder / "bundle", mode="smoke" if self.mode == "smoke" else "full",
                               gate_results=gate, config_hash=self.config["config_hash"], pool_hash=calibrated["pool_hash"],
                               selector_type="scene_router" if router else "scorer", expert_ids=serving_ids,
                               data_role_groups={g for s in self.scenes() for g in (s.source_log_id, s.source_group_id, s.parent_group_id) if g})
        if activate:
            activate_bundle(self.root, folder / "bundle")
        return {"status": "COMPLETE", "bundle_hash": result["bundle_hash"], "active_updated": activate, "path": str(folder / "bundle")}
