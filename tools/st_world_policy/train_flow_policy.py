"""Registered C0--C4: original DiT BC plus bounded official FPO feedback.

This experiment adapter reuses original dataset/sampler, strict S0 loading,
DeepSpeed ZeRO2 FP32 master/Adam state, and the common native checkpoint gate.
It never resumes or changes the parent S0. Run ``prepare --help`` or ``train
--help``. Only the lead dispatches real GPU runs after registering dependencies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import record_function

from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset, collate_training, decode_ego
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from starVLA.model.modules.flow_posttraining.fpo_objective import (
    fpo_surrogate, group_centered_advantages)
from starVLA.model.modules.flow_posttraining.policy_snapshot import (
    DiTFlowPolicy, FlowRandomStreams, FrozenPolicySnapshot, module_hash, tensor_hash)
from starVLA.model.modules.flow_posttraining.rollout_contract import (
    FeedbackBatch, FeedbackReuseGuard, prepare_fpo_batch, sample_rollout_group)
from starVLA.model.modules.st_world.consequence import (
    ConsequenceRequest, ConsequenceService, ConsequenceSpec, METRICS,
    array_sha256, fingerprint, protocol_identity)
from tools.ddpolicy_vehicle.optimizer_safety import (
    bounded_parameter_groups, capture_master_samples, master_update_evidence)
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate, validate_rank_batches
from tools.st_world_policy.base_binding import read_binding
from tools.st_world_policy.fp32_master_init import (
    SCHEMA as MASTER_INIT_SCHEMA, capture_fp32_master_initialization, restore_fp32_master_initialization)
from tools.st_world_policy.check_formal_approval import check_formal_approval, read_json
from tools.st_world_policy.flow_training_config import NO_TEACHER, REWARDS, target_contract, validate_config
from tools.st_world_policy.train_world import collective_call, verified_runtime
from tools.st_world_policy.training_state import checkpoint_receipt, resolve_latest, update_latest, terminal


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _verified_artifact(value):
    path = Path(value["path"])
    if not path.is_absolute() or set(value) != {"path", "sha256"} or file_sha256(path) != value["sha256"]:
        raise ValueError("Registered content changed: " + str(path))
    return path


def prepare(args):
    """Lock a score-independent, current-only subset of original training data.

    This step hashes real files; it does not create trajectories, run a model,
    score anything, or inspect evaluation labels. The parent exposure is known.
    """
    root = Path(args.data_root).resolve()
    binding = read_binding(args.base_binding)
    dataset = ForesightTrainingDataset(root)
    if dataset.identity["split"] != "train" or dataset.ego_identity["split"] != "train":
        raise ValueError("Only original allowed training observations may enter Flow training")
    parent = binding["parent_identity"]
    if dataset.identity != parent["data"] or dataset.ego_identity != parent["ego"]:
        raise ValueError("Current observations/original ego targets differ from bound S0")
    receipt = read_json(args.cache_receipt)
    if receipt.get("status") != "COMPLETE":
        raise ValueError("A completed official train cache receipt is required")
    cached = {}
    for log in receipt["logs"]:
        for row in log["entries"]:
            token = row["token"]
            if token in cached:
                raise ValueError("Duplicate official context in receipt: " + token)
            cached[token] = row
    indices = [i for i, row in enumerate(dataset.index) if row["token"] in cached]
    indices.sort(key=lambda i: (identity_hash(["flow_fixed_train_v1", args.seed, dataset.index[i]["token"]]), i))
    if args.max_scenes:
        indices = indices[:args.max_scenes]
    if not indices or args.scope == "SMOKE" and len(indices) > 32:
        raise ValueError("Empty or over-budget registered training population")
    protocol = protocol_identity(args.navsim_root)
    preprocessing = artifact(Path(__file__).resolve().parents[2] / "starVLA/dataloader/foresight_dataset.py")
    entries = []
    for i in indices:
        row = dataset.index[i]
        current = artifact(root / "current" / (row["token"] + ".json"))
        record = read_json(current["path"])
        cache = cached[row["token"]]
        if record["token"] != row["token"] or record["log"] != row["log"] or cache["log"] != row["log"]:
            raise ValueError("Scene/log mismatch across observation and official context")
        images = [artifact(p) for p in record["image_paths"]]
        if len(images) != 3:
            raise ValueError("Original three-view contract changed")
        cache_artifact = {"path": str(Path(cache["path"]).resolve()), "sha256": cache["sha256"]}
        _verified_artifact(cache_artifact)
        observation = identity_hash({"current": current, "images": images, "preprocess": preprocessing["sha256"]})
        entries.append({"dataset_index": i, "scene_id": row["token"], "source_log": row["log"],
                        "current": current, "images": images, "observation_hash": observation,
                        "ego": artifact(root / "ego" / (row["token"] + ".pt")), "metric_cache": cache_artifact})
    manifest = {"schema": "st_flow_training_manifest_v1", "scope": args.scope, "split": "train",
                "base_binding_hash": binding["binding_sha256"], "root": str(root),
                "data_identity": dataset.identity, "ego_identity": dataset.ego_identity,
                "current_preprocessing": preprocessing, "cache_receipt": artifact(args.cache_receipt),
                "selection": {"seed": args.seed, "rule": "seeded_hash_before_any_scores", "max_scenes": args.max_scenes},
                "score_protocol_hash": protocol["sha256"], "entries": entries,
                "base_has_seen_population": binding["parent_has_seen_training_population"]}
    manifest["identity"] = identity_hash(manifest)
    out = Path(args.output).resolve()
    atomic_json(out / "TRAINING_MANIFEST.json", manifest, immutable=True)
    atomic_json(out / "SCORE_PROTOCOL.json", protocol, immutable=True)
    atomic_json(out / "NO_TEACHER.json", NO_TEACHER, immutable=True)
    return {"status": "PREPARED", "manifest": str(out / "TRAINING_MANIFEST.json"),
            "identity": manifest["identity"], "scenes": len(entries), "logs": len({e['source_log'] for e in entries}),
            "score_protocol_hash": protocol["sha256"], "model_loaded": False, "official_score_calls": 0}


class RegisteredFlowDataset:
    """Existing image/GT path with immutable observation and context provenance."""
    def __init__(self, root, manifest_path, binding, scope, score_protocol_hash):
        self.manifest = read_json(manifest_path)
        m = self.manifest
        if (m.get("schema") != "st_flow_training_manifest_v1" or m.get("split") != "train"
                or m.get("scope") != scope or m.get("root") != str(Path(root).resolve())
                or identity_hash({k: v for k, v in m.items() if k != "identity"}) != m.get("identity")
                or m["base_binding_hash"] != binding["binding_sha256"]
                or m["score_protocol_hash"] != score_protocol_hash):
            raise ValueError("Foreign/stale Base, split, source or scoring manifest")
        self.data = ForesightTrainingDataset(root)
        if (m["data_identity"] != self.data.identity or m["ego_identity"] != self.data.ego_identity
                or self.data.identity != binding["parent_identity"]["data"]
                or self.data.ego_identity != binding["parent_identity"]["ego"]):
            raise ValueError("Training labels must remain original bound S0 GT")
        _verified_artifact(m["current_preprocessing"])
        actual_preprocessor = Path(__file__).resolve().parents[2] / "starVLA/dataloader/foresight_dataset.py"
        if file_sha256(actual_preprocessor) != m["current_preprocessing"]["sha256"]:
            raise ValueError("Registered observation preprocessing differs from actual imported implementation")
        _verified_artifact(m["cache_receipt"])
        self.entries, self.identity = m["entries"], m["identity"]
        if not self.entries or len({r["scene_id"] for r in self.entries}) != len(self.entries):
            raise ValueError("Empty/duplicated registered scene population")
        self.stamps = {}
        for row in self.entries:
            original = self.data.index[row["dataset_index"]]
            if original != {"token": row["scene_id"], "log": row["source_log"]}:
                raise ValueError("Dataset cursor does not identify the registered scene/log")
            expected = identity_hash({"current": row["current"], "images": row["images"],
                                      "preprocess": m["current_preprocessing"]["sha256"]})
            if row["observation_hash"] != expected:
                raise ValueError("Observation identity changed")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        row = self.entries[i]
        # Hash on first access, then enforce immutable inode/mtime/ctime/size.
        # We do not SHA an image/context on every repeated training exposure.
        for info in (row["current"], row["ego"], *row["images"], row["metric_cache"]):
            path = Path(info["path"])
            stat = path.stat()
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            key = (str(path), info["sha256"])
            if key in self.stamps:
                if self.stamps[key] != stamp:
                    raise ValueError("An immutable registered training dependency changed")
            else:
                _verified_artifact(info)
                self.stamps[key] = stamp
        result = self.data[row["dataset_index"]]
        if result[0]["token"] != row["scene_id"] or set(result[1]) != {"ego"}:
            raise ValueError("Only original current observations and separate ego GT are allowed")
        return result


class FlowTrainingModule(nn.Module):
    """The only trainable object: original action head and its qwen_proj."""
    def __init__(self, action_head, cfg):
        super().__init__()
        self.policy = DiTFlowPolicy(action_head)
        self.fpo = validate_config(cfg)
        self.cfg = cfg

    def forward(self, condition, targets, *, streams, global_scenes, world_size,
                actions=None, samples=None, old_cfm=None, advantages=None, valid=None):
        bc = self.policy.bc_anchor(condition, targets, noise_generator=streams["bc_noise"],
            time_generator=streams["bc_time"], global_scene_count=global_scenes, world_size=world_size)
        policy = bc * 0.
        diagnostics = {}
        if self.cfg["cell"] != "C0":
            if any(x is None for x in (actions, samples, old_cfm, advantages, valid)):
                raise ValueError("Real scored rollout/old-CFM/MC samples are required")
            current = self.policy(condition.repeat_interleave(4, 0), actions, samples,
                mc_chunk=self.cfg["runtime"]["mc_chunk"],
                checkpoint_velocity=self.cfg["runtime"]["checkpoint_velocity"])
            policy, diagnostics = fpo_surrogate(old_cfm, current, advantages, valid, self.fpo,
                                                global_count=global_scenes * 4, world_size=world_size)
        loss = bc + self.cfg["loss"]["policy_weight"] * policy
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite BC/FPO loss; no optimizer step is permitted")
        return {"loss": loss, "losses": {"bc": bc.detach(), "policy": policy.detach(), "total": loss.detach()},
                "diagnostics": diagnostics}


def numeric_execution(cfg, binding):
    from tools.st_world_policy.training_config import official_runtime_policy
    official = official_runtime_policy(cfg)
    value = {"schema": "st_flow_numeric_execution_v2", "precision": cfg["execution"]["precision_policy"],
            "master_initialization": MASTER_INIT_SCHEMA,
            "encoder": "frozen_BF16_execution_eval_original_S0_AMP",
            "policy_rollout_old_current": "BF16_execution_cuda_autocast_eval",
            "BC": "BF16_execution_cuda_autocast_train_original_dropout",
            "Euler_residual": "float32", "CFM_reduction": "float32",
            "TF32": False, "deterministic": cfg["runtime"]["deterministic"],
            "gradient_checkpointing": cfg["runtime"]["gradient_checkpointing"],
            "base_binding": binding["binding_sha256"],
            "standalone_evaluation_contract": binding["inference_contract"]}
    if official is not None:
        value.update(schema="st_flow_numeric_execution_v3",
            encoder="frozen_BF16_execution_eval_official_FA2",
            official_execution=official)
    return value


class MatchedOldMicrobatches:
    """Same action-batch shapes as current CFM; does not resample MC draws."""
    def __init__(self, old, scene_microbatch):
        self.old, self.identity, self.actions = old, old.identity, 4*scene_microbatch

    @torch.no_grad()
    def __call__(self, condition, actions, samples, *, mc_chunk):
        return torch.cat([self.old(condition[start:start+self.actions], actions[start:start+self.actions],
                          samples.slice(slice(start, start+self.actions)), mc_chunk=mc_chunk)
                          for start in range(0, len(actions), self.actions)])


def frozen_encoder_hash(encoder):
    # Include nonpersistent buffers as well as state_dict; eval alone does not
    # establish that rotary/normalization buffers stayed fixed.
    return identity_hash({"state": module_hash(encoder),
                          "all_buffers": {name: tensor_hash(value) for name, value in encoder.named_buffers()}})


def score_requests(rollout, entries, binding, feedback_cfg, execution_hash):
    """Convert detached actual generated endpoints, never GT, to official requests."""
    rollout.validate()
    if tuple(r["scene_id"] for r in entries) != rollout.scene_ids:
        raise ValueError("Official request scene order differs from actual rollout")
    version = str(rollout.policy.version)
    origin = {"kind": "in_memory_parameter_snapshot", "state_sha256": rollout.policy.state_hash,
              "optimizer_step": rollout.policy.optimizer_step, "policy_version": version,
              "parent_checkpoint_sha256": binding["checkpoint"]["sha256"],
              "parent_receipt_sha256": binding["complete_sha256"], "numeric_execution_hash": execution_hash}
    physical = decode_ego(rollout.raw_actions).detach().cpu().double().numpy()
    requests = []
    for i, row in enumerate(entries):
        for candidate in range(4):
            trajectory = physical[i, candidate]
            requests.append(ConsequenceRequest(campaign_id=feedback_cfg["campaign_id"],
                scene_id=row["scene_id"], source_log=row["source_log"],
                candidate_id=f"{rollout.rollout_id}:{i}:{candidate}", policy_version=version,
                generator_checkpoint_sha256=None, generator_kind="policy_snapshot",
                generator_source_hash=fingerprint(origin), generator_identity=origin,
                inference_runtime_hash=execution_hash, trajectory_sha256=array_sha256(trajectory),
                metric_cache_sha256=row["metric_cache"]["sha256"], score_protocol_hash=feedback_cfg["score_protocol_hash"],
                metric_cache_path=row["metric_cache"]["path"], trajectory=trajectory, observe=True))
    return requests


def _validate_scored(record, request):
    if record.get("identity") != request.identity():
        raise ValueError("Official score belongs to a different trajectory/context/policy/runtime")
    if not record.get("valid") or record.get("status") != "VALID_SCORED":
        raise ValueError("Unresolved feedback cannot enter a training update")
    if (record.get("score_units") != "0..1" or not record.get("context_unchanged")
            or set(record.get("metrics", {})) != set(METRICS)
            or any(record.get("metrics_valid", {}).get(k) is not True for k in METRICS)
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in record["metrics"].values())
            or array_sha256(record["requested_trajectory"]) != request.trajectory_sha256):
        raise ValueError("Official score schema, units, validity or endpoint changed")
    ref = record["reference_identity"]
    if ref["full_cache_sha256"] != request.metric_cache_sha256 or ref["score_protocol_hash"] != request.score_protocol_hash:
        raise ValueError("Official reference/context identity mismatch")
    trace = record["consequences"]
    metrics = record["metrics"]
    # Retain native half-severity NC. Do not replace with a boolean collision.
    pairs = ((1 - float(trace["road"]["any"]), metrics["drivable_area_compliance"]),
             (trace["collision_summary"]["nc_min"], metrics["no_at_fault_collisions"]),
             (1 - float(trace["ttc_summary"]["any"]), metrics["time_to_collision_within_bound"]))
    if any(abs(a-b) > 1e-12 for a, b in pairs):
        raise ValueError("Canonical event consequence and official metric disagree")


def feedback_from_records(rollout, requests, records, cell, protocol_hash):
    if cell not in REWARDS or cell == "C0" or len(records) != len(requests) or len(records) != len(rollout.scene_ids) * 4:
        raise ValueError("Incomplete/unregistered official feedback group")
    rewards, contexts = [], []
    for i, (request, record) in enumerate(zip(requests, records)):
        _validate_scored(record, request)
        if i % 4 == 0:
            contexts.append(request.metric_cache_sha256)
            first_context = record["context_hash"]
            first_reference = record["reference_identity"]
        elif (contexts[-1] != request.metric_cache_sha256 or first_context != record["context_hash"]
              or first_reference != record["reference_identity"]):
            raise ValueError("Four candidates did not use one identical official reference context")
        trace = record["consequences"]
        rewards.append({"C1": record["metrics"]["score"], "C2": -float(trace["road"]["any"]),
                        "C3": float(trace["collision_summary"]["nc_min"]) - 1.,
                        "C4": -float(trace["ttc_summary"]["any"])}[cell])
    reward = torch.tensor(rewards, device=rollout.raw_actions.device, dtype=torch.float32).view(-1, 4)
    result = FeedbackBatch(rollout.rollout_id, protocol_hash, tuple(contexts), REWARDS[cell], reward,
                           torch.ones_like(reward, dtype=torch.bool), tuple("valid_scored" for _ in records))
    result.validate(rollout)
    return result


def _save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_update_feedback(folder, rollout, comparison, feedback, *, numeric_hash, rank):
    """Small audit payload, not a second multi-GB policy checkpoint.

    Full generated raw endpoints, generation noise, shared CFM noise/time, old
    CFM values and actual feedback are retained. At native save boundaries this
    payload has already been consumed; recovery starts from native state/RNG.
    """
    folder = Path(folder)
    identity = {"rollout_id": rollout.rollout_id, "comparison_id": comparison.comparison_id,
                "numeric_execution_hash": numeric_hash, "old_policy": rollout.policy.__dict__,
                "feedback": {"protocol": feedback.score_protocol_hash, "contexts": feedback.context_hashes,
                             "reward_kind": feedback.reward_kind, "rewards": feedback.rewards.cpu().tolist(),
                             "statuses": feedback.statuses}}
    # Canonicalize tuples so an idempotent JSON reload compares identically.
    identity = json.loads(json.dumps(identity))
    receipt_path = folder / f"rank_{rank}_COMPLETE.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["identity"] != identity:
            raise ValueError("Regenerated boundary feedback differs; do not reuse an unrelated rollout")
        _verified_artifact(receipt["payload"])
        return receipt
    path = folder / f"rank_{rank}.pt"
    _save_torch(path, {"schema": "st_flow_update_feedback_v1", "identity": identity,
                      "rollout": rollout.state_dict(), "comparison": comparison.state_dict(),
                      "feedback": feedback.state_dict()})
    receipt = {"identity": identity, "payload": artifact(path)}
    atomic_json(receipt_path, receipt, immutable=True)
    return receipt


class OfficialFeedbackLedger:
    """Durable, explicitly bounded retries; no invalid label becomes a zero.

    Admissions are committed BEFORE CPU calls, so a worker/process failure may
    overcount the budget but never silently undercounts it. Unfinished attempts
    remain evidence. Replay reuses only checksum-verified matching records.
    The caller holds its run lock: this is a single-writer ledger. On restart
    scan admissions once, then maintain the count without an O(updates^2) walk.
    """
    def __init__(self, root, *, identity, max_calls, retry_attempts):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.retry_attempts = max_calls, retry_attempts
        atomic_json(self.root / "IDENTITY.json", {"schema": "st_flow_feedback_ledger_v1", "identity": identity,
                    "max_calls": max_calls, "retry_attempts": retry_attempts}, immutable=True)
        self._admitted_calls = sum(read_json(p)["calls"] for p in self.root.glob("*/attempt_*/ADMITTED.json"))

    @property
    def admitted_calls(self):
        return self._admitted_calls

    def score(self, requests, *, policy_version, service):
        identities = [r.identity() for r in requests]
        key = identity_hash(identities)
        folder = self.root / key
        atomic_json(folder / "REQUESTS.json", {"policy_version": policy_version, "requests": identities}, immutable=True)
        results = [None] * len(requests)
        for attempt in range(self.retry_attempts + 1):
            path = folder / f"attempt_{attempt:02d}"
            receipt = path / "COMPLETE.json"
            if receipt.exists():
                record = read_json(receipt)
                admitted = read_json(path / "ADMITTED.json")
                if record["admission_sha256"] != file_sha256(path / "ADMITTED.json"):
                    raise ValueError("Feedback admission receipt changed")
                _verified_artifact(record["result"])
                saved = torch.load(record["result"]["path"], map_location="cpu", weights_only=False)
                if len(saved) != len(admitted["indices"]):
                    raise ValueError("Incomplete cached feedback result inventory")
                for i, result in zip(admitted["indices"], saved):
                    if result.get("identity") != identities[i]:
                        raise ValueError("Cached feedback identity mismatch")
                    results[i] = result
            elif (path / "ADMITTED.json").exists():
                # Unknown whether an interrupted worker scored a trajectory.
                # Keep its admission and use the next explicitly allowed retry.
                continue
            else:
                pending = [i for i, result in enumerate(results) if result is None or not result.get("valid")]
                for i in pending:
                    result = results[i]
                    if result is not None and (result.get("status") != "TRANSIENT_SCORING_FAILURE"
                                               or not result.get("error", {}).get("retryable")):
                        raise ValueError("Nonretryable official feedback: " + json.dumps(result.get("error")))
                if not pending:
                    break
                if self.admitted_calls + len(pending) > self.max_calls:
                    raise ValueError("Official feedback call budget exhausted; no optimizer update")
                admission = {"calls": len(pending), "indices": pending, "request_hash": key,
                             "time_unix": time.time(), "policy_version": policy_version}
                atomic_json(path / "ADMITTED.json", admission, immutable=True)
                self._admitted_calls += len(pending)
                saved = service.score([requests[i] for i in pending], expected_policy_version=policy_version,
                                      require_valid=False)
                if len(saved) != len(pending):
                    raise ValueError("Official worker omitted a requested candidate")
                for i, result in zip(pending, saved):
                    if result.get("identity") != identities[i]:
                        raise ValueError("Official worker returned a foreign result")
                    results[i] = result
                _save_torch(path / "RESULTS.pt", saved)
                atomic_json(receipt, {"result": artifact(path / "RESULTS.pt"),
                                     "admission_sha256": file_sha256(path / "ADMITTED.json")}, immutable=True)
            if all(r is not None and r.get("valid") for r in results):
                break
        for request, result in zip(requests, results):
            if result is None or not result.get("valid"):
                raise RuntimeError("Unresolved official feedback after bounded retry; retained in " + str(folder))
            _validate_scored(result, request)
        return results


@torch.no_grad()
def encode_conditions(encoder, observations, micro_batch):
    """Only current observation inputs, once per microbatch, with no target API."""
    if any(p.requires_grad for p in encoder.parameters()) or any(m.training for m in encoder.modules()):
        raise ValueError("Original shared encoder must remain frozen and eval")
    output = []
    for at in range(0, len(observations), micro_batch):
        encoded = encoder.encode_current(observations[at:at+micro_batch])
        output.append(encoder.build_planner_condition(encoded).detach())
    result = torch.cat(output)
    if result.ndim != 3 or not torch.isfinite(result).all() or result.requires_grad:
        raise ValueError("Invalid frozen original planner condition")
    return result


def _native_checkpoint(engine, out, state, reason, generators, world, rank):
    """Same native save/receipt helpers as ST; all RNG and cursors are consumed."""
    tick = time.perf_counter()
    stamp = [time.time_ns() if rank == 0 else None]
    dist.broadcast_object_list(stamp, 0)
    tag = f"step_{state['completed']:06d}_{stamp[0]}"
    state.update(tag=tag, termination_reason=reason)
    saved_rng = capture_rng(generators)
    folder = out / "checkpoints" / tag
    engine.save_checkpoint(str(out / "checkpoints"), tag=tag, client_state=state, save_latest=False)
    _save_torch(folder / f"rng_rank{rank}.pt", saved_rng)
    dist.barrier()
    def commit():
        if rank == 0:
            checkpoint_receipt(folder, state, world_size=world)
            update_latest(out / "checkpoints", tag)
    collective_call(commit, "Flow native checkpoint receipt")
    restore_rng(saved_rng, generators)
    if rank == 0:
        with (out / "checkpoint_costs.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": state["completed"], "tag": tag,
                                     "seconds": time.perf_counter()-tick}) + "\n")


def run(args):
    """One synchronous on-policy update, then native optimizer-boundary save.

    No old/feedback batch survives an optimizer boundary: old refresh/reuse=1.
    A failed partial update resumes the last complete boundary, whose RNG and
    data cursor regenerate the exact rollout. The durable CPU ledger preserves
    previous official results and charges interrupted attempts conservatively.
    """
    repo = Path(__file__).resolve().parents[2]
    registration = read_json(args.registration)
    cfg = read_json(registration["artifacts"]["config"]["path"])
    fpo = validate_config(cfg)
    execution, feedback_cfg, runtime = cfg["execution"], cfg["feedback"], cfg["runtime"]
    approval = check_formal_approval(args.registration, args.approval, repo=repo, effective_execution=execution)
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != execution["world_size"]:
        raise ValueError("torchrun topology differs from registered effective global batch")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before launch")
    if runtime["deterministic"] and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic CUDA runs require an explicit CUBLAS_WORKSPACE_CONFIG before initialization")
    lease = verified_runtime(registration, cfg)
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    lock = pool = service = ledger = None
    out = Path(runtime["output_root"]) / registration["run_id"]
    started = time.perf_counter()
    try:
        binding = collective_call(lambda: read_binding(registration["artifacts"]["base_binding"]["path"],
                                                       verify_files=rank == 0), "bound original S0")
        if any(file_sha256(repo / path) != digest for path, digest in binding["source_contracts"].items()):
            raise ValueError("Protected original S0 function contract changed; explicit migration is required")
        if read_json(registration["artifacts"]["teacher_manifest"]["path"]) != NO_TEACHER:
            raise ValueError("Flow C0--C4 do not load auxiliary or external teacher targets")
        if read_json(registration["artifacts"]["target_contract"]["path"]) != target_contract(cfg):
            raise ValueError("Registered reward/BC/FPO contract differs from effective training")
        if protocol_identity(feedback_cfg["navsim_root"])["sha256"] != feedback_cfg["score_protocol_hash"]:
            raise ValueError("Actual official CPU Python/software/source protocol changed")
        data = collective_call(lambda: RegisteredFlowDataset(cfg["data"]["root"],
            registration["artifacts"]["data_manifest"]["path"], binding, execution["scope"],
            feedback_cfg["score_protocol_hash"]), "registered original current/GT/context data")
        if execution["sample_limit"] and len(data) != execution["sample_limit"]:
            raise ValueError("Use the exact registered diagnostic population, not a prefix")
        validate_rank_batches(len(data), execution["global_batch"], world, execution["micro_batch"])
        numeric = {**numeric_execution(cfg, binding),
                   "environment_sha256": registration["artifacts"]["environment"]["sha256"],
                   "source_fingerprint": registration["source_fingerprint"]}
        numeric_hash = fingerprint(numeric)
        identity = {"schema": "st_flow_training_run_v1", "config": cfg, "data": data.identity,
                    "base_binding": binding["binding_sha256"], "approval": approval,
                    "source_sha": registration["source_sha"], "registration_sha256": file_sha256(args.registration),
                    "numeric_execution": numeric, "numeric_execution_hash": numeric_hash,
                    "new_optimizer_from_common_S0": True,
                    "weights_usable_for_formal_init": registration["weights_usable_for_formal_init"]}
        signature = identity_hash(identity)
        def own_run():
            nonlocal lock
            if rank != 0:
                return
            if out.exists() and not args.resume:
                raise FileExistsError("Output already exists: explicitly resume this exact registered run")
            if args.resume and not out.is_dir():
                raise FileNotFoundError("Resume cannot silently start a fresh experiment")
            out.mkdir(parents=True, exist_ok=True)
            lock = (out / "RUN.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.resume:
                if read_json(out / "identity.json") != {"identity": signature, **identity}:
                    raise ValueError("Resume source/data/recipe/numeric identity changed")
                if (out / "STOP_REQUESTED").exists():
                    if not args.acknowledge_stop:
                        raise ValueError("Explicit acknowledgment of prior operator stop is required")
                    (out / "STOP_REQUESTED").rename(out / ("STOP_ACKNOWLEDGED_" + str(time.time_ns())))
            else:
                atomic_json(out / "identity.json", {"identity": signature, **identity}, immutable=True)
                atomic_json(out / "resolved_config.json", cfg, immutable=True)
        collective_call(own_run, "Flow run ownership")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(runtime["deterministic"])
        random.seed(execution["seed"] + rank)
        np.random.seed(execution["seed"] + rank)
        torch.manual_seed(execution["seed"] + rank)
        from tools.foresight.checkpoints import load_student
        encoder = load_student(binding["run"], binding["tag"], binding["parent_identity"],
                               device="cpu", precision="fp32", strip=False)
        # Detach BEFORE BF16 conversion. DeepSpeed must receive learned FP32
        # action masters, not rounded BF16 values promoted back to float32.
        action_head = encoder.action_model
        del encoder.action_model
        encoder.strip_auxiliary_heads()
        encoder.requires_grad_(False).eval()
        encoder.inference_fp32 = False
        encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
        from tools.st_world_policy.training_config import official_runtime_policy
        official_policy = official_runtime_policy(cfg)
        official_receipt = None
        if official_policy is not None:
            from tools.st_world_policy.official_runtime import (
                apply_official_model_runtime, build_official_optimizer, official_deepspeed_config,
                verify_official_engine)
            official_receipt = {"policy": official_policy, "model": apply_official_model_runtime(
                vlm=encoder.qwen_vl_interface.model,
                language=encoder.qwen_vl_interface.model.model.language_model,
                action_transformer=action_head.model, policy=official_policy, training_copy=True)}
            # The consequence encoder is frozen/eval in either backend; only
            # the action module participates in training/recomputation.
            encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
        encoder.bfloat16().cuda(local)
        action_head.model.gradient_checkpointing = runtime["gradient_checkpointing"]
        model = FlowTrainingModule(action_head, cfg)
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("DeepSpeed must initialize FP32 masters from actual bound master tensors")
        groups, group_records = bounded_parameter_groups(model.named_parameters())
        opt = cfg["optimizer"]
        ds_config = {"train_micro_batch_size_per_gpu": execution["micro_batch"], "gradient_accumulation_steps": 1,
            "train_batch_size": execution["micro_batch"] * world, "bf16": {"enabled": True},
            "gradient_clipping": opt["clip"], "steps_per_print": 1000000,
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                  "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000},
            "optimizer": {"type": "AdamW", "params": {"lr": opt["lr"], "betas": opt["betas"],
                           "eps": opt["eps"], "weight_decay": opt["weight_decay"]}}}
        import deepspeed
        client_optimizer = None
        if official_policy is not None:
            client_optimizer, optimizer_receipt = build_official_optimizer(model, groups, official_policy)
            official_receipt["optimizer"] = optimizer_receipt
            ds_config = official_deepspeed_config(official_policy, accumulation_owner="caller_global_valid_counts")
        fresh_master_initialization = not args.resume
        master_source = capture_fp32_master_initialization(
            model, groups, is_fresh_run=fresh_master_initialization)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=groups, config=ds_config,
            **({"optimizer": client_optimizer} if client_optimizer is not None else {}))
        master_initialization = restore_fp32_master_initialization(
            engine, master_source, is_fresh_run=fresh_master_initialization)
        del master_source
        if official_policy is not None:
            engine_check = collective_call(lambda: verify_official_engine(
                engine, client_optimizer, official_policy), "official optimizer execution")
            atomic_json(out / "official_runtime" / f"initial_engine_rank{rank}.json", engine_check, immutable=True)
            if rank == 0:
                atomic_json(out / "OFFICIAL_EXECUTION.json", official_receipt, immutable=True)
        frozen_before = collective_call(lambda: frozen_encoder_hash(encoder), "frozen encoder state hash")
        if rank == 0 and not args.resume:
            atomic_json(out / "initialization.json", {"base_binding": binding["binding_sha256"],
                "parent_checkpoint": binding["checkpoint"], "groups": group_records,
                "master_initialization": master_initialization,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "frozen_parameters": sum(p.numel() for p in encoder.parameters()),
                "frozen_execution_state_sha256": frozen_before,
                "frozen_state_hash_is_training_BF16_copy_not_FP32_parent_checksum": True,
                "numeric_execution": numeric, "external_teacher_loaded": False,
                "auxiliary_heads_executed": False, "optimizer_fresh_for_matched_C_control": True}, immutable=True)
        streams = FlowRandomStreams(execution["seed"] + rank, device=torch.device("cuda", local))
        guard = FeedbackReuseGuard(score_protocol_hash=feedback_cfg["score_protocol_hash"],
                                   max_reuses=1, max_update_age=1, max_records=1)
        state = {"identity": signature, "completed": 0, "epoch": 0, "offset": 0, "exposure": 0,
                 "termination_reason": "RUNNING_CHECKPOINT", "counts": {}, "policy_version": 0,
                 "pending_old_policy": None, "pending_feedback": None,
                 "reuse_guard": guard.state_dict(), "rng_stream_base_seed": execution["seed"],
                 "frozen_encoder_hash": frozen_before, "last_completed_policy": None,
                 "old_snapshot_lifecycle": "fresh_each_update_discard_only_at_completed_boundary"}
        if args.resume:
            resolution = collective_call(lambda: resolve_latest(out / "checkpoints", identity=signature,
                world_size=world) if rank == 0 else None, "latest complete native Flow checkpoint")
            resolutions = [resolution]
            dist.broadcast_object_list(resolutions, 0)
            resolution = resolutions[0]
            folder = Path(resolution["selected"]["folder"])
            receipt = resolution["selected"]["receipt"]
            location, restored = engine.load_checkpoint(str(out / "checkpoints"), tag=folder.name,
                load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
            if not location or any(restored.get(k) != v for k, v in receipt["state"].items()):
                raise ValueError("Native state and complete progress receipt disagree")
            state = dict(receipt["state"])
            if (engine.global_steps != state["completed"] or state["policy_version"] != state["completed"]
                    or state["pending_old_policy"] is not None or state["pending_feedback"] is not None
                    or state["rng_stream_base_seed"] != execution["seed"] or state["frozen_encoder_hash"] != frozen_before):
                raise ValueError("Native optimizer boundary/policy/RNG/frozen encoder differs from resume")
            guard.load_state_dict(state["reuse_guard"])
            # Restore LAST: initialization/strict load must not override saved RNG.
            restore_rng(torch.load(folder / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False), streams.generators)
            if rank == 0:
                atomic_json(out / "RESUME_RESOLUTION.json", resolution)
        pause = [False]
        def signal_handler(*_):
            pause[0] = True
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        pool = ThreadPoolExecutor(max_workers=runtime["loader_workers"]) if runtime["loader_workers"] else None
        status = {"identity": signature, "status": "RUNNING", "host": socket.gethostname(),
                  "pid": os.getpid(), "started_unix": time.time(), "source_sha": registration["source_sha"]}
        def write_status():
            if rank == 0:
                atomic_json(out / "status.json", {**status, **state, "updated_unix": time.time()})
        model.train()
        write_status()
        if terminal(state, execution["max_updates"]):
            status["status"] = "COMPLETE"
            if rank == 0:
                atomic_json(out / "RESULT.json", {**status, **state, "new_optimizer_updates": 0})
            write_status()
            return {**status, "new_optimizer_updates": 0}
        if rank == 0 and cfg["cell"] != "C0":
            ledger = OfficialFeedbackLedger(out / "official_feedback", identity=signature,
                max_calls=feedback_cfg["max_calls"], retry_attempts=feedback_cfg["retry_attempts"])
        # A first-update worker failure must have an actual native recovery point.
        if not args.resume:
            _native_checkpoint(engine, out, state, "RUNNING_CHECKPOINT", streams.generators, world, rank)
        calls = {"visual": 0, "language": 0}
        hooks = []
        def count(name):
            def callback(*_):
                calls[name] += 1
            return callback
        hooks.append(encoder.qwen_vl_interface.model.model.visual.register_forward_hook(count("visual")))
        hooks.append(encoder.qwen_vl_interface.model.model.language_model.register_forward_hook(count("language")))
        initial_step = state["completed"]
        while state["completed"] < execution["max_updates"]:
            batches = epoch_batches(len(data), execution["global_batch"], execution["seed"], state["epoch"])
            if state["offset"] == len(batches):
                state["epoch"] += 1
                state["offset"] = 0
                continue
            operator = pause[0] or (out / "STOP_REQUESTED").exists()
            stop = (operator or time.perf_counter()-started >= runtime["max_seconds"]
                    or time.time() >= lease["valid_until_unix"] or state["completed"] >= execution["stop_after"])
            flags = torch.tensor([int(stop), int(operator)], device="cuda")
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags[0].item():
                status["status"] = "PAUSED"
                _native_checkpoint(engine, out, state, "OPERATOR_PAUSED" if flags[1].item() else "BUDGET_PAUSED",
                                   streams.generators, world, rank)
                break
            torch.cuda.synchronize()
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            global_ids = batches[state["offset"]]
            ids = global_ids[rank::world]
            entries = [data.entries[i] for i in ids]
            def fetch():
                rows = list(pool.map(data.__getitem__, ids)) if pool else [data[i] for i in ids]
                return collate_training(rows)
            with record_function("flow.data"):
                observations, targets = collective_call(fetch, "current images/original GT/official contexts")
            data_seconds = time.perf_counter() - tick
            calls_before = dict(calls)
            with record_function("flow.frozen_shared_encoder"), torch.autocast("cuda", dtype=torch.bfloat16):
                condition = collective_call(lambda: encode_conditions(encoder, observations, execution["micro_batch"]),
                                             "one frozen Qwen condition per scene")
            expected_calls = (len(ids)+execution["micro_batch"]-1)//execution["micro_batch"]
            if any(calls[k]-calls_before[k] != expected_calls for k in calls):
                raise ValueError("Frozen visual/language must execute once per original scene microbatch")
            torch.cuda.synchronize()
            encoding_seconds = time.perf_counter()-tick-data_seconds
            rollout = feedback = comparison = advantages = old_identity = None
            scoring_seconds = 0.
            rollout_seconds = old_cfm_seconds = 0.
            official_statistics = None
            if cfg["cell"] != "C0":
                stage_tick = time.perf_counter()
                with record_function("flow.old_snapshot_and_four_rollouts"), torch.autocast("cuda", dtype=torch.bfloat16):
                    old = collective_call(lambda: FrozenPolicySnapshot(model.policy, version=state["policy_version"],
                        optimizer_step=state["completed"], base_binding_hash=binding["binding_sha256"]), "old action-only snapshot")
                    rollout = collective_call(lambda: sample_rollout_group(old, condition,
                        scene_ids=[e["scene_id"] for e in entries], observation_hashes=[e["observation_hash"] for e in entries],
                        noise_generator=streams["rollout_noise"]), "four actual ten-Euler trajectories")
                old_identity = old.identity
                old_ids = [None] * world
                dist.all_gather_object(old_ids, old_identity.__dict__)
                if any(x != old_ids[0] for x in old_ids):
                    raise ValueError("Ranks have different frozen old-policy parameter identities")
                torch.cuda.synchronize()
                rollout_seconds = time.perf_counter()-stage_tick
                local_requests = collective_call(lambda: score_requests(rollout, entries, binding, feedback_cfg, numeric_hash),
                                                 "typed actual-rollout scoring requests")
                by_rank = [None] * world
                dist.all_gather_object(by_rank, local_requests)
                stage_tick = time.perf_counter()
                def score_all():
                    nonlocal service
                    if rank != 0:
                        return None
                    if service is None:
                        spec = ConsequenceSpec(navsim_root=feedback_cfg["navsim_root"], maps_root=feedback_cfg["maps_root"],
                            cache_roots=tuple(feedback_cfg["cache_roots"]), campaign_id=feedback_cfg["campaign_id"],
                            score_protocol_hash=feedback_cfg["score_protocol_hash"])
                        service = ConsequenceService(spec, workers=feedback_cfg["workers"], max_calls=feedback_cfg["max_calls"])
                    return ledger.score([request for group in by_rank for request in group],
                                        policy_version=str(old_identity.version), service=service)
                with record_function("flow.official_CPU_feedback"):
                    records = collective_call(score_all, "bounded canonical CPU feedback")
                # Broadcast only local shards rather than full executed-state traces.
                # Rank zero has already persisted all full records and hashes.
                summaries = None
                if rank == 0:
                    offset, summaries = 0, []
                    for group in by_rank:
                        chunk = records[offset:offset+len(group)]
                        summaries.append(chunk)
                        offset += len(group)
                local_records = [None]
                dist.scatter_object_list(local_records, summaries, src=0)
                feedback = collective_call(lambda: feedback_from_records(rollout, local_requests, local_records[0],
                    cfg["cell"], feedback_cfg["score_protocol_hash"]), "official result/context/reward consistency")
                expected_contexts = tuple(e["metric_cache"]["sha256"] for e in entries)
                collective_call(lambda: guard.check(rollout, feedback, active_old=old_identity,
                    optimizer_step=state["completed"], expected_context_hashes=expected_contexts), "feedback freshness guard")
                advantages = group_centered_advantages(feedback.rewards, feedback.valid).flatten()
                official_statistics = {"valid_candidates": int(feedback.valid.sum()),
                    "zero_advantage_scenes": int((advantages.view(-1, 4) == 0).all(1).sum()),
                    "mean_reward": float(feedback.rewards.mean()),
                    "metrics_mean": {k: float(np.mean([r["metrics"][k] for r in local_records[0]])) for k in METRICS}}
                scoring_seconds = time.perf_counter()-stage_tick
                stage_tick = time.perf_counter()
                with record_function("flow.old_CFM_MC50"), torch.autocast("cuda", dtype=torch.bfloat16):
                    matched_old = MatchedOldMicrobatches(old, execution["micro_batch"])
                    comparison = collective_call(lambda: prepare_fpo_batch(matched_old, rollout, condition, config=fpo,
                        streams=streams, mc_chunk=runtime["mc_chunk"]), "coupled old CFM MC50")
                collective_call(old.verify, "frozen old policy retained")
                del matched_old
                del old
                collective_call(lambda: save_update_feedback(out / "rollouts" / f"boundary_{state['completed']:06d}",
                    rollout, comparison, feedback, numeric_hash=numeric_hash, rank=rank), "actual rollout/MC feedback audit receipt")
                torch.cuda.synchronize()
                old_cfm_seconds = time.perf_counter()-stage_tick
            lr = learning_rate(state["completed"], opt["lr"], opt["minimum_lr"], opt["warmup"], execution["max_updates"])
            for group in engine.optimizer.param_groups:
                group["lr"] = lr
            logs, ratios = {}, []
            backward_tick = time.perf_counter()
            for at in range(0, len(ids), execution["micro_batch"]):
                end = min(at+execution["micro_batch"], len(ids))
                boundary = end == len(ids)
                engine.set_gradient_accumulation_boundary(boundary)
                policy_inputs = {}
                if comparison is not None:
                    sl = slice(4*at, 4*end)
                    policy_inputs = {"actions": rollout.raw_actions.flatten(0, 1)[sl], "samples": comparison.samples.slice(sl),
                                     "old_cfm": comparison.old_cfm[sl], "advantages": advantages[sl],
                                     "valid": feedback.valid.flatten()[sl]}
                with record_function("flow.current_FPO_and_original_BC"), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = collective_call(lambda: engine(condition[at:end], targets["ego"][at:end].cuda(),
                        streams=streams, global_scenes=len(global_ids), world_size=world, **policy_inputs), "DiT BC plus FPO forward")
                with record_function("flow.backward"):
                    engine.backward(output["loss"])
                for key, value in output["losses"].items():
                    logs[key] = logs.get(key, torch.zeros((), device="cuda")) + value.float()
                if output["diagnostics"]:
                    ratios.append(output["diagnostics"]["ratio"])
                if boundary:
                    before = capture_master_samples(engine.optimizer)
                with record_function("flow.optimizer"):
                    engine.step()
                if boundary:
                    update_evidence = master_update_evidence(engine.optimizer, before, torch.device("cuda", local))
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter()-backward_tick
            if rollout is not None:
                guard.commit_consumed(rollout, feedback, active_old=old_identity, optimizer_step=state["completed"],
                                       expected_context_hashes=expected_contexts)
                state["last_completed_policy"] = old_identity.__dict__
            state["completed"] += 1
            if official_policy is not None and state["completed"] == initial_step + 1:
                engine_check = collective_call(lambda: verify_official_engine(
                    engine, client_optimizer, official_policy, require_moments=True),
                    "official FP32 Adam update state")
                atomic_json(out / "official_runtime" / f"updated_engine_rank{rank}.json",
                            engine_check, immutable=True)
            state["policy_version"] = state["completed"]
            state["offset"] += 1
            state["exposure"] += len(global_ids)
            state["termination_reason"] = "RUNNING_CHECKPOINT"
            guard.retire_before(state["completed"])
            state["reuse_guard"] = guard.state_dict()
            counts = {"original_GT_scenes": len(global_ids), "BC_repeat8_presentations": 8*len(global_ids),
                      "actual_rollouts": 4*len(global_ids) if rollout is not None else 0,
                      "policy_MC_points_per_old_and_current": 200*len(global_ids) if rollout is not None else 0}
            for key, value in counts.items():
                state["counts"][key] = state["counts"].get(key, 0) + value
            values = torch.stack([logs[k] for k in sorted(logs)])
            dist.all_reduce(values)
            logs = dict(zip(sorted(logs), (values/world).cpu().tolist()))
            per_rank = {"sample_ids": [e["scene_id"] for e in entries], "official": official_statistics,
                "policy_ratio_max_abs_from_one": float((torch.cat(ratios)-1).abs().max()) if ratios else None,
                "calls": {k: calls[k]-calls_before[k] for k in calls},
                "seconds": {"update": time.perf_counter()-tick, "data": data_seconds, "shared_encoder": encoding_seconds,
                            "snapshot_rollout": rollout_seconds, "official_feedback": scoring_seconds,
                            "old_CFM": old_cfm_seconds, "current_BC_FPO_backward_optimizer": backward_seconds},
                "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}
            all_rank = [None] * world
            dist.all_gather_object(all_rank, per_rank)
            if rank == 0:
                with (out / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": state["completed"], "losses": logs, "lr": lr,
                        "exposure": state["exposure"], "counts": counts, "per_rank": all_rank,
                        "slowest_rank_seconds": max(r["seconds"]["update"] for r in all_rank),
                        "gradient_norm": float(engine.get_global_grad_norm()), "old_policy": state["last_completed_policy"],
                        "official_admitted_calls": ledger.admitted_calls if ledger else 0, **update_evidence}) + "\n")
            write_status()
            if state["completed"] == execution["max_updates"]:
                status["status"] = "COMPLETE"
                _native_checkpoint(engine, out, state, "MAX_STEPS", streams.generators, world, rank)
                break
            if state["completed"] % runtime["save_every"] == 0:
                _native_checkpoint(engine, out, state, "RUNNING_CHECKPOINT", streams.generators, world, rank)
        for hook in hooks:
            hook.remove()
        frozen_after = collective_call(lambda: frozen_encoder_hash(encoder), "final frozen encoder hash")
        if frozen_after != frozen_before or any(p.grad is not None for p in encoder.parameters()):
            raise ValueError("Shared encoder changed during action-only training")
        result = {**status, **state, "new_optimizer_updates": state["completed"]-initial_step,
                  "frozen_encoder_unchanged": True, "scientific_result": "NOT_EVALUATED",
                  "formal_launch_approval": approval["formal_launch_allowed"]}
        if rank == 0:
            atomic_json(out / "RESULT.json", result)
        write_status()
        return result
    except BaseException as error:
        if rank == 0 and out.exists():
            atomic_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(error), "time_unix": time.time(),
                "resume": "last_complete_native_boundary_only_no_partial_optimizer_or_cursor_commit"})
        raise
    finally:
        if service:
            service.close()
        if pool:
            pool.shutdown(wait=True)
        if lock:
            lock.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Lock real original-train observations and canonical contexts; no model/score/GPU")
    for flag in ("base-binding", "data-root", "cache-receipt", "navsim-root", "output"):
        p.add_argument("--"+flag, required=True)
    p.add_argument("--scope", choices=("SMOKE", "PROFILE", "FORMAL"), required=True)
    p.add_argument("--max-scenes", type=int, required=True, help="0 only for explicitly registered FORMAL complete available population")
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("train", help="torchrun: strict native C0--C4 from registered bound S0")
    p.add_argument("--registration", required=True)
    p.add_argument("--approval")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--acknowledge-stop", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare" and (args.seed < 0 or args.max_scenes < 0 or args.max_scenes == 0 and args.scope != "FORMAL"):
        parser.error("Finite diagnostic scene bound and nonnegative seed required")
    if args.command == "train" and args.acknowledge_stop and not args.resume:
        parser.error("--acknowledge-stop requires --resume")
    try:
        result = prepare(args) if args.command == "prepare" else run(args)
        if args.command != "train" or int(os.environ.get("RANK", 0)) == 0:
            print(json.dumps(result), flush=True)
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": repr(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
