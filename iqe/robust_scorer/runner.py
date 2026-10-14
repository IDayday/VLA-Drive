"""Thin experiment layer over IQE's verified bank, context and native trainer."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import io
import subprocess
import numpy as np
import torch

from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_json, atomic_bytes, read_json, file_hash, digest, lock
from iqe.evaluation.retention import module_hash, _tensor_map_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.scorer_training import batch_fetch, fetch_context
from iqe.specialist_study.training import runtime, preload
from iqe.specialist_study.objectives import StudySampler
from iqe.specialist_study.metrics import metrics, select
from iqe.training.trainer import train
from .config import CELLS
from .model import RobustScorer
from .objectives import terms, denominators


def binding(config):
    source = config.source()
    paths = ["BASE_BINDING.json", "CAMPAIGN_BINDING.json", "DATA_PARTITION.json", "scenes.json",
             "OFFLINE_BASE_CONTRACT.json", "SERVING_CANDIDATE_POOL.json", "SCORE_PROTOCOL.json"]
    artifacts = {name: file_hash(source.root/name) for name in paths}
    roles = scenes_by_role(source)
    require(all(s.split_role == "incremental_fit" for s in roles["scorer_fit"]), "only legal D_S can fit Scorer")
    fit_logs = {s.source_log_id for s in roles["scorer_fit"]}
    for role in ("expert_fit", "stage_val", "selector_cal", "dev_report"):
        require(not fit_logs & {s.source_log_id for s in roles[role]}, "D_S log leakage")
    for role in ("scorer_fit", "stage_val"):
        meta, _ = load_bank(source, "deploy", role)
        require(meta["scene_ids"] == [s.scene_id for s in roles[role]], "fixed bank/scene order mismatch")
        artifacts[f"banks/deploy/{role}/BANK.json"] = file_hash(source.root/f"banks/deploy/{role}/BANK.json")
        artifacts[f"banks/deploy/{role}/arrays.npz"] = meta["array_hash"]
    identity = {"schema": "robust_scorer_fixed_pool_v1", "experiment_id": config.experiment_id,
                "config_hash": config.identity, "source_campaign": str(source.root), "artifacts": artifacts,
                "cells": CELLS, "max_new_trainings": 4, "updates_each": config.updates,
                "fixed_experts": pool(source)["expert_ids"], "new_experts": 0,
                "training_role": "scorer_fit", "external_scorer_weights_loaded": False,
                "source_checkpoint_policy": "same_experiment_resume_only", "navtest_access": False,
                "ACTIVE_updates": False,
                "selection": "stage_val raw argmax PDMS; tie fewer new NC/DAC then earlier checkpoint",
                "final_evaluation": "lock model using stage_val before dev; calibration only selector_cal"}
    destination = config.root/"EXPERIMENT_BINDING.json"
    with lock(config.root/"BIND.lock"):
        if destination.exists():
            old = read_json(destination)
            require(old["identity"] == identity, "immutable experiment/source changed")
            return old
        result = {"identity": identity, "registered_utc": datetime.now(timezone.utc).isoformat(),
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        atomic_json(destination, result, immutable=True)
        return result


def create(config, cell):
    require(cell in CELLS, "unknown registered Scorer cell")
    source = config.source()
    scene = scenes_by_role(source)["scorer_fit"][0]
    x = ReusedContext(source, capacity=1).read(scene)
    contract = TrajectoryContract(**read_json(source.root/"OFFLINE_BASE_CONTRACT.json")["trajectory"])
    with initialization_rng(config.seed):
        model = RobustScorer(x["world"].shape[-1], x["action"].shape[-1], contract, **CELLS[cell])
    identity = {"schema": "scratch_robust_scorer_v1", "experiment_id": config.experiment_id,
                "cell": cell, "model": CELLS[cell], "module_init_seed": config.seed,
                "initial_state_hash": module_hash(model),
                "common_initial_hash": _tensor_map_hash((k, v) for k, v in model.state_dict().items() if not k.startswith("nonzero_head.")),
                "parameters": {name: {"shape": list(p.shape), "count": p.numel()} for name, p in model.named_parameters()},
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "external_scorer_weights_loaded": False, "loaded_scorer_weights": [],
                "allowed_resume_lineage": config.experiment_id+"/"+cell,
                "Base_binding_hash": file_hash(source.root/"BASE_BINDING.json"),
                "pool_hash": pool(source)["hash"]}
    model._iqe_initialization_identity = identity
    return model, identity


@torch.no_grad()
def predict_bank(model, scenes, bank, cache, device):
    model.eval()
    results = {}
    for at in range(0, len(scenes), 32):
        sub = scenes[at:at+32]
        context = fetch_context(None, sub, device, cache)
        trajectory = torch.from_numpy(bank["trajectories"][at:at+len(sub)]).to(device)
        output = model(context, trajectory)
        for key, values in output.items():
            require(torch.isfinite(values).all().item(), "nonfinite Scorer inference")
            results.setdefault(key, []).append(values.cpu().numpy())
    return {key: np.concatenate(value) for key, value in results.items()}


def train_cell(config, cell, device="cuda:0", resume=None, stop_after=None):
    require(cell in CELLS, "unregistered Scorer cell")
    bound = binding(config)
    source = config.source()
    device = runtime(device)
    folder = config.root/"scorers"/cell
    with lock(folder/"RUN.lock"):
        meta, deploy = load_bank(source, "deploy", "scorer_fit")
        vm, vb = load_bank(source, "deploy", "stage_val")
        roles = scenes_by_role(source)
        scenes = roles["scorer_fit"]
        cache = preload(replace(source, feature_memory_gib=config.feature_memory_gib), scenes)
        val_cache = ReusedContext(source, 1024)
        model, identity = create(config, cell)
        atomic_json(folder/"SCORER_INITIALIZATION.json", identity, immutable=True)
        sampler = StudySampler([s.scene_id for s in scenes], deploy["scores"], deploy["comparison_valid"],
                               seed=config.sampler_seed, bank_hash=meta["hash"], phase_steps=config.phase_switch)
        atomic_json(folder/"COMPARISON_DISTRIBUTION.json", {
            "sampler_identity": sampler.identity, "pairs": sampler.pairs.tolist(),
            "weights": sampler.weights.tolist(), "q_hash": digest(sampler.q.tolist()) if sampler.q is not None else None,
            "gap_weight_applied_once": True, "bank_hash": meta["hash"]}, immutable=True)
        fetch = batch_fetch(source, scenes, cache, None, deploy, deploy, False)
        objective = CELLS[cell]["objective"]
        def loss(m, b):
            return terms(m(b["context"], b["trajectories"]), b, objective)
        def validate(m, step):
            predictions = predict_bank(m, roles["stage_val"], vb, val_cache, device)
            selected = select(predictions["values"], vb["valid"], expert_ids=vm["provenance"]["sources"])
            result = metrics(vb["scores"], vb["components"], selected, vm["source_logs"],
                             seed=source.bootstrap_seed, repetitions=source.bootstrap_repetitions)
            result["selection_key"] = [result["selected_01"], -sum(result["safety"][c]["new_failures"]
                                      for c in ("no_at_fault_collisions", "drivable_area_compliance"))]
            result.update(checkpoint_kind="raw_argmax", scorer_state_hash=module_hash(m),
                          bank_hash=vm["hash"], pool_hash=pool(source)["hash"], cell=cell,
                          objective=objective, dropout=CELLS[cell]["dropout"])
            buf = io.BytesIO()
            np.savez(buf, **predictions)
            path = folder/f"predictions_stage_val_{step:06d}.npz"
            atomic_bytes(path, buf.getvalue(), immutable=True)
            result["prediction_hash"] = file_hash(path)
            return result
        cfg = {"lr": config.lr, "weight_decay": config.weight_decay, "global_batch_size": config.global_batch,
               "micro_batch_size": config.micro_batch, "max_optimizer_steps": config.updates,
               "warmup_steps": config.warmup, "lr_min_ratio": 1., "grad_clip_norm": 1.,
               "schedule": "warmup_constant", "validation_steps": list(config.validation_steps),
               "checkpoint_every_steps": config.checkpoint_interval, "validation_every_steps": 1000,
               "early_stop_patience": 10**9}
        weights = {"value": 1., "nonzero": 1., "components": .2, "ranking": .2}
        source_files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("objectives.py"),
                        Path(__file__).with_name("config.py")]
        deps = {"experiment": digest(bound), "cell": cell, "data": meta["hash"], "validation": vm["hash"],
                "sampler": sampler.identity, "init": identity["initial_state_hash"],
                "source": {p.name: file_hash(p) for p in source_files}, "weights": weights,
                "training_rng_seed": config.training_seed, "sampler_seed": config.sampler_seed}
        atomic_json(folder/"RUN_IDENTITY.json", {"dependencies": deps, "training": cfg}, immutable=True)
        train(model, loss, fetch, sampler, cfg, deps, folder, device=device, weights=weights, resume=resume,
              validate=validate, stop_after=stop_after, seed=config.training_seed,
              denominator_function=lambda b: denominators(b, objective))
        return read_json(folder/"result.json")


def data_audit(config):
    bound = binding(config)
    source = config.source()
    result = {"binding_hash": digest(bound), "roles": {}, "test_or_dev_labels_read": False}
    for kind, role in (("deploy", "scorer_fit"), ("rich", "scorer_fit"), ("deploy", "stage_val")):
        meta, data = load_bank(source, kind, role)
        s, c, v, cv = (data[k] for k in ("scores", "components", "valid", "component_valid"))
        aggregate = c[:, :, 0]*c[:, :, 1]*(5*c[:, :, 2]+5*c[:, :, 3]+2*c[:, :, 4])/12
        error = float(np.abs(aggregate[v]-s[v]).max())
        require(error < 1e-6, "bank scores disagree with locked official component formula")
        row = {"scenes": len(s), "logs": len(set(meta["source_logs"])), "valid_candidates": int(v.sum()),
               "bank_hash": meta["hash"], "zeros": int((v&(s <= 1e-8)).sum()),
               "score_quantiles": np.quantile(s[v], [0, .01, .05, .1, .25, .5, .9, 1]).tolist(),
               "reconstructed_score_max_error": error, "components": {}}
        for j, name in enumerate(meta["components"]):
            mask = v&cv[:, :, j]
            row["components"][name] = {"valid": int(mask.sum()), "failures": int((mask&(c[:, :, j] < 1-1e-8)).sum()),
                                       "failure_scenes": int((mask&(c[:, :, j] < 1-1e-8)).any(1).sum())}
        result["roles"][kind+"/"+role] = row
    atomic_json(config.root/"DATA_AND_OBJECTIVE_AUDIT.json", result, immutable=True)
    return result
