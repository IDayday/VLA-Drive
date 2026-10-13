"""Explicit pre-training revision of Scorer fit scope, never evaluation roles.

The expert's D_E/D_S partition stays immutable. This separate training identity
lets a Scorer learn from both subroles while retaining their exposure diagnostics.
"""
from datetime import datetime, timezone
from iqe.contracts import require
from iqe.data.manifests import load_scenes
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from .partition import training_ids

FILE = "SCORER_FIT_SCOPE.json"


def scope(config):
    path = config.root/FILE
    if not path.exists():
        return {"scope": "expert_heldout_D_S", "amendment_hash": None}
    r = read_json(path)
    require(r["schema"] == "two_candidate_scorer_fit_scope_v1" and r["scope"] == "all_incremental_fit",
            "unknown Scorer scope")
    require(r["experiment_id"] == config.experiment_id and r["config_hash"] == config.identity,
            "Scorer scope experiment mismatch")
    for field, filename in (("base_binding_hash", "BASE_BINDING.json"), ("partition_hash", "DATA_PARTITION.json"),
                            ("scene_manifest_hash", "scenes.json")):
        require(r[field] == file_hash(config.root/filename), "Scorer scope dependency changed: " + field)
    require(r["expert_updates_at_amendment"] == r["scorer_updates_at_amendment"] == 0,
            "Scorer scope must be fixed before head training")
    return {**r, "amendment_hash": file_hash(path)}


def scorer_training_ids(config, scenes):
    r = scope(config)
    if r["scope"] == "expert_heldout_D_S":
        return training_ids(config, scenes, "scorer_fit")
    require(config.research_protocol == "two_candidate_v1" and config.fit_limit == "all_fit", "full fit protocol required")
    ids = training_ids(config, scenes, "expert_fit") | training_ids(config, scenes, "scorer_fit")
    require(ids == {s.scene_id for s in scenes if s.split_role == "incremental_fit"}, "full fit coverage/role mismatch")
    require(len(ids) == r["scenes"] and digest(sorted(ids)) == r["scene_ids_hash"], "Scorer fit population changed")
    return ids


def dependencies(config):
    r = scope(config)
    return {"scorer_fit_scope": r["amendment_hash"]} if r["amendment_hash"] else {}


def amend_scope(config, reason):
    require(config.research_protocol == "two_candidate_v1" and config.fit_limit == "all_fit", "explicit all_fit required")
    require(isinstance(reason, str) and reason.strip(), "scope change reason required")
    with lock(config.root/"scorer_fit_scope.lock"):
        if (config.root/FILE).exists():
            return scope(config)
        for kind in ("expert", "scorer"):
            folder = config.root/kind
            require(not any(folder.glob("*.pt*")) and not (folder/"status.json").exists()
                    and not (folder/"TRAINING_BUDGET.json").exists()
                    and not ((folder/"steps.jsonl").exists() and (folder/"steps.jsonl").stat().st_size),
                    "cannot change fit scope after head training starts")
        require(not (config.root/"PAIR_BANK.json").exists(), "cannot change an existing pair bank")
        scenes = load_scenes(config.root/"scenes.json")
        de, ds = (training_ids(config, scenes, name) for name in ("expert_fit", "scorer_fit"))
        ids = de | ds
        require(not de & ds and ids == {s.scene_id for s in scenes if s.split_role == "incremental_fit"}, "fit partition leakage")
        atomic_json(config.root/FILE, {"schema": "two_candidate_scorer_fit_scope_v1", "scope": "all_incremental_fit",
            "experiment_id": config.experiment_id, "config_hash": config.identity,
            "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
            "partition_hash": file_hash(config.root/"DATA_PARTITION.json"),
            "scene_manifest_hash": file_hash(config.root/"scenes.json"),
            "scene_ids_hash": digest(sorted(ids)), "scenes": len(ids),
            "expert_source_pool_scenes": len(de), "expert_heldout_subrole_scenes": len(ds),
            "expert_updates_at_amendment": 0, "scorer_updates_at_amendment": 0,
            "expert_scope_unchanged": "D_E hard/ordinary only", "evaluation_roles_fit": False,
            "diagnostics": "report D_E/D_S separately; neither is held out from this Scorer",
            "recorded_utc": datetime.now(timezone.utc).isoformat(), "reason": reason.strip()}, immutable=True)
        return scope(config)
