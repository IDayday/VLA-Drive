"""Immutable serving bundles; label/metric/optimizer dependencies never enter inference."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import shutil
import torch
from .contracts import require
from .io import atomic_json, atomic_torch, read_json, digest, file_hash, lock
from .selector import SelectionRule, RouterRule
from .scorer import TrajectoryScorer
from .router import SceneRouter
from .model import IQEModel
from .s0_adapter import S0Adapter
from .evaluation.retention import module_hash


def validate_bundle_files(path):
    path = Path(path)
    data = read_json(path / "BUNDLE.json")
    require(data["bundle_hash"] == digest({k:v for k,v in data.items() if k != "bundle_hash"}), "bundle manifest corrupt")
    require(data["registry_hash"] == digest(data["registry"]), "bundle registry corrupt")
    require(data["normalizer_hash"] == data["contract"]["normalizer_hash"], "bundle normalizer mismatch")
    require(data["rule"]["dependency_hash"] == data["pool_hash"], "bundle calibration pool mismatch")
    if "trajectory" in data["contract"]:
        require(digest(data["contract"]["trajectory"]) == data["normalizer_hash"], "normalizer parameters changed")
    for filename, expected in data["weights"].items():
        require(file_hash(path / filename) == expected, "bundle corrupt weight: " + filename)
    return data


def release_gate(evaluation, oracle, frozen_audit, calibration, minimum_groups=10):
    checks = {"oracle_or_existing_pool": bool(oracle.get("oracle_gate") or oracle.get("selector_only")), "frozen": bool(frozen_audit.get("passed")),
              "calibrated_benefit": bool(calibration.get("beneficial"))}
    for baseline in ("s0", "previous"):
        comparison = evaluation["vs_" + baseline]
        ci = evaluation["paired_group_bootstrap_vs_" + baseline]
        checks[baseline + "_positive_gain"] = comparison["mean_gain_01"] > 0 and ci["lower"] > 0 and ci["groups"] >= minimum_groups
        checks[baseline + "_no_high_to_zero"] = comparison["high_to_zero"] == 0
        checks[baseline + "_no_new_hard_safety_violation"] = all(
            comparison[c]["new_violations"] == 0 for c in ("no_at_fault_collisions", "drivable_area_compliance"))
    return {"passed": all(checks.values()), "checks": checks}


def export_bundle(model, rule, registry, path, **kwargs):
    path = Path(path)
    with lock(path.parent / (path.name + ".export.lock")):
        return _export_bundle(model,rule,registry,path,**kwargs)


def _export_bundle(model, rule, registry, path, *, mode, gate_results, config_hash, pool_hash, selector_type="scorer", expert_ids=None, data_role_groups=None):
    require(mode in {"smoke", "full"}, "bundle mode")
    require(selector_type in {"scorer", "scene_router"}, "Oracle/real-score selectors cannot be exported")
    if mode == "full":
        require(gate_results["passed"], "release gate failed: retain previous bundle")
    model.set_trainable_stage("inference"); model.eval()
    expert_ids = tuple(expert_ids or model.experts)
    require(expert_ids and expert_ids[0] == "expert_0" and all(e in model.experts for e in expert_ids), "serving pool identity")
    path = Path(path)
    if path.exists():
        previous = validate_bundle_files(path)
        require(previous["config_hash"] == config_hash and previous["pool_hash"] == pool_hash and previous["rule"] == asdict(rule)
                and previous["mode"] == mode and previous["expert_ids"] == list(expert_ids)
                and previous["base_state_hash"] == module_hash(model.adapter.framework)
                and previous["expert_state_hashes"] == {eid:module_hash(model.experts[eid]) for eid in expert_ids},
                "accepted bundle version conflict")
        selector = model.scorer if selector_type == "scorer" else model.router
        require(previous["selector_state_hash"] == (module_hash(selector) if selector is not None else None), "bundle selector changed")
        return previous
    temporary = path.with_name("." + path.name + ".building")
    if temporary.exists():
        # Interrupted private staging is preserved for diagnosis, not mistaken for
        # accepted output or mixed with a resumed generation.
        import time
        temporary.rename(temporary.with_name(temporary.name + ".interrupted." + str(time.time_ns())))
    temporary.mkdir(parents=True)
    try:
        # Actual deployment state contains no auxiliary training readout, optimizer or labels.
        base = model.adapter.framework
        contract = dict(model.adapter.contract)
        base_state = {k: v for k, v in base.state_dict().items() if not k.startswith(("future_head.", "dino_head.", "interaction_head.", "spatiotemporal_head."))}
        atomic_torch(temporary / "base.pt", {"kind": "iqe_query_base_deploy", "model": base_state,
                     "framework_contract_hash": contract["framework_contract_hash"]})
        for eid in expert_ids:
            atomic_torch(temporary / (eid + ".pt"), model.experts[eid].state_dict())
        selector = model.scorer if selector_type == "scorer" else model.router
        if selector is not None:
            atomic_torch(temporary / "selector.pt", selector.state_dict())
        events = registry.read()
        artifact_hashes = {p.name: file_hash(p) for p in temporary.glob("*.pt")}
        payload = {"schema_version": 1, "mode": mode, "contract": contract, "normalizer_hash": contract["normalizer_hash"],
                   "expert_ids": list(expert_ids), "registry": events, "registry_hash": digest(events),
                   "architecture": {eid: registry.latest(events)[eid]["architecture"] for eid in expert_ids},
                   "selector_type": selector_type, "rule": asdict(rule), "config_hash": config_hash, "pool_hash": pool_hash,
                   "gate_results": gate_results, "weights": artifact_hashes,
                   "scorer_architecture": getattr(model.scorer, "architecture", None),
                   "router_hidden": model.router.head.in_features if model.router else None,
                   "score_protocol": contract["metric"], "training_dependencies_required_at_inference": False}
        payload["data_role_groups"] = sorted(data_role_groups or [])
        payload["base_state_hash"] = module_hash(base)
        payload["expert_state_hashes"] = {eid:module_hash(model.experts[eid]) for eid in expert_ids}
        payload["selector_state_hash"] = module_hash(selector) if selector is not None else None
        payload["bundle_hash"] = digest(payload)
        atomic_json(temporary / "BUNDLE.json", payload, immutable=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(path)
    except BaseException:
        # Partial results are kept for diagnosis and never appear as a completed bundle.
        raise
    return payload


def load_bundle(path, device="cpu", *, shared_adapter=None):
    path = Path(path)
    data = validate_bundle_files(path)
    require(data["bundle_hash"] == digest({k: v for k, v in data.items() if k != "bundle_hash"}), "bundle manifest corrupt")
    require(data["registry_hash"] == digest(data["registry"]), "bundle registry corrupt")
    require(data["normalizer_hash"] == data["contract"]["normalizer_hash"], "bundle normalizer mismatch")
    require(data["selector_type"] in {"scorer", "scene_router"}, "deployment cannot access Oracle/real scores")
    for p, expected in data["weights"].items():
        require(file_hash(path / p) == expected, f"bundle weight changed: {p}")
    from omegaconf import OmegaConf
    from .query_base import build_query_framework
    c = data["contract"]
    if shared_adapter is None:
        base = build_query_framework(OmegaConf.create(c["source_config"]), c["query_architecture"], c["source_root"], c["source_commit"]).float()
        base.strip_auxiliary_heads()
        base.load_state_dict(torch.load(path / "base.pt", map_location="cpu", weights_only=True)["model"], strict=True)
        adapter = S0Adapter(base.to(device).eval(), c)
    else:
        require(digest(shared_adapter.contract) == digest(c) and not any(p.requires_grad for p in shared_adapter.parameters()), "previous-bundle shared S0 differs or is trainable")
        require(module_hash(shared_adapter.framework) == data["base_state_hash"], "previous-bundle shared encoder state mismatch")
        adapter = shared_adapter
    model = IQEModel(adapter)
    require(data["expert_ids"][0] == "expert_0" and data["expert_ids"] == sorted(set(data["expert_ids"]), key=lambda eid: int(eid[7:])), "bundle expert ID/order conflict")
    for eid in data["expert_ids"][1:]:
        model.append_expert(eid, data["architecture"][eid].get("variant", "independent"),
                            data["architecture"][eid].get("bottleneck", 32))
    for eid in model.experts:
        model.experts[eid].load_state_dict(torch.load(path / (eid + ".pt"), map_location="cpu", weights_only=True), strict=True)
    if data["selector_type"] == "scorer":
        if "selector.pt" in data["weights"]:
            require(data["scorer_architecture"] is not None, "Scorer architecture missing")
            model.scorer = TrajectoryScorer(**data["scorer_architecture"])
            model.scorer.load_state_dict(torch.load(path / "selector.pt", map_location="cpu", weights_only=True), strict=True)
        rule = SelectionRule(**data["rule"])
        require(rule.dependency_hash == data["pool_hash"], "threshold/candidate pool mismatch")
    else:
        width = c["query_architecture"]["width"]
        model.router = SceneRouter(width, width, data["expert_ids"], data["router_hidden"])
        model.router.load_state_dict(torch.load(path / "selector.pt", map_location="cpu", weights_only=True), strict=True)
        rule = RouterRule(**data["rule"])
        require(rule.dependency_hash == data["pool_hash"], "Router threshold/candidate pool mismatch")
    model.to(device).set_trainable_stage("inference")
    model.eval()
    return model, rule, data


def activate_bundle(root, bundle_path):
    root = Path(root)
    data = validate_bundle_files(bundle_path)
    require(data["mode"] == "full" and data["gate_results"]["passed"], "smoke/unaccepted bundle cannot become active")
    # Full checksum validation before atomically changing the pointer.
    for filename, expected in data["weights"].items():
        require(file_hash(Path(bundle_path) / filename) == expected, "activation corrupt weights")
    with lock(root / "ACTIVE.lock"):
        previous = read_json(root / "ACTIVE.json") if (root / "ACTIVE.json").exists() else None
        atomic_json(root / "ACTIVE.json", {"path": str(Path(bundle_path).resolve()), "bundle_hash": data["bundle_hash"], "previous": previous})


def rollback(root):
    root = Path(root)
    with lock(root / "ACTIVE.lock"):
        active = read_json(root / "ACTIVE.json")
        require(active["previous"] is not None, "no previous serving bundle")
        previous = active["previous"]
        data = validate_bundle_files(previous["path"])
        require(data["bundle_hash"] == previous["bundle_hash"], "previous bundle version conflict")
        for filename, expected in data["weights"].items():
            require(file_hash(Path(previous["path"]) / filename) == expected, "rollback corrupt weights")
        atomic_json(root / "ACTIVE.json", previous)
        return previous
