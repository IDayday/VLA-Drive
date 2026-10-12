"""Resolve latest once; offline technical binding is not a deployment release."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import torch
from iqe.io import atomic_json, atomic_torch, read_json, file_hash, digest, lock
from iqe.contracts import require
from iqe.data.manifests import load_scenes, save_scenes
from iqe.training.resume import inspect_checkpoint


def binding(config):
    b = read_json(config.root / "BASE_BINDING.json")
    require(b["experiment_id"] == config.experiment_id and b["config_hash"] == config.identity, "experiment identity changed")
    require(file_hash(b["checkpoint"]) == b["checkpoint_hash"], "bound Base changed")
    return b


def bind(config):
    with lock(config.root / "bind.lock"):
        if (config.root / "BASE_BINDING.json").exists():
            return binding(config)
        original = read_json(config.contract)
        root = Path(config.base_run).resolve()
        if config.checkpoint == "latest_complete":
            receipts = [read_json(p) | {"path": str(p).removesuffix(".COMPLETE.json")} for p in root.glob("step_*.pt.COMPLETE.json")]
            require(receipts, "no complete Base checkpoint receipts")
            # Select before reading any score. A broken newest receipt is an error,
            # not an unreported fallback to an earlier/possibly better checkpoint.
            path = max(receipts, key=lambda r: r["optimizer_step"])["path"]
        else:
            path = str(Path(config.checkpoint).resolve())
        require(Path(path).parent == root, "checkpoint must belong to the explicitly declared own Base run")
        inspected = inspect_checkpoint(path)
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(state["kind"] == "iqe_query_base" and state["optimizer_step"] > 1, "learned own Query Base required")
        require(state["framework_contract_hash"] == original["framework_contract_hash"], "Base framework contract mismatch")
        require(state.get("mode") not in {"smoke", "profile"}, "diagnostic checkpoint is not the experiment Base")
        require(state.get("mode") == "full" and state["optimizer"].get("state"), "full own training optimizer provenance required")
        repo = Path(__file__).resolve().parents[2]
        for name, expected in (original["query_model_code_hashes"] | original["reused_M0_module_hashes"]).items():
            require(file_hash(repo / name) == expected, "protected Base code changed: " + name)
        scenes = load_scenes(config.scene_manifest)
        from iqe.data.splits import validate_isolation
        validate_isolation(scenes)
        require(all(s.split_role in {"incremental_fit", "stage_val", "selector_cal", "dev_report"} for s in scenes),
                "this experiment must not open final_test/navtest")
        fit_seen = set(state.get("fit_scene_ids", []))
        chosen = []
        for role, cap in (("incremental_fit", None if config.fit_limit == "all_fit" else config.fit_limit), ("stage_val", config.val_limit),
                          ("selector_cal", config.cal_limit), ("dev_report", None)):
            rows = sorted((s for s in scenes if s.split_role == role),
                          key=lambda s: digest([config.selection_seed, role, s.scene_id]))
            require(rows, "missing inherited role: " + role)
            for s in rows[:cap]:
                require(s.source_kind == "original" and s.target_provenance == "gt" and s.input_consistency_status == "verified",
                        "baseline only accepts verified original observations and GT")
                chosen.append(replace(s, s0_seen=("yes" if s.scene_id in fit_seen else "no") if fit_seen else "unknown"))
        save_scenes(config.root / "scenes.json", chosen)
        atomic_json(config.root/"DATA_ISOLATION.json", validate_isolation(chosen), immutable=True)
        contract = dict(original, query_checkpoint=path, query_checkpoint_hash=inspected["checksum"],
                        binding_status="LOCKED_QUERY_S0", offline_experiment_binding=True,
                        permits_deployment=False, learned_driving_weights_loaded=True)
        atomic_json(config.root / "OFFLINE_BASE_CONTRACT.json", contract, immutable=True)
        head = {k.removeprefix("action_model.expert."): v.clone() for k, v in state["model"].items()
                if k.startswith("action_model.expert.")}
        require(head and "query.weight" in head, "singleton Query branch missing")
        atomic_torch(config.root / "BASE_HEAD.pt", head, immutable=True)
        counts = {r: sum(s.split_role == r for s in chosen) for r in {s.split_role for s in chosen}}
        b = {"schema": "iqe_internal_baseline_binding_v1", "experiment_id": config.experiment_id,
             "config_hash": config.identity, "checkpoint": path, "checkpoint_hash": inspected["checksum"],
             "optimizer_step": state["optimizer_step"], "parent_run_identity": inspected["run_identity"],
             "framework_contract_hash": original["framework_contract_hash"], "head_hash": file_hash(config.root / "BASE_HEAD.pt"),
             "contract_hash": digest(contract), "scene_manifest_hash": file_hash(config.root / "scenes.json"),
             "source_scene_manifest": config.scene_manifest, "source_scene_manifest_hash": file_hash(config.scene_manifest),
             "role_counts": counts, "base_exposure": {r: {v: sum(s.split_role == r and s.s0_seen == v for s in chosen)
                                                        for v in ("yes", "no", "unknown")} for r in counts},
             "selection_rule": config.checkpoint, "bound_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
             "original_training_imported_driving_weights": state.get("learned_driving_weights_loaded"),
             "own_driving_optimizer_updates": state["optimizer_step"],
             "inference_runtime": {"precision": original["precision"], "torch": str(torch.__version__),
                                   "cuda": torch.version.cuda, "visual": "legacy_fp32", "attention": "sdpa",
                                   "tf32_matmul": False, "tf32_cudnn": False, "augmentation": "disabled"},
             "production_active_changed": False, "navtest_accessed": False}
        atomic_json(config.root / "BASE_BINDING.json", b, immutable=True)
        return b


def load_head(config, device="cpu"):
    from omegaconf import OmegaConf
    from iqe.query_base import QueryActionBridge
    c = read_json(config.root / "OFFLINE_BASE_CONTRACT.json")
    head = QueryActionBridge(OmegaConf.create(c["source_config"]), c["query_architecture"]).expert
    require(file_hash(config.root / "BASE_HEAD.pt") == read_json(config.root / "BASE_BINDING.json")["head_hash"], "Base head changed")
    head.load_state_dict(torch.load(config.root / "BASE_HEAD.pt", map_location="cpu", weights_only=True), strict=True)
    return head.to(device)
