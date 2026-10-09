"""Real original-trajectory records. Metric contexts joined by token/log, never row number."""
from __future__ import annotations
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from ..contracts import SceneRecord, require
from ..io import read_json, file_hash, digest
from ..query_base import bind_source


def metric_index(path):
    path = Path(path)
    if path.suffix == ".json":
        rows = read_json(path)
        if isinstance(rows, dict):
            rows = rows.get("records", rows.get("scenes", []))
        return {r.get("token", r.get("scene_id")): {"path": r.get("cache_path", r.get("metric_context_ref")), "log": r.get("log", r.get("source_log_id"))} for r in rows}
    rows = list(csv.DictReader(path.open()))
    result = {}
    for r in rows:
        p = Path(r["file_name"])
        require(p.parent.name not in result, "duplicate metric cache token")
        result[p.parent.name] = {"path": str(p), "log": p.parents[2].name}
    return result


def import_current(root, metric_metadata, role, *, limit=None, source_commit="", scene_ids=None, workers=1):
    root = Path(root)
    identity = read_json(root / "identity.json")
    require(identity["split"] == ("train" if role == "incremental_fit" else "dev"), "original split identity mismatch")
    index = read_json(root / "index.json")
    require(identity["index_sha256"] == digest(index), "original scene index changed")
    metrics = metric_index(metric_metadata)
    require(type(workers) is int and workers > 0, "positive import I/O worker budget")
    selected = [row for row in index if scene_ids is None or row["token"] in scene_ids]
    selected = selected[:limit or len(selected)]
    def import_one(row):
        token, log = row["token"], row["log"]
        obs_ref, target_ref = root / "current" / (token + ".json"), root / "ego" / (token + ".pt")
        metric = metrics.get(token)
        if metric is None:
            return None
        require(metric["log"] == log, "metric context scene/log mismatch")
        observation = read_json(obs_ref)
        require(observation["token"] == token and observation["identity"] == identity["identity"], "observation identity changed")
        require(len(observation["image_paths"]) == 3 and all(Path(p).is_file() for p in observation["image_paths"]), "real current images unavailable")
        # Observation hash binds image contents plus legal current metadata. Future labels excluded.
        obs_hash = digest({"record": observation, "images": {p: file_hash(p) for p in observation["image_paths"]}})
        target_hash = file_hash(target_ref)
        return SceneRecord(1, token, log, log, obs_hash, role, "original", token, str(obs_ref),
                                   target_hash, str(target_ref), "gt", "verified", metric["path"], file_hash(metric["path"]),
                                   "", "no", 0)
    # Ordered map preserves the exact serial ID/hash manifest while overlapping
    # independent immutable file reads. Every image/target/context is still hashed.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(import_one, selected))
    missing = [row["token"] for row, record in zip(selected, results) if record is None]
    require(not missing, f"real metric contexts missing for {len(missing)} selected scenes; first={missing[:8]}")
    return results


def load_observation(scene, contract):
    bind_source(contract["source_root"], contract["source_commit"])
    from starVLA.dataloader.foresight_dataset import current_observation
    observation = read_json(scene.observation_ref)
    require(observation["token"] in {scene.scene_id, scene.source_scene_id}, "observation scene join mismatch")
    require(digest({"record": observation, "images": {p: file_hash(p) for p in observation["image_paths"]}}) == scene.observation_hash,
            "observation/images changed since manifest freeze")
    result = current_observation(observation)
    result["token"] = scene.scene_id
    return result


def load_target(scene, trajectory_contract):
    require(file_hash(scene.target_ref) == scene.target_id, "target changed since manifest freeze")
    label = torch.load(scene.target_ref, map_location="cpu", weights_only=True)
    require(label["token"] in {scene.scene_id, scene.source_scene_id}, "target ID join mismatch")
    raw = label["ego"].float()
    require(raw.shape == (trajectory_contract.horizon, trajectory_contract.raw_dim) and bool(torch.isfinite(raw).all()), "target geometry/time invalid")
    return raw


def validate_incoming(scene, contract, trajectory_contract):
    scene.eligible_input(external_targets=contract.get("external_targets_enabled", False))
    load_observation(scene, contract)
    load_target(scene, trajectory_contract)
    require(file_hash(scene.metric_context_ref) == scene.metric_context_hash, "incoming metric context changed")
    if scene.source_kind == "synthetic":
        evidence = read_json(scene.consistency_evidence_ref)
        require(evidence.get("status") == "verified" and evidence.get("observation_hash") == scene.observation_hash
                and evidence.get("metric_context_hash") == scene.metric_context_hash and evidence.get("source_scene_id") == scene.source_scene_id,
                "synthetic collection/render evidence does not bind actual observation and metric context")
        require(bool(evidence.get("acquisition_or_renderer")), "synthetic observation needs a real collection/render source")
    return {"scene_id": scene.scene_id, "observation": "verified", "metric_context": "hash_verified", "target": "finite_shape_verified"}
