"""DEV-only immutable context admission. Never accepted by the TRAIN reader.

Current-only feature records contain no target, score, simulator or metric cache.
Offline metric references live in the separate evaluation registration only.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .common import artifact, digest, read_json, verify_artifact, verify_identity
from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash

REGISTRATION_SCHEMA = "action_effect_evaluation_registration_v1"
MANIFEST_SCHEMA = "action_effect_evaluation_context_manifest_v1"
SHARD_SCHEMA = "action_effect_evaluation_context_shard_v1"
FEATURE_SCHEMA = "action_effect_evaluation_features_v1"
FULL_SCENES, FULL_LOGS = 1696, 16
FEATURE_FIELDS = {"schema", "scene_id", "source_log", "role", "split", "observation_hash", "base_binding_hash",
                  "evaluation_binding", "memory", "ego", "condition", "W_count", "raw", "physical", "noise",
                  "raw_hash", "physical_hash", "historical_physical_exact"}
INDEX_FIELDS = ("scene_id", "source_log", "role", "split", "observation_hash")


def validate_registration(reg):
    verify_identity(reg)
    rows = reg["population"]
    if (reg.get("schema") != REGISTRATION_SCHEMA or reg.get("split") != "dev" or reg.get("role") != "dev_report"
            or reg.get("for_training") is not False or reg.get("requested_scenes") != FULL_SCENES
            or reg.get("requested_logs") != FULL_LOGS or len(rows) != FULL_SCENES
            or len({r["scene_id"] for r in rows}) != FULL_SCENES
            or len({r["source_log"] for r in rows}) != FULL_LOGS
            or reg.get("train_dev_log_overlap") != 0 or reg.get("dtype") != "float32"
            or type(reg.get("shard_scenes")) is not int or not 1 <= reg["shard_scenes"] <= 32):
        raise ValueError("Only the fixed complete DEV1696/16log evaluation population is admitted")
    if any(r["role"] != "dev_report" or r["split"] != "dev" or any(x in r["scene_id"] for x in ("/", "\\", "..")) for r in rows):
        raise ValueError("DEV observations must not become fit/validation/navtest inputs")
    return reg


def shard_population(reg):
    n = reg["shard_scenes"]
    return [reg["population"][start:start+n] for start in range(0, len(reg["population"]), n)]


def validate_feature_record(record, expected, reg, *, compare_historical=False):
    if not isinstance(record, dict) or set(record) != FEATURE_FIELDS or record["schema"] != FEATURE_SCHEMA:
        raise ValueError("DEV feature whitelist excludes targets, labels, feedback and metric inputs")
    if (any(record[k] != expected[k] for k in INDEX_FIELDS)
            or record["base_binding_hash"] != reg["base_checkpoint_binding_hash"]
            or record["evaluation_binding"] != reg["identity"] or record["W_count"] != 144
            or record["historical_physical_exact"] is not True):
        raise ValueError("DEV feature/Base/observation identity mismatch")
    shapes = {"memory": (152,2048), "ego": (1,4), "condition": (8,2048),
              "raw": (8,4), "physical": (8,3), "noise": (8,4)}
    for key, shape in shapes.items():
        x = record[key]
        if (not isinstance(x, torch.Tensor) or tuple(x.shape) != shape or x.dtype != torch.float32
                or x.device.type != "cpu" or x.requires_grad or not torch.isfinite(x).all()):
            raise ValueError("Invalid frozen FP32 DEV feature "+key)
    if (not torch.equal(record["condition"], record["memory"][144:])
            or tensor_hash(record["raw"][None]) != record["raw_hash"]
            or tensor_hash(record["physical"][None]) != record["physical_hash"]):
        raise ValueError("DEV planner condition or trajectory bytes changed")
    from tools.foresight.checkpoints import scene_noise
    if not torch.equal(record["noise"], scene_noise(record["scene_id"], reg["seed"], "cpu")[0]):
        raise ValueError("DEV inference noise differs from the registered original scene seed")
    if compare_historical:
        path = verify_artifact(expected["historical_prediction"])
        with np.load(path, allow_pickle=False) as saved:
            trajectory = saved["trajectory"]
        if trajectory.dtype != np.float32 or not np.array_equal(record["physical"].numpy(), trajectory):
            raise ValueError("S0 DEV physical trajectory differs from the registered original FP32 baseline")
    return record


def validate_shard(done, reg, expected, *, verify_payload=True):
    verify_identity(done)
    if (done.get("schema") != SHARD_SCHEMA or done.get("evaluation_binding") != reg["identity"]
            or done.get("split") != "dev" or done.get("for_training") is not False
            or done.get("single_encoder_per_scene") is not True or done.get("historical_physical_exact") is not True
            or len(done.get("entries", [])) != len(expected)):
        raise ValueError("Incomplete or foreign DEV shard receipt")
    tag = digest([reg["identity"], [r["scene_id"] for r in expected]])
    if Path(done["shard"]["path"]).name != tag+".pt": raise ValueError("DEV shard name does not bind its population")
    for offset, (row, original) in enumerate(zip(done["entries"], expected)):
        if (any(row[k] != original[k] for k in INDEX_FIELDS) or row["shard_offset"] != offset or row["shard"] != done["shard"]):
            raise ValueError("DEV shard row/offset/order mismatch")
    if verify_payload: verify_artifact(done["shard"])
    return done


def validate_manifest(manifest, reg, *, verify_payload=True):
    verify_identity(manifest); validate_registration(reg)
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "COMPLETE"
            or manifest.get("split") != "dev" or manifest.get("role") != "dev_report"
            or manifest.get("for_training") is not False or manifest.get("dtype") != "float32"
            or manifest.get("evaluation_binding") != reg["identity"]
            or manifest.get("base_checkpoint_binding_hash") != reg["base_checkpoint_binding_hash"]
            or manifest.get("numeric_runtime_hash") != reg["numeric_runtime_hash"]
            or manifest.get("requested_scenes") != FULL_SCENES or manifest.get("requested_logs") != FULL_LOGS
            or len(manifest.get("entries",[])) != FULL_SCENES
            or len(manifest.get("shard_receipts",[])) != len(shard_population(reg))):
        raise ValueError("Partial/foreign DEV context must not masquerade as a complete evaluation manifest")
    joined = []
    for pin, expected in zip(manifest["shard_receipts"], shard_population(reg)):
        done = validate_shard(read_json(verify_artifact(pin)), reg, expected, verify_payload=verify_payload)
        joined.extend(done["entries"])
    if joined != manifest["entries"]: raise ValueError("DEV manifest differs from complete immutable shard inventory")
    return manifest


class EvaluationContextReader:
    def __init__(self, manifest, binding_path=None, max_shards=2):
        if type(max_shards) is not int or not 1 <= max_shards <= 8:
            raise ValueError("Explicit bounded DEV reader LRU required")
        self.manifest_path = Path(manifest)
        self.manifest = verify_identity(read_json(manifest))
        if self.manifest.get("schema") != MANIFEST_SCHEMA or self.manifest.get("for_training") is not False:
            raise ValueError("EvaluationContextReader accepts only DEV-only evaluation contexts")
        registered_path = verify_artifact(self.manifest["registration"])
        if binding_path is not None and artifact(binding_path) != self.manifest["registration"]:
            raise ValueError("Explicit DEV registration differs from its manifest")
        self.binding = validate_registration(read_json(registered_path))
        validate_manifest(self.manifest, self.binding)
        self.rows = {r["scene_id"]:r for r in self.manifest["entries"]}
        self.population = {r["scene_id"]:r for r in self.binding["population"]}
        self.cache = OrderedDict(); self.checked = {}; self.max_shards=max_shards; self.loads=0

    def __getitem__(self, scene_id):
        r=self.rows[scene_id]; p=Path(r["shard"]["path"]); s=p.stat()
        stamp=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if self.checked.get(str(p)) != stamp:
            verify_artifact(r["shard"]); self.checked[str(p)]=stamp; self.cache.pop(str(p),None)
        if str(p) not in self.cache:
            self.cache[str(p)]=torch.load(p,map_location="cpu",weights_only=False); self.loads+=1
            while len(self.cache)>self.max_shards:self.cache.popitem(last=False)
        self.cache.move_to_end(str(p)); records=self.cache[str(p)]
        if r["shard_offset"] >= len(records):raise ValueError("DEV shard offset out of range")
        return validate_feature_record(records[r["shard_offset"]],self.population[scene_id],self.binding,compare_historical=True)
