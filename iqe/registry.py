"""Append-only expert events. Rejection history can only be superseded by a new event."""
from __future__ import annotations
from .contracts import require
from .io import atomic_json, read_json, digest, lock
from pathlib import Path

STATUSES = {"training", "frozen", "candidate_only", "served", "rejected"}


class ExpertRegistry:
    def __init__(self, path, shared_contract_hash):
        self.path = Path(path)
        self.shared_contract_hash = shared_contract_hash

    def read(self):
        if not self.path.exists():
            return []
        payload = read_json(self.path)
        require(payload["shared_contract_hash"] == self.shared_contract_hash, "registry S0/shared contract conflict")
        require(payload["registry_hash"] == digest(payload["events"]), "registry event checksum")
        return payload["events"]

    def append(self, event):
        required = {"expert_id", "created_round", "parent", "architecture", "checkpoint_hash", "train_manifest_hash",
                    "shared_contract_hash", "status", "gates"}
        require(set(event) == required and event["status"] in STATUSES, "expert event schema")
        require(event["shared_contract_hash"] == self.shared_contract_hash, "expert shared dependency")
        with lock(str(self.path) + ".registry.lock"):
            events = self.read()
            existing = self.latest(events)
            eid = event["expert_id"]
            if eid not in existing:
                require(eid == f"expert_{event['created_round']}" and
                        (not existing and eid == "expert_0" or bool(existing) and event["created_round"] > max(e["created_round"] for e in existing.values())),
                        "expert IDs must match their append-only creation round")
            else:
                for key in ("created_round", "parent", "architecture", "train_manifest_hash", "shared_contract_hash"):
                    require(event[key] == existing[eid][key], f"expert immutable metadata conflict {key}")
                if existing[eid]["status"] != "training":
                    require(event["checkpoint_hash"] == existing[eid]["checkpoint_hash"], "frozen expert checkpoint immutable")
            if events and events[-1] == event:
                return
            events.append(event)
            atomic_json(self.path, {"shared_contract_hash": self.shared_contract_hash, "events": events, "registry_hash": digest(events)})

    @staticmethod
    def latest(events):
        return {e["expert_id"]: e for e in events}

    def candidate_pool(self):
        return {eid: e["checkpoint_hash"] for eid, e in self.latest(self.read()).items() if e["status"] in {"frozen", "candidate_only", "served"}}
