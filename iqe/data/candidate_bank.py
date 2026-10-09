from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import io
import numpy as np
from ..contracts import CandidateRecord, ScoreRecord, strict_record, require
from ..io import atomic_bytes, atomic_json, digest, file_hash, read_json, lock


class CandidateBank:
    def __init__(self, root):
        self.root = Path(root)

    def put_candidate(self, record):
        path = self.root / "candidates" / (record.key + ".json")
        require(file_hash(record.trajectory_ref) == record.trajectory_hash, "trajectory checksum")
        atomic_json(path, asdict(record), immutable=True)

    def candidate(self, key):
        record = strict_record(CandidateRecord, read_json(self.root / "candidates" / (key + ".json")))
        require(key == record.key and file_hash(record.trajectory_ref) == record.trajectory_hash, "stale candidate")
        return record

    def score_key(self, candidate, protocol_hash, context_hash, reference_hash, seed, repetition):
        return digest({"candidate": candidate.key, "protocol": protocol_hash, "context": context_hash,
                       "reference": reference_hash, "seed": seed, "repetition": repetition})

    def put_score(self, record):
        identity = {k: getattr(record, k) for k in ("candidate_key", "score_protocol_hash", "metric_context_hash",
                                                     "official_reference_hash", "eval_seed", "repetition_index")}
        if not record.score_valid:
            # Failed attempts are retained separately and cannot poison the completed label key.
            identity["failed_attempt"] = record.error_reason
        key = digest(identity)
        atomic_json(self.root / "scores" / (key + ".json"), asdict(record), immutable=True)
        return key

    def score(self, key, candidate, protocol_hash, context_hash):
        score = strict_record(ScoreRecord, read_json(self.root / "scores" / (key + ".json")))
        require(score.candidate_key == candidate.key and score.score_protocol_hash == protocol_hash
                and score.metric_context_hash == context_hash, "stale labels or scoring context")
        return score

    def pool(self, path, scenes, experts, protocol_hash):
        payload = {"schema_version": 1, "scenes": scenes, "experts": experts, "protocol_hash": protocol_hash}
        payload["pool_hash"] = digest(payload)
        atomic_json(path, payload, immutable=True)
        return payload

    def validate_pool(self, manifest, scene_records, experts):
        require(manifest["pool_hash"] == digest({k: v for k, v in manifest.items() if k != "pool_hash"}), "pool hash mismatch")
        require(manifest["experts"] == experts, "pool registry/checkpoint mismatch")
        lookup = {s.scene_id: s for s in scene_records}
        for scene_id, rows in manifest["scenes"].items():
            require(scene_id in lookup, "pool contains unknown scene")
            require(set(rows) == set(experts), "pool lacks required old/new expert replay")
            for eid, keys in rows.items():
                c = self.candidate(keys["candidate_key"])
                require(c.scene_id == scene_id and c.expert_id == eid and c.expert_checkpoint_hash == experts[eid], "candidate ID/hash join")
                if "score_key" in keys:
                    self.score(keys["score_key"], c, manifest["protocol_hash"], lookup[scene_id].metric_context_hash)
        return True


def save_trajectory(path, trajectory):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, trajectory=np.asarray(trajectory))
    atomic_bytes(path, buffer.getvalue(), immutable=True)
    return file_hash(path)
