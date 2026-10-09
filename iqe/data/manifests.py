from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from ..contracts import SceneRecord, strict_record, require
from ..io import atomic_json, digest, read_json


def save_scenes(path, scenes, *, immutable=True):
    payload = {"schema_version": 1, "records": [asdict(s) for s in scenes]}
    payload["manifest_hash"] = digest(payload)
    atomic_json(path, payload, immutable=immutable)
    return payload["manifest_hash"]


def load_scenes(path):
    payload = read_json(path)
    require(payload["manifest_hash"] == digest({k: v for k, v in payload.items() if k != "manifest_hash"}), "scene manifest corrupted")
    records = [strict_record(SceneRecord, s) for s in payload["records"]]
    require(len({s.scene_id for s in records}) == len(records), "duplicate scene IDs")
    return records


def freeze_targets(records, *, external_targets=False):
    """GT wins; ambiguous equally ranked targets must be resolved before fitting."""
    grouped = {}
    quarantined = []
    for s in records:
        try:
            s.eligible_input(external_targets=external_targets)
        except ValueError as e:
            quarantined.append({"scene_id": s.scene_id, "reason": str(e)})
            continue
        grouped.setdefault(s.observation_hash, []).append(s)
    chosen = []
    for observation, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda s: (s.target_provenance != "gt", s.target_id, s.scene_id))
        priority = candidates[0].target_provenance
        best = [s for s in candidates if s.target_provenance == priority]
        require(len({s.target_id for s in best}) == 1, f"conflicting targets for observation {observation}; freeze an audited target choice")
        chosen.append(best[0])
    return chosen, quarantined
