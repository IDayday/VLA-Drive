"""Derive diagnostic scope from an immutable, already audited Flow TRAIN list.

PROFILE preserves every entry and its order. SMOKE uses a score-independent
prefix of at most 32 entries. This changes metadata only, never hashes/targets
of an observation; normal runtime verification of accessed assets still runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash


def derive_manifest(parent_path, expected_sha256, *, scope, max_scenes=None):
    path = Path(parent_path).resolve()
    if scope not in ("SMOKE", "PROFILE"):
        raise ValueError("Diagnostic derivation cannot create a FORMAL manifest")
    if scope == "SMOKE" and (type(max_scenes) is not int or not 1 <= max_scenes <= 32):
        raise ValueError("SMOKE needs an explicit prefix budget in [1,32]")
    if scope == "PROFILE" and max_scenes is not None:
        raise ValueError("PROFILE preserves the full original population and order")
    if path.stat().st_size > 256 * 2**20:
        raise ValueError("Manifest exceeds this bounded metadata derivation budget")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Parent manifest checksum changed")
    parent = json.loads(raw)
    if (parent.get("schema") != "st_flow_training_manifest_v1"
            or parent.get("scope") != "FORMAL" or parent.get("split") != "train"
            or "diagnostic_derivation" in parent
            or identity_hash({k: v for k, v in parent.items() if k != "identity"}) != parent.get("identity")):
        raise ValueError("An intact original FORMAL TRAIN manifest is required")
    entries = parent["entries"]
    if not entries or len({e["scene_id"] for e in entries}) != len(entries):
        raise ValueError("Empty or duplicate parent population")
    if scope == "SMOKE" and len(entries) < max_scenes:
        raise ValueError("Requested smoke prefix exceeds parent population")
    result = {k: v for k, v in parent.items() if k != "identity"}
    result["scope"] = scope
    result["entries"] = entries if scope == "PROFILE" else entries[:max_scenes]
    result["diagnostic_derivation"] = {
        "parent": {"path": str(path), "sha256": expected_sha256},
        "parent_identity": parent["identity"], "parent_scene_count": len(entries),
        "rule": "full_population_same_order" if scope == "PROFILE" else "fixed_manifest_prefix_before_scores",
        "scores_read": False, "weights_usable_for_formal_init": False,
        "entries_sha256": identity_hash(result["entries"]),
    }
    result["identity"] = identity_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--scope", choices=("SMOKE", "PROFILE"), required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = derive_manifest(args.parent, args.parent_sha256,
                               scope=args.scope, max_scenes=args.max_scenes)
    output = Path(args.output).resolve()
    atomic_json(output, manifest, immutable=True)
    print(json.dumps({"status": "PREPARED", "scope": args.scope,
        "scenes": len(manifest["entries"]), "identity": manifest["identity"],
        "path": str(output), "scores_read": False, "model_loaded": False}))


if __name__ == "__main__":
    main()
