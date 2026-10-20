"""Build immutable compact effect/repair datasets from completed G0 receipts."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from .common import artifact, digest, load_config, read_json, verify_artifact, write_json
from .feedback_schema import from_jsonable, label_arrays, validate_feedback
from .navsim_feedback_adapter import make_effect_pair
from .simulate_interventions import verified_feedback


def read_population(root, *, _on_verified_scene=None, _dependency_observer=None):
    root = Path(root)
    def observe(pin):
        if _dependency_observer is not None: _dependency_observer(pin)
    observe(artifact(root / "REGISTRATION.json"))
    registration = read_json(root / "REGISTRATION.json")
    if registration.get("split") != "train" or registration["identity"] != digest({k: v for k, v in registration.items() if k != "identity"}):
        raise ValueError("Only registered immutable TRAIN feedback is admitted")
    observe(artifact(root / "COMPLETE.json"))
    completion = read_json(root / "COMPLETE.json")
    if (completion["registration_identity"] != registration["identity"]
            or len(registration["population"]) != registration["max_samples"]
            or len(completion["scenes"]) != registration["max_samples"]):
        raise ValueError("Partial/foreign G0 cannot become a complete training dataset")
    output = []
    for item, pin in zip(registration["population"], completion["scenes"]):
        observe(pin)
        receipt = read_json(verify_artifact(pin))
        if receipt["registration_identity"] != registration["identity"]:
            raise ValueError("Scene completion belongs to another registration")
        observe(receipt["scene_record"])
        scene = read_json(verify_artifact(receipt["scene_record"]))
        entry = item["context"]
        if any(scene[k] != entry[k] for k in ("scene_id", "source_log", "role")) or scene["context_entry"] != entry:
            raise ValueError("G0 scene/context join mismatch")
        rows = scene.get("candidates", [])
        if (len(rows) != 26 or len({r["candidate_id"] for r in rows}) != 26
                or rows[0]["kind"] != "identity" or scene["registration_identity"] != registration["identity"]):
            raise ValueError("Incomplete/foreign intervention population")
        feedbacks = {} if _on_verified_scene is not None else None
        for row in rows:
            if "feedback" in row:
                observe(row["feedback"]);observe(row["canonical_receipt"])
                if _dependency_observer is not None:
                    native = read_json(verify_artifact(row["canonical_receipt"]))
                    observe(native["canonical"])
                    admitted = Path(row["canonical_receipt"]["path"]).parent / "ADMITTED.json"
                    observe({"path": str(admitted), "sha256": native["admission_sha256"]})
                checked = verified_feedback(row)
                if feedbacks is not None: feedbacks[row["candidate_id"]] = checked
        # Internal streaming consumer: validation is NEVER optional. Retain only
        # one scene's checked wrappers so the compact builder does not decode
        # the same native files again or cache the whole bank in RAM.
        if _on_verified_scene is not None: _on_verified_scene(scene, feedbacks)
        output.append(scene)
    if len({s["scene_id"] for s in output}) != len(output):
        raise ValueError("Duplicate scene in G0 population")
    return registration, output


def compact_labels(feedback):
    labels = label_arrays(feedback)
    return {"names": labels["names"], "units": labels["units"],
            "values": labels["values"].tolist(), "valid": labels["valid"].tolist()}


def build(root, *, _dependency_observer=None):
    root = Path(root)
    rows, repairs, scene_index, invalid = [], [], [], []
    def consume(scene, feedbacks):
        base_row = scene["candidates"][0]
        scene_index.append({"scene_id": scene["scene_id"], "source_log": scene["source_log"],
                            "role": scene["role"], "repair_status": scene["repair"]["status"]})
        if base_row["status"] != "VALID_SCORED":
            invalid.append({"scene_id": scene["scene_id"], "reason": "base_not_valid"})
            repairs.append({**scene_index[-1], "context_entry": scene["context_entry"],
                "observation_hash": scene["observation_hash"], "base_physical": base_row["physical"],
                "target_physical": None, "coefficients": None, "base_metrics": None, "selected_metrics": None,
                "base_labels": None, "selected_labels": None, "base_feedback": base_row.get("feedback"),
                "selected_feedback": None, "base_canonical_receipt": base_row.get("canonical_receipt"),
                "selected_canonical_receipt": None, "distance_m": None, "trainable": False, "target_kind": "unavailable"})
            return
        base = feedbacks[base_row["candidate_id"]]
        base_labels = compact_labels(base)
        by_id = {r["candidate_id"]: r for r in scene["candidates"]}
        for row in scene["candidates"][1:]:
            if row["status"] != "VALID_SCORED" or row["kind"] == "policy_resample":
                continue  # Policy support is analysed separately, not a controlled local intervention.
            other = feedbacks[row["candidate_id"]]
            begin = min((interval[0] for interval in row["active_intervals_seconds"]), default=0.)
            end = max((interval[1] for interval in row["active_intervals_seconds"]), default=0.)
            pair = make_effect_pair(base, other, basis_id=row["basis_sha256"], coefficients=row["coefficients"],
                active_interval_seconds=(begin, end), geometric_distance_m=row["distance_m"],
                executable=row["executable"], intervention_metadata={"kind": row["kind"],
                    "ramp_intervals_seconds": row["ramp_intervals_seconds"],
                    "active_intervals_seconds": row["active_intervals_seconds"]})
            delta = from_jsonable(pair["delta_labels"])
            temporal = from_jsonable(pair["delta_temporal"])
            compact = {"schema": "action_effect_compact_pair_v1", "pair_identity": pair["identity"],
                "scene_id": scene["scene_id"], "source_log": scene["source_log"], "role": scene["role"],
                "context_entry": scene["context_entry"], "observation_hash": scene["observation_hash"],
                "kind": row["kind"], "candidate_id": row["candidate_id"], "coefficients": row["coefficients"],
                "basis_id": row["basis_sha256"], "active_interval_seconds": [begin, end],
                "base_physical": base_row["physical"], "modified_physical": row["physical"],
                "base_labels": base_labels, "modified_labels": compact_labels(other),
                "delta_labels": {"names": delta["names"], "units": delta["units"],
                                 "values": delta["values"].tolist(), "valid": delta["valid"].tolist()},
                "temporal_event_delta": temporal["event_delta"].tolist(),
                "temporal_event_valid": temporal["event_valid"].tolist(),
                "temporal_event_names": temporal["event_names"],
                "event_transition": pair["event_transition"], "distance_m": row["distance_m"],
                "executable": row["executable"], "eligible_repair": row["eligible_repair"],
                "base_feedback": base_row["feedback"], "modified_feedback": row["feedback"]}
            compact["identity"] = digest(compact)
            rows.append(compact)
        selected = scene["repair"].get("selected")
        chosen = by_id.get(selected)
        trainable = chosen is not None and scene["repair"]["status"] in ("NO_CHANGE_REQUIRED", "VERIFIED_LOCAL_REPAIR")
        selected_labels = compact_labels(feedbacks[chosen["candidate_id"]]) if trainable else None
        repairs.append({**scene_index[-1], "context_entry": scene["context_entry"],
            "observation_hash": scene["observation_hash"], "base_physical": base_row["physical"],
            "target_physical": chosen["physical"] if trainable else None,
            "coefficients": chosen["coefficients"] if trainable else None,
            "base_metrics": base_row["metrics"], "selected_metrics": chosen["metrics"] if trainable else None,
            "base_labels": base_labels, "selected_labels": selected_labels,
            "base_feedback": base_row["feedback"], "selected_feedback": chosen["feedback"] if trainable else None,
            "base_canonical_receipt": base_row["canonical_receipt"],
            "selected_canonical_receipt": chosen["canonical_receipt"] if trainable else None,
            "distance_m": chosen["distance_m"] if trainable else None, "trainable": trainable,
            "target_kind": ("identity_preservation" if scene["repair"]["status"] == "NO_CHANGE_REQUIRED"
                            else "verified_local_repair") if trainable else "unavailable"})
    reg, scenes = read_population(root, _on_verified_scene=consume,
        _dependency_observer=_dependency_observer)
    common = {"registration": artifact(root / "REGISTRATION.json"), "population_completion": artifact(root / "COMPLETE.json"),
        "registration_identity": reg["identity"], "requested_scenes": len(scenes), "scene_index": scene_index,
        "base_checkpoint": reg["base_checkpoint"], "score_protocol_hash": reg["spec"]["score_protocol_hash"],
        "numeric_runtime_hash": reg["numeric_runtime_hash"], "invalid_scene_evidence": invalid,
        "split": "train", "deployment_inputs": False}
    pairs = {"schema": "action_effect_pair_dataset_v1", "status": "COMPLETE", **common,
        "sampling_contract": "uniform scene within fit role, then uniform valid pair within that scene; validation logs never fit",
        "entries": rows}
    pairs["identity"] = digest(pairs); write_json(root / "PAIR_DATASET.json", pairs)
    repair = {"schema": "action_effect_repair_dataset_v1", "status": "COMPLETE", **common, "entries": repairs,
        "sampling_contract": "scene uniform within fit; identity and verified repair target statuses retained"}
    repair["identity"] = digest(repair); write_json(root / "REPAIR_DATASET.json", repair)
    return {"status": "COMPLETE", "pair_scenes": len({r["scene_id"] for r in rows}), "pairs": len(rows),
            "requested_scenes": len(scenes), "trainable_repairs_and_keeps": sum(r["trainable"] for r in repairs),
            "pair_dataset": artifact(root / "PAIR_DATASET.json"), "repair_dataset": artifact(root / "REPAIR_DATASET.json")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--max-samples", type=int, required=True)
    p.add_argument("--g0-root"); p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if not 1 <= args.max_samples <= cfg["cohort_size"]: raise ValueError("Invalid scene budget")
        root = Path(args.g0_root) if args.g0_root else Path(cfg["artifact_root"]) / "g0" / f"population_{args.max_samples:06d}"
        if args.dry_run: print(json.dumps({"status": "NOT_RUN", "simulation_calls": 0})); return 0
        from .verified_data import build_and_validate_datasets
        print(json.dumps(build_and_validate_datasets(root))); return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})); return 1


if __name__ == "__main__": raise SystemExit(main())
