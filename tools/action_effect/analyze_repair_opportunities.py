"""Report complete G0 denominators, local/support opportunity and new failures."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .common import artifact, digest, load_config, read_json, write_json
from .build_effect_pairs import read_population
from .feedback_schema import canonical_record
from .simulate_interventions import load_completed_attempt, verified_feedback
from starVLA.model.modules.st_world.consequence import ConsequenceRequest


def stage_cost(root, reg, scenes):
    folders = {Path(row["canonical_receipt"]["path"]).parent.parent
               for scene in scenes for row in scene["candidates"] if "canonical_receipt" in row}
    known, completed, scored, wall, cpu = 0, 0, 0, 0., 0.
    for folder in folders:
        for admission_path in folder.glob("attempt_*/ADMITTED.json"):
            admission = read_json(admission_path)
            if admission.get("ledger_identity") != reg["identity"]: continue
            known += 1
            if not (admission_path.parent / "COMPLETE.json").exists(): continue
            native = load_completed_attempt(admission_path.parent, ConsequenceRequest(
                **{**admission["request_identity"], "trajectory": []}))
            # Request identity excludes trajectory, so this safely validates the
            # saved record without generating/reinterpreting a prediction.
            cost = native.get("cost", {})
            completed += 1; scored += cost.get("scoring_calls", 0)
            wall += cost.get("worker_wall_seconds", 0.)
            cpu += cost.get("worker_cpu_user_seconds", 0.) + cost.get("worker_cpu_system_seconds", 0.)
    counter = read_json(Path(root) / "calls" / "COUNTER.json")
    return {**counter, "matching_admitted_attempts": known, "completed_native_attempts": completed,
        "recorded_actual_scoring_calls": scored, "unfinished_admissions": known-completed,
        "conservatively_reserved_slots_without_admission": counter["reserved_calls"]-known,
        "summed_new_native_worker_wall_seconds": wall, "summed_new_native_worker_cpu_seconds": cpu,
        "scope": "only admissions charged to this registration; historical cache reuse not charged again; no fabricated GPU cost"}


def analyze(root):
    root = Path(root); reg, scenes = read_population(root)
    statuses = Counter(s["repair"]["status"] for s in scenes)
    cohorts, kinds = {}, {}
    failures = 0; valid_base = 0; invalid_candidates = Counter(); repairs = []
    for scene in scenes:
        rows = scene["candidates"]; base = rows[0]
        invalid_candidates.update(r["status"] for r in rows if r["status"] != "VALID_SCORED")
        if base["status"] != "VALID_SCORED": continue
        valid_base += 1; score = base["metrics"]["score"]
        label = "zero" if score <= 1e-8 else "low_below_0.8" if score < .8 else "high_at_least_0.8"
        bucket = cohorts.setdefault(label, {"scenes": 0, "verified_local_repairs": 0, "support_only": 0})
        bucket["scenes"] += 1
        bucket["verified_local_repairs"] += scene["repair"]["status"] == "VERIFIED_LOCAL_REPAIR"
        bucket["support_only"] += scene["repair"]["status"] == "REQUIRES_LARGER_SUPPORT_SEARCH"
        failures += scene["repair"]["status"] != "NO_CHANGE_REQUIRED"
        if scene["repair"]["status"] == "VERIFIED_LOCAL_REPAIR":
            chosen = next(r for r in rows if r["candidate_id"] == scene["repair"]["selected"])
            native = canonical_record(verified_feedback(base))
            first_events = {"collision": native["official_first_at_fault_time_seconds"],
                            "ttc": native["official_first_ttc_state_time_seconds"],
                            "road": native["consequences"]["road"]["first_time_seconds"]}
            onset = min(x[0] for x in chosen["active_intervals_seconds"])
            repairs.append({"scene_id": scene["scene_id"], "role": scene["role"], "kind": chosen["kind"],
                            "distance_m": chosen["distance_m"], "ramp_intervals_seconds": chosen["ramp_intervals_seconds"],
                            "active_intervals_seconds": chosen["active_intervals_seconds"],
                            "base_first_event_times_seconds": first_events,
                            "observed_event_minus_intervention_onset_seconds": {
                                k: value-onset if value is not None else None for k, value in first_events.items()},
                            "credit": "not_assigned_by_event_time"})
        for row in rows[1:]:
            if row["status"] != "VALID_SCORED": continue
            item = kinds.setdefault(row["kind"], {"valid_pairs": 0, "improve_score": 0, "worsen_score": 0,
                "eligible_repair": 0, "high_to_zero": 0, "new_NC": 0, "new_DAC": 0, "new_TTC": 0,
                "score_delta_sum": 0.})
            item["valid_pairs"] += 1
            delta = row["metrics"]["score"]-score
            item["score_delta_sum"] += delta
            item["improve_score"] += delta > 1e-8; item["worsen_score"] += delta < -1e-8
            item["eligible_repair"] += row["eligible_repair"]
            item["high_to_zero"] += score >= .8 and row["metrics"]["score"] <= 1e-8
            for short, key in (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"), ("TTC", "time_to_collision_within_bound")):
                item["new_"+short] += base["metrics"][key] >= 1.-1e-8 and row["metrics"][key] < 1.-1e-8
    for item in kinds.values(): item["mean_score_delta"] = item["score_delta_sum"]/item["valid_pairs"]
    curve = []
    for budget in (1, 5, 9, 17, 24, 26):
        local, support = 0, 0
        for scene in scenes:
            if scene["repair"]["status"] == "NO_CHANGE_REQUIRED": continue
            eligible = [r for r in scene["candidates"][:budget] if r.get("eligible_repair")]
            local += any(r["kind"] not in ("identity", "larger_support", "policy_resample") for r in eligible)
            support += any(r["kind"] in ("larger_support", "policy_resample") for r in eligible)
        curve.append({"proposal_slots_including_identity_and_geometric_rejections": budget,
                      "local_repair_scenes": local, "support_success_scenes": support})
    result = {"schema": "action_effect_g0_opportunity_analysis_v1", "status": "COMPLETE",
        "registration": artifact(root / "REGISTRATION.json"), "requested_scenes": len(scenes),
        "source_logs": len({s["source_log"] for s in scenes}), "valid_base_scenes": valid_base,
        "role_counts": dict(Counter(s["role"] for s in scenes)), "execution_cost": stage_cost(root, reg, scenes),
        "base_failure_scenes": failures, "repair_status_counts": dict(statuses), "score_units": "0..1",
        "local_repair_rate_among_base_failures": statuses["VERIFIED_LOCAL_REPAIR"]/failures if failures else None,
        "base_score_cohorts": cohorts, "candidate_status_counts": dict(invalid_candidates), "by_intervention_kind": kinds,
        "proposal_budget_curve": curve, "verified_local_repairs": repairs,
        "continuous_risk_margin_cohort": "UNAVAILABLE_NATIVE_CLEARANCE_NOT_EXTRACTED",
        "event_time_vs_causal_credit": "native first event is an observation, not responsible intervention time",
        "generalization": "TRAIN-log internal fit/validation roles only; not full DEV or Navtest gain",
        "no_repair_meaning": "No verified local repair within this fixed budget; not mathematical irreparability"}
    result["identity"] = digest(result); write_json(root / "G0_RESULTS.json", result)
    lines = ["# G0 actual repair opportunity", "", f"Requested scenes: {len(scenes)}; valid original feedback: {valid_base}; logs: {result['source_logs']}.",
        "Score units: 0–1. This is a controlled TRAIN-cohort diagnostic, not deployment/generalization evidence.", "",
        f"Repair statuses: `{dict(statuses)}`.", f"Verified local repairs / original failures: {statuses['VERIFIED_LOCAL_REPAIR']} / {failures}.",
        "", "Larger-support interventions and additional original-policy samples are reported separately from local repair.",
        "Geometric rejection, duplicate proposals, invalid predictions and scoring failures are not valid zero scores.",
        "", "The proposal-budget curve includes rejected proposal slots; it is not a claim of that many actual simulator calls.",
        "First event timestamps do not assign causal responsibility. Continuous margin/clearance analysis is unavailable.",
        "", "Full counts, costs, masks and per-kind new NC/DAC/TTC/high-to-zero transitions are in G0_RESULTS.json."]
    from .common import atomic_bytes
    atomic_bytes(root / "G0_REPAIR_OPPORTUNITY.md", ("\n".join(lines)+"\n").encode())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--max-samples", type=int, required=True)
    p.add_argument("--g0-root"); p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if not 1 <= args.max_samples <= cfg["cohort_size"]: raise ValueError("Invalid scene budget")
        if args.dry_run: print(json.dumps({"status": "NOT_RUN", "simulation_calls": 0})); return 0
        root = Path(args.g0_root) if args.g0_root else Path(cfg["artifact_root"]) / "g0" / f"population_{args.max_samples:06d}"
        result = analyze(root); print(json.dumps(result)); return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})); return 1


if __name__ == "__main__": raise SystemExit(main())
