"""Paired analysis of existing score_world CSVs; never generate or score a policy.

The scorer owns canonical semantics. This reader verifies its small receipts,
prediction arrays and saved feedback, then joins by scene and log. Model weights,
images and metric-cache payloads are not opened or rehashed. Full FORMAL bindings
must be added by the lead after independent approval; SMOKE cannot select a run.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

from starVLA.model.modules.st_world.consequence import (
    METRICS, STATE_COLUMNS, file_sha256, fingerprint, trajectory_sha256,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def pinned_json(ref):
    path = Path(ref["path"])
    if not path.is_absolute() or file_sha256(path) != ref["sha256"]:
        raise ValueError("Changed pinned metadata: " + str(path))
    return read_json(path)


def keyed(rows, token="token", log="log"):
    result = {}
    for row in rows:
        scene, source = row[token], row[log]
        if (not isinstance(scene, str) or not scene or scene in result
                or not isinstance(source, str) or not source
                or any(x in scene for x in ("/", "\\", ".."))):
            raise ValueError("Empty, unsafe or duplicate scene/log identity")
        result[scene] = row
    if not result:
        raise ValueError("Empty fixed population")
    return result


def load_population(plan, path, scope):
    bound = plan["populations"]["dev"]
    full = pinned_json(bound["manifest"])
    lookup = keyed(full["scenes"])
    if (full["schema"] != "st_fixed_scene_list_v1" or full["split"] != "dev"
            or len(lookup) != bound["scenes"]
            or len({r["log"] for r in lookup.values()}) != bound["logs"]
            or full["source_index_hash"] != bound["source_index_hash"]
            or full["source_student_identity"] != bound["source_student_identity"]):
        raise ValueError("Registered full development population changed")
    selected = read_json(path)
    rows = keyed(selected["scenes"])
    if (selected.get("schema") != "st_fixed_scene_list_v1" or selected.get("split") != "dev"
            or selected.get("source_index_hash") != full["source_index_hash"]
            or selected.get("source_student_identity") != full["source_student_identity"]
            or any(lookup.get(k) != v for k, v in rows.items())):
        raise ValueError("Population is not an unchanged subset of registered dev")
    if scope == "FORMAL" and (rows != lookup or file_sha256(path) != bound["manifest"]["sha256"]):
        raise ValueError("FORMAL requires the complete registered dev manifest")
    if scope == "SMOKE" and len(rows) > 64:
        raise ValueError("SMOKE analysis is limited to 64 registered scenes")
    return selected


def _formal_binding(plan, arm, export, stage):
    """Consume frozen analysis bindings, without issuing or replacing approvals."""
    binding = plan["formal_runs"].get(arm)
    if not binding:
        raise ValueError("FORMAL run binding is pending: " + arm)
    checkpoint = export["checkpoint"]
    registration = pinned_json(binding["registration"])
    approval = pinned_json(binding["approval"])
    run_identity = pinned_json(binding["run_identity_file"])
    if (checkpoint["scope"] != "FORMAL" or not checkpoint["weights_usable_for_formal_init"]
            or checkpoint["run_identity"] != run_identity["identity"]
            or fingerprint({k: v for k, v in run_identity.items() if k != "identity"}) != run_identity["identity"]
            or run_identity["registration_sha256"] != binding["registration"]["sha256"]
            or checkpoint["training_source_sha"] != registration["source_sha"]
            or registration["execution"]["scope"] != "FORMAL"
            or registration["execution"]["sample_limit"] != 0
            or registration["execution"]["seed"] != plan["training_seed"]
            or not approval.get("formal_launch_allowed")
            or approval.get("schema") != "st_world_approval_v1"
            or set(approval.get("reviews", {})) != {"evidence", "correctness", "efficiency"}
            or approval.get("fingerprint", {}).get("registration_sha256") != binding["registration"]["sha256"]):
        raise ValueError("FORMAL checkpoint/registration/approval lineage differs")
    config = pinned_json(registration["artifacts"]["config"])
    expected_approval = {"registration_sha256": binding["registration"]["sha256"],
                         "source_fingerprint": registration["source_fingerprint"],
                         "artifacts": {k: v["sha256"] for k, v in registration["artifacts"].items()},
                         "execution": registration["execution"], "run_id": registration["run_id"]}
    if approval["fingerprint"] != expected_approval:
        raise ValueError("FORMAL approval does not bind the full run fingerprint")
    reviewers = set()
    for role, reference in approval["reviews"].items():
        review = pinned_json(reference)
        reviewer = review.get("reviewer_agent")
        if (review.get("status") != "PASS" or review.get("role") != role
                or review.get("fingerprint") != expected_approval or not reviewer
                or reviewer in registration["implementer_agents"] or reviewer in reviewers
                or not review.get("evidence_artifacts")):
            raise ValueError("Missing or nonindependent FORMAL review receipt")
        reviewers.add(reviewer)
    family = plan["arms"][arm]["family"]
    expected_cell = plan["arms"][arm]["config_cell"]
    cell = config.get("cell") if family == "consequence" else config["space_time"]["cell"]
    if (cell != expected_cell or checkpoint["completed"] != plan["endpoints"][family][stage]
            or config != run_identity["config"]
            or config["execution"]["max_updates"] != plan["endpoints"][family]["COMMON_ENDPOINT"]
            or config["optimizer"] != plan["optimizer"][family]):
        raise ValueError("Arm, common endpoint or common LR schedule differs")


def _read_ledger(root, score, requests, csv_rows):
    """Read trusted local scorer receipts. Do not reconstruct scores from means."""
    import torch
    torch.set_num_threads(2)
    ledger = root / "ledger"
    ledger_identity = read_json(ledger / "IDENTITY.json")
    if (ledger_identity["identity"] != score["identity"]
            or ledger_identity["retry_attempts"] != score["retry_attempts"]):
        raise ValueError("Scoring ledger identity differs")
    records, attempts = {}, []
    for folder in sorted(ledger.iterdir()):
        if not folder.is_dir():
            continue
        batch = read_json(folder / "REQUESTS.json")
        identities = batch["requests"]
        if (fingerprint(identities) != folder.name
                or any(requests.get(r["scene_id"]) != r for r in identities)
                or any(r["scene_id"] in records for r in identities)):
            raise ValueError("Foreign or repeated ledger request group")
        saved_rows = {}
        for attempt in range(score["retry_attempts"] + 1):
            attempt_dir = folder / f"attempt_{attempt:02d}"
            if not (attempt_dir / "ADMITTED.json").exists():
                continue
            admission = read_json(attempt_dir / "ADMITTED.json")
            indices = admission["indices"]
            if (len(set(indices)) != len(indices) or admission["calls"] != len(indices)
                    or any(type(i) is not int or not 0 <= i < len(identities) for i in indices)
                    or admission["request_hash"] != folder.name
                    or admission["policy_version"] != batch["policy_version"]):
                raise ValueError("Scoring admission identity differs")
            if not (attempt_dir / "COMPLETE.json").exists():
                attempts.append({"status": "UNRESOLVED_ATTEMPT", "calls": len(indices)})
                continue
            receipt = read_json(attempt_dir / "COMPLETE.json")
            result_path = attempt_dir / "RESULTS.pt"
            if (receipt["admission_sha256"] != file_sha256(attempt_dir / "ADMITTED.json")
                    or Path(receipt["result"]["path"]).resolve() != result_path.resolve()
                    or receipt["result"]["sha256"] != file_sha256(result_path)):
                raise ValueError("Scoring result/admission receipt changed")
            # These are hash-verified local ConsequenceService products, never uploads.
            results = torch.load(result_path, map_location="cpu", weights_only=False)
            if len(results) != len(indices):
                raise ValueError("Missing saved scoring result")
            for index, result in zip(indices, results):
                if result.get("identity") != identities[index] or saved_rows.get(index, {}).get("valid"):
                    raise ValueError("Foreign or repeated successful scoring result")
                saved_rows[index] = result
                if not result.get("valid"):
                    attempts.append({"status": result.get("status"), "calls": 1})
        for i, request in enumerate(identities):
            result = saved_rows.get(i)
            if (not result or result.get("valid") is not True or result.get("status") != "VALID_SCORED"
                    or result.get("context_unchanged") is not True
                    or set(result.get("metrics", {})) != set(METRICS)
                    or not all(result.get("metrics_valid", {}).get(k) is True for k in METRICS)):
                raise ValueError("Invalid/unresolved score is neither zero nor a removable scene")
            scene = request["scene_id"]
            if any(result["metrics"][k] != csv_rows[scene][k] for k in METRICS):
                raise ValueError("CSV metric differs from saved per-scene score")
            reference = result["reference_identity"]
            if (reference["full_cache_sha256"] != request["metric_cache_sha256"]
                    or reference["score_protocol_hash"] != request["score_protocol_hash"]):
                raise ValueError("Saved canonical reference differs")
            records[scene] = result
    if set(records) != set(requests):
        raise ValueError("Ledger does not cover the complete evaluation population")
    return records, attempts


def load_run(score_root, export_root, selected, population_path, plan, scope, arm, stage):
    from tools.st_world_policy.score_world import read_complete_export
    from tools.st_world_policy.export_world import validate_cached_prediction
    score_root, export_root = Path(score_root), Path(export_root)
    export, scenes = read_complete_export(export_root)
    expected = keyed(selected["scenes"])
    checkpoint = export["checkpoint"]
    if (keyed(scenes) != expected
            or export["observations"]["manifest_sha256"] != file_sha256(population_path)
            or export["observations"]["current_source"] != selected["source_student_identity"]
            or export["observations"].get("history") is not None
            or export["solver"] != "original_Euler_10" or export["seed"] != plan["inference_seed"]
            or export["precision"] != "FP32_master_eval_TF32_off"
            or export["scene_noise"] != "foresight-action-v1"
            or checkpoint["base_binding"] != plan["base_binding"]["binding_sha256"]
            or fingerprint({k: v for k, v in checkpoint.items() if k != "sha256"}) != checkpoint["sha256"]):
        raise ValueError("Export population, checkpoint, input or inference contract differs")
    if scope == "FORMAL":
        _formal_binding(plan, arm, export, stage)
    score = read_json(score_root / "IDENTITY.json")
    complete = read_json(score_root / "COMPLETE.json")
    protocol = score["protocol"]
    if (score["schema"] != "st_world_official_evaluation_v1"
            or fingerprint({k: v for k, v in score.items() if k != "identity"}) != score["identity"]
            or score["export_identity"] != export["identity"]
            or score["export_file_sha256"] != file_sha256(export_root / "IDENTITY.json")
            or score["checkpoint"] != checkpoint
            or fingerprint({k: v for k, v in protocol.items() if k != "sha256"}) != protocol["sha256"]
            or protocol["sha256"] != plan["score_protocol_sha256"]
            or score["spec"]["score_protocol_hash"] != protocol["sha256"]):
        raise ValueError("Scorer/export/protocol identity differs")
    if (complete.get("status") != "COMPLETE" or complete["identity"] != score["identity"]
            or complete["requested_scenes"] != len(expected) or complete["valid_scored"] != len(expected)
            or complete["invalid"] != 0 or complete["unresolved"] != 0
            or not complete["official_single_candidate_reference"]
            or complete["deployment_candidates_per_scene"] != 1
            or complete["checkpoint_scope"] != checkpoint["scope"]
            or complete["diagnostic_weights"] != (not checkpoint["weights_usable_for_formal_init"])
            or complete["score_units"] != "0..1"):
        raise ValueError("Incomplete/invalid scoring population or diagnostic provenance")
    csv_path = score_root / "SCENES.csv"
    if file_sha256(csv_path) != complete["scenes_csv_sha256"]:
        raise ValueError("SCENES.csv hash differs")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["scene_id", "source_log", *METRICS]:
            raise ValueError("Canonical CSV schema differs")
        rows = keyed(list(reader), "scene_id", "source_log")
    if set(rows) != set(expected):
        raise ValueError("CSV is missing scenes or contains extra scenes")
    for scene, row in rows.items():
        if set(row) != {"scene_id", "source_log", *METRICS}:
            raise ValueError("Malformed CSV row width")
        if row["source_log"] != expected[scene]["log"]:
            raise ValueError("CSV scene belongs to a foreign log")
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError("Nonfinite/out-of-range canonical metric")
    requests = keyed(score["request_identities"], "scene_id", "source_log")
    if set(requests) != set(expected):
        raise ValueError("Scoring request population differs")
    contexts = keyed(pinned_json(score["metric_index"]))
    runtime = fingerprint({k: export[k] for k in ("environment", "seed", "solver", "precision", "scene_noise")})
    predictions = {}
    for scene, request in requests.items():
        context = contexts.get(scene)
        if (context is None or context["log"] != expected[scene]["log"]
                or request["source_log"] != expected[scene]["log"]
                or str(Path(context["cache_path"]).resolve()) != request["metric_cache_path"]
                or context.get("sha256", request["metric_cache_sha256"]) != request["metric_cache_sha256"]
                or request["candidate_id"] != "single_policy" or request["generator_kind"] != "policy"
                or request["policy_version"] != checkpoint["sha256"]
                or request["generator_checkpoint_sha256"] != checkpoint["native_checkpoint_sha256"]
                or request["generator_source_hash"] != fingerprint(checkpoint["deployment_source_contract"])
                or request["inference_runtime_hash"] != runtime
                or request["score_protocol_hash"] != protocol["sha256"]):
            raise ValueError("Scene, native checkpoint or canonical context binding differs")
        path = export_root / "predictions" / (scene + ".npz")
        meta = read_json(path.with_suffix(".json"))
        validate_cached_prediction(path, meta, identity=export["identity"], token=scene, log=request["source_log"])
        with np.load(path, allow_pickle=False) as arrays:
            trajectory = arrays["trajectory"].astype(np.float64)
        if trajectory_sha256(trajectory) != request["trajectory_sha256"]:
            raise ValueError("Scored trajectory differs from the exported prediction")
        predictions[scene] = trajectory
    records, attempts = _read_ledger(score_root, score, requests, rows)
    return {"export": export, "score": score, "rows": rows, "requests": requests,
            "records": records, "predictions": predictions, "failed_attempts": attempts,
            "score_root": str(score_root.resolve()), "export_root": str(export_root.resolve()),
            "scenes_csv_sha256": complete["scenes_csv_sha256"]}


def trajectory_diagnostics(run, selected, plan):
    import torch
    from starVLA.dataloader.foresight_dataset import decode_ego
    torch.set_num_threads(2)
    root = Path(plan["populations"]["dev"]["current_root"])
    if file_sha256(root / "ego_identity.json") != selected["ego_identity_sha256"]:
        raise ValueError("Ground-truth label identity changed")
    ego_identity = read_json(root / "ego_identity.json")
    if (ego_identity["schema"] != "foresight_ego_labels_v1" or ego_identity["split"] != "dev"
            or ego_identity["index_sha256"] != selected["source_index_hash"]):
        raise ValueError("Ground-truth label role/index differs")
    result = {}
    for row in selected["scenes"]:
        scene = row["token"]
        path = root / "ego" / (scene + ".pt")
        if file_sha256(path) != row["ego_record_sha256"]:
            raise ValueError("Ground-truth scene label changed")
        label = torch.load(path, map_location="cpu", weights_only=True)
        if (label["identity"] != ego_identity["identity"] or label["token"] != scene
                or tuple(label["ego"].shape) != (8, 4) or not torch.isfinite(label["ego"]).all()):
            raise ValueError("Invalid ground-truth scene label")
        gt = decode_ego(label["ego"].float()).numpy().astype(np.float64)
        errors = np.linalg.norm(run["predictions"][scene][:, :2] - gt[:, :2], axis=1)
        record = run["records"][scene]
        states = np.asarray(record["executed_states"], dtype=np.float64)
        if (states.shape != (41, 11) or not np.isfinite(states).all()
                or tuple(record["state_columns"]) != STATE_COLUMNS
                or not np.array_equal(record["executed_state_times_seconds"], np.arange(41) * .1)):
            raise ValueError("Executed state/time/units contract differs")
        progress = record["progress"]
        raw, reference = float(progress["raw_meters"]), float(progress["reference_raw_meters"])
        if not np.isfinite([raw, reference]).all():
            raise ValueError("Nonfinite raw progress")
        result[scene] = {"ade_m": float(errors.mean()), "fde_m": float(errors[-1]),
                         "raw_progress_m": raw, "reference_progress_m": reference,
                         "mean_executed_speed_mps": float(np.hypot(states[1:, 3], states[1:, 4]).mean())}
    return result


def cluster_draws(logs, *, replicates, seed):
    unique, inverse = np.unique(np.asarray(logs), return_inverse=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    return unique, inverse, draws


def estimate(values, inverse, draws, mask=None):
    """Scene-weighted ratio, resampling whole paired logs including empty strata."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("Missing metric cannot be dropped or filled")
    clusters = int(inverse.max()) + 1
    counts = np.bincount(inverse[mask], minlength=clusters)
    totals = np.bincount(inverse[mask], weights=values[mask], minlength=clusters)
    boot_counts = counts[draws].sum(axis=1)
    valid = boot_counts > 0
    bootstrap = totals[draws].sum(axis=1)[valid] / boot_counts[valid]
    return {"mean": float(values[mask].mean()) if mask.any() else None,
            "denominator_scenes": int(mask.sum()), "represented_logs": int((counts > 0).sum()),
            "ci95": np.quantile(bootstrap, [.025, .975], method="linear").tolist() if len(bootstrap) else None,
            "bootstrap_defined": int(valid.sum()), "bootstrap_undefined": int((~valid).sum())}


def paired_statistics(control, treatment, logs, definitions):
    if set(control) != set(treatment):
        raise ValueError("Paired metric fields differ")
    unique, inverse, draws = cluster_draws(logs, replicates=definitions["bootstrap_replicates"],
                                         seed=definitions["bootstrap_seed"])
    metrics = {}
    for key in control:
        a, b = np.asarray(control[key]), np.asarray(treatment[key])
        if a.shape != (len(logs),) or b.shape != a.shape:
            raise ValueError("Paired metric population differs")
        metrics[key] = {"control": estimate(a, inverse, draws), "treatment": estimate(b, inverse, draws),
                        "delta": estimate(b - a, inverse, draws),
                        "equal_log_delta_sensitivity": float(np.mean([
                            (b - a)[inverse == i].mean() for i in range(len(unique))]))}
    a, b = np.asarray(control["score"]), np.asarray(treatment["score"])
    za, zb = a <= definitions["zero_epsilon"], b <= definitions["zero_epsilon"]
    high = a >= definitions["high_score_threshold"]
    flags = {"zero_repaired": za & ~zb, "new_zero": ~za & zb,
             "zero_persisted": za & zb, "nonzero_persisted": ~za & ~zb,
             "high_any_drop": high & (b < a - definitions["zero_epsilon"]),
             "high_large_drop": high & (b < a - definitions["high_drop_threshold"]),
             "high_to_zero": high & zb,
             "ep_drop": np.asarray(treatment["ego_progress"]) < np.asarray(control["ego_progress"]) - definitions["ep_drop_threshold"]}
    safety_gain = np.any([np.asarray(treatment[k]) > np.asarray(control[k]) + definitions["zero_epsilon"]
                          for k in ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")], axis=0)
    flags["safety_gain_with_ep_drop"] = safety_gain & flags["ep_drop"]
    events = {key: {"count": int(value.sum()), "full_population_rate": estimate(value, inverse, draws)}
              for key, value in flags.items()}
    for key, mask in (("zero_repaired", za), ("new_zero", ~za), ("high_any_drop", high),
                      ("high_large_drop", high), ("high_to_zero", high)):
        events[key]["conditional_rate"] = estimate(flags[key], inverse, draws, mask)
    events["net_zero_repair_rate"] = estimate(flags["zero_repaired"].astype(float) - flags["new_zero"], inverse, draws)
    strata = {"control_zero": za, "control_high": high, "control_middle": ~za & ~high}
    stratified = {name: {k: estimate(np.asarray(treatment[k]) - np.asarray(control[k]), inverse, draws, mask)
                        for k in control} for name, mask in strata.items()}
    stopping = None
    if "reference_progress_m" in control:
        if not np.array_equal(control["reference_progress_m"], treatment["reference_progress_m"]):
            raise ValueError("Fixed reference progress changed across the pair")
        moving = np.asarray(control["reference_progress_m"]) >= definitions["moving_reference_m"]
        stop_flags = {}
        for side, values in (("control", control), ("treatment", treatment)):
            stop_flags[side] = moving & (np.asarray(values["raw_progress_m"]) <= definitions["low_progress_m"]) & (
                np.asarray(values["mean_executed_speed_mps"]) <= definitions["near_stop_speed_mps"])
        stopping = {side: {"count": int(value.sum()), "full_population_rate": estimate(value, inverse, draws),
                           "moving_reference_rate": estimate(value, inverse, draws, moving)}
                    for side, value in stop_flags.items()}
        stopping["delta_full_population_rate"] = estimate(stop_flags["treatment"].astype(float) - stop_flags["control"], inverse, draws)
        stopping["new_stop_count"] = int((stop_flags["treatment"] & ~stop_flags["control"]).sum())
        flags["new_stop_proxy"] = stop_flags["treatment"] & ~stop_flags["control"]
    return {"scenes": len(logs), "logs": len(unique), "metrics": metrics, "events": events,
            "control_defined_strata": stratified, "stopping_proxy": stopping,
            "by_log": {log: {"scenes": int((inverse == i).sum()), "delta": {
                k: float((np.asarray(treatment[k]) - np.asarray(control[k]))[inverse == i].mean()) for k in control}}
                for i, log in enumerate(unique.tolist())}}, flags


def analyze(plan, population_path, control_score, control_export, treatment_score, treatment_export,
            *, scope, comparison, stage):
    if plan.get("schema") != "st_world_evaluation_plan_v1" or scope not in ("SMOKE", "FORMAL"):
        raise ValueError("Explicit registered analysis scope required")
    if scope == "SMOKE":
        if comparison != "SMOKE_AA" or stage != "SMOKE":
            raise ValueError("SMOKE is only an engineering A/A analysis, never a campaign contrast")
        arms = {"control": "SMOKE", "treatment": "SMOKE"}
    else:
        arms = plan["comparisons"][comparison]
        if stage not in plan["endpoints"][plan["arms"][arms["control"]]["family"]]:
            raise ValueError("Unregistered checkpoint stage")
    selected = load_population(plan, population_path, scope)
    a = load_run(control_score, control_export, selected, population_path, plan, scope, arms["control"], stage)
    b = a if (Path(control_score).resolve() == Path(treatment_score).resolve()
              and Path(control_export).resolve() == Path(treatment_export).resolve()) else load_run(
        treatment_score, treatment_export, selected, population_path, plan, scope, arms["treatment"], stage)
    if scope == "SMOKE" and a["export"]["identity"] != b["export"]["identity"]:
        raise ValueError("SMOKE A/A must use the same immutable export")
    if scope == "FORMAL" and a["export"]["checkpoint"]["run_identity"] == b["export"]["checkpoint"]["run_identity"]:
        raise ValueError("A campaign contrast requires distinct registered runs")
    if (a["export"]["observations"] != b["export"]["observations"]
            or a["export"]["environment"] != b["export"]["environment"]):
        raise ValueError("Paired input or inference environment differs")
    for scene in a["rows"]:
        if (a["requests"][scene]["metric_cache_sha256"] != b["requests"][scene]["metric_cache_sha256"]
                or a["records"][scene]["context_hash"] != b["records"][scene]["context_hash"]
                or a["records"][scene]["reference_identity"] != b["records"][scene]["reference_identity"]):
            raise ValueError("Paired canonical context/reference changed")
    da = trajectory_diagnostics(a, selected, plan)
    db = da if b is a else trajectory_diagnostics(b, selected, plan)
    tokens = sorted(a["rows"])
    metric_names = [*METRICS, *next(iter(da.values()))]
    arrays = [{k: np.array([({**run["rows"][s], **diag[s]})[k] for s in tokens]) for k in metric_names}
              for run, diag in ((a, da), (b, db))]
    logs = [a["rows"][s]["source_log"] for s in tokens]
    statistics, flags = paired_statistics(*arrays, logs, plan["statistics"])
    paired = []
    for i, scene in enumerate(tokens):
        row = {"scene_id": scene, "source_log": logs[i]}
        for k in metric_names:
            row.update({"control_" + k: float(arrays[0][k][i]), "treatment_" + k: float(arrays[1][k][i]),
                        "delta_" + k: float(arrays[1][k][i] - arrays[0][k][i])})
        row.update({k: int(v[i]) for k, v in flags.items()})
        paired.append(row)
    def identity(run):
        return {k: run[k] for k in ("score_root", "export_root", "scenes_csv_sha256", "failed_attempts")} | {
            "score_identity": run["score"]["identity"], "export_identity": run["export"]["identity"],
            "checkpoint": run["export"]["checkpoint"]}
    return {"schema": "st_world_paired_analysis_v1", "status": "COMPLETE", "scope": scope,
            "comparison": comparison, "stage": stage, "scientific_result": scope == "FORMAL",
            "analysis_source_sha256": file_sha256(Path(__file__)),
            "analysis_environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                     "bootstrap_rng": "PCG64", "gpu_used": False, "scoring_calls": 0},
            "formal_launch_authorized_by_this_analysis": False,
            "population_sha256": file_sha256(population_path), "statistics_definition": plan["statistics"],
            "control": identity(a), "treatment": identity(b), "statistics": statistics,
            "verification_boundary": "Verified CSV, export, prediction, ledger, GT and metadata hashes; reused scorer cache/native-weight receipts; no cache or model byte rehash, no scoring.",
            "validity": {"requested": len(tokens), "paired_valid": len(tokens), "invalid": 0, "unresolved": 0}}, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "population", "control-score", "control-export", "treatment-score", "treatment-export", "comparison", "stage", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--scope", required=True, choices=("SMOKE", "FORMAL"))
    args = parser.parse_args()
    try:
        result, rows = analyze(read_json(args.plan), args.population, args.control_score, args.control_export,
                               args.treatment_score, args.treatment_export, scope=args.scope,
                               comparison=args.comparison, stage=args.stage)
        result["plan_sha256"] = file_sha256(args.plan)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
        csv_path = root / "PAIRED_SCENES.csv"
        content = buffer.getvalue().encode()
        if csv_path.exists() and csv_path.read_bytes() != content:
            raise ValueError("Existing paired CSV differs; use a new immutable analysis directory")
        if not csv_path.exists():
            with csv_path.open("xb") as stream:
                stream.write(content)
        result["paired_scenes_csv_sha256"] = file_sha256(csv_path)
        from tools.st_world_policy.base_binding import immutable_json
        immutable_json(root / "ANALYSIS.json", result)
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_ANALYSIS", "error": repr(error), "scientific_result": False}))
        return 2
    print(json.dumps({k: result[k] for k in ("status", "scope", "comparison", "stage", "scientific_result", "validity")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
