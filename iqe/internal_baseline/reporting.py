from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import csv
import io
import json
import time
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import atomic_json, atomic_torch, atomic_bytes, read_json, file_hash, digest
from iqe.data.manifests import load_scenes
from iqe.evaluation.retention import module_hash, module_hash_parameters, module_hash_buffers
from iqe.training.checkpoint import capture_rng, restore_rng
from .binding import load_head
from .features import JointCache, encode_joint, planner_bundle
from .training import load_bank, make_scorer, predict_scores, score_context
from .evaluation import argmax_base_tie, selection_metrics, paired_bootstrap


def markdown(config, name, text):
    atomic_bytes(config.root/name, text.encode())


def evaluate(config, device="cuda:0"):
    bank, data = load_bank(config)
    result = read_json(config.root/"scorer/result.json")
    require(result["status"] == "COMPLETE", "Scorer must finish its fixed budget before dev")
    best = config.root/"scorer/best.pt"
    require(file_hash(best) == result["best_checkpoint_hash"], "scorer best hash mismatch")
    model = make_scorer(config).to(device)
    model.load_state_dict(torch.load(best, map_location=device, weights_only=True), strict=True)
    scenes = load_scenes(config.root/"scenes.json")
    lookup = {s.scene_id: s for s in scenes}
    manifest = read_json(config.root/"EXPERT_MANIFEST.json")
    base_scores = read_json(config.root/"BASE_FIT_VAL_SCORES.json")
    hard_stats = {}
    cache = JointCache(config)
    expert = load_head(config, device).eval()
    expert.load_state_dict(torch.load(config.root/"expert/best.pt", map_location=device, weights_only=True), strict=True)
    for name, ids in (("H_train", manifest["hard"]), ("H_val", manifest["hard_val"]), ("ordinary_train", manifest["ordinary"])):
        mask = np.array([r["scene_id"] in set(ids) for r in bank["rows"]])
        indices = np.flatnonzero(mask)
        il_base, il_expert = [], []
        from iqe.losses import expert_il_terms
        with torch.no_grad():
            for at in range(0, len(indices), 64):
                ix = indices[at:at+64]
                rows = [lookup[bank["rows"][i]["scene_id"]] for i in ix]
                x = cache.batch(rows, device)
                predicted = expert(planner_bundle(config, x, [s.scene_id for s in rows]))
                il_expert.extend((predicted.raw-x["target_raw"]).abs().sum(-1).mean(-1).cpu().tolist())
                il_base.extend((x["base_raw"]-x["target_raw"]).abs().sum(-1).mean(-1).cpu().tolist())
        delta = data["scores"][mask, 1]-data["scores"][mask, 0]
        hard_stats[name] = {"n": int(mask.sum()), "base_PDMS": float(data["scores"][mask, 0].mean()),
                            "expert_PDMS": float(data["scores"][mask, 1].mean()), "expert_minus_base": float(delta.mean()),
                            "base_IL": float(np.mean(il_base)), "expert_IL": float(np.mean(il_expert)),
                            "log_bootstrap": paired_bootstrap(delta, [r["log"] for r, yes in zip(bank["rows"], mask) if yes],
                                                              config.bootstrap_repetitions, config.bootstrap_seed)}
    atomic_json(config.root/"EXPERT_SPECIALIZATION.json", hard_stats, immutable=True)
    # Select optional margin on natural selector_cal before any dev predictions.
    calix = [i for i, r in enumerate(bank["rows"]) if r["role"] == "selector_cal"]
    calscenes = [lookup[bank["rows"][i]["scene_id"]] for i in calix]
    cp, _ = predict_scores(config, model, calscenes, data["trajectories"][calix], device)
    cal_results = {}
    for margin in (0., .01, .02, .05):
        choices = (cp[:, 1]-cp[:, 0] > margin).astype(np.int64)
        cal_results[str(margin)] = selection_metrics(data["scores"][calix], data["components"][calix], choices,
                            [s.source_log_id for s in calscenes], bootstrap_repetitions=config.bootstrap_repetitions, bootstrap_seed=config.bootstrap_seed)
    # Predeclared tie-break: PDMS, fewer new hard violations, fewer high->zero,
    # then larger margin. No claim of per-scene safety; raw argmax remains primary.
    selected_margin = max(cal_results, key=lambda k: (cal_results[k]["selected"],
                         -sum(x["new_violations"] for x in cal_results[k]["safety"].values()), -cal_results[k]["high_to_zero"], float(k)))
    atomic_json(config.root/"SINGLE_MARGIN.json", {"margin": float(selected_margin), "role": "selector_cal", "results": cal_results,
                "scorer_hash": file_hash(best), "bank_hash": file_hash(config.root/"PAIR_BANK.json")}, immutable=True)
    output, csv_rows = {}, []
    for role in ("stage_val", "dev_report"):
        ix = [i for i, r in enumerate(bank["rows"]) if r["role"] == role]
        rows = [lookup[bank["rows"][i]["scene_id"]] for i in ix]
        predictions, _ = predict_scores(config, model, rows, data["trajectories"][ix], device)
        scores, components = data["scores"][ix], data["components"][ix]
        selectors = {"B0_Base": np.zeros(len(ix), dtype=np.int64), "B1_Expert": np.ones(len(ix), dtype=np.int64),
                     "B2_Oracle_at_2": scores.argmax(1), "B3_ScratchScorer_argmax": argmax_base_tie(predictions),
                     "B4_ScratchScorer_margin": (predictions[:, 1]-predictions[:, 0] > float(selected_margin)).astype(np.int64)}
        output[role] = {}
        for name, choices in selectors.items():
            metrics = selection_metrics(scores, components, choices, [s.source_log_id for s in rows],
                          bootstrap_repetitions=config.bootstrap_repetitions, bootstrap_seed=config.bootstrap_seed)
            output[role][name] = metrics
            csv_rows.append({"role": role, "method": name, "n": len(rows), "PDMS_01": metrics["selected"],
                             "gain_vs_same_base": metrics["selected_gain"], "regret": metrics["selection_regret"],
                             "oracle_capture": metrics["oracle_capture"], "switch_rate": metrics["switch_rate"],
                             "new_NC": metrics["safety"]["NC"]["new_violations"], "new_DAC": metrics["safety"]["DAC"]["new_violations"],
                             "high_to_zero": metrics["high_to_zero"]})
        atomic_json(config.root/f"{role}_decisions.json", {"rows": [{"scene_id": s.scene_id, "log": s.source_log_id,
                    "predicted_values": predictions[j].tolist(), "true_scores": scores[j].tolist(),
                    "choices": {name: int(choices[j]) for name, choices in selectors.items()}} for j, s in enumerate(rows)]}, immutable=True)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    atomic_bytes(config.root/"FOUR_WAY_RESULTS.csv", stream.getvalue().encode())
    atomic_json(config.root/"RESULTS.json", output, immutable=True)
    d = output["dev_report"]["B3_ScratchScorer_argmax"]
    if hard_stats["H_train"]["expert_minus_base"] <= 0:
        verdict = "EXPERT_NOT_SPECIALIZED"
    elif d["oracle_gain"] <= 0:
        verdict = "NO_HELDOUT_COVERAGE_GAIN"
    elif d["selected_gain"] <= 0:
        verdict = "CANDIDATE_GAIN_NOT_REALIZED"
    elif any(v["new_violations"] for v in d["safety"].values()):
        verdict = "SCORE_GAIN_WITH_SAFETY_REGRESSION"
    elif hard_stats["H_val"]["expert_minus_base"] > 0:
        verdict = "POSITIVE_BASELINE_SIGNAL"
    else:
        verdict = "NO_HELDOUT_SPECIALIZATION_GAIN"
    result = {"verdict": verdict, "single_seed": True, "expert_train_specialized": hard_stats["H_train"]["expert_minus_base"] > 0,
              "expert_heldout_specialized": hard_stats["H_val"]["expert_minus_base"] > 0,
              "scorer_all_new_parameters_random": True, "external_scorer_weights_loaded": False,
              "dev_result": d, "formal_ACTIVE_updated": False, "navtest_accessed": False,
              "base_binding_hash": file_hash(config.root/"BASE_BINDING.json")}
    atomic_json(config.root/"VERDICT.json", result)
    markdown(config, "EXPERT_SPECIALIZATION_REPORT.md", "# Expert specialization\n\nRaw IL is inherited channel-sum/time-mean L1. All PDMS are [0,1].\n\n```json\n"+json.dumps(hard_stats, indent=2)+"\n```\n")
    markdown(config, "SELECTION_REGRET_DECOMPOSITION.md", "# Selection evidence\n\nSame locked Base, same fixed development scenes. Positive gains and negative costs are totals over all requested scenes; rates divide by n, switch precision by switched scenes. Oracle capture is unbounded and NA when its denominator vanishes.\n\n```json\n"+json.dumps(output, indent=2)+"\n```\n")
    return result


def audit_online(config, device="cuda:0"):
    from iqe.s0_adapter import S0Adapter
    from iqe.data.sources import load_observation
    from .inference import InternalAlgorithm
    # Same-batch original and wrapper must be exactly equal. Cross-batch cache
    # comparison is reported separately, never used to loosen this requirement.
    protocol = {"same_batch_raw_max_abs": 0., "same_batch_physical_max_abs": 0., "label_dependence_max_abs": 0.,
                "scenes": 32, "new_parameters_init": config.module_init_seed, "latency_warmup": 5, "latency_repetitions": 30}
    atomic_json(config.root/"ONLINE_AUDIT_PROTOCOL.json", protocol, immutable=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    adapter = S0Adapter.load_and_validate_s0(read_json(config.root/"OFFLINE_BASE_CONTRACT.json"), device)
    base_before = {"parameters": module_hash_parameters(adapter), "buffers": module_hash_buffers(adapter)}
    language = adapter.framework.qwen_vl_interface.model.model.language_model
    calls = []; handle = language.register_forward_hook(lambda *args: calls.append(1))
    scenes = [s for s in load_scenes(config.root/"scenes.json") if s.split_role == "incremental_fit"][:32]
    cache = JointCache(config)
    evidence = []
    for at in range(0, len(scenes), 4):
        rows = scenes[at:at+4]; obs = [load_observation(s, adapter.contract) for s in rows]
        calls.clear()
        with torch.no_grad():
            original = adapter.framework.predict_action(obs)
        require(len(calls) == 1, "original Base VLM forward count")
        calls.clear(); planner, context = encode_joint(adapter, obs)
        with torch.no_grad(): wrapped = adapter.predict_s0(planner)
        require(len(calls) == 1 and torch.equal(original, wrapped.physical), "K=1 changes original Base output/call count")
        poisoned = [dict(o, target=torch.full((8, 4), float("nan")), future_actor_gt="deleted/shuffled", score=-999.) for o in obs]
        calls.clear(); p2, c2 = encode_joint(adapter, poisoned)
        require(len(calls) == 1 and torch.equal(context.world, c2.world) and torch.equal(context.action, c2.action)
                and torch.equal(planner.scene, p2.scene), "training labels affect current W/A/planning")
        cached = cache.batch(rows, device)
        evidence.append({"scene_ids": [s.scene_id for s in rows], "original_vs_K1_max_abs": float((original-wrapped.physical).abs().max()),
                         "label_shuffle_W_max_abs": float((context.world-c2.world).abs().max()),
                         "cache_rebatch_W_max_abs": float((cached["world"]-context.world).abs().max()),
                         "cache_rebatch_A_max_abs": float((cached["action"]-context.action).abs().max()),
                         "cache_rebatch_physical_max_abs": float((cached["base_physical"]-wrapped.physical).abs().max())})
    expert = adapter.clone_s0_expert("expert_1").to(device).eval()
    model = make_scorer(config).to(device).eval()
    algorithm = InternalAlgorithm(adapter, expert, model)
    obs = [load_observation(scenes[0], adapter.contract)]
    projection_calls = []; ph = model.world_projection.register_forward_hook(lambda *args: projection_calls.append(1))
    calls.clear(); output = algorithm(obs, return_debug=True)
    require(len(calls) == len(projection_calls) == 1, "two-candidate forward must share VLM and score projections")
    ph.remove(); handle.remove()
    after = {"parameters": module_hash_parameters(adapter), "buffers": module_hash_buffers(adapter)}
    require(base_before == after, "read-only context extraction mutated Base")
    result = {"status": "PASS", "evidence": evidence, "frozen_hashes_before": base_before, "frozen_hashes_after": after,
              "W_shape": list(context.world.shape), "A_shape": list(context.action.shape), "W_dtype": str(context.world.dtype),
              "A_dtype": str(context.action.dtype), "vlm_calls_per_image_to_selected_trajectory": 1,
              "score_context_projections_per_scene": 1, "auxiliary_prediction_heads_executed": 0,
              "GT_in_score_context": False, "poisoned_labels_ignored": True,
              "base_shared_parameter_count": sum(p.numel() for p in adapter.parameters())-sum(p.numel() for p in adapter.framework.action_model.expert.parameters()),
              "base_head_parameter_count": sum(p.numel() for p in adapter.framework.action_model.expert.parameters()),
              "new_expert_parameter_count": sum(p.numel() for p in expert.parameters()), "new_scorer_parameter_count": sum(p.numel() for p in model.parameters())}
    atomic_json(config.root/"SCORE_CONTEXT_AUDIT.json", result, immutable=True)
    markdown(config, "SCORE_CONTEXT_AUDIT.md", "# Actual current-input W/A audit\n\nOne frozen original VLM; no auxiliary prediction head, external encoder, extra tokens or future labels. Rebatch differences below are numerical observations, not new tolerance gates.\n\n```json\n"+json.dumps(result, indent=2)+"\n```\n")
    return result


def diagnostics(config, device="cuda:0"):
    bank, data = load_bank(config)
    scenes = {s.scene_id: s for s in load_scenes(config.root/"scenes.json")}
    eligible = [i for i, r in enumerate(bank["rows"]) if r["role"] == "incremental_fit" and abs(data["scores"][i, 1]-data["scores"][i, 0]) > .01]
    eligible.sort(key=lambda i: digest([config.selection_seed, "overfit64", bank["rows"][i]["scene_id"]]))
    chosen = eligible[:64]
    if not chosen:
        result = {"status": "SCORER_COMPARISON_SIGNAL_ABSENT", "real_unique_scenes": 0}
        atomic_json(config.root/"SCORER_DIAGNOSTICS.json", result)
        return result
    rows = [scenes[bank["rows"][i]["scene_id"]] for i in chosen]
    x = JointCache(config).batch(rows, device)
    context = score_context(config, x)
    trajectories = torch.as_tensor(data["trajectories"][chosen], device=device)
    scores = torch.as_tensor(data["scores"][chosen], device=device)
    targets = torch.as_tensor(data["components"][chosen], device=device)
    from . import objectives
    from iqe.internal_baseline.scorer import initialization_rng
    # A fresh diagnostic head, never used as the main head's warm start.
    model = make_scorer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    labels = {"scores": scores, "valid": torch.ones_like(scores, dtype=torch.bool), "trajectories": trajectories,
              "components": targets, "component_valid": torch.ones_like(targets, dtype=torch.bool),
              "natural": torch.ones(len(rows), device=device, dtype=torch.bool), "comparison": torch.ones(len(rows), device=device, dtype=torch.bool)}
    history = []
    torch.manual_seed(config.training_rng_seed)
    model.eval()
    with torch.no_grad(): initial = model(context, trajectories)["values"].cpu().numpy()
    initial_metrics = selection_metrics(scores.cpu().numpy(), targets.cpu().numpy(), argmax_base_tie(initial),
                                       [s.source_log_id for s in rows], bootstrap_repetitions=100)
    for step in range(config.diagnostic_updates):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(context, trajectories)
        ts = objectives.terms(prediction, labels)
        # Diagnostic population uses the actual global gap-weighted objective,
        # rather than pretending a uniform 64-scene probe is a q-sampled stream.
        rank = objectives.weighted_population_ranking(prediction["values"], scores, labels["valid"])
        loss = ts["value"].numerator/ts["value"].denominator + .2*ts["components"].numerator/ts["components"].denominator
        loss = loss + .2*rank.numerator/rank.denominator.clamp_min(torch.finfo(torch.float32).tiny)
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        require(torch.isfinite(norm).item(), "diagnostic gradient nonfinite")
        optimizer.step()
        if step == 0 or (step+1) % 25 == 0:
            model.eval()
            with torch.no_grad(): p = model(context, trajectories)["values"].cpu().numpy()
            metrics = selection_metrics(scores.cpu().numpy(), targets.cpu().numpy(), argmax_base_tie(p), [s.source_log_id for s in rows], bootstrap_repetitions=100)
            history.append({"step": step+1, "loss": float(loss), "gradient_norm": float(norm), "selection": metrics})
    # Readout diagnostics use the selected MAIN scorer on the fixed held-out set.
    main = make_scorer(config).to(device)
    main.load_state_dict(torch.load(config.root/"scorer/best.pt", weights_only=True, map_location=device), strict=True)
    main.eval()
    valix = [i for i, r in enumerate(bank["rows"]) if r["role"] == "stage_val"]
    conditions = {k: [] for k in ("correct", "scene_mismatch", "time_shuffle", "remove_W")}
    cache = JointCache(config)
    for at in range(0, len(valix), config.scorer_micro_batch):
        ix = valix[at:at+config.scorer_micro_batch]
        vr = [scenes[bank["rows"][i]["scene_id"]] for i in ix]
        c = score_context(config, cache.batch(vr, device))
        t = torch.as_tensor(data["trajectories"][ix], device=device)
        wrong = {k: v.roll(1, 0) for k, v in c.items()}
        removed = c | {"world": torch.zeros_like(c["world"])}
        with torch.no_grad():
            for name, ctx, tau in (("correct", c, t), ("scene_mismatch", wrong, t), ("time_shuffle", c, t.flip(-2)), ("remove_W", removed, t)):
                conditions[name].append(main(ctx, tau)["values"].cpu().numpy())
    diagnostics_report = {name: selection_metrics(data["scores"][valix], data["components"][valix], argmax_base_tie(np.concatenate(pred)),
                           [bank["rows"][i]["log"] for i in valix], bootstrap_repetitions=config.bootstrap_repetitions,
                           bootstrap_seed=config.bootstrap_seed) for name, pred in conditions.items()}
    result = {"status": "COMPLETE", "purpose": "real64 fitability only; not generalization", "real_unique_scenes": len(chosen),
              "scene_ids": [s.scene_id for s in rows], "updates": config.diagnostic_updates, "history": history,
              "initial_selection": initial_metrics,
              "comparison_fit_improved": history[-1]["selection"]["selection_regret"] < initial_metrics["selection_regret"],
              "heldout_pairing_diagnostics": diagnostics_report, "main_scorer_warm_started_from_probe": False}
    atomic_json(config.root/"SCORER_DIAGNOSTICS.json", result)
    verdict = read_json(config.root/"VERDICT.json") if (config.root/"VERDICT.json").exists() else {}
    if verdict:
        verdict["scorer_fitability_diagnostic"] = {"unique_scenes": len(chosen), "updates": config.diagnostic_updates,
                      "initial_regret": initial_metrics["selection_regret"], "final_regret": history[-1]["selection"]["selection_regret"]}
        if not result["comparison_fit_improved"] and history[-1]["selection"]["selection_regret"] > 1e-6:
            verdict["scorer_diagnostic_observation"] = "SCORER_UNDERFIT"
        atomic_json(config.root/"VERDICT.json", verdict)
    return result


def export_bundle(config, device="cuda:0"):
    require((config.root/"VERDICT.json").exists(), "finish fixed evaluation before diagnostic bundle export")
    path = config.root/"bundle"
    weights = {"expert_1": torch.load(config.root/"expert/best.pt", weights_only=True, map_location="cpu"),
               "internal_scorer": torch.load(config.root/"scorer/best.pt", weights_only=True, map_location="cpu")}
    checked_expert = load_head(config, device)
    checked_expert.load_state_dict(weights["expert_1"], strict=True)
    checked_scorer = make_scorer(config).to(device)
    checked_scorer.load_state_dict(weights["internal_scorer"], strict=True)
    atomic_torch(path/"algorithm.pt", weights, immutable=True)
    init = read_json(config.root/"SCORER_INITIALIZATION.json")
    manifest = {"schema": "own_internal_scorer_bundle_v1", "experiment_id": config.experiment_id,
                "base_contract": read_json(config.root/"OFFLINE_BASE_CONTRACT.json"),
                "weights_hash": file_hash(path/"algorithm.pt"), "scorer_initialization": read_json(config.root/"SCORER_INITIALIZATION.json"),
                "scorer_architecture": {"world_dim": init["parameters"]["world_projection.0.weight"]["shape"][0],
                                         "action_dim": init["parameters"]["action_projection.0.weight"]["shape"][0],
                                         "dropout": config.scorer_dropout, "context": config.score_context},
                "selection": "predicted_total_argmax_base_tie", "candidate_ids": ["expert_0", "expert_1"],
                "extra_vlm": False, "metric_cache_required_at_inference": False, "GT_required_at_inference": False,
                "deployment_release": False, "ACTIVE_updated": False}
    atomic_json(path/"BUNDLE.json", manifest, immutable=True)
    return {"status": "COMPLETE", "bundle": str(path), "formal_ACTIVE_updated": False}
