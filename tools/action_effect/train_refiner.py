"""Train A0/A1 small EGR heads from immutable, physically verified G0 repairs.

No optimizer, sampler, resume or checkpoint implementation is duplicated here.
A step contains no simulator; official evaluation is a separate bounded command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .common import (artifact, digest, load_config, read_json, verify_artifact,
                     verify_identity, write_json)
from .context_data import ContextReader, batch_context
from .feedback_schema import from_jsonable, label_arrays
from .training import run_training
from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner
from starVLA.model.modules.action_effect.effect_model import EffectModelConfig, FeedbackLabelSpec
from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner, RefinerConfig
from starVLA.model.modules.action_effect.intervention_basis import reconstruct, BasisSpec

ARMS = ("A0", "A1", "ScoreRefiner")
ACTIVE_GRID_COEFFICIENTS = (True, False, True, True, True, False, True, True)


def _labels_match(compact, actual):
    return (compact["names"] == actual["names"] and compact["units"] == actual["units"]
            and np.array_equal(np.asarray(compact["valid"], dtype=np.bool_), actual["valid"])
            and np.array_equal(np.asarray(compact["values"], dtype=np.float64), actual["values"]))


class RepairDataset:
    """Load once, verify native receipts, retain only explicit role/target records.

    The dataset contains supervision, but context handed to the model comes only
    from ContextReader/batch_context's strict memory/ego/mask whitelist.
    """
    def __init__(self, dataset_path, context_reader, *, verify_feedback=True):
        self.path = Path(dataset_path)
        self.manifest = verify_identity(read_json(self.path))
        m = self.manifest
        if (m.get("schema") != "action_effect_repair_dataset_v1" or m.get("status") != "COMPLETE"
                or m.get("split") != "train" or m.get("deployment_inputs") is not False):
            raise ValueError("Complete registered TRAIN repair dataset required")
        self.registration = verify_identity(read_json(verify_artifact(m["registration"])))
        completion = read_json(verify_artifact(m["population_completion"]))
        if (self.registration["identity"] != m["registration_identity"]
                or completion["registration_identity"] != m["registration_identity"]
                or self.registration["base_checkpoint"] != context_reader.binding["base_checkpoint"]
                or self.registration["context_manifest"] != artifact(context_reader.manifest_path)
                or len(completion["scenes"]) != m["requested_scenes"]):
            raise ValueError("Repair/Base/context/population completion mismatch")
        if (m["base_checkpoint"] != self.registration["base_checkpoint"]
                or m["numeric_runtime_hash"] != self.registration["numeric_runtime_hash"]
                or m["score_protocol_hash"] != self.registration["spec"]["score_protocol_hash"]):
            raise ValueError("Repair simulation identity changed")
        if verify_feedback:
            # Reconstruct compact rows from immutable native scene receipts.
            # Self-hashing a modified compact JSON is not native provenance.
            # This performs no simulation and reuses deterministic completed data.
            from .build_effect_pairs import build
            rebuilt = build(Path(m["registration"]["path"]).parent)
            if rebuilt["repair_dataset"]["sha256"] != artifact(self.path)["sha256"]:
                raise ValueError("Compact repairs differ from deterministic native receipt reconstruction")
        self.reader = context_reader
        self.entries, self.by_id, logs = [], {}, {}
        all_entries = m["entries"]
        if len(all_entries) != m["requested_scenes"] or len({r["scene_id"] for r in all_entries}) != len(all_entries):
            raise ValueError("Repair inventory missing or duplicate scene")
        self.label_spec = None
        for row in all_entries:
            sid, role, log = row["scene_id"], row["role"], row["source_log"]
            if type(row["trainable"]) is not bool:
                raise ValueError("Trainable status must be an explicit boolean")
            if role not in ("fit", "validation") or (log in logs and logs[log] != role):
                raise ValueError("Repair log/source group crosses fitting and validation")
            logs[log] = role
            if (sid not in self.reader.rows or row["context_entry"] != self.reader.rows[sid]
                    or row["observation_hash"] != self.reader.rows[sid]["observation_hash"]
                    or log != self.reader.rows[sid]["source_log"] or role != self.reader.rows[sid]["role"]):
                raise ValueError("Repair scene/observation/context join mismatch")
            current = self.reader[sid]
            base = np.asarray(row["base_physical"], dtype=np.float64)
            if base.shape != (8, 3) or not np.isfinite(base).all() or not np.array_equal(base, current["physical"].double().numpy()):
                raise ValueError("Repair Base trajectory differs from its frozen context")
            if row["trainable"]:
                if row["target_kind"] not in ("identity_preservation", "verified_local_repair"):
                    raise ValueError("Unverified target cannot enter refiner training")
                target, coefficients = np.asarray(row["target_physical"]), np.asarray(row["coefficients"])
                if (target.shape != (8, 3) or coefficients.shape != (8,) or not np.isfinite(target).all()
                        or not np.isfinite(coefficients).all() or np.any(coefficients[[1, 5]] != 0)):
                    raise ValueError("Invalid target or unexposed intervention window")
                if (np.any(np.abs(coefficients) > np.asarray(RefinerConfig().coefficient_scales))
                        or not np.allclose(reconstruct(base, coefficients, BasisSpec()), target, rtol=0, atol=1e-10)):
                    raise ValueError("Verified target coefficients do not reconstruct its physical trajectory")
                if row["target_kind"] == "identity_preservation" and (not np.array_equal(target, base) or np.any(coefficients != 0)):
                    raise ValueError("Keep target must be exactly Base with zero coefficients")
                spec = FeedbackLabelSpec.from_label_arrays(row["base_labels"])
                if spec != FeedbackLabelSpec.from_label_arrays(row["selected_labels"]):
                    raise ValueError("Repair endpoint labels use different units/schema")
                if self.label_spec is not None and self.label_spec != spec:
                    raise ValueError("Dataset label schemas differ")
                self.label_spec = spec
            if verify_feedback and row.get("base_feedback") is not None:
                from .simulate_interventions import verified_feedback
                base_feedback = verified_feedback({"feedback": row["base_feedback"],
                                                   "canonical_receipt": row["base_canonical_receipt"]})
                if (base_feedback["scene_id"] != sid or base_feedback["source_log"] != log
                        or base_feedback["observation_sha256"] != row["observation_hash"]):
                    raise ValueError("Base feedback belongs to another observation")
                if row["base_labels"] is None:
                    if base_feedback["valid"] or row["trainable"]:
                        raise ValueError("Missing Base labels cannot be treated as valid")
                elif (not _labels_match(row["base_labels"], label_arrays(base_feedback))
                        or not np.array_equal(from_jsonable(base_feedback["submitted_trajectory"])["poses"], base)):
                    raise ValueError("Compact Base labels differ from independent native receipt")
                if row["trainable"]:
                    target_feedback = verified_feedback({"feedback": row["selected_feedback"],
                                                         "canonical_receipt": row["selected_canonical_receipt"]})
                    if (not _labels_match(row["selected_labels"], label_arrays(target_feedback))
                            or target_feedback["scene_id"] != sid
                            or any(target_feedback[k] != base_feedback[k] for k in
                                   ("context_hash", "numeric_runtime_hash", "score_protocol_hash", "reference_input_sha256"))
                            or not np.array_equal(from_jsonable(target_feedback["submitted_trajectory"])["poses"], row["target_physical"])):
                        raise ValueError("Verified repair endpoint or paired simulation context mismatch")
            self.entries.append(row); self.by_id[sid] = row
        self.fit = [r for r in self.entries if r["role"] == "fit" and r["trainable"]]
        self.validation = [r for r in self.entries if r["role"] == "validation"]
        if self.label_spec is None:
            raise ValueError("No verified repair or identity-preservation supervision")


def prepare_dataset(cfg, dataset_path, context_manifest, campaign_binding=None):
    binding = Path(campaign_binding) if campaign_binding else Path(cfg["artifact_root"]) / "CAMPAIGN_BINDING.json"
    reader = ContextReader(context_manifest, binding)
    # Pin the actual manifest path, never infer it from a timestamp/directory scan.
    reader.manifest_path = Path(context_manifest)
    return RepairDataset(dataset_path, reader)


def model_config_from_record(record, cfg, seed):
    memory = record["memory"]
    return EffectModelConfig(context_dim=memory.shape[1], world_tokens=record["W_count"],
                             action_tokens=len(memory)-record["W_count"], ego_dim=record["ego"].numel(),
                             hidden_dim=cfg["training"]["hidden"], layers=cfg["training"]["layers"],
                             heads=cfg["training"]["heads"], init_seed=seed+301)


def build_model(cfg, dataset, *, arm, seed, device, effect_checkpoint=None, effect_identity=None):
    if arm not in ARMS or (arm == "A0") != (effect_checkpoint is None):
        raise ValueError("A0 must not load a teacher; A1/ScoreRefiner require an explicit effect checkpoint")
    record = dataset.reader[dataset.fit[0]["scene_id"]]
    config = RefinerConfig(model=model_config_from_record(record, cfg, seed),
                           active_coefficients=ACTIVE_GRID_COEFFICIENTS, init_seed=seed+301)
    teacher = None
    if effect_checkpoint is not None:
        from .train_effect_model import load_effect_teacher
        teacher = load_effect_teacher(effect_checkpoint,
            expected_base_binding_hash=dataset.reader.binding["base_checkpoint_binding_hash"], identity_path=effect_identity)
        expected_mode = "delta_feedback" if arm == "A1" else "absolute_score"
        if teacher.config.mode != expected_mode or teacher.label_spec != dataset.label_spec:
            raise ValueError("Refiner arm requires matching actually trained teacher mode and feedback schema")
    model = EffectGuidedRefiner(CoefficientRefiner(config), effect_teacher=teacher).to(device)
    return model


def loss_weights(arm):
    result = {"supervision": 1., "magnitude": .01, "preserve": 1.}
    if arm != "A0": result.update(effect_consistency=1., predicted_safety=1.)
    return result


def make_step(dataset, *, arm, microbatch_scenes, device):
    if type(microbatch_scenes) is not int or microbatch_scenes < 1:
        raise ValueError("Bounded positive microbatch required")
    weights = loss_weights(arm)
    protected = [dataset.label_spec.names.index(n) for n in
                 ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")]
    def step(model, indices, optimizer_step):
        selected = [dataset.fit[i] for i in indices]
        if not selected: raise ValueError("Empty fitting update")
        high = torch.tensor([r["target_kind"] == "identity_preservation" for r in selected], device=device)
        masks = torch.as_tensor(np.stack([np.asarray(r["base_labels"]["valid"], dtype=np.bool_)
                                          & np.asarray(r["selected_labels"]["valid"], dtype=np.bool_)
                                          for r in selected]), dtype=torch.bool, device=device)
        if arm == "ScoreRefiner":
            masks &= torch.tensor([u == "0..1" for u in dataset.label_spec.units], device=device)
        denominators = {"supervision": len(selected), "magnitude": len(selected), "preserve": int(high.sum())}
        if arm != "A0":
            denominators.update(effect_consistency=int(masks.any(1).sum()), predicted_safety=int(masks[:, protected].any(1).sum()))
        numerators = {name: 0. for name in weights}
        for start in range(0, len(selected), microbatch_scenes):
            rows = selected[start:start+microbatch_scenes]
            context_rows = [dataset.reader[r["scene_id"]] for r in rows]
            base = torch.tensor([r["base_physical"] for r in rows], dtype=torch.float32, device=device)
            target = torch.tensor([r["target_physical"] for r in rows], dtype=torch.float32, device=device)
            coefficients = torch.tensor([r["coefficients"] for r in rows], dtype=torch.float32, device=device)
            args = {"target_trajectory": target, "target_coefficients": coefficients,
                    "high_quality": high[start:start+len(rows)]}
            if arm != "A0":
                effects = np.stack([np.asarray(r["selected_labels"]["values"])-np.asarray(r["base_labels"]["values"])
                                    for r in rows])
                args.update(target_effect=torch.as_tensor(effects, dtype=torch.float32, device=device),
                            effect_valid=masks[start:start+len(rows)], label_spec=dataset.label_spec)
            result = model.training_loss(batch_context(context_rows, device), base,
                                         use_effect_guidance=arm != "A0", **args)
            # Complete-update denominators are known from metadata; release each
            # microbatch graph immediately without changing any task's objective.
            loss = sum(result.contributions[name].numerator * weights[name] / max(denominators[name], 1)
                       for name in weights)
            loss.backward()
            for name in weights: numerators[name] += float(result.contributions[name].numerator.detach())
        terms = {name: numerators[name]/max(denominators[name], 1) for name in weights}
        return {**terms, "total": sum(terms[n]*weights[n] for n in weights),
                **{n+"_denominator": d for n, d in denominators.items()},
                "identity_scenes": int(high.sum()), "repair_scenes": len(selected)-int(high.sum())}
    return step


def train(cfg, *, dataset_path, context_manifest, output, arm, device, seed, max_updates,
          stop_after=None, resume=False, campaign_binding=None, effect_checkpoint=None, effect_identity=None):
    if not 1 <= max_updates <= cfg["training"]["max_updates"]:
        raise ValueError("Explicit bounded updates exceed the campaign recipe")
    dataset = prepare_dataset(cfg, dataset_path, context_manifest, campaign_binding)
    if not dataset.fit: raise ValueError("NO_VERIFIED_FIT_REPAIR_OR_KEEP_TARGETS")
    model = build_model(cfg, dataset, arm=arm, seed=seed, device=device,
                        effect_checkpoint=effect_checkpoint, effect_identity=effect_identity)
    source = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip()
    identity = {"arm": arm, "source_commit": source, "base_binding_hash": dataset.reader.binding["base_checkpoint_binding_hash"],
        "data_artifact": artifact(dataset_path), "context_artifact": artifact(context_manifest),
        "model_config": asdict(model.refiner.config), "label_spec": asdict(dataset.label_spec),
        "effect_teacher": getattr(model.effect_teacher, "provenance", None),
        "fit_scene_ids": [r["scene_id"] for r in dataset.fit], "fit_logs": sorted({r["source_log"] for r in dataset.fit}),
        "role": "fit_only", "event_guidance_enabled": False,
        "continuous_combinations": "OOD relative to the discrete fixed24 grid; requires actual simulator evaluation"}
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output)/"RESOLVED_CONFIG.json", cfg)
    return run_training(model, output=output, identity=identity, dataset_size=len(dataset.fit),
        step_fn=make_step(dataset, arm=arm, microbatch_scenes=cfg["training"]["microbatch_scenes"], device=device),
        cfg=cfg, device=device, seed=seed, max_updates=max_updates, stop_after=stop_after, resume=resume)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--repair-dataset", required=True)
    p.add_argument("--context-manifest", required=True); p.add_argument("--campaign-binding")
    p.add_argument("--output", required=True); p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int)
    p.add_argument("--max-updates", type=int, required=True); p.add_argument("--stop-after", type=int)
    p.add_argument("--effect-checkpoint"); p.add_argument("--effect-identity")
    p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.dry_run:
            print(json.dumps({"status": "NOT_RUN", "arm": a.arm, "config_hash": digest(cfg), "simulation_calls": 0})); return 0
        r = train(cfg, dataset_path=a.repair_dataset, context_manifest=a.context_manifest,
                  output=a.output, arm=a.arm, device=a.device, seed=cfg["seed"] if a.seed is None else a.seed,
                  max_updates=a.max_updates, stop_after=a.stop_after, resume=a.resume,
                  campaign_binding=a.campaign_binding, effect_checkpoint=a.effect_checkpoint, effect_identity=a.effect_identity)
        print(json.dumps(r)); return 0 if r["status"] in ("COMPLETE", "PAUSED") else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})); return 1


if __name__ == "__main__": raise SystemExit(main())
