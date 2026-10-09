"""32 audited GT observations, fixed cached inputs, isolated diagnostic weights."""
from copy import deepcopy
from pathlib import Path
import torch
from .contracts import SceneRecord, strict_record, require
from .data.sources import load_target
from .data.sampler import ConsumedSampler
from .evaluation.retention import frozen_snapshot, audit_frozen
from .pipeline import stack_features
from .training.trainer import train
from .io import read_json, digest, atomic_json


def overfit_32(pipeline, number, steps=32, resume=None):
    manifest = read_json(pipeline.round_root(number) / "expert_manifest.json")
    require(not manifest["probe"], "overfit_32 requires 32 audited targets; smoke probes are separate evidence")
    records = {r["scene_id"]: strict_record(SceneRecord, r) for r in manifest["records"]}
    ids = sorted(set(manifest["buckets"]["hard_original"]))[:32]
    require(len(ids) == 32, "overfit_32 needs 32 distinct audit-passing coverage scenes")
    rows = [records[s] for s in ids]
    model = pipeline.model(number - 1)
    eid = f"expert_{number}"
    require(eid not in model.experts, "overfit is an isolated job; run in a fresh CLI process")
    model.append_expert(eid)
    model.set_trainable_stage("expert_train", eid)
    features = stack_features([pipeline.cached(s) for s in rows]).to(pipeline.device)
    targets = torch.stack([load_target(s, pipeline.trajectory_contract) for s in rows]).to(pipeline.device)
    before = frozen_snapshot(model, eid)
    references = {old: {"raw": o.raw, "physical": o.physical} for old in model.experts if old != eid
                  for o in [model.adapter.expert_forward(model.experts[old], features)]}
    def metrics():
        model.eval()
        with torch.no_grad():
            out = model.experts[eid](features)
            truth = pipeline.trajectory_contract.physical(targets)
            return {"ADE_m": float((out.physical[..., :2] - truth[..., :2]).norm(dim=-1).mean()),
                    "raw_L1": float((out.raw - targets).abs().mean())}
    initial = metrics()
    entries = [{"scene_id": sid, "index": i, "bucket": "diagnostic", "source_group_id": rows[i].source_group_id}
               for _ in range(steps) for i, sid in enumerate(ids)]
    plan = {"entries": entries, "plan_hash": digest(entries)}
    cfg = deepcopy(pipeline.config["expert_train"])
    cfg.update(global_batch_size=32, max_optimizer_steps=steps, warmup_steps=min(steps - 1, cfg["warmup_steps"]),
               validation_every_steps=steps, diagnostic_disable_dropout=True)
    def fetch(entries, device):
        ii = [e["index"] for e in entries]
        return {"features": features.subset(ii).to(device), "target": targets[ii].to(device)}
    def loss(expert, batch):
        return model.adapter.compute_expert_il_loss(expert(batch["features"]), batch["target"])
    folder = pipeline.root / "diagnostics" / "overfit_32" / f"round_{number:03d}"
    train(model.experts[eid], loss, fetch, ConsumedSampler(plan), cfg, {"manifest": manifest["manifest_hash"], "frozen": before}, folder,
          device=pipeline.device, resume=resume,
          denominator_function=lambda d: {"il": d["target"].new_tensor(len(d["target"]))})
    final = metrics()
    model.set_trainable_stage("expert_train", eid)
    frozen = audit_frozen(model, before, references, features, pipeline.config["tolerances"]["fp32_cuda" if pipeline.device.startswith("cuda") else "fp32_cpu"])
    result = read_json(folder / "result.json") | {"diagnostic_only": True, "science": "UNTESTED", "scene_ids": ids,
        "fixed_targets": [s.target_id for s in rows], "before": initial, "after": final, "frozen_audit": frozen}
    atomic_json(folder / "diagnostic.json", result)
    return result
