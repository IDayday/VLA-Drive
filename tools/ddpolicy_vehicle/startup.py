"""Bounded REAL camera full-recipe forward/backward; not a formal experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .run_meter import metered_run


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("asset-root", "source-manifest", "processed-root", "vehicle-root", "depth-root",
                 "tokens", "campaign-root", "run-id"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--arm", choices=["A", "B", "C"], default="B")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--backward", action="store_true")
    a = p.parse_args()
    with metered_run(a.campaign_root, a.run_id, 1, {"kind": "actual_camera_full_recipe_startup",
            "arm": a.arm, "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_diff_sha256": __import__("hashlib").sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
            "real_optimizer_updates": 0}) as (record, out, save):
        import numpy as np
        import torch
        from accelerate import Accelerator
        from omegaconf import OmegaConf
        from .configuration import make_config
        from starVLA.dataloader.ddpolicy_vehicle_dataset import DDPVehicleDataset
        from starVLA.model.framework.DDPVehicle import DDPVehicle
        from starVLA.model.modules.vehicle_joint.initialization import module_manifest
        from starVLA.model.modules.vehicle_joint.masks import VehicleRoleScheduler
        accelerator = Accelerator(mixed_precision="bf16")
        os.environ["DEPTH_MODEL_CKPTS"] = str(Path(a.asset_root)/"depth_model_ckpts")
        cfg = make_config(a.asset_root, a.source_manifest, a.arm)
        OmegaConf.save(cfg, out/"config.yaml")
        ds = DDPVehicleDataset(a.tokens, a.processed_root, a.vehicle_root, a.depth_root, cfg,
                              sensor_roots=["/mnt/navsim/trainval_sensor_blobs/trainval"])
        batch = [ds[i] for i in range(min(a.samples, len(ds)))]
        load_start = time.monotonic()
        model = DDPVehicle(cfg, accelerator=accelerator).to("cuda")
        record["load_seconds"] = time.monotonic()-load_start
        record["parameters"] = sum(p.numel() for p in model.parameters())
        record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record["generic_ppd_load"] = getattr(model, "ppd_generic_load_report", None)
        # Driving modules are locally initialized tensors. Hash them before any
        # backward/optimizer operation; public model files were checked on load.
        names = ["action_model", "action_input_model", "vehicle_reader", "vehicle_heads",
                 "traj_emb", "rgb_act_pre", "gs_traj_emb", "gs_act_pre"]
        initial = {n: module_manifest(getattr(model, n)) for n in names if hasattr(model, n)}
        from starVLA.model.modules.vehicle_joint.initialization import tensor_hash
        for name in ("rgb_query", "gs_query", "traj_emb_h0", "gs_traj_emb_h0"):
            if hasattr(model, name): initial[name] = tensor_hash(getattr(model, name))
        if hasattr(model, "rgb_model"):
            initial["rgb_model.qwen_proj_video"] = module_manifest(model.rgb_model.qwen_proj_video)
        if hasattr(model, "gs_model"):
            initial["gs_model.dit.qwen_proj"] = module_manifest(model.gs_model.dit.qwen_proj)
            initial["gs_model.dit.qwen_cross_attn"] = module_manifest(model.gs_model.dit.qwen_cross_attn)
        initial["qwen_driving_token_embeddings"] = model.qwen_vl_interface.driving_token_initialization
        if hasattr(model, "vehicle_token_initialization"):
            initial["qwen_vehicle_token_embeddings"] = model.vehicle_token_initialization
        (out/"driving_initialization.json").write_text(json.dumps(initial, indent=2))
        save()
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model.train()
        start = time.monotonic()
        output = model(batch, role_scheduler=VehicleRoleScheduler(2042), completed_updates=0)
        torch.cuda.synchronize()
        record["forward_seconds"] = time.monotonic()-start
        record["losses"] = {k: float(v.detach()) for k,v in output["losses"].items()}
        record["metrics"] = output["metrics"]
        if not torch.isfinite(output["loss"]): raise FloatingPointError("Nonfinite actual camera loss")
        if a.backward:
            start = time.monotonic(); output["loss"].backward(); torch.cuda.synchronize()
            record["backward_seconds"] = time.monotonic()-start
            grads = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None: continue
                if not torch.isfinite(parameter.grad).all(): raise FloatingPointError("Nonfinite gradient: "+name)
                group = name.split(".")[0]
                grads[group] = grads.get(group, 0.) + float(parameter.grad.float().square().sum())
            record["gradient_l2"] = {k:v**.5 for k,v in grads.items()}
            record["backward_scenes"] = len(batch)
        record["inference_scenes"] = len(batch)
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        print(json.dumps({k:v for k,v in record.items() if k not in ("metrics", "generic_ppd_load")}), flush=True)


if __name__ == "__main__": main()
