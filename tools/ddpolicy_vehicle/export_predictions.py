"""Resumable FP32 current-camera export, one joint sample and sole ego slot0.

No labels/metric environment are read here. CPU official scoring runs separately
and consumes atomic scene exports. Navtest requires a frozen final-model lock.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .prepare_data import atomic_json
from .run_meter import metered_run


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ("training-run", "checkpoint-tag", "current-root", "output", "campaign-root", "run-id"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--sampling-seed", type=int, required=True)
    p.add_argument("--rank", type=int, default=0); p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--final-lock")
    p.add_argument("--local-checkpoint-cache", help="Optional allocated local disk for exact copies of this campaign's evaluation checkpoints")
    a = p.parse_args()
    if not 0 <= a.rank < a.world_size or a.limit < 0 or a.max_seconds < 1: raise ValueError("Invalid shard/budget")
    with metered_run(a.campaign_root, a.run_id, 1, {"kind": "camera_prediction_export", "real_optimizer_updates": 0}) as (record, _, save):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from types import SimpleNamespace
        from starVLA.dataloader.ddpolicy_current import CurrentCameraDataset
        from starVLA.model.framework.DDPVehicle import DDPVehicle
        from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
        from .checkpoints import checkpoint_identity, scene_noise, stage_checkpoint
        dataset = CurrentCameraDataset(a.current_root)
        training, checkpoint = checkpoint_identity(a.training_run, a.checkpoint_tag)
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"]): raise ValueError("Commit evaluation source before export")
        cfg = OmegaConf.create(training["config"])
        protocol = {"precision": "FP32", "samples_per_scene": 1, "sampling_seed": a.sampling_seed,
                    "solver": "original Euler", "steps": int(cfg.framework.action_model.num_inference_timesteps),
                    "scorer": None, "camera_only": True, "gt_future_conditioning": False,
                    "noise_protocol": "ddpolicy-v1 SHA256 token/seed; CPU randn 1x9x8x4; ego slot0"}
        if dataset.metadata.get("split") == "navtest":
            if not a.final_lock or a.limit or checkpoint["startup"]: raise ValueError("Navtest requires full, formally trained, frozen endpoint")
            lock = json.loads(Path(a.final_lock).read_text())
            if checkpoint["sha256"] not in lock["checkpoint_sha256"] or source != lock["evaluation_source_sha"]:
                raise ValueError("Model/source was not frozen before Navtest")
            if a.sampling_seed not in lock["sampling_seeds"] or protocol["steps"] != lock["inference_steps"]:
                raise ValueError("Navtest protocol differs from frozen protocol")
            if len(dataset) != 12146 or len({r["log"] for r in dataset.index}) != 136: raise ValueError("Incomplete Navtest population")
        identity = {"checkpoint": checkpoint, "current_identity": dataset.identity, "evaluation_source": source,
                    "protocol": protocol, "world_size": a.world_size, "limit": a.limit}
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True); (out/"predictions").mkdir(exist_ok=True)
        identity_path = out/"identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity: raise ValueError("Export resume identity mismatch")
        else: atomic_json(identity_path, identity)
        record.update(checkpoint=checkpoint["sha256"], evaluation_source=source, protocol=protocol); save()
        checkpoint_root = Path(a.training_run)/"checkpoints"
        if a.local_checkpoint_cache:
            checkpoint_root = stage_checkpoint(a.training_run, a.checkpoint_tag, checkpoint, a.local_checkpoint_cache)
        sources = json.loads(Path(cfg.from_scratch.source_manifest).read_text())
        os.environ["DEPTH_MODEL_CKPTS"] = sources["ppd"]["root"]
        cfg.framework.qwenvl.device_map = "cpu"
        model = DDPVehicle(cfg, accelerator=SimpleNamespace(process_index=0, device=torch.device("cpu")))
        # Restore FP32 masters, not a BF16 round-trip of the learned parameters.
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_root), tag=a.checkpoint_tag)
        model.float(); model.load_state_dict(state, strict=True); del state
        model.to("cuda").eval(); model.inference_fp32 = True
        indices = list(range(min(a.limit or len(dataset), len(dataset))))[a.rank::a.world_size]
        failed = complete = 0
        for index in indices:
            if time.time()-record["start_unix"] >= a.max_seconds:
                record["status"] = "PAUSED"; break
            token = dataset.index[index]["token"]; destination = out/"predictions"/(token+".npz")
            metadata_path = destination.with_suffix(".json")
            if metadata_path.exists():
                previous = json.loads(metadata_path.read_text())
                if previous["identity_sha256"] != identity_hash(identity): raise ValueError("Scene export identity changed")
                if previous["status"] == "ok" and file_sha256(destination) != previous["proposal_sha256"]:
                    raise ValueError("Scene export artifact changed")
                failed += previous["status"] != "ok"; complete += 1; continue
            row = {"token": token, "log": dataset.index[index]["log"], "identity_sha256": identity_hash(identity), "status": "ok"}
            try:
                example = dataset[index]
                noise = scene_noise(token, a.sampling_seed, 9, "cuda")
                if not model.joint_enabled: noise = noise[:, 0]
                start = time.monotonic()
                result = model.predict_action([example], initial_noise=noise)
                torch.cuda.synchronize(); row["inference_seconds"] = time.monotonic()-start
                arrays = {"trajectory": result["ego"][0].cpu().numpy(), "joint_encoded": result["joint_encoded"][0].float().cpu().numpy()}
                for key in ("vehicle_xy", "active_actor_mask", "selected_query_indices"):
                    if result[key] is not None: arrays[key] = result[key][0].cpu().numpy()
                if result["vehicle_prediction"] is not None:
                    arrays.update({"vehicle_"+key:value[0].float().cpu().numpy() for key,value in result["vehicle_prediction"].items()})
                if any(not np.isfinite(value).all() for value in arrays.values()): raise FloatingPointError("Nonfinite prediction field")
                temp = destination.with_suffix(f".{os.getpid()}.tmp")
                with temp.open("wb") as stream: np.savez_compressed(stream, **arrays)
                os.replace(temp, destination)
                row["proposal_sha256"] = file_sha256(destination); row["graph"] = result["graph_audit"]
            except Exception as error:
                row.update(status="failed", error=repr(error)); failed += 1
            atomic_json(metadata_path, row); complete += 1
            record.update(inference_scenes=complete, failed=failed); save()
        shard = {"status": "complete" if complete == len(indices) else "paused", "completed": complete,
                 "requested": len(indices), "failed": failed, "identity_sha256": identity_hash(identity)}
        atomic_json(out/f"shard_{a.rank}.json", shard)
        record.update(inference_scenes=complete, failed=failed)
        if failed: raise RuntimeError("Failed predictions retained; evaluation is invalid")


if __name__ == "__main__": main()
