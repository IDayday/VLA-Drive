"""Original DDP DA3 metric depth supervision in a NEW independently identified cache."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .run_meter import metered_run


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("source-manifest", "observation-root", "tokens", "output", "campaign-root", "run-id"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    with metered_run(a.campaign_root, a.run_id, int(a.device.startswith("cuda")),
                     {"kind": "original_generic_depth_labels", "max_seconds": a.max_seconds}) as (record, run_dir, save):
        started = time.monotonic()
        import numpy as np
        import torch
        from starVLA.model.modules.vehicle_joint.initialization import verify_generic_source, identity_hash
        sys.path.insert(0, str(Path(__file__).parents[2]/"depth_process/Depth-Anything-3/src"))
        from depth_anything_3.api import DepthAnything3
        source = json.loads(Path(a.source_manifest).read_text())["da3"]
        verify_generic_source(source["root"], source)
        identity = {"schema": "ddpolicy_generic_da3_labels_v1", "repository": source["repository"],
                    "revision": source["revision"], "files": source["files"], "process_res": 252,
                    "camera_order": ["CAM_F0", "CAM_L0", "CAM_R0"], "input_time": "current_only"}
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
        if (out/"identity.json").exists():
            if json.loads((out/"identity.json").read_text()) != identity:
                raise ValueError("Depth cache provenance mismatch")
        else: (out/"identity.json").write_text(json.dumps(identity, indent=2))
        record["identity"] = identity_hash(identity); save()
        model = DepthAnything3.from_pretrained(source["root"]).to(a.device).eval()
        tokens = json.loads(Path(a.tokens).read_text())
        if isinstance(tokens, dict): tokens = tokens["train_tokens"]+tokens.get("dev_tokens", [])
        if a.limit: tokens = tokens[:a.limit]
        completed = 0
        for token in tokens:
            if time.monotonic()-started >= a.max_seconds:
                record["incomplete_reason"] = "startup time allocation exhausted"; break
            dst = out/(token+".npz")
            if dst.exists(): continue
            with np.load(Path(a.observation_root)/"observations"/(token+".npz")) as obs:
                images = obs["image_paths"].tolist()
            with torch.inference_mode():
                prediction = model.inference(images, process_res=252)
            depth = np.asarray(prediction.depth, dtype=np.float32)
            if depth.ndim != 3 or depth.shape[0] != 3 or not np.isfinite(depth).all() or (depth <= 0).all():
                raise ValueError("Invalid original generic depth output")
            tmp = dst.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, depth=depth, identity=np.array(record["identity"]))
            os.replace(tmp, dst)
            completed += 1; record["inference_scenes"] = completed; save()
        record["requested_scenes"] = len(tokens)
        record["written_scenes"] = completed
        print(json.dumps({"written_scenes": completed, "requested_scenes": len(tokens)}), flush=True)


if __name__ == "__main__": main()
