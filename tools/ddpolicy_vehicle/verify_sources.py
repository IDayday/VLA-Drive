"""Verify generic local files against public immutable HF revisions (no downloads)."""
import argparse
import hashlib
import json
from pathlib import Path
import requests
from starVLA.model.modules.vehicle_joint.initialization import file_sha256

SOURCES = {
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "models/Qwen3-VL-2B-Instruct"),
    "wan": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "fc913c34361f4ec879e2f9c78b4f11ae50a937d1", "models/Wan2.1-Fun-V1.1-1.3B-InP"),
    "ppd": ("gangweix/Pixel-Perfect-Depth", "be33763bc1bc1c581869a20ffde66c36b304b1a7", "depth_model_ckpts"),
    "depth_v2": ("depth-anything/Depth-Anything-V2-Large", None, "depth_model_ckpts"),
    "da3": ("depth-anything/DA3METRIC-LARGE", None, "models/da3metric-large"),
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--asset-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for key, (repo, revision, relative) in SOURCES.items():
        root = Path(a.asset_root) / relative
        if revision is None:
            metadata = root / ".cache/huggingface/download/config.json.metadata"
            revision = metadata.read_text().splitlines()[0] if metadata.is_file() else "main"
        r = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=45)
        r.raise_for_status()
        public = r.json()
        # main only discovers a revision; it is never a mutable training identity.
        revision = public["sha"]
        (out / (key + "_public.json")).write_text(json.dumps(public, indent=2))
        files, checked = {}, []
        for item in public["siblings"]:
            name = item["rfilename"]
            path = root / name
            if not path.is_file(): continue
            sha = file_sha256(path)
            if "lfs" in item:
                if sha != item["lfs"]["sha256"]:
                    raise ValueError(f"Public weight mismatch: {key}/{name}")
                checked.append(name)
            else:
                data = path.read_bytes()
                git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if git_blob != item["blobId"]:
                    raise ValueError(f"Public configuration mismatch: {key}/{name}")
            files[name] = sha
        if not checked: raise ValueError(f"No verified public weights for {key}")
        records[key] = {"repository": repo, "revision": revision, "root": str(root),
                        "files": files, "public_lfs_verified": checked,
                        "generic_pretraining_data_fully_auditable": False}
        (out / "sources.json").write_text(json.dumps(records, indent=2))
        print(f"{key}: {len(files)} files verified against {revision}", flush=True)


if __name__ == "__main__": main()
