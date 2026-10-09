from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from ..contracts import FeatureBundle, FeatureRecord, strict_record, require
from ..io import atomic_json, atomic_torch, file_hash, read_json, lock


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, record: FeatureRecord, bundle: FeatureBundle):
        require(bundle.scene_ids == (record.scene_id,), "feature cache requires a named single-scene record")
        key = record.cache_key
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        with lock(folder / "WRITE.lock"):
            if (folder / "COMPLETE.json").exists():
                return self.get(record)
            f = bundle.detached().to("cpu")
            payload = {"scene": f.scene, "ego": f.ego, "valid_tokens": f.valid_tokens, "conditions": f.conditions,
                       "scene_ids": f.scene_ids, "contract_hash": f.contract_hash}
            atomic_torch(folder / "features.pt", payload)
            d = asdict(record) | {"tensor_ref": str(folder / "features.pt"), "checksum": file_hash(folder / "features.pt"),
                                  "shapes": {k: list(payload[k].shape) for k in ("scene", "ego", "valid_tokens")}, "valid": True, "error": None}
            atomic_json(folder / "COMPLETE.json", d, immutable=True)
        return f

    def get(self, expected: FeatureRecord):
        folder = self.root / expected.cache_key
        if not (folder / "COMPLETE.json").is_file():
            raise FileNotFoundError(f"feature cache missing/incomplete: {folder}")
        actual = strict_record(FeatureRecord, read_json(folder / "COMPLETE.json"))
        require(actual.cache_key == expected.cache_key and actual.valid, "feature cache contract mismatch")
        require(file_hash(actual.tensor_ref) == actual.checksum, "feature cache checksum mismatch")
        t = torch.load(actual.tensor_ref, map_location="cpu", weights_only=True)
        require(all(list(t[k].shape) == v for k, v in actual.shapes.items()), "cache shapes mismatch")
        result = FeatureBundle(**t)
        require(result.scene_ids == (expected.scene_id,), "feature tensor scene identity mismatch")
        return result
