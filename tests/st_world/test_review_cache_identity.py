"""Independent review counterexample: verify must bind the requested population.

This fixture is an IO identity test, not DINO or real-image integration evidence.
The test deliberately requires rejection; it fails on audited cc9057c3.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from starVLA.dataloader.st_world_history import image_reference
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheWriter
from tools.st_world_policy import cache_teacher


def test_verify_rejects_a_different_valid_scene_manifest(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 18), "blue").save(image)
    ref = image_reference(image)
    contract = TeacherFeatureContract(
        image_hw=(32, 32), patch_hw=(16, 16), native_grid_hw=(2, 2),
        output_grid_hw=(2, 2), channels=8, storage_dtype="float32",
        compute_dtype="float32", view_names=("CAM_F0", "CAM_L0", "CAM_R0"),
        teacher_identity="a" * 64, transform_identity="b" * 64,
        pooling={"kind": "none", "kernel_hw": (1, 1), "stride_hw": (1, 1)},
    )
    manifest = {
        "schema": "st_teacher_requests_v1", "split": "train", "times_s": [0.],
        "view_names": list(contract.view_names), "timestamp_tolerance_s": .05,
        "scenes": [{"token": "original_scene", "log": "original_log", "split": "train",
                    "timestamp_us": 0, "frames": [{"requested_s": 0., "timestamp_us": 0,
                                                     "views": [ref] * 3}]}],
    }
    manifest["identity"] = identity_hash(manifest)
    writer = TeacherCacheWriter(tmp_path / "cache", contract, manifest, images_per_shard=1)
    with writer.shard_lock(0):
        writer.commit_shard(0, torch.zeros(1, 2, 2, 8), torch.ones(1, 2, 2, dtype=torch.bool))
    writer.finalize()
    other = copy.deepcopy(manifest)
    other["scenes"][0].update(token="different_scene", log="different_log")
    other["identity"] = identity_hash({k: v for k, v in other.items() if k != "identity"})
    requests = tmp_path / "different_requests.json"
    requests.write_text(json.dumps(other))
    # Only replace the independent teacher-loading descriptor. The real verify
    # CLI branch and immutable writer/reader remain the code under review.
    monkeypatch.setattr(cache_teacher, "make_contract", lambda *_args, **_kwargs: (contract, {}, {}))
    args = SimpleNamespace(
        requests=str(requests), command="verify", output=str(writer.root),
        teacher_root="unused_fixture_teacher", variant="SP2", storage_dtype="float32",
        legacy_c1_root=None, max_scenes=1, images_per_shard=1, batch_size=1,
        workers=1, cpu_threads=1, max_seconds=10., min_free_gib=0.,
        mode="smoke", registration=None, rank=0, world_size=1,
    )
    with pytest.raises(ValueError, match="[Mm]anifest|[Pp]opulation|[Rr]equest"):
        cache_teacher.run(args)
