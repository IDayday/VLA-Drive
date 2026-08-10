import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.field2plan.dynamics_teachers import (
    DynamicsTeacherSample,
    OfficialVJEPA2Adapter,
)


VIEWS = ("cam_f0", "cam_l0", "cam_r0")


def _sample() -> DynamicsTeacherSample:
    return DynamicsTeacherSample(
        token="token-a",
        view_names=VIEWS,
        features=np.ones((8, 3, 6, 4, 5), dtype=np.float16),
        confidence=np.ones((8, 3, 4, 5), dtype=np.float32),
        valid_mask=np.ones((8, 3, 4, 5), dtype=np.bool_),
        frame_indices=np.arange(4, 12, dtype=np.int64),
        frame_times_s=np.arange(1, 9, dtype=np.float32) * 0.5,
        source_image_hw=np.full((8, 3, 2), [1080, 1920], dtype=np.int64),
        feature_hw=np.full((8, 3, 2), [4, 5], dtype=np.int64),
        spatial_layout="per_view_patch_grid",
        feature_normalization="l2",
        metadata={"selected_layers": [-1], "temporal_stride": 2},
    )


def test_dynamics_teacher_sample_schema_is_strict() -> None:
    sample = _sample().validate()
    assert sample.features.shape == (8, 3, 6, 4, 5)
    assert sample.features.dtype == np.float16

    broken = _sample()
    object.__setattr__(broken, "confidence", np.ones((8, 3, 4, 4), np.float32))
    with pytest.raises(ValueError, match="confidence"):
        broken.validate()


def test_vjepa_adapter_is_lazy_and_never_downloads(tmp_path, monkeypatch) -> None:
    imported = []
    original_import = importlib.import_module

    def tracked_import(name, package=None):
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked_import)
    adapter = OfficialVJEPA2Adapter(
        local_repo=tmp_path / "missing-repo",
        checkpoint=tmp_path / "missing.pt",
        model_variant="vjepa2_1_vit_large",
        device="cpu",
    )
    assert not any("vjepa" in name.lower() for name in imported)

    with pytest.raises(FileNotFoundError, match="local repo"):
        adapter.load_model()
    assert not any("vjepa" in name.lower() for name in imported)


def test_vjepa_preprocess_uses_manifest_center_square_crop(tmp_path) -> None:
    adapter = OfficialVJEPA2Adapter(
        local_repo=tmp_path,
        checkpoint=tmp_path / "unused.pt",
        device="cpu",
    )
    frames = np.zeros((1, 12, 4, 8, 3), dtype=np.uint8)
    frames[..., :2, 0] = 255
    frames[..., 6:, 0] = 255

    video = adapter.preprocess_video(frames)

    assert video.shape == (1, 3, 12, 384, 384)
    expected_red = torch.tensor(-0.485 / 0.229)
    torch.testing.assert_close(video[:, 0].mean(), expected_red, atol=1e-5, rtol=0)


def test_vjepa_rope_compatibility_keeps_query_dtype() -> None:
    observed = {}

    def upstream(x, pos, n_registers, has_cls_first):
        observed["position_dtype"] = pos.dtype
        return x.float()

    module = SimpleNamespace(rotate_queries_or_keys=upstream)
    OfficialVJEPA2Adapter._install_rope_dtype_compatibility(module)
    query = torch.ones(1, 1, 2, 2, dtype=torch.bfloat16)
    position = torch.arange(2, dtype=torch.float32)

    output = module.rotate_queries_or_keys(query, position, 0, False)

    assert observed["position_dtype"] == torch.bfloat16
    assert output.dtype == torch.bfloat16
