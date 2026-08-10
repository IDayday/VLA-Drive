import importlib
import pickle

import numpy as np
import pytest

from starVLA.model.modules.field2plan.geometry_teachers import (
    DA3LegacyDepthAdapter,
    GeometryTeacherSample,
    OfficialVGGTMetricDepthAdapter,
    VGGTAdapter,
    estimate_metric_scale_from_depth_reference,
    estimate_metric_scale_from_camera_rig,
)


VIEW_NAMES = ("cam_f0", "cam_l0", "cam_r0")


def _write_legacy_da3(path, shape=(4, 6)) -> None:
    payload = {
        view: np.full(shape, index + 2.0, dtype=np.float32)
        for index, view in enumerate(VIEW_NAMES)
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def test_da3_legacy_adapter_exposes_honest_metric_depth_schema(tmp_path) -> None:
    token = "scene-token"
    _write_legacy_da3(tmp_path / f"{token}.pkl-depth.pkl")
    adapter = DA3LegacyDepthAdapter(tmp_path)

    sample = adapter.load_cached(
        token=token,
        source_image_hw=np.asarray([[1080, 1920]] * 3, dtype=np.int64),
    )

    assert isinstance(sample, GeometryTeacherSample)
    sample.validate()
    assert sample.depth_m.shape == (3, 4, 6)
    assert sample.confidence.shape == (3, 4, 6)
    assert sample.valid_mask.shape == (3, 4, 6)
    assert sample.depth_m.dtype == np.float32
    assert sample.confidence.dtype == np.float32
    assert sample.valid_mask.dtype == np.bool_
    assert sample.view_names == VIEW_NAMES
    assert sample.coordinate_frame == "camera_optical_z_depth_m"
    assert sample.metadata["confidence_source"] == "finite_positive_validity"
    np.testing.assert_array_equal(sample.depth_hw, [[4, 6]] * 3)
    np.testing.assert_allclose(
        sample.resize_scale_xy,
        [[6 / 1920, 4 / 1080]] * 3,
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_array_equal(sample.confidence, 1.0)


def test_da3_legacy_adapter_masks_invalid_values_without_silent_fallback(tmp_path) -> None:
    token = "scene-token"
    _write_legacy_da3(tmp_path / f"{token}.pkl-depth.pkl")
    path = tmp_path / f"{token}.pkl-depth.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["cam_f0"][0, 0] = np.nan
    payload["cam_l0"][0, 1] = 0.0
    with path.open("wb") as stream:
        pickle.dump(payload, stream)

    sample = DA3LegacyDepthAdapter(tmp_path).load_cached(
        token, np.asarray([[1080, 1920]] * 3)
    )

    assert not sample.valid_mask[0, 0, 0]
    assert not sample.valid_mask[1, 0, 1]
    assert sample.confidence[0, 0, 0] == 0.0
    assert sample.depth_m[0, 0, 0] == 0.0


@pytest.mark.parametrize("failure", ["missing", "corrupt", "wrong_views", "shape"])
def test_da3_legacy_adapter_fails_fast_on_bad_cache(tmp_path, failure) -> None:
    token = "scene-token"
    path = tmp_path / f"{token}.pkl-depth.pkl"
    if failure == "corrupt":
        path.write_bytes(b"not-a-pickle")
    elif failure == "wrong_views":
        with path.open("wb") as stream:
            pickle.dump({"cam_f0": np.ones((4, 6), dtype=np.float32)}, stream)
    elif failure == "shape":
        with path.open("wb") as stream:
            pickle.dump(
                {
                    "cam_f0": np.ones((4, 6), dtype=np.float32),
                    "cam_l0": np.ones((5, 6), dtype=np.float32),
                    "cam_r0": np.ones((4, 6), dtype=np.float32),
                },
                stream,
            )

    adapter = DA3LegacyDepthAdapter(tmp_path)
    expected = FileNotFoundError if failure == "missing" else ValueError
    with pytest.raises(expected):
        adapter.load_cached(token, np.asarray([[1080, 1920]] * 3))


def test_geometry_teacher_sample_rejects_shape_mismatch() -> None:
    sample = GeometryTeacherSample(
        token="x",
        view_names=VIEW_NAMES,
        depth_m=np.ones((3, 4, 6), dtype=np.float32),
        confidence=np.ones((3, 4, 5), dtype=np.float32),
        valid_mask=np.ones((3, 4, 6), dtype=np.bool_),
        source_image_hw=np.asarray([[1080, 1920]] * 3),
        depth_hw=np.asarray([[4, 6]] * 3),
        resize_scale_xy=np.asarray([[6 / 1920, 4 / 1080]] * 3),
        coordinate_frame="camera_optical_z_depth_m",
        metadata={},
    )
    with pytest.raises(ValueError, match="confidence"):
        sample.validate()


def test_vggt_adapter_is_lazy_and_requires_explicit_local_api(tmp_path, monkeypatch) -> None:
    imported = []
    original_import = importlib.import_module

    def tracked_import(name, package=None):
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked_import)
    adapter = VGGTAdapter(
        local_repo=tmp_path / "missing-repo",
        checkpoint=tmp_path / "missing.ckpt",
        module_name="user_vggt_adapter",
        factory_name="build_teacher",
    )
    assert "user_vggt_adapter" not in imported

    with pytest.raises(FileNotFoundError, match="local repo"):
        adapter.infer([])
    assert "user_vggt_adapter" not in imported


def test_vggt_metric_scale_uses_known_camera_baselines() -> None:
    predicted_world_to_camera = np.repeat(
        np.eye(4, dtype=np.float32)[None], 3, axis=0
    )[:, :3]
    # For R=I, camera center is -t.  The predicted rig is exactly half-scale.
    predicted_world_to_camera[:, 0, 3] = np.asarray([0.5, 0.0, -0.5])
    known_camera_centers_m = np.asarray(
        [[-1.0, 0.0, 1.5], [0.0, 0.0, 1.5], [1.0, 0.0, 1.5]],
        dtype=np.float32,
    )

    scale, diagnostics = estimate_metric_scale_from_camera_rig(
        predicted_world_to_camera, known_camera_centers_m
    )

    assert scale == pytest.approx(2.0)
    assert diagnostics["valid_pair_count"] == 3
    assert diagnostics["pair_scale_median"] == pytest.approx(2.0)


@pytest.mark.parametrize("failure", ["shape", "degenerate_prediction", "degenerate_reference"])
def test_vggt_metric_scale_fails_fast_on_invalid_rig(failure) -> None:
    predicted = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)[:, :3]
    known = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    if failure == "shape":
        predicted = predicted[:2]
    elif failure == "degenerate_prediction":
        pass
    else:
        predicted[:, 0, 3] = np.asarray([1.0, 0.0, -1.0])
        known[:] = 0.0

    with pytest.raises(ValueError):
        estimate_metric_scale_from_camera_rig(predicted, known)


def test_official_vggt_adapter_is_lazy_and_requires_local_assets(
    tmp_path, monkeypatch
) -> None:
    imported = []
    original_import = importlib.import_module

    def tracked_import(name, package=None):
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked_import)
    adapter = OfficialVGGTMetricDepthAdapter(
        local_repo=tmp_path / "missing-repo",
        checkpoint=tmp_path / "missing.safetensors",
        device="cpu",
    )
    assert not any(name.startswith("vggt") for name in imported)

    with pytest.raises(FileNotFoundError, match="local repo"):
        adapter.load_model()
    assert not any(name.startswith("vggt") for name in imported)


def test_vggt_metric_scale_uses_robust_metric_depth_anchor() -> None:
    relative = np.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32
    )
    reference = relative * 7.5
    reference[0, 0, 0] = 180.0  # One finite but inconsistent outlier.

    scale, diagnostics = estimate_metric_scale_from_depth_reference(
        relative,
        reference,
        min_valid_pixels=4,
    )

    assert scale == pytest.approx(7.5)
    assert diagnostics["valid_pixel_count"] == 6
    assert diagnostics["method"] == "robust_log_median_metric_depth_ratio"


def test_vggt_metric_depth_anchor_rejects_missing_or_degenerate_overlap() -> None:
    relative = np.ones((3, 4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        estimate_metric_scale_from_depth_reference(relative, relative[:, :, :-1])
    with pytest.raises(ValueError, match="valid pixels"):
        estimate_metric_scale_from_depth_reference(
            relative,
            np.zeros_like(relative),
            min_valid_pixels=4,
        )
