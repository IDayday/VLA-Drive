"""Small trusted NAVSIM fixtures exercise the production finalizer, no recaching."""

from pathlib import Path
import copy
import json
import lzma
import pickle
import os
import pytest
from starVLA.rl.flow_grpo.asset_publication import (
    finalize_assets,
    validate_cache_assets,
    verify_publication,
    cache_builder_status,
)
from starVLA.rl.flow_grpo.transactions import atomic_json


@pytest.fixture(scope="module")
def real_cache():
    cache_root = Path(
        "/mnt/project/DriveDreamer-Policy/navsim_exp/qds_metric_cache_navtrain"
    )
    path = next(cache_root.glob("*/*/*/metric_cache.pkl"))
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)


@pytest.fixture
def assets(tmp_path, real_cache):
    root = tmp_path / "assets"
    root.mkdir()
    cache = copy.copy(real_cache)
    token, log = "fixture_token", cache.log_name
    path = (
        root
        / "metric_cache_navtrain_v2"
        / log
        / "original"
        / token
        / "metric_cache.pkl"
    )
    path.parent.mkdir(parents=True)
    cache.file_path = path
    path.write_bytes(lzma.compress(pickle.dumps(cache)))
    metadata, image, test = root / "meta.pkl", root / "image.jpg", root / "test.json"
    metadata.write_bytes(b"trusted-fixture-metadata-content")
    image.write_bytes(b"fixture-image-content")
    test.write_text("[]")
    record = {
        "token": token,
        "log": log,
        "time_us": cache.timepoint.time_us,
        "metadata": str(metadata),
        "images": [str(image)] * 3,
    }
    for name, data in [
        ("metadata_records.json", [record]),
        ("navtrain_tokens.json", [token]),
        ("split_manifest.json", {"train_tokens": [token], "dev_tokens": []}),
        ("dev_tokens.json", []),
        ("rl_train_tokens.json", [token]),
    ]:
        atomic_json(root / name, data)
    (root / "cache_build.log").write_text(
        "Completed dataset caching! All 1 features and targets were cached successfully.\nDone storing metadata csv file.\n"
    )
    atomic_json(
        root / "cache_job.json",
        {
            "pid": os.getpid(),
            "cache": str(root / "metric_cache_navtrain_v2"),
            "command": [
                "python",
                "cache_metrics_spawn.py",
                f"metric_cache_path={root / 'metric_cache_navtrain_v2'}",
            ],
            "log": str(root / "cache_build.log"),
        },
    )
    templates = {}
    for variant in ("frozen_visual", "unfrozen_visual"):
        file = tmp_path / f"{variant}.json"
        atomic_json(file, {"paths": {"test_list": str(test)}, "runtime": {}})
        templates[variant] = str(file)
    return root, path, record, templates


@pytest.mark.parametrize(
    "damage,category",
    [
        ("empty", "corrupt"),
        ("compressed", "corrupt"),
        ("schema", "schema_mismatch"),
        ("identity", "schema_mismatch"),
        ("duplicate", "duplicates"),
        ("missing", "missing"),
    ],
)
def test_cache_content_errors_are_reported_and_never_published(
    assets, damage, category
):
    root, path, record, templates = assets
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "compressed":
        path.write_bytes(b"invalid xz")
    elif damage == "schema":
        path.write_bytes(lzma.compress(pickle.dumps({"human_trajectory": []})))
    elif damage == "identity":
        record["time_us"] += 1
        atomic_json(root / "metadata_records.json", [record])
    elif damage == "missing":
        path.rename(path.with_suffix(".preserved_missing"))
    else:
        duplicate = (
            root
            / "metric_cache_navtrain_v2/another/type"
            / record["token"]
            / "metric_cache.pkl"
        )
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(path.read_bytes())
    result = validate_cache_assets(root, [record], workers=0)
    assert result["status"] == "FAIL" and result[category]
    with pytest.raises(ValueError, match="cache validation failed"):
        finalize_assets(root, templates, workers=0)
    assert not (root / "published_v2").exists()
    evidence = list(root.glob(".published_v2.attempt-*/cache_validation.json"))
    assert len(evidence) == 1 and json.loads(evidence[0].read_text())[category]


def test_manifest_interruption_retries_without_deleting_evidence(assets):
    root, _, _, templates = assets

    def fail(phase):
        assert phase == "after_manifest"
        raise RuntimeError("injected finalize interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        finalize_assets(root, templates, workers=0, fault=fail)
    attempts = list(root.glob(".published_v2.attempt-*"))
    assert len(attempts) == 1 and (attempts[0] / "asset_manifest.json").is_file()
    assert not (root / "published_v2").exists()
    result = finalize_assets(root, templates, workers=0)
    assert result["status"] == "ASSETS_READY_ONLY"
    assert attempts[0].exists()
    final = root / "published_v2"
    from omegaconf import OmegaConf

    for config in (final / "configs").glob("*.yaml"):
        cfg = OmegaConf.load(config)
        assert Path(cfg.paths.asset_manifest) == final / "asset_manifest.json"
        assert Path(cfg.paths.asset_publication) == final
        assert (
            verify_publication(final)["asset_identity"]
            == cfg.paths.asset_manifest_identity
        )


def test_complete_publish_is_idempotent_and_replacement_rejected(assets):
    root, path, _, templates = assets
    first = finalize_assets(root, templates, workers=0)
    files = {
        str(p): p.stat().st_mtime_ns
        for p in (root / "published_v2").rglob("*")
        if p.is_file()
    }
    assert finalize_assets(root, templates, workers=0) == first
    assert files == {
        str(p): p.stat().st_mtime_ns
        for p in (root / "published_v2").rglob("*")
        if p.is_file()
    }
    path.write_bytes(b"replaced after publication")
    with pytest.raises(ValueError, match="immutable asset changed"):
        finalize_assets(root, templates, workers=0)


def test_published_identity_conflict_and_running_build_refused(assets):
    root, _, _, templates = assets
    # A reused PID is not enough to call an unrelated process an active builder.
    assert cache_builder_status(root, 1)["finished"]
    (root / "cache_build.log").write_text("build not completed")
    with pytest.raises(ValueError, match="has not finished"):
        finalize_assets(root, templates, workers=0)
    (root / "cache_build.log").write_text(
        "Completed dataset caching! All 1 features and targets were cached successfully.\nDone storing metadata csv file.\n"
    )
    finalize_assets(root, templates, workers=0)
    Path(templates["frozen_visual"]).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="conflicts"):
        finalize_assets(root, templates, workers=0)
