"""Validate trusted NAVSIM v2 caches and atomically publish immutable asset bundles."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
import json
import lzma
import multiprocessing as mp
import os
import pickle
import re
import uuid
import numpy as np
from omegaconf import OmegaConf
from .loading import file_sha
from .reproducibility import write_asset_manifest, verify_asset_manifest
from .transactions import atomic_json, publication_lock, preserve_attempt


def cache_builder_status(root, expected_count):
    root = Path(root)
    job = json.loads((root / "cache_job.json").read_text())
    cache = str((root / "metric_cache_navtrain_v2").resolve())
    if (
        str(Path(job["cache"]).resolve()) != cache
        or f"metric_cache_path={cache}" not in job["command"]
    ):
        raise ValueError("cache job identity differs from target")
    active = []
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = [x.decode() for x in proc.read_bytes().split(b"\0") if x]
        except (OSError, UnicodeError):
            continue
        if f"metric_cache_path={cache}" in argv and any(
            Path(x).name in {"cache_metrics_spawn.py", "run_metric_caching.py"}
            for x in argv
        ):
            active.append({"pid": int(proc.parent.name), "argv": argv})
    log = Path(job["log"])
    text = log.read_text(errors="replace")
    counts = re.findall(
        r"Completed dataset caching! All (\d+) features and targets were cached successfully",
        text,
    )
    complete = (
        bool(counts)
        and int(counts[-1]) == expected_count
        and "Done storing metadata csv file." in text
    )
    temporary = [str(p) for p in Path(cache).glob("*/*/*/metric_cache.pkl.tmp-*")]
    return {
        "registered_pid": job["pid"],
        "active_matching_processes": active,
        "completed_count": int(counts[-1]) if counts else None,
        "complete_log": complete,
        "log_sha256": file_sha(log),
        "temporary_files": temporary,
        "finished": complete and not active and not temporary,
    }


def expected_cache_records(root, raw_root=None):
    root = Path(root)
    records = json.loads((root / "metadata_records.json").read_text())
    tokens = json.loads((root / "navtrain_tokens.json").read_text())
    if len(tokens) != len(set(tokens)) or len(records) != len(
        {r["token"] for r in records}
    ):
        raise ValueError("duplicate target token in metadata/token list")
    if {r["token"] for r in records} != set(tokens):
        raise ValueError("target metadata/token lists differ")
    if raw_root is not None:
        groups = defaultdict(list)
        for row in records:
            groups[row["log"]].append(row)

        def add_time(item):
            log, rows = item
            path = Path(raw_root) / "navsim_logs/trainval" / f"{log}.pkl"
            with path.open("rb") as stream:
                frames = pickle.load(stream)
            times = {f["token"]: int(f["timestamp"]) for f in frames}
            return [{**r, "time_us": times[r["token"]]} for r in rows]

        with ThreadPoolExecutor(8) as pool:
            records = [
                row for group in pool.map(add_time, groups.items()) for row in group
            ]
    if any("time_us" not in row for row in records):
        raise ValueError(
            "authoritative raw scene timestamps required for cache identity"
        )
    return records


def _check_cache(job):
    path, expected = job
    path = Path(path)
    try:
        before = path.stat()
        if before.st_size == 0:
            raise ValueError("zero-byte cache")
        with lzma.open(path, "rb") as stream:
            cache = pickle.load(stream)  # only this project's trusted generated caches
    except Exception as exc:
        return {
            "token": expected["token"],
            "path": str(path),
            "category": "corrupt",
            "error": repr(exc),
        }
    try:
        from navsim.planning.metric_caching.metric_cache import MetricCache

        if (
            not isinstance(cache, MetricCache)
            or not set(MetricCache.__dataclass_fields__) <= vars(cache).keys()
        ):
            raise ValueError("not this project's complete NAVSIM v2 MetricCache schema")
        if any(
            getattr(cache, field) is None for field in MetricCache.__dataclass_fields__
        ):
            raise ValueError("required original-scene v2 data is unavailable")
        if (
            cache.log_name != expected["log"]
            or path.parents[2].name != expected["log"]
            or Path(cache.file_path).parent.name != expected["token"]
            or Path(cache.file_path).parents[2].name != expected["log"]
            or cache.timepoint.time_us != expected["time_us"]
            or cache.ego_state.time_point.time_us != expected["time_us"]
        ):
            raise ValueError("cache token/log/time identity mismatch")
        poses = cache.human_trajectory.poses
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError("invalid official human trajectory")
        if not cache.future_tracked_objects or len(cache.current_tracked_objects) != 1:
            raise ValueError("v2 traffic/reference data unavailable")
        for field in ("map_root", "map_name", "map_version"):
            if not getattr(cache.map_parameters, field, None):
                raise ValueError("v2 map parameters unavailable")
        sha = file_sha(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("cache changed while validating")
        return {
            "token": expected["token"],
            "path": str(path.resolve()),
            "sha256": sha,
            "category": "valid",
        }
    except Exception as exc:
        return {
            "token": expected["token"],
            "path": str(path),
            "category": "schema_mismatch",
            "error": repr(exc),
        }


def validate_cache_assets(root, records, workers=8):
    root = Path(root)
    expected = {r["token"]: r for r in records}
    if len(expected) != len(records):
        raise ValueError("duplicate expected cache tokens")
    found = defaultdict(list)
    for path in (root / "metric_cache_navtrain_v2").glob("*/*/*/metric_cache.pkl"):
        found[path.parent.name].append(path)
    result = {
        "missing": sorted(set(expected) - set(found)),
        "extra": sorted(set(found) - set(expected)),
        "duplicates": {
            t: list(map(str, ps)) for t, ps in found.items() if len(ps) != 1
        },
        "corrupt": [],
        "schema_mismatch": [],
        "valid": [],
        "target_count": len(records),
    }
    jobs = [
        (str(paths[0]), expected[token])
        for token, paths in found.items()
        if token in expected and len(paths) == 1
    ]
    if workers:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            rows = pool.map(_check_cache, jobs, chunksize=16)
            for row in rows:
                result[row["category"]].append(row)
    else:
        for job in jobs:
            row = _check_cache(job)
            result[row["category"]].append(row)
    result["status"] = (
        "PASS"
        if not any(
            result[k]
            for k in ("missing", "extra", "duplicates", "corrupt", "schema_mismatch")
        )
        else "FAIL"
    )
    return result


def publication_recipe(root, templates):
    root = Path(root)
    return {
        "schema_version": 2,
        "inputs": {
            name: file_sha(root / name)
            for name in (
                "metadata_records.json",
                "navtrain_tokens.json",
                "split_manifest.json",
                "dev_tokens.json",
                "rl_train_tokens.json",
            )
        },
        "templates": {name: file_sha(path) for name, path in templates.items()},
        "validator_sha256": file_sha(__file__),
    }


def verify_publication(publication, verify_inputs=True):
    publication = Path(publication)
    marker = json.loads((publication / "COMPLETE").read_text())
    if marker.get("schema_version") != 2 or marker.get("status") != "ASSETS_READY_ONLY":
        raise ValueError("asset publication is not complete")
    for relative, expected in marker["files"].items():
        if file_sha(publication / relative) != expected:
            raise ValueError(f"published asset bundle changed: {relative}")
    if verify_inputs:
        verify_asset_manifest(
            publication / "asset_manifest.json", marker["asset_identity"]
        )
    return marker


def finalize_assets(root, templates, *, raw_root=None, workers=8, fault=None):
    root = Path(root).resolve()
    final = root / "published_v2"
    with publication_lock(root / ".publish_v2.lock"):
        recipe = publication_recipe(root, templates)
        if (final / "COMPLETE").is_file():
            marker = verify_publication(final)
            if marker["recipe"] != recipe:
                raise ValueError(
                    "published asset identity conflicts with requested inputs/templates/validator"
                )
            state = cache_builder_status(root, marker["scenes"])
            if not state["finished"]:
                raise ValueError(
                    "cache builder still writing or completion log invalid"
                )
            return marker
        if final.exists():
            preserve_attempt(final)
        stage = root / (".published_v2.attempt-" + uuid.uuid4().hex)
        stage.mkdir()
        records = expected_cache_records(root, raw_root)
        state = cache_builder_status(root, len(records))
        atomic_json(stage / "builder_status.json", state)
        if not state["finished"]:
            raise ValueError(
                "cache builder has not finished; no finalization or mutation of cache"
            )
        result = validate_cache_assets(root, records, workers)
        atomic_json(stage / "cache_validation.json", result)
        if result["status"] != "PASS":
            raise ValueError(
                f"cache validation failed; full lists preserved at {stage}"
            )
        if not cache_builder_status(root, len(records))["finished"]:
            raise ValueError("cache builder changed during validation")
        paths = [
            root / name
            for name in (
                "navtrain_tokens.json",
                "split_manifest.json",
                "dev_tokens.json",
                "rl_train_tokens.json",
                "metadata_records.json",
            )
        ]
        for row in records:
            paths.extend([row["metadata"], *row["images"]])
        paths.extend(row["path"] for row in result["valid"])
        configs = {name: OmegaConf.load(path) for name, path in templates.items()}
        paths.extend(cfg.paths.test_list for cfg in configs.values())
        manifest = write_asset_manifest(paths, stage / "asset_manifest.json")
        actual = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        if any(actual[row["path"]] != row["sha256"] for row in result["valid"]):
            raise ValueError("validated cache content changed before publication")
        if fault:
            fault("after_manifest")
        (stage / "configs").mkdir()
        for name, cfg in configs.items():
            cfg.paths.data_root = str(root / "dataset")
            cfg.paths.train_list = str(root / "navtrain_tokens.json")
            cfg.paths.split_manifest = str(root / "split_manifest.json")
            cfg.paths.metric_cache = str(root / "metric_cache_navtrain_v2")
            cfg.paths.asset_manifest = str(final / "asset_manifest.json")
            cfg.paths.asset_manifest_identity = manifest["identity"]
            cfg.paths.asset_publication = str(final)
            cfg.runtime.acceptance_record = (
                f"reports/ddp_flow_grpo_paired/release_full_{name}.json"
            )
            OmegaConf.save(cfg, stage / "configs" / f"paired_{name}.yaml")
        files = {
            str(p.relative_to(stage)): file_sha(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        marker = dict(
            schema_version=2,
            status="ASSETS_READY_ONLY",
            recipe=recipe,
            scenes=len(records),
            asset_identity=manifest["identity"],
            files=files,
            production_acceptance="NOT_READY; asset completion never releases training",
        )
        atomic_json(stage / "COMPLETE", marker)
        os.rename(stage, final)  # all files + COMPLETE become visible in one operation
        return marker
