"""Verify and export the portable NAVSIM trajectory release using NumPy only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


DEFAULT_RELEASE = (Path(__file__).resolve().parents[1] / "data" /
                   "optimized_trajectories" / "navtrain_20261004_v3")
FIELDS = {"tokens", "trajectories", "accepted", "acceptance_noise_scale"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_release(directory=DEFAULT_RELEASE):
    """Return verified metadata and arrays; preserve original float64 poses."""
    directory = Path(directory)
    metadata = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if (metadata["schema"] != "navsim_optimized_trajectories_v1" or
        metadata["trajectory_file"] != "trajectories.npz" or
        metadata["split"] != "navtrain" or
        metadata["frame"] != "ego_t0_local_rear_axle" or
        metadata["pose_fields"] != ["x_m", "y_m", "yaw_rad"] or
        metadata["times_s"] != [.5 * i for i in range(1, 9)]):
        raise ValueError("Unsupported trajectory release or coordinate contract")
    source = directory / "trajectories.npz"
    if (source.stat().st_size != metadata["trajectory_bytes"] or
        digest(source) != metadata["trajectory_sha256"]):
        raise ValueError("Trajectory file checksum mismatch")
    with np.load(source, allow_pickle=False) as bank:
        if set(bank.files) != FIELDS:
            raise ValueError("Unexpected trajectory arrays")
        arrays = {key: bank[key].copy() for key in FIELDS}
    tokens, poses, accepted, scale = (arrays[key] for key in (
        "tokens", "trajectories", "accepted", "acceptance_noise_scale"))
    count = metadata["scene_count"]
    if (tokens.shape != (count,) or tokens.dtype.kind != "U" or
        len(set(tokens.tolist())) != count or
        poses.shape != (count, 8, 3) or poses.dtype != np.float64 or
        not np.isfinite(poses).all() or
        accepted.shape != (count,) or accepted.dtype != np.bool_ or
        scale.shape != (count,) or scale.dtype != np.float32 or
        not np.isfinite(scale).all() or
        not np.isin(scale[accepted], [.25, .5, 1.]).all() or
        not (scale[~accepted] == 0).all()):
        raise ValueError("Invalid token coverage, trajectories, masks or noise scales")
    if (int(accepted.sum()) != metadata["verified_scene_count"] or
        int((~accepted).sum()) != metadata["original_gt_fallback_count"]):
        raise ValueError("Verified/fallback counts changed")
    summary = metadata["source_summary"]
    if (not summary["complete"] or
        summary["run_identity"] != metadata["source_run_identity"] or
        summary["accepted_geometry_failures"] != 0 or
        summary["exported_scenes"] != count or
        summary["accepted_count"] != int(accepted.sum())):
        raise ValueError("Incomplete or inconsistent source release")
    for key, array in arrays.items():
        expected = metadata["arrays"][key]
        if (list(array.shape) != expected["shape"] or
            array.dtype.str != expected["dtype"] or
            hashlib.sha256(array.tobytes(order="C")).hexdigest() != expected["sha256"]):
            raise ValueError("Array contents changed: " + key)
    return metadata, arrays


def select_tokens(arrays, requested_tokens):
    """Align to the caller's existing train split; missing tokens are an error."""
    requested = list(requested_tokens)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested training tokens are not unique")
    lookup = {token: i for i, token in enumerate(arrays["tokens"].tolist())}
    missing = [token for token in requested if token not in lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} training tokens: {missing[:5]}")
    indices = np.asarray([lookup[token] for token in requested], dtype=np.int64)
    return {key: value[indices].copy() for key, value in arrays.items()}


def export_campaign(directory, destination):
    """Produce final/{NPZ,manifest,summary,hashes} for existing prepare loaders."""
    metadata, arrays = load_release(directory)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new campaign directory: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + "-",
                                     dir=destination.parent))
    try:
        final = temporary / "final"
        final.mkdir()
        shutil.copyfile(Path(directory) / "trajectories.npz", final / "trajectories.npz")
        if digest(final / "trajectories.npz") != metadata["trajectory_sha256"]:
            raise ValueError("Exported trajectory checksum mismatch")
        identity = hashlib.sha256(json.dumps({
            "schema": "lossless_portable_navtrain_campaign_v1",
            "source_run_identity": metadata["source_run_identity"],
            "trajectory_sha256": metadata["trajectory_sha256"],
        }, sort_keys=True).encode()).hexdigest()
        write_json(final / "manifest.json", {
            "run_identity": identity,
            "derived_from_run_identity": metadata["source_run_identity"],
            "tokens": arrays["tokens"].tolist(),
            "interval_s": .5,
            "frame": "local_rear_axle",
            "source_release": metadata,
        })
        write_json(final / "summary.json", {
            **metadata["source_summary"], "run_identity": identity,
            "source_run_identity": metadata["source_run_identity"],
            "lossless_repack": True,
        })
        write_json(final / "artifact_hashes.json", {
            name: digest(final / name) for name in (
                "trajectories.npz", "manifest.json", "summary.json")})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "export-campaign"):
        command = sub.add_parser(name)
        command.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
        if name == "export-campaign":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        metadata, _ = load_release(args.release)
        print(json.dumps({"verified": True, "release": metadata["release_id"],
                          "scenes": metadata["scene_count"],
                          "accepted": metadata["verified_scene_count"],
                          "gt_fallbacks": metadata["original_gt_fallback_count"],
                          "npz_sha256": metadata["trajectory_sha256"]}, indent=2))
    else:
        print(export_campaign(args.release, args.output))


if __name__ == "__main__":
    main()
