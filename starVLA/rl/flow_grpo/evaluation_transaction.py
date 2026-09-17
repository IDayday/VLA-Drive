"""Shared CLI/orchestrator evaluation lifecycle and result integrity checks."""

from pathlib import Path
import json
import os
import socket
import time
import numpy as np
import pandas as pd
from .contracts import digest
from .loading import file_sha
from .transactions import atomic_json, publication_lock, preserve_attempt


def check_existing_identity(root, identity):
    for name in ("evaluation_state.json", "evaluation.json", "COMPLETE"):
        path = Path(root) / name
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("identity") is not None and saved["identity"] != identity:
                raise ValueError(f"evaluation identity conflict: {path}")


def result_artifacts(root, report, tokens):
    """Validate semantics before hashes; unavailable official comfort is explicit."""
    root = Path(root)
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("evaluation requires unique tokens")
    csv = root / "original_protocol_scores.csv"
    frame = pd.read_csv(csv, dtype={"token": str, "log_name": str})
    if not {"token", "log_name", "score", "valid"} <= set(frame):
        raise ValueError("evaluation CSV missing required columns")
    if (
        len(frame) != len(tokens)
        or frame.token.duplicated().any()
        or set(frame.token) != set(tokens)
    ):
        raise ValueError("evaluation CSV token set/count mismatch")
    if frame.log_name.isna().any() or not frame.valid.eq(True).all():
        raise ValueError("invalid evaluation rows")
    for name in frame.select_dtypes(include="number"):
        if name == "two_frame_extended_comfort":
            values = frame[name].dropna().to_numpy()
            declared = report.get("aggregation", {}).get("two_frame_available")
            if declared != len(values):
                raise ValueError("unavailable comfort count is not declared")
        else:
            values = frame[name].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite evaluation CSV: {name}")
    if not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite evaluation score")
    if report.get("scene_count") != len(tokens) or report.get("valid") != len(tokens):
        raise ValueError("evaluation report count mismatch")
    if not np.isfinite(report.get("epdms", np.nan)) or not np.isclose(
        report["epdms"], frame.score.mean(), atol=1e-12, rtol=0
    ):
        raise ValueError("evaluation report score mismatch")
    files = [csv, root / "trajectories.npz", root / "metric_cache_identity.json"]
    prediction_dir = root / "predictions" / report["split"]
    if {p.stem for p in prediction_dir.glob("*.npy")} != set(tokens):
        raise ValueError("prediction token set mismatch")
    with np.load(files[1], allow_pickle=False) as archive:
        if archive["tokens"].tolist() != list(tokens):
            raise ValueError("trajectory tokens/order mismatch")
        for name, width in (("physical", 3), ("normalized", 4)):
            if (
                archive[name].shape != (len(tokens), 8, width)
                or not np.isfinite(archive[name]).all()
            ):
                raise ValueError(f"invalid trajectories: {name}")
        for i, token in enumerate(tokens):
            path = root / "predictions" / report["split"] / f"{token}.npy"
            if not np.array_equal(
                np.load(path, allow_pickle=False), archive["physical"][i]
            ):
                raise ValueError(f"prediction differs from trajectory archive: {token}")
            files.append(path)
    cache = json.loads(files[2].read_text())
    if set(cache) != set(tokens) or digest(cache) != report["metric_cache_identity"]:
        raise ValueError("evaluation metric identity token/content mismatch")
    if report["identity"]["metric_assets_sha256"] != digest(cache):
        raise ValueError("evaluation metric identity differs from actual input")
    return {str(path.relative_to(root)): file_sha(path) for path in files}


def completed_evaluation(output, identity, tokens):
    root = Path(output)
    if not root.exists():
        return None
    check_existing_identity(root, identity)
    marker = root / "COMPLETE"
    if not marker.is_file():
        return None
    complete = json.loads(marker.read_text())
    report_path = root / "evaluation.json"
    report = json.loads(report_path.read_text())
    if complete.get("schema_version") != 2 or complete.get("status") != "COMPLETE":
        raise ValueError("invalid evaluation completion marker")
    if report.get("status") != "COMPLETE" or report.get("schema_version") != 2:
        raise ValueError("evaluation report is not complete")
    if complete.get("report_sha256") != file_sha(report_path):
        raise ValueError("evaluation report digest changed")
    if result_artifacts(root, report, tokens) != report.get("artifacts"):
        raise ValueError("evaluation artifact digest changed")
    return report


def request_evaluation(output, identity, tokens, execute):
    """Used by the actual paired controller, independently of its --resume flag.

    The CLI owns the filesystem lock; do not hold it while waiting for that child.
    Concurrent requests are serialized by evaluation_transaction in the child.
    """
    saved = completed_evaluation(output, identity, tokens)
    if saved is not None:
        return saved
    execute()
    saved = completed_evaluation(output, identity, tokens)
    if saved is None:
        raise RuntimeError("evaluation executor exited without a complete publication")
    return saved


def evaluation_transaction(output, identity, tokens, execute):
    root = Path(output)
    with publication_lock(root.with_name(root.name + ".lock")):
        saved = completed_evaluation(root, identity, tokens)
        if saved is not None:
            return saved
        if root.exists():
            preserve_attempt(root)
        root.mkdir(parents=True)
        state = dict(
            schema_version=2,
            identity=identity,
            status="RUNNING",
            pid=os.getpid(),
            host=socket.gethostname(),
            started=time.time(),
        )
        atomic_json(root / "evaluation_state.json", state)
        try:
            report = execute(root)
            report.update(schema_version=2, status="COMPLETE", identity=identity)
            report["artifacts"] = result_artifacts(root, report, tokens)
            atomic_json(root / "evaluation.json", report)
            atomic_json(root / "evaluation_state.json", {**state, "status": "COMPLETE"})
            # Final atomic publication is the ONLY authority for completion.
            atomic_json(
                root / "COMPLETE",
                dict(
                    schema_version=2,
                    status="COMPLETE",
                    identity=identity,
                    report_sha256=file_sha(root / "evaluation.json"),
                ),
            )
            return report
        except BaseException as exc:
            atomic_json(
                root / "evaluation_state.json",
                {**state, "status": "FAILED", "error": repr(exc)},
            )
            raise
