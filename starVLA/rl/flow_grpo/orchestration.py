"""The paired controller's restart plan; training and artifacts have separate cursors."""

from pathlib import Path
import re
from .transactions import atomic_json, preserve_attempt


def checkpoint_inventory(run, validate):
    complete, incomplete = {}, []
    root = Path(run) / "checkpoints"
    for path in sorted(root.glob("*")) if root.exists() else []:
        match = re.fullmatch(r"\.?update_(\d+)(?:\.incomplete)?", path.name)
        if not path.is_dir() or not match:
            continue
        if not (path / "COMPLETE").is_file():
            incomplete.append(path)
            continue
        # Every complete checkpoint is checked, including steps above this target.
        # A later conflicting initialization must never be silently ignored.
        state = validate(path)
        update = int(match[1])
        if state["update"] != update or path.name != f"update_{update:06d}":
            raise ValueError(f"checkpoint name/update mismatch: {path}")
        if update in complete:
            raise ValueError("duplicate checkpoint update")
        complete[update] = path
    return complete, incomplete


def advance_target(run, target, *, validate, train, export, evaluate):
    """Execute the real orchestration with injectable external actions for tests."""
    run = Path(run)
    complete, incomplete = checkpoint_inventory(run, validate)
    latest = max(complete, default=0)
    if target not in complete:
        if latest > target:
            raise ValueError(
                f"missing historical target {target}, latest checkpoint is {latest}; cannot replay completed updates"
            )
        for path in incomplete:
            preserve_attempt(
                path
            )  # incomplete evidence retained; COMPLETE dirs untouched
        train(complete.get(latest), target)
        complete, _ = checkpoint_inventory(run, validate)
        if target not in complete:
            raise RuntimeError(f"training exited without validated checkpoint {target}")
        latest = max(complete)
    checkpoint = complete[target]
    progress_path = run / "orchestration_progress.json"
    progress = (
        __import__("json").loads(progress_path.read_text())
        if progress_path.exists()
        else {}
    )
    progress["training_latest_update"] = latest
    progress.setdefault("exports", {})
    progress.setdefault("dev_evaluations", {})
    atomic_json(progress_path, progress)
    exported = export(checkpoint, target)
    progress["exports"][str(target)] = str(exported)
    atomic_json(progress_path, progress)
    report = evaluate(exported, target)
    progress["dev_evaluations"][str(target)] = report
    atomic_json(progress_path, progress)
    return checkpoint, exported, report
