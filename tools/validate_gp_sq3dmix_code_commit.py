#!/usr/bin/env python3
"""Permit report-only descendants of an immutable GP implementation commit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ALLOWED_REPORT_PREFIXES = (
    "docs/experiments/results/",
    "docs/experiments/decisions/",
)
ALLOWED_REPORT_FILES = {
    "docs/experiments/GP_SQ3DMIX_STAGE_A_V2.md",
    "docs/experiments/GP_SQ3DMIX_STAGE_B_MULTI_SEED.md",
    "docs/experiments/GP_SQ3DMIX_FORMAL.md",
}


def validate(repo: Path, bound: str, current: str) -> list[str]:
    for value, name in ((bound, "bound"), (current, "current")):
        result = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{value}^{{commit}}"],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Unknown {name} commit: {value}")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", bound, current],
        check=False,
    )
    if ancestor.returncode:
        raise RuntimeError(f"Bound implementation commit {bound} is not an ancestor of {current}")
    changed = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only", f"{bound}..{current}"],
        text=True,
    ).splitlines()
    invalid = [
        path
        for path in changed
        if path not in ALLOWED_REPORT_FILES
        and not path.startswith(ALLOWED_REPORT_PREFIXES)
    ]
    if invalid:
        raise RuntimeError(
            "Code/config changed after the bound implementation commit: "
            + ", ".join(invalid[:20])
        )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bound", required=True)
    parser.add_argument("--current", required=True)
    args = parser.parse_args()
    changed = validate(Path(args.repo).resolve(), args.bound, args.current)
    print(
        f"validated implementation commit {args.bound}; "
        f"report-only descendants={len(changed)} current={args.current}"
    )


if __name__ == "__main__":
    main()
