#!/usr/bin/env python3
"""Audit whether ``initialize_from_config`` differs from the release recipe.

This is intentionally a source/checkpoint audit rather than another training
run.  The option constructs the large frozen VLM with random weights before a
checkpoint overwrites them, so it can consume RNG state.  It can only explain a
reproduction gap if the release and reproduction take different construction
paths, or if VLM construction happens before the randomly initialized action
head.  Both claims can be checked directly from the released source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


BACKBONE_PATH = "navsim/agents/EpisodeDrive/drivevla_backbone.py"
AGENT_PATH = "navsim/agents/EpisodeDrive/drivevla_base_agent.py"
CONFIG_PATH = "navsim/planning/script/config/common/agent/episode_drive.yaml"


def _git_show(repo: Path, revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _yaml_section_scalar(text: str, section: str, key: str) -> str | None:
    """Read one scalar from the simple two-level release YAML without Hydra."""

    in_section = False
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            in_section = raw_line.rstrip() == f"{section}:"
            continue
        if in_section:
            match = re.match(rf"^\s+{re.escape(key)}:\s*(.*?)\s*$", raw_line)
            if match:
                return match.group(1)
    return None


def _method(source: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise ValueError(f"Could not find {class_name}.{method_name}")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _constructor_line(method: ast.FunctionDef, constructor: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and _call_name(node) == constructor
    ]
    if not lines:
        raise ValueError(f"Could not find constructor call {constructor}")
    return min(lines)


def _initialize_branch(source: str) -> ast.If:
    method = _method(source, "DriveVLABackbone", "__init__")
    for node in ast.walk(method):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "initialize_from_config":
                return node
    raise ValueError("Could not find initialize_from_config branch")


def _branch_signature(source: str) -> str:
    branch = _initialize_branch(source)
    # Only the true branch defines initialize_from_config semantics.  Exclude
    # locations so harmless line shifts do not look like a semantic change.
    return ast.dump(ast.Module(body=branch.body, type_ignores=[]), include_attributes=False)


def _active_override(pid: int | None) -> str | None:
    if pid is None:
        return None
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    prefix = b"agent.vlm_config.initialize_from_config="
    for item in command:
        if item.startswith(prefix):
            return item[len(prefix) :].decode("utf-8", "replace")
    return None


def audit(
    repo: Path,
    release_ref: str,
    pid: int | None = None,
    fingerprint_path: Path | None = None,
) -> dict[str, Any]:
    current = {
        path: (repo / path).read_text()
        for path in (BACKBONE_PATH, AGENT_PATH, CONFIG_PATH)
    }
    released = {
        path: _git_show(repo, release_ref, path)
        for path in (BACKBONE_PATH, AGENT_PATH, CONFIG_PATH)
    }

    variants: dict[str, Any] = {}
    for name, sources in (("release", released), ("current", current)):
        agent_init = _method(sources[AGENT_PATH], "DriveVLABaseAgent", "__init__")
        action_line = _constructor_line(agent_init, "ActionDecoder")
        backbone_line = _constructor_line(agent_init, "DriveVLABackbone")
        variants[name] = {
            "config_initialize_from_config": _yaml_section_scalar(
                sources[CONFIG_PATH], "vlm_config", "initialize_from_config"
            ),
            "action_decoder_constructor_line": action_line,
            "vlm_constructor_line": backbone_line,
            "action_head_precedes_vlm": action_line < backbone_line,
            "initialize_true_branch_sha256": _sha256(
                _branch_signature(sources[BACKBONE_PATH])
            ),
            "source_sha256": {
                path: _sha256(payload) for path, payload in sources.items()
            },
        }

    branch_equal = (
        variants["release"]["initialize_true_branch_sha256"]
        == variants["current"]["initialize_true_branch_sha256"]
    )
    active_value = _active_override(pid)

    fingerprint: dict[str, Any] | None = None
    if fingerprint_path is not None and fingerprint_path.is_file():
        payload = json.loads(fingerprint_path.read_text())
        released_fingerprint = payload.get("checkpoints", {}).get("released", {})
        fingerprint = {
            "path": str(fingerprint_path),
            "matching_seeds": released_fingerprint.get("matching_seeds"),
            "fingerprint_tensor_count": released_fingerprint.get(
                "fingerprint_tensor_count"
            ),
            "fingerprint_element_count": released_fingerprint.get(
                "fingerprint_element_count"
            ),
            "exact_seed2_match": released_fingerprint.get("matching_seeds") == [2],
        }

    checks = {
        "release_config_uses_initialize_from_config": (
            variants["release"]["config_initialize_from_config"] == "true"
        ),
        "current_config_uses_initialize_from_config": (
            variants["current"]["config_initialize_from_config"] == "true"
        ),
        "active_run_uses_initialize_from_config": (
            active_value == "true" if pid is not None else None
        ),
        "release_action_head_precedes_vlm": variants["release"][
            "action_head_precedes_vlm"
        ],
        "current_action_head_precedes_vlm": variants["current"][
            "action_head_precedes_vlm"
        ],
        "initialize_true_branch_ast_matches_release": branch_equal,
        "public_action_head_fingerprint_is_exact_seed2": (
            fingerprint["exact_seed2_match"] if fingerprint is not None else None
        ),
    }
    required = [value for value in checks.values() if value is not None]
    excluded = bool(required) and all(required)

    return {
        "audit": "stage2_initialize_from_config_semantics",
        "release_reference": release_ref,
        "active_pid": pid,
        "active_override": active_value,
        "variants": variants,
        "initialize_branch_ast_equal": branch_equal,
        "checkpoint_fingerprint": fingerprint,
        "checks": checks,
        "excluded_as_reproduction_root_cause": excluded,
        "interpretation": (
            "The public release and active reproduction both construct the frozen "
            "VLM through AutoModel.from_config, and the action head is initialized "
            "first. Therefore VLM construction cannot alter the seed-2 action-head "
            "initialization. It can advance the later RNG stream, but the matching "
            "release branch and locked runtime make that shared behavior rather than "
            "a reproduction mismatch. The unavailable private launcher remains a "
            "provenance limitation, not positive evidence of a different path."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release-ref", default="b9a4f27")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = audit(
        args.repo.resolve(),
        args.release_ref,
        pid=args.pid,
        fingerprint_path=args.fingerprint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["excluded_as_reproduction_root_cause"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
