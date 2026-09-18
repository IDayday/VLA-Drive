"""Bind the additional cluster/verification layer without changing actor code.

Native acceptance still validates the actor, recipe, assets and measured GPU
evidence. A content-addressed release directory additionally binds these new
launchers, observers, tests and allocation plans. An edited file selects an
unreleased directory; direct native CLI calls default to an unpublished path.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def orchestration_identity(root=None):
    root=Path(root or ROOT)
    paths=[]
    for name in ["scripts/cluster_flow_grpo","tests/cluster_flow_grpo"]:
        paths.extend((root/name).rglob("*.py"))
    paths.extend((root/"configs/cluster_flow_grpo").glob("*.json"))
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(paths)}
    if not files:raise ValueError("missing cluster source inventory")
    sha=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"schema_version":1,"sha256":sha,"files":files}


def configure_release(root=None):
    root=Path(root or ROOT);identity=orchestration_identity(root)
    directory=root/"reports/ddp_flow_grpo_world16/releases"/identity["sha256"]
    os.environ["FLOW_GRPO_CLUSTER_RELEASE_DIR"]=str(directory)
    return identity,directory


def validate_binding(cfg, expected):
    if orchestration_identity()!=expected:
        raise ValueError("cluster/verification code or plan changed; qualification must be republished")
    record=json.loads(Path(cfg["runtime"]["acceptance_record"]).read_text())
    paths={p["path"] for p in record["gates"].values()}
    for path in paths:
        bundle=json.loads(Path(path).read_text())
        if bundle.get("orchestration_identity")!=expected:
            raise ValueError("native release lacks the matching cluster/verification source binding")
        from scripts.cluster_flow_grpo.test_release import validate_cpu_receipt
        receipt=bundle["cluster_cpu_validation"]
        if hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest()!=receipt["sha256"]:
            raise ValueError("cluster CPU evidence changed")
        validate_cpu_receipt(receipt["path"],expected)
