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


def placement_evidence(root=None):
    """Validate explicitly requested hardware relocations before publication/use.

    The original profile gates remain required. A replacement node additionally
    needs an actual completed two-update run and an exact native boundary check.
    """
    root=Path(root or ROOT)
    plan=json.loads((root/"configs/cluster_flow_grpo/paired_world16.json").read_text())
    receipts={}
    for variant,group in plan["groups"].items():
        relocation=group.get("placement_validation")
        if not relocation:continue
        loaded={}
        for kind,relative in relocation.items():
            path=root/relative;raw=path.read_bytes();loaded[kind]=json.loads(raw)
            receipts[str(path.resolve())]=hashlib.sha256(raw).hexdigest()
        spec,control,comparison=(loaded[k] for k in ("spec","control","comparison"))
        if spec["nodes"]!=group["nodes"] or spec["entry"][spec["entry"].index("--config")+1]!=group["config"]:
            raise ValueError("relocation does not match the allocated nodes/config")
        if control.get("status")!="PASS" or control.get("exit_codes")!=[0]*len(group["nodes"]):
            raise ValueError("relocation execution failed")
        files=comparison.get("files",{})
        if (comparison.get("status")!="PASS" or len(files)!=50 or
            any(f.get("status")!="PASS" or not f.get("values") or
                any(v.get("allclose") is not True for v in f["values"].values()) for f in files.values())):
            raise ValueError("relocation exact boundary comparison failed/incomplete")
    return receipts


def validate_binding(cfg, expected):
    if orchestration_identity()!=expected:
        raise ValueError("cluster/verification code or plan changed; qualification must be republished")
    record=json.loads(Path(cfg["runtime"]["acceptance_record"]).read_text())
    paths={p["path"] for p in record["gates"].values()}
    for path in paths:
        bundle=json.loads(Path(path).read_text())
        if bundle.get("orchestration_identity")!=expected:
            raise ValueError("native release lacks the matching cluster/verification source binding")
        if bundle.get("placement_evidence",{})!=placement_evidence():
            raise ValueError("relocation evidence changed")
        from scripts.cluster_flow_grpo.test_release import validate_cpu_receipt
        receipt=bundle["cluster_cpu_validation"]
        if hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest()!=receipt["sha256"]:
            raise ValueError("cluster CPU evidence changed")
        validate_cpu_receipt(receipt["path"],expected)
