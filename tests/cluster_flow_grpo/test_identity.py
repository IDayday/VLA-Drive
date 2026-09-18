"""The additional orchestration release namespace is fail-closed on edits."""
import json
from pathlib import Path
import pytest
from scripts.cluster_flow_grpo import identity


def test_code_test_or_plan_edit_selects_unreleased_namespace(tmp_path, monkeypatch):
    for name in ["scripts/cluster_flow_grpo/paired.py","tests/cluster_flow_grpo/test_a.py","configs/cluster_flow_grpo/paired_world16.json"]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("{}\n")
    first,old=identity.configure_release(tmp_path)
    old.mkdir(parents=True);(old/"release_frozen_visual.json").write_text('"unit fixture only"')
    for name in first["files"]:
        p=tmp_path/name;before=p.read_text();p.write_text(before+"\n")
        changed,new=identity.configure_release(tmp_path)
        assert changed!=first and new!=old and not (new/"release_frozen_visual.json").exists()
        assert Path(__import__("os").environ["FLOW_GRPO_CLUSTER_RELEASE_DIR"])==new
        p.write_text(before)
    assert identity.orchestration_identity(tmp_path)==first


def test_binding_reads_actual_evidence_header_and_rejects_change(tmp_path, monkeypatch):
    from scripts.cluster_flow_grpo import test_release
    import hashlib
    expected={"schema_version":1,"sha256":"fixture","files":{"observer.py":"digest"}}
    monkeypatch.setattr(identity,"orchestration_identity",lambda:expected)
    receipt=tmp_path/"cpu.json";receipt.write_text("unit fixture; GPU gates remain separate")
    monkeypatch.setattr(test_release,"validate_cpu_receipt",lambda *a:None)
    bundle=tmp_path/"bundle.json";bundle.write_text(json.dumps({"orchestration_identity":expected,
        "cluster_cpu_validation":{"path":str(receipt),"sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()}}))
    record=tmp_path/"record.json";record.write_text(json.dumps({"gates":{"model-independent-fixture":{"path":str(bundle)}}}))
    cfg={"runtime":{"acceptance_record":str(record)}}
    identity.validate_binding(cfg,expected)
    bundle.write_text(json.dumps({"orchestration_identity":{"sha256":"other"}}))
    with pytest.raises(ValueError,match="binding"):identity.validate_binding(cfg,expected)
    with pytest.raises(ValueError,match="code or plan changed"):identity.validate_binding(cfg,{"sha256":"changed"})
