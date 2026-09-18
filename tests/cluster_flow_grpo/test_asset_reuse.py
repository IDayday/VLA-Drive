"""Explicit user-authorized reuse; no claim of fresh data-byte verification."""
import hashlib
import json
import pytest
from scripts.cluster_flow_grpo import assets,cluster
from starVLA.rl.flow_grpo import reproducibility,acceptance


@pytest.mark.parametrize("damage",[None,"receipt","context","failed","source","path","identity"])
def test_reuse_requires_completed_matching_receipt_without_rescanning(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(acceptance,"executable_identity",lambda:"native")
    monkeypatch.setattr(reproducibility,"verify_asset_manifest",lambda *a:pytest.fail("must not rescan corpus"))
    context=tmp_path/"context.json";control=tmp_path/"control.json";manifest=tmp_path/"manifest.json"
    context.write_text(json.dumps({"executable_sha256":"changed" if damage=="source" else "native",
                                  "resume_identity":{"data_metric_manifest":"locked"}}))
    control.write_text(json.dumps({"status":"FAIL" if damage=="failed" else "PASS","exit_codes":[0,0,0]}))
    def ref(p):return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    value={"mode":"reuse_completed_full_verification_at_user_request","context":ref(context),
           "control":ref(control),"manifest":str(manifest)}
    receipt=tmp_path/"receipt.json";receipt.write_text(json.dumps(value));sha=ref(receipt)["sha256"]
    if damage=="receipt":receipt.write_text("changed")
    if damage=="context":context.write_text("changed")
    def execute():
        assets.install_receipt(receipt,sha)
        return reproducibility.verify_asset_manifest(tmp_path/"other" if damage=="path" else manifest,
                                                    "changed" if damage=="identity" else "locked")
    if damage:
        with pytest.raises(ValueError):execute()
    else:
        assert execute()=="locked"
        assert not manifest.exists()  # Explicit reuse, not a fake read/full check.


def test_real_launcher_explicitly_wraps_only_native_training():
    spec={"nodes":[{"devices":[0,1]}],"master_addr":"localhost","master_port":1234,
          "entry":["-m","starVLA.rl.flow_grpo.cli","train","--config","fixed.yaml"],
          "asset_verification_receipt":{"path":"receipt.json","sha256":"digest"}}
    command=cluster.command_for(spec,0)
    assert command[-3:]==["train","--config","fixed.yaml"]
    assert "scripts.cluster_flow_grpo.assets" in command
    assert command[command.index("--sha256")+1]=="digest"
    spec["entry"]=["some-other-program"]
    with pytest.raises(ValueError,match="native train"):cluster.command_for(spec,0)


@pytest.mark.parametrize("damage", [None, "target", "origin", "archive", "verifier", "missing"])
def test_actor_edit_rebind_requires_unchanged_actual_asset_verifiers(tmp_path, monkeypatch, damage):
    verifier = tmp_path/"verifier.py"
    verifier.write_text("# real fixture verifier source\n")
    files = {str(verifier): hashlib.sha256(verifier.read_bytes()).hexdigest()}
    monkeypatch.setattr(assets, "ASSET_VERIFIERS", tuple(files))
    source = tmp_path/"source.json"
    source.write_text(json.dumps({"source_sha256": files}))
    binding = {"target_executable_sha256": "new", "origin_executable_sha256": "old",
        "source_environment": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "verifier_files": files}
    if damage in {"target", "origin"}:
        binding[damage+"_executable_sha256"] = "changed"
    if damage == "archive":
        source.write_text("changed")
    if damage == "verifier":
        verifier.write_text("# changed verifier\n")
    if damage == "missing":
        binding["verifier_files"] = {}
    call = lambda: assets.validate_source_rebind({"source_rebind": binding},
        {"executable_sha256": "old"}, "new")
    if damage:
        with pytest.raises(ValueError):
            call()
    else:
        call()
