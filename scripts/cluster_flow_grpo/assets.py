"""Explicit reuse of this job's completed full-asset verification.

Authorized for this deployment by the user to avoid repeated full-corpus scans.
This does not assert a fresh byte check, and never grants model acceptance.
Processor, numerical settings, dependencies, token hashes and resume state keep
their native checks. Changed data requires a new full verification receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ASSET_VERIFIERS = (
    "starVLA/rl/flow_grpo/reproducibility.py",
    "starVLA/rl/flow_grpo/loading.py",
    "starVLA/rl/flow_grpo/contracts.py",
    "starVLA/rl/flow_grpo/asset_publication.py",
)


def validate_source_rebind(data, context, current_executable):
    """Reuse data verification across an explicitly bound actor-only edit.

    This is NOT model acceptance or exact resume. The old, unmodified execution
    evidence remains the source of the corpus check. All actual asset-verifier
    files must be byte-identical to its archived source inventory.
    """
    binding = data.get("source_rebind", {})
    if binding.get("target_executable_sha256") != current_executable:
        raise ValueError("asset reuse source rebind does not match this executable")
    source_ref = binding.get("source_environment", {})
    raw = Path(source_ref["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_ref["sha256"]:
        raise ValueError("archived asset-verifier source inventory changed")
    source = json.loads(raw)["source_sha256"]
    if binding.get("origin_executable_sha256") != context["executable_sha256"]:
        raise ValueError("source rebind origin differs from verified execution")
    if set(binding.get("verifier_files", {})) != set(ASSET_VERIFIERS):
        raise ValueError("incomplete asset verifier source binding")
    for path in ASSET_VERIFIERS:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if source.get(path) != actual or binding["verifier_files"][path] != actual:
            raise ValueError("asset verifier changed: " + path)


def install_receipt(receipt, sha256):
    from starVLA.rl.flow_grpo import reproducibility as repro
    from starVLA.rl.flow_grpo.acceptance import executable_identity
    raw = Path(receipt).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("asset verification receipt changed")
    data = json.loads(raw)
    if data.get("mode") != "reuse_completed_full_verification_at_user_request":
        raise ValueError("explicit asset reuse authorization missing")
    context_raw = Path(data["context"]["path"]).read_bytes()
    control_raw = Path(data["control"]["path"]).read_bytes()
    for value, key in [(context_raw,"context"),(control_raw,"control")]:
        if hashlib.sha256(value).hexdigest() != data[key]["sha256"]:
            raise ValueError("asset verification evidence changed")
    context, control = json.loads(context_raw), json.loads(control_raw)
    if (control.get("status") != "PASS"
            or not control.get("exit_codes") or any(control["exit_codes"])):
        raise ValueError("asset reuse requires a completed matching native execution")
    current = executable_identity()
    if context["executable_sha256"] != current:
        validate_source_rebind(data, context, current)
    expected_identity = context["resume_identity"]["data_metric_manifest"]
    manifest = Path(data["manifest"]).resolve()
    def reuse(path, expected=None):
        if Path(path).resolve() != manifest or expected != expected_identity:
            raise ValueError("asset reuse manifest identity/path mismatch")
        return expected_identity
    repro.verify_asset_manifest = reuse
    print(json.dumps({"asset_verification":"REUSED_PREVIOUS_FULL_CHECK",
                      "receipt":str(Path(receipt).resolve()),"sha256":sha256,
                      "fresh_data_bytes_checked":False,
                      "asset_verifier_source_rebound":bool(data.get("source_rebind")),
                      "model_acceptance_granted":False}),flush=True)
    return data


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--receipt",required=True);p.add_argument("--sha256",required=True)
    p.add_argument("--module",required=True)
    p.add_argument("arguments",nargs=argparse.REMAINDER)
    a = p.parse_args()
    install_receipt(a.receipt,a.sha256)
    sys.argv = [a.module,*a.arguments]
    runpy.run_module(a.module,run_name="__main__")


if __name__ == "__main__": main()
