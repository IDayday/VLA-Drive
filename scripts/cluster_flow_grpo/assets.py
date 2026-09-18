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
    if (context["executable_sha256"] != executable_identity() or control.get("status") != "PASS"
            or not control.get("exit_codes") or any(control["exit_codes"])):
        raise ValueError("asset reuse requires a completed matching native execution")
    expected_identity = context["resume_identity"]["data_metric_manifest"]
    manifest = Path(data["manifest"]).resolve()
    def reuse(path, expected=None):
        if Path(path).resolve() != manifest or expected != expected_identity:
            raise ValueError("asset reuse manifest identity/path mismatch")
        return expected_identity
    repro.verify_asset_manifest = reuse
    print(json.dumps({"asset_verification":"REUSED_PREVIOUS_FULL_CHECK",
                      "receipt":str(Path(receipt).resolve()),"sha256":sha256,
                      "fresh_data_bytes_checked":False}),flush=True)
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
