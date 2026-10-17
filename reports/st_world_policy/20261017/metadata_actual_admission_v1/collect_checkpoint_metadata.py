"""Read only the nine explicit ST run directories; never open checkpoint payloads."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import time


def small_json(path):
    if path.stat().st_size > 65536:
        raise ValueError(f"Refusing large metadata: {path}")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def inspect(root):
    rows = []
    for family, names in (("spatial", [f"SP{i}_fulltrain_seed42_official_5k_v1" for i in range(4)]),
                          ("consequence", [f"C{i}_fulltrain_seed42_official_4096_v1" for i in range(5)])):
        for name in names:
            run = root / "formal" / family / "runs" / name
            config, config_sha = small_json(run / "resolved_config.json")
            status, status_sha = small_json(run / "status.json")
            registration, reg_sha = small_json(root / "formal" / family / "registrations" / name / "REGISTRATION.json")
            if registration["run_id"] != name:
                raise ValueError("Run registration mismatch")
            result = None
            if (run / "RESULT.json").exists():
                result, _ = small_json(run / "RESULT.json")
            checkpoints = []
            for folder in sorted((run / "checkpoints").iterdir()):
                if not folder.is_dir():
                    continue
                children = list(folder.iterdir())
                if len(children) > 32:
                    raise ValueError("Unexpected checkpoint inventory size")
                total = 0
                files = {}
                for child in children:
                    info = child.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("Unexpected checkpoint entry type")
                    files[child.name] = {"bytes": info.st_size, "allocated_bytes": info.st_blocks * 512,
                                        "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
                    total += max(info.st_size, info.st_blocks * 512)
                complete = folder / "COMPLETE.json"
                cp = {"tag": folder.name, "stat_charged_bytes": total, "files": files,
                      "complete_receipt_exists": complete.exists()}
                if complete.exists():
                    receipt, receipt_sha = small_json(complete)
                    body = {k: v for k, v in receipt.items() if k != "receipt_identity"}
                    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    expected_names = {f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt" for i in range(4)}
                    expected_names |= {"mp_rank_00_model_states.pt"} | {f"rng_rank{i}.pt" for i in range(4)}
                    checks = {
                        "receipt_identity": digest == receipt["receipt_identity"],
                        "schema": receipt["schema"] == "st_native_checkpoint_v1",
                        "world_size": receipt["world_size"] == config["execution"]["world_size"] == 4,
                        "run_identity": receipt["state"]["identity"] == status["identity"],
                        "tag": receipt["state"]["tag"] == folder.name,
                        "native_inventory": set(receipt["files"]) == expected_names,
                        "actual_inventory": {n for n in files if n.endswith(".pt")} == expected_names,
                        "recorded_sizes": all(n in files and files[n]["bytes"] == d["bytes"]
                                              for n, d in receipt["files"].items()),
                    }
                    cp.update(receipt_path=str(complete), receipt_sha256=receipt_sha,
                              receipt_checks=checks, completed=receipt["state"]["completed"],
                              termination_reason=receipt["state"]["termination_reason"],
                              declared_payload_bytes=sum(x["bytes"] for x in receipt["files"].values()))
                checkpoints.append(cp)
            latest = (run / "checkpoints" / "latest").read_text().strip()
            rows.append({"run_id": name, "root": str(run), "registration_sha256": reg_sha,
                         "config_sha256": config_sha, "status_sha256": status_sha,
                         "status_reported": status["status"], "status_step": status["completed"],
                         "status_updated_unix": status["updated_unix"],
                         "process_liveness_checked": False,
                         "result_status": None if result is None else result.get("status"),
                         "max_updates": config["execution"]["max_updates"],
                         "stop_after": config["execution"]["stop_after"],
                         "save_every": config["runtime"]["save_every"], "latest_tag": latest,
                         "latest_has_matching_receipt": any(c["tag"] == latest and c["complete_receipt_exists"]
                             and all(c.get("receipt_checks", {}).values()) for c in checkpoints),
                         "checkpoints": checkpoints,
                         "retained_checkpoint_stat_bytes": sum(c["stat_charged_bytes"] for c in checkpoints)})
    fs = os.statvfs(root)
    return {"schema": "st_nine_run_checkpoint_stat_review_v1", "checked_at_unix": time.time(),
            "runs": rows, "filesystem_free_bytes": fs.f_bavail * fs.f_frsize,
            "checkpoint_payloads_opened": 0, "payload_hashes_revalidated": False,
            "scope": "Receipt content/hash, native inventory and stat sizes only; not loadability or live process verification",
            "suggested_remaining_checkpoint_reserve_bytes": 1983 * 2**30,
            "reservation_basis": "Retain all original 4*240 +187 +4*209 GiB commitments; do not subtract existing files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect(args.root), sort_keys=True, indent=2) + "\n")
