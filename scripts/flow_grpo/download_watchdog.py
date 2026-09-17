"""Retry failed ranges and refresh persistently slow connections in our aria2 job."""
import argparse
import json
from pathlib import Path
import time
from download_status import rpc, root


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    assets = {
        a["name"]: a for a in json.loads((root / "release.json").read_text())["assets"]
    }
    resets, failures = {}, {}
    unavailable = 0
    while True:
        try:
            active = rpc("aria2.tellActive")
            waiting = rpc("aria2.tellWaiting", [0, 100])
            stopped = rpc("aria2.tellStopped", [0, 100])
            unavailable = 0
            active_names = {Path(e["files"][0]["path"]).name for e in active + waiting}
            for entry in stopped:
                filename = Path(entry["files"][0]["path"]).name
                if entry["status"] != "error" or filename in active_names:
                    continue
                name = filename.removesuffix(".partial")
                asset = assets[name]
                failures[name] = failures.get(name, 0) + 1
                if failures[name] > 12:
                    raise RuntimeError("repeated failure beyond retry budget: " + name)
                options = {
                    "dir": str(root.resolve()),
                    "out": filename,
                    "continue": "true",
                    "check-integrity": "true",
                    "checksum": asset["digest"].replace("sha256:", "sha-256="),
                    "lowest-speed-limit": "0",
                }
                gid = rpc("aria2.addUri", [[asset["browser_download_url"]], options])
                rpc("aria2.removeDownloadResult", [entry["gid"]])
                print(
                    json.dumps(
                        {
                            "event": "retry_failed",
                            "file": filename,
                            "gid": gid,
                            "time": time.time(),
                        }
                    ),
                    flush=True,
                )
            now = time.monotonic()
            for entry in active:
                gid = entry["gid"]
                resets.setdefault(gid, now)
                if int(entry["downloadSpeed"]) >= 100_000 or now - resets[gid] < (
                    0 if args.once else 180
                ):
                    continue
                if (
                    int(entry["completedLength"]) >= int(entry["totalLength"])
                    or int(entry["totalLength"]) == 0
                ):
                    continue  # Never interrupt checksum validation.
                rpc("aria2.pause", [gid])
                for _ in range(50):
                    status = rpc("aria2.tellStatus", [gid, ["status"]])["status"]
                    if status != "pausing":
                        break
                    time.sleep(0.1)
                if status == "paused":
                    rpc("aria2.unpause", [gid])
                    resets[gid] = now
                    print(
                        json.dumps(
                            {
                                "event": "refresh_slow_connection",
                                "gid": gid,
                                "file": Path(entry["files"][0]["path"]).name,
                                "time": time.time(),
                            }
                        ),
                        flush=True,
                    )
        except OSError:
            unavailable += 1
            if unavailable >= 12 or args.once:
                return
        if args.once:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
