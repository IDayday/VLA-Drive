"""Read real aria2 completed bytes (sparse file size is not download progress)."""
import json
from pathlib import Path
import urllib.request
import urllib.error

root = Path("artifacts/action-only-checkpoints-v1")
secret_path = root / "aria2-rpc-secret"
secret = secret_path.read_text() if secret_path.exists() else ""
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def rpc(method, params=None):
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "status",
            "method": method,
            "params": ["token:" + secret] + (params or []),
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:16800/jsonrpc",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    result = json.load(opener.open(request, timeout=5))
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["result"]


def main():
    try:
        stats = rpc("aria2.getGlobalStat")
    except (urllib.error.URLError, TimeoutError):
        from download_action_only import checksums, sha

        rows = []
        for name, expected in checksums("SHA256SUMS.reconstructed").items():
            path = root / name
            actual = sha(path) if path.is_file() else None
            rows.append(
                {
                    "file": name,
                    "status": "complete_verified"
                    if actual == expected
                    else "incomplete_or_invalid",
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
        print(json.dumps({"downloader_running": False, "files": rows}, indent=2))
        return
    entries = (
        rpc("aria2.tellActive")
        + rpc("aria2.tellWaiting", [0, 100])
        + rpc("aria2.tellStopped", [0, 100])
    )
    rows = []
    for entry in entries:
        rows.append(
            {
                "file": Path(entry["files"][0]["path"]).name,
                "completed_bytes": int(entry["completedLength"]),
                "total_bytes": int(entry["totalLength"]),
                "status": entry["status"],
                "bytes_per_second": int(entry["downloadSpeed"]),
                "connections": int(entry.get("connections", 0)),
                "error": entry.get("errorMessage"),
            }
        )
    print(
        json.dumps(
            {"bytes_per_second": int(stats["downloadSpeed"]), "files": rows}, indent=2
        )
    )


if __name__ == "__main__":
    main()
