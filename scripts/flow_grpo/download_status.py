"""Read real aria2 completed bytes (sparse file size is not download progress)."""
import json
from pathlib import Path
import urllib.request

root = Path("artifacts/action-only-checkpoints-v1")
secret = (root / "aria2-rpc-secret").read_text()
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


if __name__ == "__main__":
    stats = rpc("aria2.getGlobalStat")
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
