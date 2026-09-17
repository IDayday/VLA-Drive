"""Download immutable release parts with bounded, independently verified HTTP ranges.

The durable .partial file is append-only. A failed curl attempt can only replace
its disposable .chunk, avoiding curl's built-in retry truncation of large files.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import os
import secrets
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path("artifacts/action-only-checkpoints-v1")
URL = "https://api.github.com/repos/IDayday/VLA-Drive/releases/tags/action-only-checkpoints-v1"
CHUNK_BYTES = 16 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksums(filename):
    return {
        name.lstrip("*"): expected
        for expected, name in (
            line.split(maxsplit=1)
            for line in (ROOT / filename).read_text().splitlines()
        )
    }


def verify_file(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}: {actual}")
    print("VERIFIED", path.name, actual, flush=True)


def download(asset, expected):
    target = ROOT / asset["name"]
    if target.exists():
        if target.stat().st_size != asset["size"]:
            raise IOError("unexpected existing size: " + str(target))
        verify_file(target, expected)
        return
    temporary = Path(str(target) + ".partial")
    chunk = Path(str(target) + ".chunk")
    headers = Path(str(target) + ".headers")
    failures = 0
    while (temporary.stat().st_size if temporary.exists() else 0) < asset["size"]:
        start = temporary.stat().st_size if temporary.exists() else 0
        end = min(start + CHUNK_BYTES, asset["size"]) - 1
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--http1.1",
                "--retry",
                "0",
                "--speed-time",
                "45",
                "--speed-limit",
                "10000",
                "--max-time",
                "180",
                "--connect-timeout",
                "20",
                "--silent",
                "--show-error",
                "--range",
                f"{start}-{end}",
                "--dump-header",
                str(headers),
                "--output",
                str(chunk),
                asset["browser_download_url"],
            ]
        )
        ranges = re.findall(
            r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
            headers.read_text(errors="replace") if headers.exists() else "",
        )
        valid = (
            result.returncode == 0
            and ranges
            and tuple(map(int, ranges[-1])) == (start, end, asset["size"])
            and chunk.stat().st_size == end - start + 1
        )
        if not valid:
            failures += 1
            print(
                "RETRY",
                target.name,
                "durable_bytes",
                start,
                "attempt",
                failures,
                "curl",
                result.returncode,
                "range",
                ranges[-1:] or None,
                flush=True,
            )
            if failures >= 40:
                raise IOError("repeated range download failure: " + str(target))
            time.sleep(min(failures * 2, 15))
            continue
        with temporary.open("ab") as output, chunk.open("rb") as source:
            shutil.copyfileobj(source, output)
            output.flush()
        chunk.unlink()
        failures = 0
        print("PROGRESS", target.name, end + 1, "/", asset["size"], flush=True)
    if temporary.stat().st_size != asset["size"]:
        raise IOError("unexpected partial size: " + str(temporary))
    verify_file(temporary, expected)
    temporary.replace(target)


def aria2_download(parts, expected):
    inputs = []
    for asset in parts:
        target = ROOT / asset["name"]
        if target.exists():
            actual = sha(target)
            if actual == expected[asset["name"]]:
                print("VERIFIED", target.name, actual, flush=True)
                continue
            quarantine = Path(str(target) + ".invalid-" + actual[:12])
            target.replace(quarantine)
            print("QUARANTINED", target.name, "checksum mismatch", flush=True)
        inputs.extend(
            [
                asset["browser_download_url"],
                " dir=" + str(ROOT.resolve()),
                " out=" + asset["name"] + ".partial",
                " checksum=sha-256=" + expected[asset["name"]],
            ]
        )
    if not inputs:
        return
    input_path = ROOT / "aria2-input.txt"
    input_path.write_text("\n".join(inputs) + "\n")
    secret_path = ROOT / "aria2-rpc-secret"
    if not secret_path.exists():
        secret_path.write_text(secrets.token_hex(24))
        secret_path.chmod(0o600)
    conf = ROOT / "aria2.conf"
    conf.write_text(
        "\n".join(
            [
                "continue=true",
                "check-integrity=true",
                "auto-file-renaming=false",
                "allow-overwrite=false",
                "file-allocation=none",
                "max-connection-per-server=8",
                "split=8",
                "max-concurrent-downloads=6",
                "min-split-size=8M",
                "max-tries=100",
                "retry-wait=2",
                "timeout=30",
                "connect-timeout=10",
                "lowest-speed-limit=0",
                "disk-cache=64M",
                "summary-interval=15",
                "console-log-level=notice",
                "download-result=full",
                "enable-rpc=true",
                "rpc-listen-all=false",
                "rpc-listen-port=16800",
                "rpc-secret=" + secret_path.read_text(),
                "all-proxy="
                + os.environ.get("https_proxy", os.environ.get("HTTPS_PROXY", "")),
            ]
        )
        + "\n"
    )
    conf.chmod(0o600)
    for attempt in range(12):
        process = subprocess.Popen(
            ["aria2c", "--conf-path=" + str(conf), "--input-file=" + str(input_path)]
        )
        from download_status import rpc

        while process.poll() is None:
            time.sleep(2)
            try:
                active = rpc("aria2.tellActive")
                waiting = rpc("aria2.tellWaiting", [0, 100])
                stopped = rpc("aria2.tellStopped", [0, 100])
                if stopped and not active and not waiting:
                    rpc("aria2.shutdown")
                    break
            except OSError:
                pass  # RPC may not yet be listening or may already have exited.
        result_code = process.wait()
        if result_code == 0:
            break
        print("ARIA2_RETRY", attempt + 1, "exit", result_code, flush=True)
        time.sleep(5)
    else:
        raise RuntimeError(
            "aria2 retries exhausted; preserve .aria2 controls to resume"
        )
    for asset in parts:
        target = ROOT / asset["name"]
        if not target.exists():
            partial = Path(str(target) + ".partial")
            if partial.stat().st_size != asset["size"]:
                raise IOError("aria2 incomplete file: " + str(partial))
            verify_file(partial, expected[asset["name"]])
            partial.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["aria2", "curl"], default="aria2")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "release.json").exists():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--silent",
                "--show-error",
                URL,
                "-o",
                str(ROOT / "release.json"),
            ],
            check=True,
        )
    data = json.loads((ROOT / "release.json").read_text())
    for name in ("SHA256SUMS.parts", "SHA256SUMS.reconstructed", "CHECKPOINTS.md"):
        if not (ROOT / name).exists():
            asset = next(a for a in data["assets"] if a["name"] == name)
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "4",
                    "--silent",
                    "--show-error",
                    asset["browser_download_url"],
                    "-o",
                    str(ROOT / name),
                ],
                check=True,
            )
    expected = checksums("SHA256SUMS.parts")
    parts = [a for a in data["assets"] if a["name"] in expected]
    if len(parts) != len(expected):
        raise ValueError("release assets and checksum manifest disagree")
    if args.backend == "aria2":
        aria2_download(parts, expected)
    else:
        if any(ROOT.glob("*.aria2")):
            raise RuntimeError(
                "aria2 control files present; resume with --backend aria2"
            )
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(
                pool.map(lambda asset: download(asset, expected[asset["name"]]), parts)
            )
    for name, digest in checksums("SHA256SUMS.reconstructed").items():
        target = ROOT / name
        if not target.exists():
            temporary = Path(str(target) + ".reconstructing")
            with temporary.open("wb") as output:
                for part_name in sorted(
                    n for n in expected if n.startswith(name + ".part-")
                ):
                    with (ROOT / part_name).open("rb") as source:
                        shutil.copyfileobj(source, output, 16 * 1024**2)
            verify_file(temporary, digest)
            temporary.replace(target)
        else:
            verify_file(target, digest)


if __name__ == "__main__":
    main()
