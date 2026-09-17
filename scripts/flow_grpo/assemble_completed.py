"""Publish one SHA-verified model while the other release model is downloading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from download_status import root, rpc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("variant", choices=["frozen", "unfrozen"])
    args = parser.parse_args()
    complete = {
        Path(e["files"][0]["path"]).name
        for e in rpc("aria2.tellStopped", [0, 100])
        if e["status"] == "complete"
    }

    def sums(name):
        return {
            filename.lstrip("*"): digest
            for digest, filename in (
                line.split(maxsplit=1)
                for line in (root / name).read_text().splitlines()
            )
        }

    expected = sums("SHA256SUMS.reconstructed")
    parts = sums("SHA256SUMS.parts")
    name = next(
        n for n in expected if n.startswith("action-only-" + args.variant + "-")
    )
    target = root / name
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(str(target) + ".ready-reconstructing")
    if temporary.exists():
        raise FileExistsError(temporary)
    paths = []
    for part in sorted(n for n in parts if n.startswith(name + ".part-")):
        path = root / part
        if not path.exists():
            if part + ".partial" not in complete:
                raise ValueError("part not checksum-complete: " + part)
            path = Path(str(path) + ".partial")
        paths.append((part, path))
    whole = hashlib.sha256()
    with temporary.open("xb") as output:
        for part, path in paths:
            checksum = hashlib.sha256()
            with path.open("rb") as source:
                for data in iter(lambda: source.read(16 * 1024**2), b""):
                    output.write(data)
                    checksum.update(data)
                    whole.update(data)
            if checksum.hexdigest() != parts[part]:
                raise ValueError("part SHA mismatch: " + part)
            print("VERIFIED", part, flush=True)
    if whole.hexdigest() != expected[name]:
        raise ValueError("reconstructed SHA mismatch")
    os.link(
        temporary, target
    )  # Atomic publication without overwriting an existing checkpoint.
    temporary.unlink()
    record = {
        "model": str(target),
        "sha256": whole.hexdigest(),
        "bytes": target.stat().st_size,
        "status": "VERIFIED",
    }
    (root / (name + ".verified.json")).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
