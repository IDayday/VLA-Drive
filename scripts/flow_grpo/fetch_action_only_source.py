"""Fetch only reviewed, SHA256-locked files from the user's exact source commit."""
import hashlib
import json
from pathlib import Path
import subprocess


def fetch():
    lock = json.loads(Path("reference_lock.json").read_text())
    source = next(
        r
        for r in lock["references"]
        if r["url"] == "https://github.com/IDayday/VLA-Drive"
    )
    root = Path("artifacts/action-only-source")
    for name, expected in source["reviewed_files"].items():
        path = root / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("modified reference file: " + str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + ".partial")
        url = (
            "https://raw.githubusercontent.com/IDayday/VLA-Drive/"
            + source["sha"]
            + "/"
            + name
        )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--max-time",
                "60",
                "--silent",
                "--show-error",
                url,
                "--output",
                str(temporary),
            ],
            check=True,
        )
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError("reference SHA256 mismatch: " + name)
        temporary.replace(path)
        print("VERIFIED", name, flush=True)


if __name__ == "__main__":
    fetch()
