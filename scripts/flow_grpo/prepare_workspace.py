"""Idempotent local asset links; inherited user source changes are now committed."""
from pathlib import Path
import argparse
import json
import subprocess


def link_existing(source, target):
    source, target = Path(source).resolve(), Path(target)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(f"refusing to replace existing {target}")
    else:
        target.symlink_to(source, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root", default="/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts"
    )
    parser.add_argument(
        "--model-root", default="/mnt/project/DriveDreamer-Policy/models"
    )
    args = parser.parse_args()
    result = subprocess.run(
        [
            "git",
            "apply",
            "--reverse",
            "--check",
            "reports/ddp_flow_grpo/inherited_dirty.patch",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "inherited user source snapshot differs; do not blindly reapply the historical patch: "
            + result.stderr
        )
    link_existing(args.asset_root, "artifacts")
    link_existing(args.model_root, "models")
    print(
        json.dumps(
            {
                "status": "PASS",
                "inherited_source": "already committed in 4acd8e7; historical user contribution; patch not reapplied",
                "asset_root": str(Path("artifacts").resolve()),
                "model_root": str(Path("models").resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
