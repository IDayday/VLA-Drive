"""Prove source/archive versus installed-package equality without changing either.

The installed nuPlan wheel omits test directories. A whole-tree digest thus
differs even when every installed Python file is byte-identical. This narrow
receipt permits only that case; it never excuses a differing runtime file.
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, identity, read, source_sha


def tree_manifest(root):
    root = Path(root)
    manifest, tree = {}, hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        name, data = str(path.relative_to(root)), path.read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        tree.update(name.encode() + b'\0' + data)
    if not manifest:
        raise ValueError('Empty nuPlan source tree')
    return manifest, tree.hexdigest()


def compare(reference, installed):
    old, old_sha = tree_manifest(reference)
    new, new_sha = tree_manifest(installed)
    if any(old.get(name) != value for name, value in new.items()):
        raise ValueError('Installed runtime file differs from reference')
    omitted = sorted(old.keys() - new.keys())
    if any(not {'test', 'tests'}.intersection(Path(name).parts) for name in omitted):
        raise ValueError('A non-test reference file is missing from the installed package')
    return dict(schema='nuplan_test_only_packaging_difference_v1', passed=True,
        reference_root=str(reference), installed_root=str(installed),
        reference_tree_sha256=old_sha, installed_tree_sha256=new_sha,
        shared_files=len(new), shared_manifest_identity=identity(new),
        differing_shared_files=0, installed_extra_files=0,
        omitted_test_files=omitted, omitted_test_file_count=len(omitted))


def verify_receipt(path, reference_sha, installed_sha):
    receipt = read(path)
    current = compare(receipt['reference_root'], receipt['installed_root'])
    if (any(receipt.get(k) != v for k, v in current.items()) or
            current['reference_tree_sha256'] != reference_sha or
            current['installed_tree_sha256'] != installed_sha):
        raise ValueError('Source-equivalence receipt does not match the actual banks/trees')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'installed', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the earlier source audit')
    report = compare(args.reference, args.installed)
    report.update(resources=check_host(), report_source_sha=source_sha(ROOT))
    atomic_json(args.output, report)
    print({k: v for k, v in report.items() if k not in ('omitted_test_files', 'resources')})


if __name__ == '__main__':
    main()
