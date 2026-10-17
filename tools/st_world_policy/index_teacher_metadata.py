"""CPU-only metadata preparation; never instantiates a teacher or opens tensors."""
import argparse
import json
from pathlib import Path

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, require_hash
from starVLA.dataloader.st_world_metadata_index import build_index, MetadataIndex, MiB, GiB


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='Index the exact complete declared manifest; no subsetting')
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache-root', type=Path)
    source.add_argument('--requests', type=Path)
    build.add_argument('--expected-cache-identity')
    build.add_argument('--expected-request-identity', required=True)
    build.add_argument('--contract-receipt', type=Path, required=True,
                       help='Small existing teacher cache identity or qualification plan with top-level contract')
    build.add_argument('--contract-receipt-sha256', required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--max-scenes', type=int, required=True, help='Hard population limit, not sampling')
    build.add_argument('--max-disk-bytes', type=int, default=2*GiB)
    build.add_argument('--max-builder-rss-bytes', type=int, default=4*GiB)
    verify = sub.add_parser('verify', help='Verify source/index hashes and every metadata row, no tensor reads')
    verify.add_argument('--index', type=Path, required=True)
    verify.add_argument('--expected-request-identity', required=True)
    verify.add_argument('--expected-index-identity', required=True)
    verify.add_argument('--max-metadata-cache-bytes', type=int, default=16*MiB)
    return p


def run(args):
    if args.command == 'build':
        if bool(args.cache_root) != bool(args.expected_cache_identity):
            raise ValueError('A cache root and its original COMPLETE identity are required together')
        require_hash(args.contract_receipt_sha256, 'contract receipt hash')
        if args.contract_receipt.stat().st_size > MiB:
            raise ValueError('Contract receipt must be a small pinned existing artifact')
        if file_sha256(args.contract_receipt) != args.contract_receipt_sha256:
            raise ValueError('Contract receipt checksum mismatch')
        receipt = json.loads(args.contract_receipt.read_text())
        if receipt.get('schema') not in ('st_teacher_cache_v1', 'st_fixed_teacher_qualification_plan_v1',
                                          'st_ondemand_qualification_plan_v1'):
            raise ValueError('Unrecognized teacher contract provenance receipt')
        contract = TeacherFeatureContract.from_dict(receipt['contract'])
        d = build_index(args.output, requests=args.cache_root/'requests.json' if args.cache_root else args.requests,
            expected_request_identity=args.expected_request_identity, contract=contract, cache_root=args.cache_root,
            expected_cache_identity=args.expected_cache_identity, max_scenes=args.max_scenes,
            max_disk_bytes=args.max_disk_bytes, max_builder_rss_bytes=args.max_builder_rss_bytes)
        return {k: d[k] for k in ('identity','request_identity','cache_identity','scene_count','record_count',
            'database_bytes','builder_peak_rss_bytes','tensor_files_opened','image_files_opened','tensor_disk_bytes')}
    with MetadataIndex(args.index, expected_request_identity=args.expected_request_identity,
                       expected_index_identity=args.expected_index_identity,
                       max_metadata_cache_bytes=args.max_metadata_cache_bytes) as index:
        # Stream the original row order. All per-row hashes are checked by __getitem__.
        for rows in (index.manifest['scenes'], index.records):
            for _ in rows:
                pass
        for key in index.receipt_hashes:
            index.receipt_hashes[key]
        return {'status': 'METADATA_VERIFIED', 'identity': index.receipt['identity'],
            'scenes': len(index.manifest['scenes']), 'records': len(index.records),
            'image_files_opened': 0, 'tensor_files_opened': 0, 'training_admission': False}


def main():
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
