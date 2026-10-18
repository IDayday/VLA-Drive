#!/usr/bin/env python3
"""Explicit source-bridged call to the existing exact metadata-v3 migration API."""
import argparse
import json
from pathlib import Path
import sys

from storage_binding import RUNTIME, install

OLD_POLICY_SHA = 'c77dca4739f3026940b8a56821cfd4bd222c643cdf9e4e07e5b9efba21a82163'
NEW_POLICY = RUNTIME / 'reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/metadata_v3/PACKAGE_ONLY_POLICY_PROPOSAL.json'
NEW_POLICY_SHA = '744666144432007438c1030ca8144369de38df521bf34fbe2d4a092913d97b0e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Lead-only explicit live migration; never implied by help/import')
    for name in ('old-policy', 'old-policy-sha256', 'new-policy', 'new-policy-sha256',
                 'expected-old-ledger-sha256', 'quiescence-receipt', 'quiescence-sha256',
                 'pilot-inventory', 'pilot-inventory-sha256', 'checkpoint-receipt', 'checkpoint-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--renew-quiescence-receipt')
    parser.add_argument('--renew-quiescence-sha256')
    args = vars(parser.parse_args())
    if not args.pop('execute'):
        raise ValueError('Explicit execution required; this is not an automatic policy refresh')
    if (args['old_policy_sha256'] != OLD_POLICY_SHA or args['new_policy_sha256'] != NEW_POLICY_SHA or
            Path(args['new_policy']) != NEW_POLICY):
        raise ValueError('Only the reviewed package-only v2→v3 transition is permitted')
    _, receipt = install()
    from tools.st_world_policy import metadata_storage as m
    result = m.migrate_policy(**args)
    print(json.dumps({'migration': result, 'w_source_compatibility': receipt}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
