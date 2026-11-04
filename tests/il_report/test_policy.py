from copy import deepcopy
import unittest

from tools.il_report.policy import authorized_host_rows
from tools.il_report.resource_review import permission_signature


class PolicyTests(unittest.TestCase):
    def test_legacy_mapping_keeps_declared_order_and_metadata(self):
        policy = {'allowed_hosts': {'first': {}, 'second': {'temporary': True,
                  'expires_unix': 200}}}
        self.assertEqual(list(authorized_host_rows(policy, now=100)), ['first', 'second'])
        self.assertEqual(authorized_host_rows(policy, now=100)['second']['expires_unix'], 200)

    def test_current_list_never_expands_from_metadata_or_aliases(self):
        hosts = ['training-vlawm-tsinghua-zt3', 'training-vlawm-zt',
                 'training-vlawm-tsinghua-zt5', 'training-vlawm-tsinghua-zt',
                 'training-vlawm-tsinghua-zt2']
        policy = {'allowed_hosts': hosts, 'hosts': {h: {'allowed': True} for h in hosts}}
        policy['hosts']['training-vla-zt'] = {'allowed': True}
        before = deepcopy(policy)
        self.assertEqual(list(authorized_host_rows(policy)), hosts)
        self.assertEqual(policy, before)

    def test_withdrawal_and_expiry_are_applied_to_both_schemas(self):
        metadata = {'withdrawn': {'allowed': False}, 'disabled': {'new_task_dispatch_allowed': False},
                    'denied': {}, 'expired': {'temporary': True, 'expires_unix': 99},
                    'active': {'temporary': True, 'expires_unix': 101},
                    'unbounded': {'temporary': True, 'unbounded': True}}
        for declared in (metadata, list(metadata)):
            with self.subTest(schema=type(declared).__name__):
                policy = {'allowed_hosts': declared, 'hosts': metadata,
                          'global_dispatch_prohibitions': {'denied': {'new_task_dispatch_allowed': False}}}
                self.assertEqual(list(authorized_host_rows(policy, now=100)), ['active', 'unbounded'])

    def test_current_metadata_can_withdraw_legacy_entry(self):
        policy = {'allowed_hosts': {'old': {'allowed': True}},
                  'hosts': {'old': {'allowed': False}}}
        self.assertEqual(authorized_host_rows(policy), {})

    def test_unknown_allowlist_shape_is_refused(self):
        for declared in ('host', [None], [3], {'host': False}):
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                authorized_host_rows({'allowed_hosts': declared})

    def test_metadata_authority_change_invalidates_cached_inventory(self):
        before = {'allowed_hosts': ['host'], 'hosts': {'host': {'allowed': True}}}
        after = deepcopy(before)
        after['hosts']['host']['allowed'] = False
        self.assertNotEqual(permission_signature(before), permission_signature(after))


if __name__ == '__main__':
    unittest.main()
