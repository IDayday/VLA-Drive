"""CPU admission: authorization is necessary, not an exclusive resource lease."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.structured_world import benchmark_engineering_execution as bench


class ReservationTests(unittest.TestCase):
    def authorize(self, policy):
        with patch.object(bench, 'read', return_value=policy), patch.object(bench, 'digest', return_value='policy-hash'):
            return bench.authorize('host')

    def test_allowed_and_owned_reservations(self):
        policy = {'task_authorizations': {bench.TASK: {'allowed_hosts': ['host']}}}
        self.assertEqual(self.authorize(policy)['host_reservation'], {})
        policy['host_reservations'] = {'host': {'task': bench.TASK, 'exclude_new_other_workloads': True}}
        self.assertEqual(self.authorize(policy)['host_reservation']['task'], bench.TASK)

    def test_other_task_and_global_withdrawal_reject_before_GPU(self):
        base = {'task_authorizations': {bench.TASK: {'allowed_hosts': ['host']}}}
        for extra in ({'global_dispatch_prohibitions': {'host': {'new_task_dispatch_allowed': False}}},
                      {'host_reservations': {'host': {'task': 'iqe_v1', 'exclude_new_other_workloads': True}}}):
            policy = {**copy.deepcopy(base), **extra}
            with self.assertRaises(ValueError): self.authorize(policy)
        with self.assertRaises(ValueError):
            self.authorize({'task_authorizations': {bench.TASK: {'allowed_hosts': []}}})


if __name__ == '__main__':
    unittest.main(verbosity=2)
