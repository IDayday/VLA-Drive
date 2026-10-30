from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[2]/'tools/qwen_world_compare/delayed_eval_hosts.py'
spec = importlib.util.spec_from_file_location('delayed_eval_hosts', MODULE)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class DelayedHostTests(unittest.TestCase):
    def setUp(self):
        self.host = 'training-vlawm-zt2'
        self.request = dict(requested_at_unix=100, not_before_unix=700)
        self.policy = dict(allowed_hosts={'training-vla-zt': {}}, allowed_aliases=['training-vla-zt'],
                           task_authorizations={t:dict(record='/old/grant', permanent_hosts=['training-vla-zt'],
                           temporary_hosts=[], unbounded_temporary_hosts=[], allowed_hosts=['training-vla-zt'],
                           starts_unix=1, expires_unix=1, training_run_overrides={'stopped':{'training_allowed':False}})
                           for t in operator.TASKS},
                           global_dispatch_prohibitions={self.host:dict(new_task_dispatch_allowed=False, utc='1970-01-01T00:00:50+00:00'),
                           'training-vla-zt3':dict(new_task_dispatch_allowed=False, utc='1970-01-01T00:01:30+00:00')})

    def test_no_early_grant(self):
        with self.assertRaises(PermissionError):
            operator.permit(self.policy, self.host, self.request, Path('/new'), 699.999)

    def test_newer_withdrawal_wins(self):
        self.policy['global_dispatch_prohibitions'][self.host]['utc']='1970-01-01T00:01:41+00:00'
        with self.assertRaises(PermissionError):
            operator.permit(self.policy, self.host, self.request, Path('/new'), 701)

    def test_addition_retains_other_hosts_stopped_jobs_and_immutable_ancestry(self):
        original = deepcopy(self.policy)
        changed, grants = operator.permit(self.policy, self.host, self.request, Path('/new'), 700)
        self.assertEqual(self.policy, original)
        self.assertFalse(changed['global_dispatch_prohibitions']['training-vla-zt3']['new_task_dispatch_allowed'])
        self.assertNotIn('training-vla-zt3', changed['allowed_hosts'])
        self.assertNotIn('training-vla-zt2', changed['allowed_hosts'])
        for task, grant in grants.items():
            self.assertEqual(grant['previous_authorization_record'], '/old/grant')
            self.assertFalse(grant['training_run_overrides']['stopped']['training_allowed'])
            field = 'allowed_hosts' if task == operator.TASKS[0] else 'unbounded_temporary_hosts'
            self.assertIn(self.host, grant[field])
            if task != operator.TASKS[0]:
                self.assertIn(self.host, grant['shared_training_hosts'])
            self.assertEqual(grant['starts_unix'], 1)
            self.assertEqual(grant['permanent_hosts'], ['training-vla-zt'])
            self.assertEqual(changed['task_authorizations'][task], grant)

    def test_idempotent_host_list_and_second_host_keeps_first(self):
        changed, _ = operator.permit(self.policy, self.host, self.request, Path('/new'), 700)
        changed, _ = operator.permit(changed, 'training-vlawm-zt', self.request, Path('/new'), 701)
        changed, grants = operator.permit(changed, self.host, self.request, Path('/new'), 702)
        self.assertEqual(changed['allowed_aliases'].count(self.host), 1)
        for task, grant in grants.items():
            field = 'allowed_hosts' if task == operator.TASKS[0] else 'unbounded_temporary_hosts'
            self.assertEqual(grant[field].count(self.host), 1)
            self.assertIn('training-vlawm-zt', grant[field])


if __name__ == '__main__':
    unittest.main()
