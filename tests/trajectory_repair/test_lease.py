import unittest
from starVLA.rl.trajectory_repair.lease import Lease


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.lease = Lease(('fixed',), ('temporary',), 1000, 29800, 30700, 31600, 33340, 33400)

    def test_no_start_before_grant(self):
        with self.assertRaises(PermissionError):
            self.lease.require_training_time('temporary', 999)

    def test_finite_admission_before_drain(self):
        self.lease.require_training_time('temporary', 2000, 60)
        with self.assertRaises(PermissionError):
            self.lease.require_training_time('temporary', 2000, 30000)

    def test_absolute_boundaries(self):
        for now, prefix in [(29800, 'NO_NEW_ADMISSION'), (30700, 'COMMIT_NATIVE'),
                            (31600, 'MIGRATE_TO_FIXED'), (33340, 'TEMPORARY_WORKERS'),
                            (33400, 'EXPIRED_NO_USE'), (100000, 'EXPIRED_NO_USE')]:
            with self.subTest(now=now):
                self.assertTrue(self.lease.phase('temporary', now).startswith(prefix))
                with self.assertRaises(PermissionError):
                    self.lease.require_training_time('temporary', now)

    def test_only_fixed_migration_destination(self):
        self.lease.require_migration_destination('fixed')
        with self.assertRaises(PermissionError):
            self.lease.require_migration_destination('temporary')

    def test_unlisted_host_refused(self):
        with self.assertRaises(PermissionError):
            self.lease.phase('unlisted', 2000)

    def test_fixed_host_not_constrained_by_temporary_clock(self):
        self.lease.require_training_time('fixed', 100000)

    def test_nan_clock_cannot_reopen_expired_lease(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.lease.require_training_time('temporary', value)

    def test_nan_checkpoint_latency_is_invalid(self):
        with self.assertRaises(ValueError):
            self.lease.require_training_time('temporary', 2000, float('nan'))


if __name__ == '__main__':
    unittest.main()
