"""Operational recovery tests; no changes to the scientific training code."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.il_report import health


class HealthTests(unittest.TestCase):
    def journal(self, directory, updates):
        p = Path(directory)/'metrics_rank0.jsonl'
        p.write_text(''.join(json.dumps(dict(updates=n, seconds=2.5))+'\n' for n in updates))
        return p

    def test_archive_rolls_back_only_uncommitted_suffix_and_retains_physical_cost(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.journal(d, range(1, 6))
            row = health.rewind_journal(p, 3, Path(d)/'aborted')
            self.assertEqual(row['aborted_updates'], 2)
            self.assertEqual(row['physical_seconds'], 5)
            self.assertEqual([json.loads(r)['updates'] for r in p.read_text().splitlines()], [1, 2, 3])
            self.assertEqual(health.rewind_journal(p, 3, Path(d)/'aborted'), row)

    def test_duplicate_or_hole_refused_without_mutation(self):
        for updates in ([1, 2, 2, 3], [1, 3, 4]):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as d:
                p = self.journal(d, updates); before = p.read_bytes()
                with self.assertRaises(ValueError):
                    health.rewind_journal(p, 2, Path(d)/'aborted')
                self.assertEqual(before, p.read_bytes())

    def test_missing_native_boundary_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.journal(d, range(1, 4)); before = p.read_bytes()
            with self.assertRaises(ValueError):
                health.rewind_journal(p, 10, Path(d)/'aborted')
            self.assertEqual(before, p.read_bytes())

    def test_archived_evidence_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.journal(d, range(1, 6))
            row = health.rewind_journal(p, 3, Path(d)/'aborted')
            Path(row['aborted_file']).write_text('bad evidence')
            with self.assertRaises(ValueError):
                health.rewind_journal(p, 3, Path(d)/'aborted')

    def test_unreviewed_recovery_refuses_before_learner_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(health.subprocess, 'Popen') as launch:
            with self.assertRaises(PermissionError):
                health.recover_huber(Path(d))
            launch.assert_not_called()

    def test_health_does_not_restart_stopped_or_completed_experiments(self):
        runs = [dict(id='RAW', status='已完成', updates=100000),
                dict(id='PHYSICAL', status='筛选后未继续', updates=10000),
                dict(id='DELTA', status='筛选后未继续', updates=10000),
                dict(id='ARCH_A', status='训练更新中', updates=13000)]
        with tempfile.TemporaryDirectory() as d, patch.object(health.subprocess, 'Popen') as launch:
            value = health.check(Path(d), runs)
            launch.assert_not_called()
            self.assertTrue(all(r['action'] == 'OBSERVE' for r in value['runs']))
            self.assertTrue(all(r['intentional_stop_preserved'] for r in value['runs'][:3]))

    def test_unknown_fault_is_review_required_not_unbounded_retry(self):
        with tempfile.TemporaryDirectory() as d, patch.object(health.subprocess, 'Popen') as launch:
            value = health.check(Path(d), [dict(id='ARCH_D', status='训练故障 / 待恢复',
                                                updates=7000, worker_status='FAILED')])
            self.assertEqual(value['runs'][0]['action'], 'REVIEW_REQUIRED')
            launch.assert_not_called()

    def test_dead_or_reused_owner_is_not_live(self):
        self.assertFalse(health.alive(None))
        self.assertFalse(health.alive(dict(pid=999999999, start_ticks=0)))

    def test_fresh_gpu_journal_does_not_hide_dead_refresh_controller(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)/'assets'; root = Path(d)/'report'
            lock = base/'fresh_loss_IL_20261029/TIME_RESIDUAL/owner.lock'
            lock.parent.mkdir(parents=True); lock.touch()
            with patch.object(health, 'BASE', base):
                result = health.check(root, [dict(id='TIME_RESIDUAL', updates=33000,
                    status='训练更新中', worker_status='RUNNING', journal_age_seconds=1)])
            self.assertFalse(result['runs'][0]['controller_owner_lock_held'])
            self.assertEqual(result['runs'][0]['action'], 'REVIEW_REQUIRED')


if __name__ == '__main__':
    unittest.main()
