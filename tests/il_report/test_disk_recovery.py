import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import Mock

from tools.il_report.disk_recovery import rewind_at_boundary,eligible_sample,prepare


class DiskRecoveryTests(unittest.TestCase):
    def test_shared_recovery_only_admits_visible_registered_evaluation_owners(self):
        sample=dict(gpus=[dict(free_MiB=70000)for _ in range(8)],owners=[dict(visible=True,output='/mnt/project/qwen-world-postcot-20261025/periodic_NAVTEST_TIME/evaluation')])
        self.assertTrue(eligible_sample(sample,65000))
        sample['owners'][0]['visible']=False;self.assertFalse(eligible_sample(sample,65000))
        sample['owners'][0].update(visible=True,output='/unrelated/task');self.assertFalse(eligible_sample(sample,65000))

    def test_user_terminated_run_rejected_before_remote_or_native_changes(self):
        with tempfile.TemporaryDirectory()as d,ExitStack()as stack:
            root=Path(d);run=root/'runs/IL_full_seed42';run.mkdir(parents=True)
            (root/'USER_TERMINATION.json').write_text(json.dumps(dict(schema='experiment_user_termination_v1',status='TERMINATED_BY_USER',run=str(run))))
            remote=Mock()
            with self.assertRaises(PermissionError):prepare(run,root,{},root/'receipt',remote,stack)
            remote.assert_not_called()

    def test_truncated_uncommitted_bytes_preserved_and_valid_boundary_retained(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'metrics.jsonl';prefix=b'{"updates":1}\n{"updates":2}\n';tail=b'{"updates":3}\n{"updates":4'
            p.write_bytes(prefix+tail);r=rewind_at_boundary(p,2,Path(d)/'archive')
            self.assertEqual(p.read_bytes(),prefix);self.assertEqual(Path(r['aborted_file']).read_bytes(),tail)

    def test_corruption_before_commit_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'metrics.jsonl';p.write_bytes(b'{"updates":1}\nbad\n{"updates":3}\n')
            with self.assertRaises(ValueError):rewind_at_boundary(p,3,Path(d)/'archive')
            self.assertIn(b'bad',p.read_bytes())

    def test_valid_journal_uses_existing_receipt_and_is_idempotent(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'metrics.jsonl';p.write_text(''.join(json.dumps(dict(updates=i,seconds=2))+'\n'for i in range(1,5)))
            archive=Path(d)/'archive';first=rewind_at_boundary(p,2,archive)
            self.assertEqual(first,rewind_at_boundary(p,2,archive))
