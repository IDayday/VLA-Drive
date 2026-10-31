import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.il_report.lifecycle import user_termination
from tools.il_report import navtest_cadence as cadence
from tools.il_report import evidence as evidence


class LifecycleTests(unittest.TestCase):
    def marker(self,root):
        run=root/'runs/IL_full_seed42';run.mkdir(parents=True)
        (root/'USER_TERMINATION.json').write_text(json.dumps(dict(schema='experiment_user_termination_v1',status='TERMINATED_BY_USER',run=str(run))))
        return run

    def test_explicit_stop_overrides_fresh_journals_without_marking_budget_complete(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);run=self.marker(root)
            for i in range(8):(run/f'metrics_rank{i}.jsonl').write_text('{"updates":68084,"optimizer_global_steps":68084}\n')
            row=evidence.journal_snapshot(evidence.Evidence(),'V12',run,{})
            self.assertEqual(row['status'],'用户已终止');self.assertEqual(row['updates'],68084)
            self.assertIsNone(row['estimated_remaining_hours'])

    def test_missing_marker_keeps_active_run_eligible(self):
        with tempfile.TemporaryDirectory()as d:self.assertIsNone(user_termination(Path(d)/'runs/IL_full_seed42'))

    def test_misbound_marker_is_refused(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);run=self.marker(root);(root/'USER_TERMINATION.json').write_text(json.dumps(dict(schema='experiment_user_termination_v1',status='TERMINATED_BY_USER',run='/wrong/run')))
            with self.assertRaises(ValueError):user_termination(run)

    def test_closed_v12_excluded_without_removing_other_architectures(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);old=root/'old';old.mkdir();four=root/'four';four.mkdir()
            self.marker(old/'optimized_labels_IL_20261028')
            with patch.object(cadence,'OLD',old),patch.object(cadence,'FOUR',four):
                active=cadence.runs()
            self.assertNotIn('V12',active);self.assertTrue(all('ARCH_'+a in active for a in'ABCD'))
